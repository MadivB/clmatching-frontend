"""PURITY LOSS BUDGET on the B3 chain: attribute every foreign-energy hit in
every final label to the stage that mixed it in.
  ransac  - foreign hit inside a segment whose dominant group = label's group
  stitch  - foreign-dominant RANSAC segment stitched into the label at build
  absorb  - leftover/DBSCAN hit (no segment) absorbed at build
  rejoin  - foreign build-label merged by the endpoint rejoin
  vp      - foreign post-rejoin label merged by vertex pinpointing
Usage: python budget_worker.py <ptname> [...]"""
import copy, inspect, json, os, pickle, sys
from types import SimpleNamespace
import numpy as np
import torch

REL = r'F:\CLMatching_v0.5.6_release'
SCR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REL)
from recluster import truth_groups, _file_load
import vertex_pinpoint as vp

OUT = os.path.join(SCR, 'budget')
os.makedirs(OUT, exist_ok=True)
tb = _file_load('tb_b', os.path.join(REL, 'clustering_v055',
                                     'global_track_clustering_toolbox_v11_2.py'))
sd = _file_load('sd_b', os.path.join(REL, 'stitch_directional.py'))
sp = _file_load('sp_b', os.path.join(REL, 'segment_split.py'))
er = _file_load('er_b', os.path.join(REL, 'endpoint_rejoin.py'))
params = json.load(open(os.path.join(REL, 'params_full.json')))
sig = set(inspect.signature(tb.build_global_labels_toolbox).parameters)
kw_base = {k: v for k, v in params.items() if k in sig}
kw_base['return_label_info'] = True
tb._match_segments_across_tpcs_toolbox = sd.make_matcher(
    SimpleNamespace(**params), tb._fit_line_metrics)

CLASSES = ('ransac', 'stitch', 'absorb', 'rejoin', 'vp')
rows, cases = [], []
for ptname in sys.argv[1:]:
    blob = torch.load(os.path.join(REL, 'pt', ptname + '.pt'), weights_only=False)
    for ev_id, ev in blob['events'].items():
        with open(os.path.join(SCR, 'segs', f'{ptname}_ev{ev_id}.pkl'), 'rb') as f:
            segments = pickle.load(f)
        x = ev['x'].numpy().astype(np.float64)
        y = ev['y'].numpy().astype(np.float64)
        z = ev['z'].numpy().astype(np.float64)
        xyz = np.stack([x, y, z], 1)
        e = np.clip(ev['energy'].numpy().astype(np.float64), 0, None)
        tg = truth_groups(ev['truth_t0'].numpy().astype(np.float64))
        tpcv = ev['tpc'].numpy().astype(int)
        segments, _ = sp.split_segments(segments, xyz, SimpleNamespace(**params))
        tb._build_tpc_segments_toolbox = lambda *a, **k: (segments, {})
        labels0, si, li0 = tb.build_global_labels_toolbox(
            x, y, z, ev['io_group'].numpy(), **kw_base)
        labels0 = np.asarray(labels0)
        li = copy.deepcopy(li0)
        pr_ = dict(params); pr_['er_bridge_occupancy'] = 0.5
        labels1, st = er.rejoin_endpoints(labels0, int(si), li, xyz, tpcv,
                                          SimpleNamespace(**pr_))
        labels1 = np.asarray(labels1)
        labels_f, stv = vp.pinpoint_vertex_merge(
            labels_global=labels1, split_index=int(si), label_info=li,
            x=x, y=y, z=z, config=SimpleNamespace(**params), tpc=tpcv)
        labels_f = np.asarray(labels_f)
        seg_of = np.full(len(xyz), -1, int)
        segdom = []
        for k, s in enumerate(segments):
            h = np.asarray(s['hits'], int)
            seg_of[h] = k
            mm = tg[h] >= 0
            segdom.append(int(np.bincount(tg[h][mm], weights=e[h][mm]).argmax())
                          if mm.any() else -1)
        segdom = np.array(segdom + [-99])          # seg -1 -> sentinel
        ok = (labels_f >= 0) & (tg >= 0)
        ev_budget = dict.fromkeys(CLASSES, 0.0)
        for F in np.unique(labels_f[ok]):
            m = ok & (labels_f == F)
            r = np.bincount(tg[m], weights=e[m])
            dom = int(r.argmax())
            e_for = float(r.sum() - r.max())
            if e_for < 1.0:
                continue
            own = m & (tg == dom)
            l0o = labels0[own]; l0o = l0o[l0o >= 0]
            l1o = labels1[own]; l1o = l1o[l1o >= 0]
            if not len(l0o) or not len(l1o):
                continue
            l0_own = int(np.bincount(l0o).argmax())
            l1_own = int(np.bincount(l1o).argmax())
            fidx = np.flatnonzero(m & (tg != dom))
            lab_budget = dict.fromkeys(CLASSES, 0.0)
            for i in fidx:
                if labels0[i] == l0_own:
                    sgi = seg_of[i]
                    if sgi >= 0 and segdom[sgi] == dom:
                        c = 'ransac'
                    elif sgi >= 0:
                        c = 'stitch'
                    else:
                        c = 'absorb'
                elif labels1[i] == l1_own:
                    c = 'rejoin'
                else:
                    c = 'vp'
                lab_budget[c] += e[i]
            for c in CLASSES:
                ev_budget[c] += lab_budget[c]
            worst = max(lab_budget, key=lab_budget.get)
            cases.append(dict(pt=ptname, ev=int(ev_id), lab=int(F), e_for=e_for,
                              e_tot=float(e[labels_f == F].sum()), **lab_budget,
                              worst=worst))
        rows.append(dict(pt=ptname, ev=int(ev_id),
                         e_ok=float(e[ok].sum()), **ev_budget))
        print(ptname, ev_id, 'done', flush=True)
with open(os.path.join(OUT, f'rows_{sys.argv[1]}.pkl'), 'wb') as f:
    pickle.dump(dict(rows=rows, cases=cases), f)
print('worker done')
