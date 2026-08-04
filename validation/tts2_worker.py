"""S2 two-track splitter validation: adopted chain (build->rejoin->vp->refine)
then split_two_track; P/C/N with and without + per-split truth audit.
Usage: python tts_worker.py <ptname> [...]"""
import copy, inspect, json, os, pickle, sys
from types import SimpleNamespace
import numpy as np
import torch

REL = r'F:\CLMatching_v0.5.6_release'
SCR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REL)
from recluster import truth_groups, _file_load
import vertex_pinpoint as vp

OUT = os.path.join(SCR, 'tts2')
os.makedirs(OUT, exist_ok=True)
tb = _file_load('tb_t', os.path.join(REL, 'clustering_v055',
                                     'global_track_clustering_toolbox_v11_2.py'))
sd = _file_load('sd_t', os.path.join(REL, 'stitch_directional.py'))
sp = _file_load('sp_t', os.path.join(REL, 'segment_split.py'))
er = _file_load('er_t', os.path.join(REL, 'endpoint_rejoin.py'))
br = _file_load('br_t', os.path.join(REL, 'blob_refine.py'))
tts = _file_load('tts_t', os.path.join(REL, 'two_track_split.py'))
params = json.load(open(os.path.join(REL, 'params_full.json')))
sig = set(inspect.signature(tb.build_global_labels_toolbox).parameters)
kw_base = {k: v for k, v in params.items() if k in sig}
kw_base['return_label_info'] = True
tb._match_segments_across_tpcs_toolbox = sd.make_matcher(
    SimpleNamespace(**params), tb._fit_line_metrics)

def metrics(labels, si, e, tg):
    n_raw = len(np.unique(labels[labels >= 0]))
    ok = (labels >= 0) & (tg >= 0)
    ulab, li = np.unique(labels[ok], return_inverse=True)
    M = np.zeros((len(ulab), int(tg[ok].max()) + 1))
    np.add.at(M, (li, tg[ok]), e[ok])
    P = float(M.max(1).sum() / M.sum())
    C = float(M.max(0).sum() / M.sum())
    tr = ulab < si
    Pt = float(M[tr].max(1).sum() / max(M[tr].sum(), 1e-9))
    return P, C, Pt, n_raw

rows, audits = [], []
for ptname in sys.argv[1:]:
    blob = torch.load(os.path.join(REL, 'pt', ptname + '.pt'), weights_only=False)
    for ev_id, ev in blob['events'].items():
        with open(os.path.join(SCR, 'segs', f'{ptname}_ev{ev_id}.pkl'), 'rb') as f:
            seg0 = pickle.load(f)
        x = ev['x'].numpy().astype(np.float64)
        y = ev['y'].numpy().astype(np.float64)
        z = ev['z'].numpy().astype(np.float64)
        xyz = np.stack([x, y, z], 1)
        e = np.clip(ev['energy'].numpy().astype(np.float64), 0, None)
        tg = truth_groups(ev['truth_t0'].numpy().astype(np.float64))
        tpcv = ev['tpc'].numpy().astype(int)
        segments, _ = sp.split_segments(seg0, xyz, SimpleNamespace(**params))
        tb._build_tpc_segments_toolbox = lambda *a, **k: (segments, {})
        labels0, si, li0 = tb.build_global_labels_toolbox(
            x, y, z, ev['io_group'].numpy(), **kw_base)
        labels0 = np.asarray(labels0)
        li = copy.deepcopy(li0)
        pr_ = dict(params); pr_['er_bridge_occupancy'] = 0.5
        labels1, st = er.rejoin_endpoints(labels0, int(si), li, xyz, tpcv,
                                          SimpleNamespace(**pr_))
        labels2, stv = vp.pinpoint_vertex_merge(
            labels_global=labels1, split_index=int(si), label_info=li,
            x=x, y=y, z=z, config=SimpleNamespace(**params), tpc=tpcv)
        labels_b, brs = br.refine_blobs(np.asarray(labels2), int(si), xyz,
                                        SimpleNamespace(**params))
        si = int(si)
        P0, C0, Pt0, N0 = metrics(labels_b, si, e, tg)
        labels_s, sts = tts.split_two_track(labels_b, si, li, xyz, e, tpcv,
                                            SimpleNamespace(**params))
        P1, C1, Pt1, N1 = metrics(labels_s, si, e, tg)
        labels_s = np.asarray(labels_s)
        for s_ in sts['splits']:
            child = (np.asarray(labels_b) == s_['label']) & (labels_s == s_['donee'])
            keeper = labels_s == s_['label']
            donee0 = np.asarray(labels_b) == s_['donee']
            def dom(m):
                mm = m & (tg >= 0)
                if not mm.any():
                    return -1
                return int(np.bincount(tg[mm], weights=e[mm]).argmax())
            gn, gk, gd = dom(child), dom(keeper), dom(donee0)
            v_sep = ('SEPARATED' if gn != gk and gn >= 0 and gk >= 0 else
                     'no-truth' if gn < 0 or gk < 0 else 'BROKE-SAME')
            v_don = ('GOOD' if gn == gd and gn >= 0 else
                     'no-truth' if gn < 0 or gd < 0 else 'WRONG')
            audits.append(dict(pt=ptname, ev=int(ev_id), verdict=v_sep,
                               donation=v_don, e_new=float(e[child].sum()), **s_))
        rows.append(dict(pt=ptname, ev=int(ev_id), P0=P0, C0=C0, Pt0=Pt0, N0=N0,
                         P1=P1, C1=C1, Pt1=Pt1, N1=N1,
                         n_splits=len(sts['splits'])))
        print(ptname, ev_id, 'done', flush=True)
with open(os.path.join(OUT, f'rows_{sys.argv[1]}.pkl'), 'wb') as f:
    pickle.dump(dict(rows=rows, audits=audits), f)
print('worker done')
