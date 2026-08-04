"""Leftover-DBSCAN granularity scan on the B3 chain: eps {4,3,2.5,2} x min 3.
Full chain per variant; reports P, C, track-only purity, blob-only purity,
total foreign MeV, n_labels. Usage: python dbscan_worker.py <ptname> [...]"""
import copy, inspect, json, os, pickle, sys
from types import SimpleNamespace
import numpy as np
import torch

REL = r'F:\CLMatching_v0.5.6_release'
SCR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REL)
from recluster import truth_groups, _file_load
import vertex_pinpoint as vp

OUT = os.path.join(SCR, 'dbscan')
os.makedirs(OUT, exist_ok=True)
tb = _file_load('tb_d', os.path.join(REL, 'clustering_v055',
                                     'global_track_clustering_toolbox_v11_2.py'))
sd = _file_load('sd_d', os.path.join(REL, 'stitch_directional.py'))
sp = _file_load('sp_d', os.path.join(REL, 'segment_split.py'))
er = _file_load('er_d', os.path.join(REL, 'endpoint_rejoin.py'))
params = json.load(open(os.path.join(REL, 'params_full.json')))
sig = set(inspect.signature(tb.build_global_labels_toolbox).parameters)
tb._match_segments_across_tpcs_toolbox = sd.make_matcher(
    SimpleNamespace(**params), tb._fit_line_metrics)

VARIANTS = [4.0, 3.0, 2.5, 2.0]

def metrics(labels, si, e, tg):
    ok = (labels >= 0) & (tg >= 0)
    ulab, li = np.unique(labels[ok], return_inverse=True)
    M = np.zeros((len(ulab), int(tg[ok].max()) + 1))
    np.add.at(M, (li, tg[ok]), e[ok])
    P = float(M.max(1).sum() / M.sum())
    C = float(M.max(0).sum() / M.sum())
    tr = ulab < si
    Pt = float(M[tr].max(1).sum() / max(M[tr].sum(), 1e-9))
    Pb = float(M[~tr].max(1).sum() / max(M[~tr].sum(), 1e-9))
    frn = float(M.sum() - M.max(1).sum())
    return P, C, Pt, Pb, frn, len(ulab), float(M[~tr].sum())

rows = []
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
        for eps in VARIANTS:
            cfg = dict(params)
            cfg['leftover_dbscan_eps'] = eps
            kw = {k: v for k, v in cfg.items() if k in sig}
            kw['return_label_info'] = True
            labels0, si, li0 = tb.build_global_labels_toolbox(
                x, y, z, ev['io_group'].numpy(), **kw)
            labels0 = np.asarray(labels0)
            li = copy.deepcopy(li0)
            pr_ = dict(cfg); pr_['er_bridge_occupancy'] = 0.5
            labels1, st = er.rejoin_endpoints(labels0, int(si), li, xyz, tpcv,
                                              SimpleNamespace(**pr_))
            labels_f, stv = vp.pinpoint_vertex_merge(
                labels_global=labels1, split_index=int(si), label_info=li,
                x=x, y=y, z=z, config=SimpleNamespace(**cfg), tpc=tpcv)
            P, C, Pt, Pb, frn, nlab, eb = metrics(np.asarray(labels_f), int(si), e, tg)
            rows.append(dict(pt=ptname, ev=int(ev_id), eps=eps, P=P, C=C,
                             P_track=Pt, P_blob=Pb, foreign=frn, n_labels=nlab,
                             e_blob=eb))
        print(ptname, ev_id, 'done', flush=True)
with open(os.path.join(OUT, f'rows_{sys.argv[1]}.pkl'), 'wb') as f:
    pickle.dump(rows, f)
print('worker done')
