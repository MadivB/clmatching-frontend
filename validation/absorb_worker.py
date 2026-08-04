"""Small-fragment absorption scan (squeeze N toward 1500): after the full
adopted chain, labels with <= mh hits are absorbed into the label of their
nearest hit within radius r (variants: any target vs blob-only targets).
Usage: python absorb_worker.py <ptname> [...]"""
import copy, inspect, json, os, pickle, sys
from types import SimpleNamespace
import numpy as np
import torch
from scipy.spatial import cKDTree

REL = r'F:\CLMatching_v0.5.6_release'
SCR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REL)
from recluster import truth_groups, _file_load
import vertex_pinpoint as vp

OUT = os.path.join(SCR, 'absorb')
os.makedirs(OUT, exist_ok=True)
tb = _file_load('tb_a', os.path.join(REL, 'clustering_v055',
                                     'global_track_clustering_toolbox_v11_2.py'))
sd = _file_load('sd_a', os.path.join(REL, 'stitch_directional.py'))
sp = _file_load('sp_a', os.path.join(REL, 'segment_split.py'))
er = _file_load('er_a', os.path.join(REL, 'endpoint_rejoin.py'))
br = _file_load('br_a', os.path.join(REL, 'blob_refine.py'))
tts = _file_load('tts_a', os.path.join(REL, 'two_track_split.py'))
params = json.load(open(os.path.join(REL, 'params_full.json')))
sig = set(inspect.signature(tb.build_global_labels_toolbox).parameters)
kw_base = {k: v for k, v in params.items() if k in sig}
kw_base['return_label_info'] = True
tb._match_segments_across_tpcs_toolbox = sd.make_matcher(
    SimpleNamespace(**params), tb._fit_line_metrics)

# (max_hits, radius_cm, targets)
VARIANTS = [(6, 3.0, 'any'), (10, 3.0, 'any'), (10, 5.0, 'any'),
            (15, 5.0, 'any'), (20, 5.0, 'any'), (10, 5.0, 'blob')]

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
    Pb = float(M[~tr].max(1).sum() / max(M[~tr].sum(), 1e-9))
    return P, C, Pt, Pb, n_raw

def absorb(labels, si, xyz, e, mh, r, targets):
    lab = labels.copy()
    u, cnt = np.unique(lab[lab >= 0], return_counts=True)
    small = set(int(l) for l, c in zip(u, cnt) if c <= mh)
    if targets == 'blob':
        big_mask = (lab >= si) & ~np.isin(lab, list(small))
    else:
        big_mask = (lab >= 0) & ~np.isin(lab, list(small))
    if not big_mask.any() or not small:
        return lab, 0, 0.0
    tree = cKDTree(xyz[big_mask])
    big_lab = lab[big_mask]
    n_abs = 0
    e_moved = 0.0
    for l in sorted(small):
        idx = np.flatnonzero(lab == l)
        d, j = tree.query(xyz[idx], k=1)
        k = int(np.argmin(d))
        if d[k] <= r:
            lab[idx] = int(big_lab[j[k]])
            n_abs += 1
            e_moved += float(e[idx].sum())
    return lab, n_abs, e_moved

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
        labels0, si, li0 = tb.build_global_labels_toolbox(
            x, y, z, ev['io_group'].numpy(), **kw_base)
        li = copy.deepcopy(li0)
        pr_ = dict(params); pr_['er_bridge_occupancy'] = 0.5
        labels1, _st = er.rejoin_endpoints(np.asarray(labels0), int(si), li,
                                           xyz, tpcv, SimpleNamespace(**pr_))
        labels2, _sv = vp.pinpoint_vertex_merge(
            labels_global=labels1, split_index=int(si), label_info=li,
            x=x, y=y, z=z, config=SimpleNamespace(**params), tpc=tpcv)
        labels3, _br = br.refine_blobs(np.asarray(labels2), int(si), xyz,
                                       SimpleNamespace(**params))
        labels_f, _ts = tts.split_two_track(labels3, int(si), li, xyz, e,
                                            tpcv, SimpleNamespace(**params))
        labels_f = np.asarray(labels_f)
        si = int(si)
        P, C, Pt, Pb, N = metrics(labels_f, si, e, tg)
        rows.append(dict(pt=ptname, ev=int(ev_id), variant='base', P=P, C=C,
                         Pt=Pt, Pb=Pb, N=N, n_abs=0, e_moved=0.0))
        for mh, r, tgt in VARIANTS:
            lab_v, n_abs, e_moved = absorb(labels_f, si, xyz, e, mh, r, tgt)
            P, C, Pt, Pb, N = metrics(lab_v, si, e, tg)
            rows.append(dict(pt=ptname, ev=int(ev_id),
                             variant=f'h{mh}_r{r}_{tgt}', P=P, C=C, Pt=Pt,
                             Pb=Pb, N=N, n_abs=n_abs, e_moved=e_moved))
        print(ptname, ev_id, 'done', flush=True)
with open(os.path.join(OUT, f'rows_{sys.argv[1]}.pkl'), 'wb') as f:
    pickle.dump(rows, f)
print('worker done')
