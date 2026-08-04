"""Endpoint-rejoin validation: build -> REJOIN -> vp. Audits rejoin pairs.
Usage: python tier2n_worker.py <ptname> [...]"""
import copy, inspect, json, os, pickle, sys
from types import SimpleNamespace
import numpy as np
import torch

REL = r'F:\CLMatching_v0.5.6_release'
SCR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REL)
from recluster import truth_groups, _file_load
import vertex_pinpoint as vp

OUT = os.path.join(SCR, 'tier2r')
os.makedirs(OUT, exist_ok=True)
tb = _file_load('tb_n', os.path.join(REL, 'clustering_v055',
                                     'global_track_clustering_toolbox_v11_2.py'))
sd = _file_load('sd_n', os.path.join(REL, 'stitch_directional.py'))
sp = _file_load('sp_n', os.path.join(REL, 'segment_split.py'))
er = _file_load('er_n', os.path.join(REL, 'endpoint_rejoin.py'))
params = json.load(open(os.path.join(REL, 'params_full.json')))
sig = set(inspect.signature(tb.build_global_labels_toolbox).parameters)
kw_base = {k: v for k, v in params.items() if k in sig}
kw_base['return_label_info'] = True
tb._match_segments_across_tpcs_toolbox = sd.make_matcher(
    SimpleNamespace(**params), tb._fit_line_metrics)

def eval_pc(labels, e, tg):
    ok = (labels >= 0) & (tg >= 0)
    ulab, li = np.unique(labels[ok], return_inverse=True)
    M = np.zeros((len(ulab), int(tg[ok].max()) + 1))
    np.add.at(M, (li, tg[ok]), e[ok])
    return float(M.max(1).sum() / M.sum()), float(M.max(0).sum() / M.sum())

rows, audits = [], []
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
        segments, _n = sp.split_segments(segments, xyz, SimpleNamespace(**params))
        tb._build_tpc_segments_toolbox = lambda *a, **k: (segments, {})
        labels0, si, li0 = tb.build_global_labels_toolbox(
            x, y, z, ev['io_group'].numpy(), **kw_base)
        labels0 = np.asarray(labels0)
        def dom(l):
            m = (labels0 == l) & (tg >= 0)
            return int(np.bincount(tg[m], weights=e[m]).argmax()) if m.any() else -1
        for variant, do_rejoin, vpmode, er_over in (
                ('B3', True, 'legacy', {'er_bridge_occupancy': 0.5}),):
            li = copy.deepcopy(li0)
            labels1 = labels0
            st = {'pairs': []}
            if do_rejoin:
                pr_ = dict(params); pr_.update(er_over)
                labels1, st = er.rejoin_endpoints(labels0, int(si), li, xyz, tpcv,
                                                  SimpleNamespace(**pr_))
            p2 = dict(params)
            p2['vp_mode'] = vpmode
            labels_f, stv = vp.pinpoint_vertex_merge(
                labels_global=labels1, split_index=int(si), label_info=li,
                x=x, y=y, z=z, config=SimpleNamespace(**p2), tpc=tpcv)
            P, C = eval_pc(labels_f, e, tg)
            if variant == 'B3':
                for pr in st['pairs']:
                    a, b = pr['labels']
                    ga, gb = dom(a), dom(b)
                    v = ('correct' if ga == gb and ga >= 0 else
                         'no-truth' if ga < 0 or gb < 0 else 'WRONG')
                    ea_ = float(e[labels0 == a].sum())
                    eb_ = float(e[labels0 == b].sum())
                    audits.append(dict(pt=ptname, ev=int(ev_id), stage='rejoin',
                                       verdict=v, e_min=min(ea_, eb_),
                                       e_max=max(ea_, eb_), **pr))
                for at in stv.get('attaches', []):
                    ga, gb = dom(at['blob']), dom(at['owner'])
                    v = ('correct' if ga == gb and ga >= 0 else
                         'no-truth' if ga < 0 or gb < 0 else 'WRONG')
                    audits.append(dict(pt=ptname, ev=int(ev_id), stage='attach',
                                       verdict=v, kind='attach',
                                       labels=[at['blob'], at['owner']],
                                       gap_cm=at['dist_cm'], dot=0.0, trans_cm=0.0,
                                       active_frac=0.0, n_bridge=0))
                np.savez_compressed(os.path.join(OUT, f'{ptname}_ev{ev_id}.npz'),
                                    labels=labels_f)
            rows.append(dict(pt=ptname, ev=int(ev_id), variant=variant, P=P, C=C,
                             n_pairs=len(st['pairs']),
                             n_att=len(stv.get('attaches', []))))
        print(ptname, ev_id, 'done', flush=True)
with open(os.path.join(OUT, f'rows_{sys.argv[1]}.pkl'), 'wb') as f:
    pickle.dump(dict(rows=rows, audits=audits), f)
print('worker done')
