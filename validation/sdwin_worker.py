"""Stitch end-window scan, stage 1 (EDGE level): mode {z, euclid} x W {4,5,6,8,10,12}.
Standalone matcher on cached+split segments; verdict per accepted edge from the
dominant truth group of each segment. First event per worker also does a full
z/8 build vs the tier2r (adopted B3) labels as a bit-identity guard.
Usage: python sdwin_worker.py <ptname> [...]"""
import inspect, json, os, pickle, sys
from types import SimpleNamespace
import numpy as np
import torch

REL = r'F:\CLMatching_v0.5.6_release'
SCR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REL)
from recluster import truth_groups, _file_load

OUT = os.path.join(SCR, 'sdwin')
os.makedirs(OUT, exist_ok=True)
tb = _file_load('tb_w', os.path.join(REL, 'clustering_v055',
                                     'global_track_clustering_toolbox_v11_2.py'))
sd = _file_load('sd_w', os.path.join(REL, 'stitch_directional.py'))
sp = _file_load('sp_w', os.path.join(REL, 'segment_split.py'))
params = json.load(open(os.path.join(REL, 'params_full.json')))
sig = set(inspect.signature(tb.build_global_labels_toolbox).parameters)
kw_base = {k: v for k, v in params.items() if k in sig}
kw_base['return_label_info'] = True

VARIANTS = [(m, w) for m in ('z', 'euclid') for w in (4, 5, 6, 8, 10, 12)]

rows, wrong_edges = [], []
checked_identity = False
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
        segments, _n = sp.split_segments(segments, xyz, SimpleNamespace(**params))
        # per-segment dominant truth group + energy (verdict basis)
        segdom, sege = [], []
        for s in segments:
            h = np.asarray(s['hits'], int)
            m = tg[h] >= 0
            sege.append(float(e[h].sum()))
            segdom.append(int(np.bincount(tg[h][m], weights=e[h][m]).argmax())
                          if m.any() else -1)
        if not checked_identity:
            tb._build_tpc_segments_toolbox = lambda *a, **k: (segments, {})
            tb._match_segments_across_tpcs_toolbox = sd.make_matcher(
                SimpleNamespace(**params), tb._fit_line_metrics)
            lz, si_, li_ = tb.build_global_labels_toolbox(
                x, y, z, ev['io_group'].numpy(), **kw_base)
            # tier2r holds FINAL labels (post rejoin+vp); compare pre-stage
            # builds instead: default-mode matcher must equal explicit z/8
            m2 = sd.make_matcher(SimpleNamespace(**{**params,
                'sd_end_window_mode': 'z', 'sd_end_window_cm': 8.0}),
                tb._fit_line_metrics)
            tb._match_segments_across_tpcs_toolbox = m2
            lz2, _, _ = tb.build_global_labels_toolbox(
                x, y, z, ev['io_group'].numpy(), **kw_base)
            ident = bool(np.array_equal(np.asarray(lz), np.asarray(lz2)))
            print(f'IDENTITY default-vs-z8 {ptname} ev{ev_id}: {ident}', flush=True)
            if not ident:
                sys.exit('identity check FAILED')
            checked_identity = True
        for mode, W in VARIANTS:
            cfg = dict(params)
            cfg['sd_end_window_mode'] = mode
            cfg['sd_end_window_cm'] = float(W)
            matcher = sd.make_matcher(SimpleNamespace(**cfg), tb._fit_line_metrics)
            _tracks, dbg = matcher(segments, x, y, z)
            nc = nw = nn = 0
            ew = 0.0
            for ed in dbg['accepted_edges']:
                i, j = ed['i'], ed['j']
                ga, gb = segdom[i], segdom[j]
                if ga < 0 or gb < 0:
                    nn += 1
                elif ga == gb:
                    nc += 1
                else:
                    nw += 1
                    ew += min(sege[i], sege[j])
                    wrong_edges.append(dict(pt=ptname, ev=int(ev_id), mode=mode,
                        W=W, i=i, j=j, dot=ed['dot'], ep=ed['endpoint_dist'],
                        dseg=ed['segment_dist'], e_min=min(sege[i], sege[j]),
                        tpcs=(ed['tpc_i'], ed['tpc_j']), gs=(ga, gb)))
            rows.append(dict(pt=ptname, ev=int(ev_id), mode=mode, W=W,
                             n_corr=nc, n_wrong=nw, n_notruth=nn, e_wrong=ew))
        print(ptname, ev_id, 'done', flush=True)
with open(os.path.join(OUT, f'rows_{sys.argv[1]}.pkl'), 'wb') as f:
    pickle.dump(dict(rows=rows, wrong_edges=wrong_edges), f)
print('worker done')
