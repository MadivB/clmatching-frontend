"""Stage-by-stage wall-time profile of the current frontier chain on two
events (median-size and largest), plus a cProfile hot-function list."""
import copy, cProfile, inspect, io, json, os, pickle, pstats, sys, time
from types import SimpleNamespace
import numpy as np
import torch

REL = r'F:\CLMatching_v0.5.6_release'
SCR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REL)
from recluster import truth_groups, _file_load
import vertex_pinpoint as vp

tb = _file_load('tb_p', os.path.join(REL, 'clustering_v055',
                                     'global_track_clustering_toolbox_v11_2.py'))
sd = _file_load('sd_p', os.path.join(REL, 'stitch_directional.py'))
sp = _file_load('sp_p', os.path.join(REL, 'segment_split.py'))
er = _file_load('er_p', os.path.join(REL, 'endpoint_rejoin.py'))
br = _file_load('br_p', os.path.join(REL, 'blob_refine.py'))
tts = _file_load('tts_p', os.path.join(REL, 'two_track_split.py'))
fa = _file_load('fa_p', os.path.join(REL, 'frag_absorb.py'))
params = json.load(open(os.path.join(REL, 'params_full.json')))
sig = set(inspect.signature(tb.build_global_labels_toolbox).parameters)
kw_base = {k: v for k, v in params.items() if k in sig}
kw_base['return_label_info'] = True

# wrap the two expensive internals to time them separately
t_acc = {}
orig_build = tb._build_tpc_segments_toolbox
orig_match = sd.make_matcher(SimpleNamespace(**params), tb._fit_line_metrics)

def timed_build(*a, **k):
    t0 = time.perf_counter()
    r = orig_build(*a, **k)
    t_acc['ransac'] = t_acc.get('ransac', 0) + time.perf_counter() - t0
    return r

def timed_match(*a, **k):
    t0 = time.perf_counter()
    r = orig_match(*a, **k)
    t_acc['stitch'] = t_acc.get('stitch', 0) + time.perf_counter() - t0
    return r

tb._build_tpc_segments_toolbox = timed_build
tb._match_segments_across_tpcs_toolbox = timed_match

def run(ev, do_cprofile=False):
    global t_acc
    t_acc = {}
    x = ev['x'].numpy().astype(np.float64)
    y = ev['y'].numpy().astype(np.float64)
    z = ev['z'].numpy().astype(np.float64)
    xyz = np.stack([x, y, z], 1)
    e = np.clip(ev['energy'].numpy().astype(np.float64), 0, None)
    tpcv = ev['tpc'].numpy().astype(int)
    T = {}
    t0 = time.perf_counter()
    labels0, si, li0 = tb.build_global_labels_toolbox(
        x, y, z, ev['io_group'].numpy(), **kw_base)
    T['build_total'] = time.perf_counter() - t0
    T['ransac'] = t_acc.get('ransac', 0)
    T['stitch'] = t_acc.get('stitch', 0)
    T['dbscan+absorb'] = T['build_total'] - T['ransac'] - T['stitch']
    li = copy.deepcopy(li0)
    pr_ = dict(params); pr_['er_bridge_occupancy'] = 0.5
    t0 = time.perf_counter()
    labels1, _ = er.rejoin_endpoints(np.asarray(labels0), int(si), li, xyz,
                                     tpcv, SimpleNamespace(**pr_))
    T['rejoin'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    labels2, _ = vp.pinpoint_vertex_merge(
        labels_global=labels1, split_index=int(si), label_info=li,
        x=x, y=y, z=z, config=SimpleNamespace(**params), tpc=tpcv)
    T['vertex_pinpoint'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    labels3, _ = br.refine_blobs(np.asarray(labels2), int(si), xyz,
                                 SimpleNamespace(**params))
    T['blob_refine'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    labels4, _ = tts.split_two_track(labels3, int(si), li, xyz, e, tpcv,
                                     SimpleNamespace(**params))
    T['two_track_split'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    labels5, _ = fa.absorb_fragments(np.asarray(labels4), int(si), xyz,
                                     SimpleNamespace(**params))
    T['frag_absorb'] = time.perf_counter() - t0
    return T

cases = [('flow0000007', 9), ('flow0000001', 0)]
for ptname, ev_id in cases:
    blob = torch.load(os.path.join(REL, 'pt', ptname + '.pt'), weights_only=False)
    ev = blob['events'][ev_id]
    n = ev['x'].shape[0]
    T = run(ev)
    tot = sum(v for k, v in T.items() if k != 'build_total') + T['build_total'] \
        - T['ransac'] - T['stitch'] - T['dbscan+absorb']
    tot = (T['ransac'] + T['stitch'] + T['dbscan+absorb'] + T['rejoin']
           + T['vertex_pinpoint'] + T['blob_refine'] + T['two_track_split']
           + T['frag_absorb'])
    print(f'\n{ptname} ev{ev_id} ({n:,} hits): TOTAL {tot:.1f} s')
    for k in ('ransac', 'stitch', 'dbscan+absorb', 'rejoin', 'vertex_pinpoint',
              'blob_refine', 'two_track_split', 'frag_absorb'):
        print(f'  {k:16s} {T[k]:7.2f} s  ({T[k]/tot*100:5.1f}%)', flush=True)

# hot functions on the medium event
blob = torch.load(os.path.join(REL, 'pt', 'flow0000007.pt'), weights_only=False)
ev = blob['events'][9]
pr = cProfile.Profile()
pr.enable()
run(ev)
pr.disable()
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(18)
print('\n--- cProfile top (cumulative), flow0000007 ev9 ---')
print('\n'.join(s.getvalue().splitlines()[4:40]))
