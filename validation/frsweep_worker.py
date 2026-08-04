"""fast_ransac 126-event identity sweep: legacy vs fast(auto/GPU) full chain.
Usage: python frsweep_worker.py <ptname> [...]"""
import importlib.util, json, os, pickle, sys, time
import numpy as np
import torch

REL = r'F:\CLMatching_v0.5.6_release'
SCR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REL)

OUT = os.path.join(SCR, 'frsweep')
os.makedirs(OUT, exist_ok=True)
params0 = json.load(open(os.path.join(REL, 'params_full.json')))

def fresh(tag):
    spec = importlib.util.spec_from_file_location(
        f'rc_{tag}', os.path.join(REL, 'recluster.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f'rc_{tag}'] = mod
    spec.loader.exec_module(mod)
    return mod

rc_legacy = fresh('legacy')
rc_fast = fresh('fast')

rows = []
for ptname in sys.argv[1:]:
    blob = torch.load(os.path.join(REL, 'pt', ptname + '.pt'),
                      weights_only=False)
    for ev_id, ev in blob['events'].items():
        p1 = dict(params0); p1['fast_ransac_enable'] = False
        t0 = time.perf_counter()
        l1, *_ = rc_legacy.run_clustering(ev, p1)
        dt1 = time.perf_counter() - t0
        p2 = dict(params0)
        p2.update(fast_ransac_enable=True, fast_ransac_backend='auto')
        t0 = time.perf_counter()
        l2, *_ = rc_fast.run_clustering(ev, p2)
        dt2 = time.perf_counter() - t0
        ident = bool(np.array_equal(np.asarray(l1), np.asarray(l2)))
        rows.append(dict(pt=ptname, ev=int(ev_id), identical=ident,
                         t_legacy=dt1, t_fast=dt2,
                         n_hits=int(ev['x'].shape[0])))
        print(f'{ptname} ev{ev_id}: identical={ident} '
              f'legacy {dt1:.1f}s fast {dt2:.1f}s', flush=True)
with open(os.path.join(OUT, f'rows_{sys.argv[1]}.pkl'), 'wb') as f:
    pickle.dump(rows, f)
print('worker done')
