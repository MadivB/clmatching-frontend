"""Three-way verification of fast_ransac on two events:
  legacy (flag off)  vs  fast-CPU (must be BIT-IDENTICAL)  vs  fast-GPU
Reports label agreement + wall time of the full run_clustering call.
Each variant runs in a fresh module namespace so patches don't leak."""
import importlib.util, json, os, sys, time
import numpy as np
import torch

REL = r'F:\CLMatching_v0.5.6_release'
sys.path.insert(0, REL)

params0 = json.load(open(os.path.join(REL, 'params_full.json')))

def fresh_recluster(tag):
    spec = importlib.util.spec_from_file_location(
        f'recluster_{tag}', os.path.join(REL, 'recluster.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f'recluster_{tag}'] = mod
    spec.loader.exec_module(mod)
    return mod

CASES = [('flow0000007', 9), ('flow0000006', 1)]
VARIANTS = [
    ('legacy', dict(fast_ransac_enable=False)),
    ('fast-cpu', dict(fast_ransac_enable=True, fast_ransac_backend='cpu')),
    ('fast-gpu', dict(fast_ransac_enable=True, fast_ransac_backend='auto')),
]

print('torch cuda available:', torch.cuda.is_available(), flush=True)
results = {}
for ptname, ev_id in CASES:
    blob = torch.load(os.path.join(REL, 'pt', ptname + '.pt'),
                      weights_only=False)
    ev = blob['events'][ev_id]
    for tag, over in VARIANTS:
        rc = fresh_recluster(f'{tag}_{ptname}_{ev_id}'.replace('-', '_'))
        p = dict(params0)
        p.update(over)
        t0 = time.perf_counter()
        labels, si, li, dbg = rc.run_clustering(ev, p)
        dt = time.perf_counter() - t0
        results[(ptname, ev_id, tag)] = (np.asarray(labels), dt)
        print(f'{ptname} ev{ev_id} {tag:9s}: {dt:6.1f} s', flush=True)
    ref = results[(ptname, ev_id, 'legacy')][0]
    for tag in ('fast-cpu', 'fast-gpu'):
        lab = results[(ptname, ev_id, tag)][0]
        same = np.array_equal(lab, ref)
        agree = float(np.mean(lab == ref))
        print(f'  {tag} vs legacy: identical={same} agreement={agree:.6f}',
              flush=True)
print('done')
