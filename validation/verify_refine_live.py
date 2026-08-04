"""Verify the wired blob-refine chain: LIVE recluster.py (fresh RANSAC) must
reproduce the refine-scan T500_e2.5 P/C exactly on two check events."""
import glob, json, os, pickle, sys
import numpy as np
import torch

REL = r'F:\CLMatching_v0.5.6_release'
SCR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REL)
from recluster import run_clustering, truth_groups

params = json.load(open(os.path.join(REL, 'params_full.json')))
assert params.get('blob_refine_enable') is True

ref = {}
for fp in glob.glob(os.path.join(SCR, 'refine', 'rows_*.pkl')):
    for r in pickle.load(open(fp, 'rb')):
        if r['variant'] == 'T500_e2.5':
            ref[(r['pt'], r['ev'])] = r

def eval_pc(labels, ev):
    e = np.clip(ev['energy'].numpy().astype(np.float64), 0, None)
    tg = truth_groups(ev['truth_t0'].numpy().astype(np.float64))
    ok = (labels >= 0) & (tg >= 0)
    ulab, li = np.unique(labels[ok], return_inverse=True)
    M = np.zeros((len(ulab), int(tg[ok].max()) + 1))
    np.add.at(M, (li, tg[ok]), e[ok])
    return float(M.max(1).sum() / M.sum()), float(M.max(0).sum() / M.sum()), \
        len(np.unique(labels[labels >= 0]))

for ptname, ev_id in [('flow0000006', 1), ('flow0000007', 9)]:
    blob = torch.load(os.path.join(REL, 'pt', ptname + '.pt'), weights_only=False)
    ev = blob['events'][ev_id]
    labels, si, li, dbg = run_clustering(ev, params)
    P, C, N = eval_pc(labels, ev)
    r = ref[(ptname, ev_id)]
    okP = abs(P - r['P']) < 1e-9
    okC = abs(C - r['C']) < 1e-9
    okN = N == r['n_raw']
    br = dbg.get('blob_refine', {})
    print(f'{ptname} ev{ev_id}: P {P:.6f} (scan {r["P"]:.6f} {"OK" if okP else "MISMATCH"}) | '
          f'C {C:.6f} ({"OK" if okC else "MISMATCH"}) | N {N} (scan {r["n_raw"]} '
          f'{"OK" if okN else "MISMATCH"}) | blobs refined {len(br.get("refined", []))} '
          f'new labels {br.get("n_new_labels", 0)}', flush=True)
print('done')
