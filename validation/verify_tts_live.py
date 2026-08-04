"""Verify the fully wired frontier chain (stitch->rejoin->vp->refine->S2v2):
LIVE recluster.py must reproduce the tts2-scan P1/C1/N1 exactly, including an
event with actual splits (flow0000009 ev6)."""
import glob, json, os, pickle, sys
import numpy as np
import torch

REL = r'F:\CLMatching_v0.5.6_release'
SCR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REL)
from recluster import run_clustering, truth_groups

params = json.load(open(os.path.join(REL, 'params_full.json')))
assert params.get('two_track_split_enable') is True

ref = {}
for fp in glob.glob(os.path.join(SCR, 'tts2', 'rows_*.pkl')):
    for r in pickle.load(open(fp, 'rb'))['rows']:
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

for ptname, ev_id in [('flow0000009', 6), ('flow0000007', 9)]:
    blob = torch.load(os.path.join(REL, 'pt', ptname + '.pt'), weights_only=False)
    ev = blob['events'][ev_id]
    labels, si, li, dbg = run_clustering(ev, params)
    P, C, N = eval_pc(labels, ev)
    r = ref[(ptname, ev_id)]
    okP = abs(P - r['P1']) < 1e-9
    okC = abs(C - r['C1']) < 1e-9
    okN = N == r['N1']
    ns = len(dbg.get('two_track_split', {}).get('splits', []))
    print(f'{ptname} ev{ev_id}: P {P:.6f} ({"OK" if okP else "MISMATCH vs " + format(r["P1"], ".6f")}) | '
          f'C {C:.6f} ({"OK" if okC else "MISMATCH"}) | N {N} ({"OK" if okN else "MISMATCH"}) | '
          f'splits {ns} (scan {r["n_splits"]})', flush=True)
print('done')
