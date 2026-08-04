"""Bit-identity check: LIVE recluster.py (fresh RANSAC, params_full.json with
endpoint_rejoin_enable) vs the tier2r cached-segment B3 labels."""
import json, os, sys
import numpy as np
import torch

REL = r'F:\CLMatching_v0.5.6_release'
SCR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REL)
from recluster import run_clustering, truth_groups

params = json.load(open(os.path.join(REL, 'params_full.json')))
assert params.get('endpoint_rejoin_enable') is True

def eval_pc(labels, ev):
    e = np.clip(ev['energy'].numpy().astype(np.float64), 0, None)
    tg = truth_groups(ev['truth_t0'].numpy().astype(np.float64))
    ok = (labels >= 0) & (tg >= 0)
    pn = pd = cn = cd = 0.0
    for lab in np.unique(labels[ok]):
        w = np.bincount(tg[ok & (labels == lab)], weights=e[ok & (labels == lab)])
        pn += w.max(); pd += w.sum()
    for g_ in np.unique(tg[ok]):
        w = np.bincount(labels[ok & (tg == g_)], weights=e[ok & (tg == g_)])
        cn += w.max(); cd += w.sum()
    return pn / max(pd, 1e-9), cn / max(cd, 1e-9)

for ptname, ev_id in [('flow0000006', 1), ('flow0000007', 9)]:
    blob = torch.load(os.path.join(REL, 'pt', ptname + '.pt'), weights_only=False)
    ev = blob['events'][ev_id]
    labels, si, li, dbg = run_clustering(ev, params)
    ref = np.load(os.path.join(SCR, 'tier2r', f'{ptname}_ev{ev_id}.npz'))['labels']
    ident = bool(np.array_equal(labels, ref))
    p, c = eval_pc(labels, ev)
    er = dbg.get('endpoint_rejoin', {})
    print(f'{ptname} ev{ev_id}: identical={ident} | live P {p:.4f} C {c:.4f} | '
          f'rejoin pairs {len(er.get("pairs", []))} groups {len(er.get("groups", []))}',
          flush=True)
    if not ident:
        d = labels != ref
        print('  MISMATCH on', int(d.sum()), 'hits; labels involved:',
              np.unique(labels[d])[:10], 'vs', np.unique(ref[d])[:10])
print('done')
