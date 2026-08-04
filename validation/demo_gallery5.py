"""Demo gallery v5: failure modes of the B3 FRONTIER chain (tier2r labels).
A: contaminated TRACK-like clusters (shower-embedded excluded)
B: broken tracks with substantial pieces
C: the surviving wrong rejoins (endpoint-exclusive + capped pass)
Renders top cases + demo_index5.html.
"""
import glob, inspect, json, os, pickle, sys
from types import SimpleNamespace
import numpy as np
import torch
import plotly.graph_objects as go

REL = r'F:\CLMatching_v0.5.6_release'
SCR = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(REL, 'plots', 'html')
sys.path.insert(0, REL)
from recluster import truth_groups, _file_load
sp = _file_load('sp_g5', os.path.join(REL, 'segment_split.py'))
er = _file_load('er_g5', os.path.join(REL, 'endpoint_rejoin.py'))

def pca_full(pts):
    c = pts.mean(0)
    q = pts - c
    _, s, vt = np.linalg.svd(q, full_matrices=False)
    u = vt[0]
    t = q @ u
    lin = float(s[0]**2 / max(float((s**2).sum()), 1e-12))
    resid = q - np.outer(t, u)
    w = float(np.median(np.linalg.norm(resid, axis=1)))
    return c, u, float(t.min()), float(t.max()), lin, w

# ---- audits: the surviving wrong rejoins (B3) --------------------------------
audits = []
for fp in glob.glob(os.path.join(SCR, 'tier2r', 'rows_*.pkl')):
    audits += pickle.load(open(fp, 'rb'))['audits']
wrong_pairs = [a for a in audits if a.get('verdict') == 'WRONG']
wrong_by_ev = {}
for a in wrong_pairs:
    wrong_by_ev.setdefault((a['pt'], a['ev']), []).append(a)

# ---- scan all 126 events on B3 final labels ----------------------------------
rowsA, rowsB = [], []
blobs = {}
for fp in sorted(glob.glob(os.path.join(SCR, 'tier2r', 'flow*.npz'))):
    base = os.path.basename(fp).replace('.npz', '')
    ptname, ev_s = base.split('_ev')
    ev_id = int(ev_s)
    labels = np.load(fp)['labels']
    if ptname not in blobs:
        blobs[ptname] = torch.load(os.path.join(REL, 'pt', ptname + '.pt'), weights_only=False)
    ev = blobs[ptname]['events'][ev_id]
    xyz = np.stack([ev['x'].numpy(), ev['y'].numpy(), ev['z'].numpy()], 1).astype(np.float64)
    e = np.clip(ev['energy'].numpy().astype(np.float64), 0, None)
    tt = ev['truth_t0'].numpy().astype(np.float64)
    tg = truth_groups(tt)
    tpc = ev['tpc'].numpy().astype(int)
    flags = sp.two_track_flag(labels, xyz, e, tpc, SimpleNamespace())
    wr_here = set()
    for a in wrong_by_ev.get((ptname, ev_id), []):
        wr_here.update(a['labels'])
    ok = (labels >= 0) & (tg >= 0)
    geo = {}
    for lab in np.unique(labels[labels >= 0]):
        m = labels == lab
        if m.sum() < 30:
            continue
        mo = m & ok
        if not mo.any():
            continue
        r = np.bincount(tg[mo], weights=e[mo])
        dg = int(r.argmax())
        e_for = float(r.sum() - r.max())
        c, u, t0_, t1_, lin, w = pca_full(xyz[m])
        L = t1_ - t0_
        if lin >= 0.90 and L >= 30:
            geo[int(lab)] = (c, u, t0_, t1_, dg, float(e[m].sum()), lin)
        if e_for >= 30 and lin >= 0.90 and L >= 30 and w <= 3.0:
            fm = m & ok & (tg != dg)
            per_tpc = {int(x): float(e[fm & (tpc == x)].sum()) for x in np.unique(tpc[fm])}
            hint = 'interleaved along track (RANSAC-level overlap)'
            if per_tpc:
                wt, we = max(per_tpc.items(), key=lambda kv: kv[1])
                own_t = float(e[m & ok & (tg == dg) & (tpc == wt)].sum())
                if we > 0.5 * e_for and we > 4 * max(own_t, 1e-9):
                    hint = f'foreign concentrated in tpc{wt} (stitch/merge origin)'
            if int(lab) in wr_here:
                hint = 'REJOIN pass did this (see section C)'
            fg = [g for g in np.argsort(r)[::-1] if g != dg and r[g] > 0][:1]
            rowsA.append(dict(pt=ptname, ev=ev_id, lab=int(lab), e=float(e[m].sum()),
                L=L, lin=lin, w=w, pur=float(r.max() / r.sum()), foreign=e_for,
                hint=hint, flagged=int(lab) in flags,
                t0own=float(np.nanmean(tt[m & (tg == dg)])),
                t0for=float(np.nanmean(tt[tg == fg[0]])) if fg else np.nan))
    by_g = {}
    for lab, (c, u, t0_, t1_, dg, E, lin) in geo.items():
        if E >= 100:
            by_g.setdefault(dg, []).append(lab)
    for g, labs in by_g.items():
        for i in range(len(labs)):
            for j in range(i + 1, len(labs)):
                a, b = labs[i], labs[j]
                ca, ua, ta0, ta1, _, Ea, _ = geo[a]
                cb, ub, tb0, tb1, _, Eb, _ = geo[b]
                ang = np.degrees(np.arccos(min(1.0, abs(float(ua @ ub)))))
                if ang > 15:
                    continue
                ea_ = [ca + ta0 * ua, ca + ta1 * ua]
                eb_ = [cb + tb0 * ub, cb + tb1 * ub]
                gap = min(float(np.linalg.norm(p - q)) for p in ea_ for q in eb_)
                if gap > 80:
                    continue
                pbn = min(eb_, key=lambda q: min(float(np.linalg.norm(p - q)) for p in ea_))
                off = float(np.linalg.norm((pbn - ca) - ((pbn - ca) @ ua) * ua))
                if off > 8:
                    continue
                rowsB.append(dict(pt=ptname, ev=ev_id, a=a, b=b, gap=gap, ang=float(ang),
                                  off=off, ea=Ea, eb=Eb))

topA = sorted(rowsA, key=lambda r: -r['foreign'])[:8]
usedB = set()
topB = []
for r in sorted(rowsB, key=lambda r: -(min(r['ea'], r['eb']))):
    k1, k2 = (r['pt'], r['ev'], r['a']), (r['pt'], r['ev'], r['b'])
    if k1 in usedB or k2 in usedB:
        continue
    usedB.add(k1)
    usedB.add(k2)
    topB.append(r)
    if len(topB) >= 6:
        break
topC = sorted(wrong_pairs, key=lambda a: -a['e_min'])[:8]
print(f'A contaminated tracks: {len(rowsA)} | B broken pairs: {len(rowsB)} | '
      f'C wrong rejoins total: {len(wrong_pairs)}', flush=True)

# ---- clean out the stale v4 galleries ----------------------------------------
for pat in ('v4fail_*.html', 'v4break_*.html', 'rejoinfail_*.html',
            'demo_index3.html', 'demo_index4_rejoin.html'):
    for f in glob.glob(os.path.join(HTML, pat)):
        os.remove(f)
print('removed stale v4 gallery files', flush=True)

made = {1: [], 2: [], 3: []}

def render(ptname, ev_id, labs, tag, title):
    labels = np.load(os.path.join(SCR, 'tier2r', f'{ptname}_ev{ev_id}.npz'))['labels']
    ev = blobs[ptname]['events'][ev_id]
    xyz = np.stack([ev['x'].numpy(), ev['y'].numpy(), ev['z'].numpy()], 1).astype(np.float64)
    e = np.clip(ev['energy'].numpy().astype(np.float64), 0, None)
    tt = ev['truth_t0'].numpy().astype(np.float64)
    tg = truth_groups(tt)
    tpc = ev['tpc'].numpy().astype(int)
    fig = go.Figure()
    tsel = set()
    for i, lab in enumerate(labs):
        m = labels == lab
        tsel |= set(np.unique(tpc[m]).tolist())
        mo = m & (tg >= 0)
        row = np.bincount(tg[mo], weights=e[mo]) if mo.any() else np.array([0.0])
        dg = int(row.argmax())
        own = m & (tg == dg)
        foreign = m & (tg >= 0) & (tg != dg)
        fig.add_trace(go.Scatter3d(x=xyz[own, 0], y=xyz[own, 1], z=xyz[own, 2],
            mode='markers', marker=dict(size=2.3, color=['#1f77b4', '#ff7f0e'][i % 2]),
            name=f'label {lab} own (g{dg})',
            text=[f'g{gv} t0={tv:.0f} tpc{pv}' for gv, tv, pv in
                  zip(tg[own], tt[own], tpc[own])], hoverinfo='text'))
        if foreign.any():
            fig.add_trace(go.Scatter3d(x=xyz[foreign, 0], y=xyz[foreign, 1], z=xyz[foreign, 2],
                mode='markers', marker=dict(size=5, color='red', symbol='diamond',
                                            line=dict(width=0.5, color='black')),
                name=f'label {lab} FOREIGN',
                text=[f'g{gv} t0={tv:.0f} tpc{pv}' for gv, tv, pv in
                      zip(tg[foreign], tt[foreign], tpc[foreign])], hoverinfo='text'))
    ctx = np.isin(tpc, sorted(tsel)) & ~np.isin(labels, labs)
    fig.add_trace(go.Scatter3d(x=xyz[ctx, 0], y=xyz[ctx, 1], z=xyz[ctx, 2], mode='markers',
        marker=dict(size=1.3, color='rgba(160,175,195,0.25)'), name='context',
        hoverinfo='skip'))
    fig.update_layout(title=title, legend=dict(orientation='h', y=-0.02))
    fig.update_scenes(aspectmode='data', xaxis_title='x [cm]', yaxis_title='y [cm]',
                      zaxis_title='z [cm]')
    out = os.path.join(HTML, f'{tag}.html')
    fig.write_html(out, include_plotlyjs=True, default_width='100%', default_height='94vh')
    print('saved:', os.path.basename(out), flush=True)
    return f'{tag}.html'

for r in topA:
    fs = 'FLAGGED two-track' if r['flagged'] else 'NOT flagged (blind spot)'
    name = render(r['pt'], r['ev'], [r['lab']], f"v5fail_{r['pt']}_ev{r['ev']}_lab{r['lab']}",
        f"TRACK contamination: {r['pt']} ev {r['ev']} label {r['lab']} | {r['e']:.0f} MeV "
        f"len {r['L']:.0f} cm lin {r['lin']:.2f} width {r['w']:.1f} cm | purity {r['pur']:.3f} "
        f"foreign {r['foreign']:.0f} MeV | {r['hint']} | {fs} | t0 {r['t0own']:.0f} vs {r['t0for']:.0f}")
    made[1].append((name, f"{r['pt']} ev{r['ev']} lab{r['lab']}: {r['e']:.0f} MeV track, "
        f"purity {r['pur']:.3f}, foreign {r['foreign']:.0f} MeV, {r['hint']}, "
        f"{'flagged' if r['flagged'] else 'NOT flagged'}"))
for r in topB:
    name = render(r['pt'], r['ev'], [r['a'], r['b']], f"v5break_{r['pt']}_ev{r['ev']}_lab{r['a']}_{r['b']}",
        f"BROKEN track: {r['pt']} ev {r['ev']} labels {r['a']}+{r['b']} | gap {r['gap']:.1f} cm "
        f"angle {r['ang']:.1f} deg offset {r['off']:.1f} cm | E {r['ea']:.0f}+{r['eb']:.0f} MeV "
        f"| SURVIVED the collinear rejoin")
    made[2].append((name, f"{r['pt']} ev{r['ev']} {r['a']}+{r['b']}: gap {r['gap']:.1f} cm, "
                          f"angle {r['ang']:.1f} deg, E {r['ea']:.0f}+{r['eb']:.0f} MeV"))

# ---- section C: pre-rejoin pieces of the surviving wrong rejoins --------------
tb = _file_load('tb_g5', os.path.join(REL, 'clustering_v055',
                                      'global_track_clustering_toolbox_v11_2.py'))
sd = _file_load('sd_g5', os.path.join(REL, 'stitch_directional.py'))
params = json.load(open(os.path.join(REL, 'params_full.json')))
sig = set(inspect.signature(tb.build_global_labels_toolbox).parameters)
kw_base = {k: v for k, v in params.items() if k in sig}
kw_base['return_label_info'] = True
tb._match_segments_across_tpcs_toolbox = sd.make_matcher(
    SimpleNamespace(**params), tb._fit_line_metrics)
byC = {}
for a in topC:
    byC.setdefault((a['pt'], a['ev']), []).append(a)
for (ptname, ev_id), cases in byC.items():
    ev = blobs[ptname]['events'][ev_id]
    x = ev['x'].numpy().astype(np.float64)
    y = ev['y'].numpy().astype(np.float64)
    z = ev['z'].numpy().astype(np.float64)
    xyz = np.stack([x, y, z], 1)
    e = np.clip(ev['energy'].numpy().astype(np.float64), 0, None)
    tt = ev['truth_t0'].numpy().astype(np.float64)
    tg = truth_groups(tt)
    tpcv = ev['tpc'].numpy().astype(int)
    with open(os.path.join(SCR, 'segs', f'{ptname}_ev{ev_id}.pkl'), 'rb') as f:
        segments = pickle.load(f)
    segments, _ = sp.split_segments(segments, xyz, SimpleNamespace(**params))
    tb._build_tpc_segments_toolbox = lambda *a_, **k_: (segments, {})
    labels0, si, li = tb.build_global_labels_toolbox(x, y, z, ev['io_group'].numpy(), **kw_base)
    labels0 = np.asarray(labels0)
    for a in cases:
        la, lb = a['labels']
        fig = go.Figure()
        tsel = set()
        anchors = []
        for lab, col in ((la, '#1f77b4'), (lb, '#ff7f0e')):
            m = labels0 == lab
            tsel |= set(np.unique(tpcv[m]).tolist())
            mo = m & (tg >= 0)
            r = np.bincount(tg[mo], weights=e[mo]) if mo.any() else np.array([0.0])
            dg = int(r.argmax())
            fig.add_trace(go.Scatter3d(x=xyz[m, 0], y=xyz[m, 1], z=xyz[m, 2],
                mode='markers', marker=dict(size=2.4, color=col),
                name=f'label {lab} ({e[m].sum():.0f} MeV, g{dg}, '
                     f't0 {np.nanmean(tt[m & (tg == dg)]):.0f})',
                text=[f'g{gv} t0={tv:.0f} tpc{pv}' for gv, tv, pv in
                      zip(tg[m], tt[m], tpcv[m])], hoverinfo='text'))
            fr = er._line_frame(xyz[m])
            if fr is not None:
                c_, u_, t_, _ = fr
                anchors.append((c_ + t_.min() * u_, c_ + t_.max() * u_))
        if len(anchors) == 2:
            best = min(((p, q) for p in anchors[0] for q in anchors[1]),
                       key=lambda pq: np.linalg.norm(pq[0] - pq[1]))
            link = np.array(best)
            fig.add_trace(go.Scatter3d(x=link[:, 0], y=link[:, 1], z=link[:, 2],
                mode='lines', line=dict(width=8, color='red'),
                name=f'the rejoin ({a["kind"]}, gap {a["gap_cm"]:.1f} cm)'))
        ctx = np.isin(tpcv, sorted(tsel)) & ~np.isin(labels0, [la, lb])
        fig.add_trace(go.Scatter3d(x=xyz[ctx, 0], y=xyz[ctx, 1], z=xyz[ctx, 2],
            mode='markers', marker=dict(size=1.3, color='rgba(160,175,195,0.25)'),
            name='context', hoverinfo='skip'))
        fig.update_layout(
            title=(f'SURVIVING wrong rejoin (B3): {ptname} ev {ev_id} labels {la}+{lb} | '
                   f'{a["kind"]} | gap {a["gap_cm"]:.1f} cm dot {a["dot"]:.3f} '
                   f'trans {a["trans_cm"]:.2f} cm bridge {a["n_bridge"]} hits '
                   f'(active {a["active_frac"]:.2f}) | wrongly merged {a["e_min"]:.0f} MeV'),
            legend=dict(orientation='h', y=-0.02))
        fig.update_scenes(aspectmode='data', xaxis_title='x [cm]',
                          yaxis_title='y [cm]', zaxis_title='z [cm]')
        name = f'v5rejoin_{ptname}_ev{ev_id}_lab{la}_{lb}.html'
        fig.write_html(os.path.join(HTML, name), include_plotlyjs=True,
                       default_width='100%', default_height='94vh')
        made[3].append((name, f'{ptname} ev{ev_id} {la}+{lb}: {a["kind"]}, gap '
                              f'{a["gap_cm"]:.1f} cm, dot {a["dot"]:.3f}, trans '
                              f'{a["trans_cm"]:.2f}, bridge {a["n_bridge"]}, wrong '
                              f'{a["e_min"]:.0f} MeV'))
        print('saved:', name, flush=True)

SEC = {1: ('A. Contaminated TRACK-like clusters (shower-embedded excluded)',
           'Victims are clean tracks: linearity >= 0.90, length >= 30 cm, median width '
           '<= 3 cm. Red diamonds = energy from another interaction (hover for t0). '
           'Titles say whether the two-track flag marks the cluster and whether the '
           'rejoin pass caused it.'),
       2: ('B. Broken tracks with SUBSTANTIAL pieces (both >= 100 MeV, lin >= 0.90, >= 30 cm)',
           'One particle, two clean track labels, and the collinear rejoin still did '
           'not connect them - these show what the rejoin gates refuse and why.'),
       3: ('C. The surviving wrong rejoins (52 total across 126 events, top 8 by energy)',
           'What remains after endpoint exclusivity + remnant caps. The red line is the '
           'accepted connection between the two pre-rejoin pieces. These are the '
           'collinear doppelgangers: geometry is as clean as a true continuation.')}
rows_html = ['<p><b>Chain (B3 FRONTIER, params_full.json):</b> directional stitch '
             '(dot&le;-0.97/W8/ep40 + trans&le;5cm + back-wall veto) &rarr; splitter S1 '
             '&rarr; endpoint rejoin (exclusive best-first, remnant caps, bridge evidence) '
             '&rarr; blob-inclusive pinpointing v0.4 + two-track flag. '
             '<b>P 0.9879 / C 0.3916</b>. Shower-embedded mixing excluded per '
             'track-first criterion.</p>']
for cat in (1, 2, 3):
    t, d = SEC[cat]
    rows_html.append(f'<h2>{t}</h2><p>{d}</p><ul>')
    for name, desc in made[cat]:
        rows_html.append(f'<li><a href="{name}">{name}</a> - {desc}</li>')
    rows_html.append('</ul>')
html = ('<html><head><meta charset="utf-8"><title>Failure modes - B3 frontier chain</title>'
        '<style>body{font-family:sans-serif;max-width:900px;margin:2em auto;line-height:1.5}</style>'
        '</head><body><h1>Failure modes of the B3 frontier chain</h1>'
        + ''.join(rows_html) + '</body></html>')
open(os.path.join(HTML, 'demo_index5.html'), 'w', encoding='utf-8').write(html)
print('saved: demo_index5.html')
