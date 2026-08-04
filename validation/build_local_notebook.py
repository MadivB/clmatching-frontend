# Builds event_display_local.ipynb in the CLMatching release folder.
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
cells = []

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells.append(md("""# Event display (LOCAL): shipped v0.5.5 clustering vs your tuned re-clustering vs truth

Local twin of `event_display_compare.ipynb` (which needs NERSC modules/data and
cannot run here). Everything below reads only the released `pt/flow*.pt` files.

Available labelings:
- **truth** - hits colored by true interaction (`truth_t0` grouped with a 3-tick gap,
  same definition as `recluster.py`)
- **shipped** - the v0.5.5 production clustering shipped inside the `.pt`
- **tuned** - a LIVE re-run of the full v0.5.5 clustering with *your* parameter
  json (`PARAMS` below). Cached on disk, so each (file, event, params) runs once.

Extra view for the full-chain performance: `show_matching()` colors energy that the
v0.4.1 matching placed at the wrong time (|t0 - truth_t0| > 10 ticks) as red diamonds.

The tuning loop this notebook is the window for:
```
cp clustering_v055/clustering_defaults.json my.json   # edit parameters
# set PARAMS = 'my.json' below, re-run, compare panels + eval_event()
python recluster.py pt/*.pt --params my.json          # full 126-event measure
```"""))

cells.append(code(r"""import os, sys, json, hashlib, colorsys, importlib.util, inspect, time
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
import torch
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.renderers.default = 'iframe'   # robust in classic Jupyter; standalone HTML is saved anyway

HERE = os.path.abspath('')
if not os.path.isdir(os.path.join(HERE, 'pt')):   # launched from elsewhere
    HERE = r'F:\CLMatching_v0.5.6_release'
PT_DIR = os.path.join(HERE, 'pt')
CLU = os.path.join(HERE, 'clustering_v055')
HTML_DIR = os.path.join(HERE, 'plots', 'html')
CACHE_DIR = os.path.join(HERE, 'recluster_cache')
os.makedirs(HTML_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

LABELINGS = {'truth': 'Truth interactions', 'shipped': 'Shipped v0.5.5 clusters',
             'tuned': 'Tuned re-clustering (PARAMS)'}
_BRIGHTS = ['#1f77b4', '#2ca02c', '#9467bd', '#17becf', '#bcbd22', '#e377c2',
            '#7f7f7f', '#8c564b', '#3987e5', '#1baf7a', '#eda100', '#4a3aa7']
_REDS = ['#8b0000', '#dc143c', '#ff4500', '#b22222', '#ff6b6b', '#a52a2a',
         '#ff0000', '#cd5c5c', '#e9967a', '#800000']
_pt_cache, _cache = {}, {}

_PHI = 0.6180339887498949
def _distinct_rgb(i):
    h = (0.11 + i * _PHI) % 1.0
    s = 0.95 - 0.30 * (i % 3 == 2)
    v = 0.97 - 0.32 * (i % 2)
    return colorsys.hsv_to_rgb(h, s, v)

def _mpl_colors(labels):
    u = sorted(np.unique(labels[labels >= 0]).tolist())
    lut = {int(l): (*_distinct_rgb(i), 1.0) for i, l in enumerate(u)}
    return np.array([lut.get(int(l), (0.82, 0.82, 0.82, 0.45)) for l in labels])

def _hex_colors(labels):
    u = sorted(np.unique(labels[labels >= 0]).tolist())
    lut = {}
    for i, l in enumerate(u):
        r, g, b = _distinct_rgb(i)
        lut[int(l)] = f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'
    return [lut.get(int(l), 'rgba(200,200,200,0.35)') for l in labels]"""))

cells.append(md("## Tune here"))

cells.append(code("""FNUM  = 7      # FLOW file number 1..10 (pt/flow0000001.pt ...)
EVENT = 9      # event (spill) id inside the file - run list_events() to see them
TPC   = None   # charge-TPC id 0..69, or None for the whole event
PROJ  = 'zx'   # 2D projection: 'zy', 'zx' or 'xy'
MIN_E_LABEL = 0.0   # only color clusters with E >= this [MeV]; smaller -> gray
PARAMS = 'params_full.json'  # FRONTIER chain (rejoin + vp + refine + S2 + frag absorb); None -> tuned=shipped

# CURRENT CASE (now FIXED by the adopted chain): two parallel muons (t0 626
# vs 413) used to be fused by the legacy stage-3.5 endpoint vertexing into one
# 1665 MeV label. params_full.json now runs stage 3.5 OFF + blob-inclusive
# pinpointing v0.4 (eps5/in1.5): the muons come out as separate tuned labels
# (133 and 132); the event's worst residual foreign is 410 MeV (was 868).
# Inspect the fixed event:
#   worst_tracks(5, algo='tuned', tracks_only=False)
#   focus2d(worst_tracks(1, algo='tuned', tracks_only=False), algo='tuned', proj='zx')
#   focus([133, 132], algo='tuned')        <- the two muons, now separate"""))

cells.append(code(r"""def _file_load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def truth_groups(tt, gap=3.0):
    '''Truth interaction id per hit from truth_t0 (> gap ticks splits); -1 = no truth.'''
    g = np.full(tt.shape, -1, np.int64)
    idx = np.flatnonzero(np.isfinite(tt))
    idx = idx[np.argsort(tt[idx])]
    if idx.size:
        g[idx] = np.concatenate([[0], np.cumsum(np.diff(tt[idx]) > gap)])
    return g


def run_clustering(ev, params):
    '''v0.5.5 = toolbox RANSAC clustering + v0.2.5 vertex merge (same as recluster.py).'''
    toolbox = _file_load('clu_toolbox',
                         os.path.join(CLU, 'global_track_clustering_toolbox_v11_2.py'))
    if params.get('fast_ransac_enable', False):
        fr = _file_load('clu_fr', os.path.join(HERE, 'fast_ransac.py'))
        fr.enable(toolbox, SimpleNamespace(**params))
    if params.get('stitch_directional_enable', False):
        sd = _file_load('clu_sd', os.path.join(HERE, 'stitch_directional.py'))
        toolbox._match_segments_across_tpcs_toolbox = sd.make_matcher(
            SimpleNamespace(**params), toolbox._fit_line_metrics)
    sig = set(inspect.signature(toolbox.build_global_labels_toolbox).parameters)
    kw = {k: v for k, v in params.items() if k in sig}
    kw['return_label_info'] = True
    kw['return_debug_info'] = True
    labels, split_index, label_info, _dbg = toolbox.build_global_labels_toolbox(
        ev['x'].numpy(), ev['y'].numpy(), ev['z'].numpy(), ev['io_group'].numpy(), **kw)
    if params.get('vertex_merge_enable', True):
        vm = _file_load('clu_vm', os.path.join(CLU, 'vertex_merge.py'))
        # returns (labels, stats); mutates label_info in place, split_index unchanged
        labels, _vm_stats = vm.merge_vertex_tracks(
            labels_global=labels, split_index=split_index, label_info=label_info,
            x=ev['x'].numpy(), y=ev['y'].numpy(), z=ev['z'].numpy(),
            config=SimpleNamespace(**params))
    if params.get('endpoint_rejoin_enable', False):
        err = _file_load('clu_er', os.path.join(HERE, 'endpoint_rejoin.py'))
        xyz_ = np.stack([ev['x'].numpy(), ev['y'].numpy(), ev['z'].numpy()], 1)
        labels, _er_stats = err.rejoin_endpoints(
            labels, int(split_index), label_info, xyz_.astype(np.float64),
            ev['tpc'].numpy(), SimpleNamespace(**params))
    if params.get('vertex_pinpoint_enable', False):
        vpp = _file_load('clu_vp', os.path.join(HERE, 'vertex_pinpoint.py'))
        labels, _vp_stats = vpp.pinpoint_vertex_merge(
            labels_global=labels, split_index=split_index, label_info=label_info,
            x=ev['x'].numpy(), y=ev['y'].numpy(), z=ev['z'].numpy(),
            config=SimpleNamespace(**params), tpc=ev['tpc'].numpy())
    if params.get('blob_refine_enable', False):
        br = _file_load('clu_br', os.path.join(HERE, 'blob_refine.py'))
        xyz_ = np.stack([ev['x'].numpy(), ev['y'].numpy(), ev['z'].numpy()], 1)
        labels, _br_stats = br.refine_blobs(labels, int(split_index),
                                            xyz_.astype(np.float64),
                                            SimpleNamespace(**params))
    if params.get('two_track_split_enable', False):
        tts = _file_load('clu_tts', os.path.join(HERE, 'two_track_split.py'))
        xyz_ = np.stack([ev['x'].numpy(), ev['y'].numpy(), ev['z'].numpy()], 1)
        labels, _tts_stats = tts.split_two_track(
            labels, int(split_index), label_info, xyz_.astype(np.float64),
            ev['energy'].numpy().astype(np.float64), ev['tpc'].numpy(),
            SimpleNamespace(**params))
    if params.get('frag_absorb_enable', False):
        fa = _file_load('clu_fa', os.path.join(HERE, 'frag_absorb.py'))
        xyz_ = np.stack([ev['x'].numpy(), ev['y'].numpy(), ev['z'].numpy()], 1)
        labels, _fa_stats = fa.absorb_fragments(labels, int(split_index),
                                                xyz_.astype(np.float64),
                                                SimpleNamespace(**params))
    return np.asarray(labels), int(split_index)


def _load_pt(fnum):
    if fnum not in _pt_cache:
        _pt_cache[fnum] = torch.load(os.path.join(PT_DIR, f'flow{fnum:07d}.pt'),
                                     weights_only=False)
    return _pt_cache[fnum]


def list_events(fnum=None):
    blob = _load_pt(FNUM if fnum is None else fnum)
    print(os.path.basename(blob['flow_file']), '-', blob['n_events'], 'events')
    for ev_id, ev in blob['events'].items():
        print(f"  event {ev_id:3d}: {ev['x'].shape[0]:7d} hits | "
              f"ns_total_eff {ev['meta']['ns_total_eff']*100:.2f}%")


def load_event(fnum, ev):
    '''Per-hit arrays + all labelings (cached). tuned = live re-cluster if PARAMS set.'''
    key = (fnum, ev, PARAMS)
    if key in _cache:
        return _cache[key]
    d = _load_pt(fnum)['events'][ev]
    xyz = np.stack([d['x'].numpy(), d['y'].numpy(), d['z'].numpy()], 1).astype(np.float64)
    ev_data = dict(
        xyz=xyz, E=np.clip(d['energy'].numpy().astype(np.float64), 0, None),
        tpc=d['tpc'].numpy().astype(int),
        truth=truth_groups(d['truth_t0'].numpy().astype(np.float64)),
        shipped=d['labels'].numpy().astype(int),
        t0=d['t0'].numpy().astype(np.float64),
        truth_t0=d['truth_t0'].numpy().astype(np.float64),
        clusters=d['clusters'], meta=d['meta'], _raw=d)
    if PARAMS is None:
        ev_data['tuned'] = ev_data['shipped'].copy()
    else:
        ppath = PARAMS if os.path.isabs(PARAMS) else os.path.join(HERE, PARAMS)
        tag = hashlib.md5(open(ppath, 'rb').read()).hexdigest()[:10]
        cpath = os.path.join(CACHE_DIR, f'f{fnum:07d}_ev{ev}_{tag}.npz')
        if os.path.exists(cpath):
            ev_data['tuned'] = np.load(cpath)['labels']
        else:
            print(f'[tuned] re-clustering f{fnum} ev{ev} with {PARAMS} '
                  f'({xyz.shape[0]} hits, ~1-2 min) ...')
            t0 = time.time()
            labels, split_index = run_clustering(d, json.load(open(ppath)))
            np.savez_compressed(cpath, labels=labels, split_index=split_index)
            print(f'[tuned] done in {time.time()-t0:.0f} s -> cached {os.path.basename(cpath)}')
            ev_data['tuned'] = labels
    _cache[key] = ev_data
    return ev_data


def tpc_summary(fnum=None, ev=None, min_hits=200):
    '''Rank TPCs by deposited energy; cluster count per labeling.'''
    dat = load_event(FNUM if fnum is None else fnum, EVENT if ev is None else ev)
    keys = ['truth', 'shipped', 'tuned']
    print(' tpc | n_hits | E [MeV] | ' + ' | '.join(f'{k:>8s}' for k in keys))
    for t in np.argsort(np.bincount(dat['tpc'], weights=dat['E'], minlength=70))[::-1]:
        m = dat['tpc'] == t
        if m.sum() < min_hits:
            continue
        row = [f'{int(t):4d}', f'{int(m.sum()):6d}', f'{dat["E"][m].sum():7.0f}']
        for k in keys:
            row.append(f'{len(np.unique(dat[k][m][dat[k][m] >= 0])):8d}')
        print(' | '.join(row))"""))

cells.append(code("""list_events()
tpc_summary()   # which TPCs are busy in this event?"""))

cells.append(md("""## 2D projection view

`show()` - matplotlib panels, one per labeling. `min_e=5` grays out clusters below 5 MeV."""))

cells.append(code(r"""def show(fnum=None, ev=None, tpc=None, proj=None, min_e=None, s=2.0,
         panels=('truth', 'shipped', 'tuned')):
    fnum = FNUM if fnum is None else fnum
    ev = EVENT if ev is None else ev
    tpc = TPC if tpc is None else tpc
    proj = PROJ if proj is None else proj
    min_e = MIN_E_LABEL if min_e is None else min_e
    dat = load_event(fnum, ev)
    m = np.ones(len(dat['E']), bool) if tpc is None else (dat['tpc'] == tpc)
    ax_i = {'zy': (2, 1), 'zx': (2, 0), 'xy': (0, 1)}[proj]
    X, Y = dat['xyz'][m][:, ax_i[0]], dat['xyz'][m][:, ax_i[1]]
    nrow = int(np.ceil(len(panels) / 2))
    fig, axes = plt.subplots(nrow, 2, figsize=(14, 5.5 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.ravel()
    for ax, key in zip(axes, panels):
        lab = dat[key][m].copy()
        if min_e > 0 and key != 'truth':
            for c in np.unique(lab[lab >= 0]):
                if dat['E'][m][lab == c].sum() < min_e:
                    lab[lab == c] = -1
        order = np.argsort(lab >= 0)
        ax.scatter(X[order], Y[order], c=_mpl_colors(lab)[order], s=s, lw=0)
        ax.set_title(f'{LABELINGS[key]}  ({len(np.unique(lab[lab >= 0]))} in view)', fontsize=11)
        ax.set_aspect('equal'); ax.grid(alpha=0.2)
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    lbl = {0: 'x [cm]', 1: 'y [cm]', 2: 'z [cm]'}
    for ax in axes:
        ax.set_xlabel(lbl[ax_i[0]]); ax.set_ylabel(lbl[ax_i[1]])
    where = f'TPC {tpc}' if tpc is not None else 'all TPCs'
    fig.suptitle(f'file {fnum}  event {ev}  {where}  ({m.sum()} hits, proj {proj})',
                 y=0.995, fontsize=13)
    fig.tight_layout(); plt.show()

show()"""))

cells.append(md("""## 3D plotly (interactive, saves standalone HTML)

Whole-event views subsample to `max_points` for speed. HTML files land in `plots/html/`."""))

cells.append(code(r"""def show3d(fnum=None, ev=None, tpc=None, min_e=None, max_points=60000, size=1.6,
           panels=('truth', 'shipped', 'tuned'), show=True, save=True):
    fnum = FNUM if fnum is None else fnum
    ev = EVENT if ev is None else ev
    tpc = TPC if tpc is None else tpc
    min_e = MIN_E_LABEL if min_e is None else min_e
    dat = load_event(fnum, ev)
    m = np.ones(len(dat['E']), bool) if tpc is None else (dat['tpc'] == tpc)
    idx = np.flatnonzero(m)
    if len(idx) > max_points:
        idx = np.random.default_rng(1).choice(idx, max_points, replace=False)
    x, y, z = dat['xyz'][idx, 0], dat['xyz'][idx, 1], dat['xyz'][idx, 2]
    ncol = 2; nrow = int(np.ceil(len(panels) / ncol))
    fig = make_subplots(rows=nrow, cols=ncol,
                        specs=[[{'type': 'scene'}] * ncol for _ in range(nrow)],
                        subplot_titles=[LABELINGS[k] for k in panels],
                        horizontal_spacing=0.01, vertical_spacing=0.04)
    for i, key in enumerate(panels):
        lab = dat[key][idx].copy()
        if min_e > 0 and key != 'truth':
            E_sel = dat['E'][idx]
            for c in np.unique(lab[lab >= 0]):
                if E_sel[lab == c].sum() < min_e:
                    lab[lab == c] = -1
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode='markers',
            marker=dict(size=size, color=_hex_colors(lab)),
            text=[f'{key} label {int(l)}' for l in lab],
            hoverinfo='text', showlegend=False),
            row=i // ncol + 1, col=i % ncol + 1)
    where = f'TPC {tpc}' if tpc is not None else 'all TPCs'
    fig.update_layout(autosize=True, height=520 * nrow,
                      title=f'file {fnum}  event {ev} - {where}  ({len(idx)} of {int(m.sum())} hits shown)')
    fig.update_scenes(aspectmode='data', xaxis_title='x [cm]',
                      yaxis_title='y [cm]', zaxis_title='z [cm]')
    if save:
        tag = f'tpc{tpc}' if tpc is not None else 'allTPC'
        path = os.path.join(HTML_DIR, f'display_f{fnum:07d}_ev{ev}_{tag}.html')
        fig.write_html(path, include_plotlyjs=True,
                       default_width='100%', default_height='94vh')
        print(f'saved: {path}  ({os.path.getsize(path)/1e6:.1f} MB)')
    if show:
        fig.show()
    return fig

show3d();"""))

cells.append(md("""## Over-merge mistakes (MC truth) as red diamonds

Hits whose truth interaction differs from their cluster's energy-dominant truth
interaction. `algo='shipped'` or `'tuned'` - run both to see what your parameters fixed."""))

cells.append(code(r"""def show_mistakes(fnum=None, ev=None, tpc=None, algo='shipped',
                  size=2.2, size_bad=6.0, show=True, save=True):
    fnum = FNUM if fnum is None else fnum
    ev = EVENT if ev is None else ev
    tpc = TPC if tpc is None else tpc
    dat = load_event(fnum, ev)
    lab_all, truth_all, E_all = dat[algo], dat['truth'], dat['E']
    dom = {}
    for c in np.unique(lab_all[lab_all >= 0]):
        cm = (lab_all == c) & (truth_all >= 0)
        if cm.any():
            u, inv = np.unique(truth_all[cm], return_inverse=True)
            dom[int(c)] = int(u[np.bincount(inv, weights=E_all[cm]).argmax()])
    m = np.ones(len(E_all), bool) if tpc is None else (dat['tpc'] == tpc)
    lab, truth, E, xyz = lab_all[m], truth_all[m], E_all[m], dat['xyz'][m]
    bad = np.array([(l >= 0 and t >= 0 and int(l) in dom and t != dom[int(l)])
                    for l, t in zip(lab, truth)])
    good = ~bad
    truth_col = np.array(_hex_colors(np.where(truth >= 0, truth, -1)))
    lab_col = np.array(_hex_colors(lab))
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}] * 2],
        subplot_titles=['Truth interactions', f'{LABELINGS[algo]}: over-merges = red diamonds'])
    fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode='markers',
        marker=dict(size=size, color=truth_col.tolist()),
        text=[f'truth inter {int(t)}' for t in truth], hoverinfo='text',
        showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=xyz[good, 0], y=xyz[good, 1], z=xyz[good, 2], mode='markers',
        marker=dict(size=size, color=lab_col[good].tolist()),
        text=[f'clus {int(l)}' for l in lab[good]], hoverinfo='text',
        name='grouped', showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter3d(x=xyz[bad, 0], y=xyz[bad, 1], z=xyz[bad, 2], mode='markers',
        marker=dict(size=size_bad, color='red', symbol='diamond',
                    line=dict(width=0.5, color='black')),
        text=[f'clus {int(l)} but truth inter {int(t)}' for l, t in zip(lab[bad], truth[bad])],
        hoverinfo='text', name='OVER-MERGED (wrong interaction)', showlegend=True), row=1, col=2)
    where = f'TPC {tpc}' if tpc is not None else 'all TPCs'
    e_tot = float(E[(lab >= 0) & (truth >= 0)].sum()); e_bad = float(E[bad].sum())
    fig.update_layout(autosize=True, height=560, legend=dict(orientation='h', y=-0.02),
        title=f'{LABELINGS[algo]} over-merge mistakes - file {fnum} event {ev}, {where}  |  '
              f'{int(bad.sum())} wrong hits, {100*e_bad/max(e_tot,1e-9):.1f}% of clustered energy')
    fig.update_scenes(aspectmode='data', xaxis_title='x [cm]',
                      yaxis_title='y [cm]', zaxis_title='z [cm]')
    if save:
        tag = f'tpc{tpc}' if tpc is not None else 'allTPC'
        path = os.path.join(HTML_DIR, f'mistakes_{algo}_f{fnum:07d}_ev{ev}_{tag}.html')
        fig.write_html(path, include_plotlyjs=True,
                       default_width='100%', default_height='94vh')
        print(f'saved: {path}')
    if show:
        fig.show()
    return fig

show_mistakes();"""))

cells.append(md("""## Matching performance: where did the chain place energy WRONG?

The full-chain view (this is what NS total-eff measures): red diamonds = hits with
truth whose final matched `t0` is more than `tol` ticks from `truth_t0`."""))

cells.append(code(r"""def show_matching(fnum=None, ev=None, tpc=None, tol=10.0, algo='shipped',
                  size=1.8, size_bad=5.0, max_points=60000, show=True, save=True):
    fnum = FNUM if fnum is None else fnum
    ev = EVENT if ev is None else ev
    tpc = TPC if tpc is None else tpc
    dat = load_event(fnum, ev)
    m = np.ones(len(dat['E']), bool) if tpc is None else (dat['tpc'] == tpc)
    has_truth = np.isfinite(dat['truth_t0']) & m
    wrong = has_truth & (np.abs(dat['t0'] - dat['truth_t0']) > tol)
    right = has_truth & ~wrong
    e_h = dat['E'][has_truth].sum(); e_w = dat['E'][wrong].sum()
    ri = np.flatnonzero(right)
    if len(ri) > max_points:
        ri = np.random.default_rng(1).choice(ri, max_points, replace=False)
    xyz, lab = dat['xyz'], dat[algo]
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}] * 2],
        subplot_titles=['Truth interactions',
                        f'matched OK (cluster colors) vs WRONG t0 (red diamonds, >{tol:g} ticks)'])
    ti = np.flatnonzero(m)
    if len(ti) > max_points:
        ti = np.random.default_rng(2).choice(ti, max_points, replace=False)
    fig.add_trace(go.Scatter3d(x=xyz[ti, 0], y=xyz[ti, 1], z=xyz[ti, 2], mode='markers',
        marker=dict(size=size, color=np.array(_hex_colors(np.where(dat['truth'] >= 0, dat['truth'], -1)))[ti].tolist()),
        hoverinfo='skip', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=xyz[ri, 0], y=xyz[ri, 1], z=xyz[ri, 2], mode='markers',
        marker=dict(size=size, color=np.array(_hex_colors(lab))[ri].tolist()),
        text=[f'clus {int(l)}  dt={abs(d):.1f}' for l, d in
              zip(lab[ri], (dat['t0'] - dat['truth_t0'])[ri])],
        hoverinfo='text', name='placed OK', showlegend=False), row=1, col=2)
    wi = np.flatnonzero(wrong)
    fig.add_trace(go.Scatter3d(x=xyz[wi, 0], y=xyz[wi, 1], z=xyz[wi, 2], mode='markers',
        marker=dict(size=size_bad, color='red', symbol='diamond',
                    line=dict(width=0.5, color='black')),
        text=[f'clus {int(l)}  t0={t:.0f} truth {tt:.0f}' for l, t, tt in
              zip(lab[wi], dat['t0'][wi], dat['truth_t0'][wi])],
        hoverinfo='text', name=f'WRONG t0 ({100*e_w/max(e_h,1e-9):.1f}% of truth energy)',
        showlegend=True), row=1, col=2)
    where = f'TPC {tpc}' if tpc is not None else 'all TPCs'
    fig.update_layout(autosize=True, height=560, legend=dict(orientation='h', y=-0.02),
        title=f'matching errors - file {fnum} event {ev}, {where}  |  event ns_total_eff '
              f'{dat["meta"]["ns_total_eff"]*100:.2f}%')
    fig.update_scenes(aspectmode='data', xaxis_title='x [cm]',
                      yaxis_title='y [cm]', zaxis_title='z [cm]')
    if save:
        tag = f'tpc{tpc}' if tpc is not None else 'allTPC'
        path = os.path.join(HTML_DIR, f'matching_f{fnum:07d}_ev{ev}_{tag}.html')
        fig.write_html(path, include_plotlyjs=True,
                       default_width='100%', default_height='94vh')
        print(f'saved: {path}')
    if show:
        fig.show()
    return fig

show_matching();"""))

cells.append(md("""## Low-energy focus: highlight sub-MeV clusters

Over-splitting at 2-6 MeV halves per-cluster matching efficiency - this view shows
where the small clusters live. Diamonds = clusters below `e_tiny` MeV."""))

cells.append(code(r"""def show3d_lowE(fnum=None, ev=None, tpc=None, algos=('shipped', 'tuned'),
                e_small=5.0, e_tiny=1.0, max_points=60000,
                size_bg=1.4, size_small=3.0, size_tiny=5.0, show=True, save=True):
    fnum = FNUM if fnum is None else fnum
    ev = EVENT if ev is None else ev
    tpc = TPC if tpc is None else tpc
    dat = load_event(fnum, ev)
    m = np.ones(len(dat['E']), bool) if tpc is None else (dat['tpc'] == tpc)
    fig = make_subplots(rows=1, cols=len(algos), specs=[[{'type': 'scene'}] * len(algos)],
                        subplot_titles=[LABELINGS[k] for k in algos], horizontal_spacing=0.02)
    rng = np.random.default_rng(1)
    for col, key in enumerate(algos, start=1):
        lab_all = dat[key]
        e_of = {int(c): dat['E'][lab_all == c].sum() for c in np.unique(lab_all[lab_all >= 0])}
        lab = lab_all[m]; xyz = dat['xyz'][m]
        cl_e = np.array([e_of.get(int(l), np.inf) if l >= 0 else np.inf for l in lab])
        tiny = cl_e < e_tiny; small = (cl_e >= e_tiny) & (cl_e < e_small); bg = ~(tiny | small)
        bi = np.flatnonzero(bg)
        if len(bi) > max_points:
            bi = rng.choice(bi, max_points, replace=False)
        fig.add_trace(go.Scatter3d(x=xyz[bi, 0], y=xyz[bi, 1], z=xyz[bi, 2], mode='markers',
            marker=dict(size=size_bg, color='rgba(140,160,185,0.35)'), hoverinfo='skip',
            name=f'>= {e_small:g} MeV / unclustered', showlegend=(col == 1)), row=1, col=col)
        si = np.flatnonzero(small)
        if si.size:
            o = {int(c): k for k, c in enumerate(np.unique(lab[si]))}
            fig.add_trace(go.Scatter3d(x=xyz[si, 0], y=xyz[si, 1], z=xyz[si, 2], mode='markers',
                marker=dict(size=size_small, color=[_BRIGHTS[o[int(l)] % len(_BRIGHTS)] for l in lab[si]]),
                text=[f'label {int(l)}  E={e_of[int(l)]:.2f} MeV' for l in lab[si]], hoverinfo='text',
                name=f'{e_tiny:g}-{e_small:g} MeV clusters', showlegend=(col == 1)), row=1, col=col)
        ti = np.flatnonzero(tiny)
        if ti.size:
            o = {int(c): k for k, c in enumerate(np.unique(lab[ti]))}
            fig.add_trace(go.Scatter3d(x=xyz[ti, 0], y=xyz[ti, 1], z=xyz[ti, 2], mode='markers',
                marker=dict(size=size_tiny, color=[_REDS[o[int(l)] % len(_REDS)] for l in lab[ti]], symbol='diamond'),
                text=[f'label {int(l)}  E={e_of[int(l)]:.3f} MeV' for l in lab[ti]], hoverinfo='text',
                name=f'< {e_tiny:g} MeV clusters (diamonds)', showlegend=(col == 1)), row=1, col=col)
        n_s = len(np.unique(lab[si])) if si.size else 0
        n_t = len(np.unique(lab[ti])) if ti.size else 0
        print(f'{key}: {n_s} clusters in [{e_tiny:g},{e_small:g}) MeV, {n_t} below {e_tiny:g} MeV (in view)')
    where = f'TPC {tpc}' if tpc is not None else 'all TPCs'
    fig.update_layout(autosize=True, height=560, legend=dict(orientation='h', y=-0.02),
                      title=f'low-energy clusters - file {fnum} event {ev}, {where}')
    fig.update_scenes(aspectmode='data', xaxis_title='x [cm]',
                      yaxis_title='y [cm]', zaxis_title='z [cm]')
    if save:
        tag = f'tpc{tpc}' if tpc is not None else 'allTPC'
        path = os.path.join(HTML_DIR, f'lowE_f{fnum:07d}_ev{ev}_{tag}.html')
        fig.write_html(path, include_plotlyjs=True,
                       default_width='100%', default_height='94vh')
        print(f'saved: {path}')
    if show:
        fig.show()
    return fig

show3d_lowE();"""))

cells.append(md("""## Focus on one mistake: worst_tracks() + focus() / focus2d()

`worst_tracks(n)` ranks the current event's clusters by foreign (other-interaction)
energy. `focus([labels])` shows ONLY those clusters and their TPC corridor in 3D
(own hits colored, foreign = red diamonds, context gray); `focus2d([labels])` is
the matplotlib version. This is the fastest way to inspect a case from the
tracks_shipped.csv / broken_shipped.csv catalogs: set FNUM/EVENT above, then
`focus([131])` or `focus([121, 122])` for a broken pair."""))

cells.append(code(r"""def _foreign_table(fnum, ev, algo):
    dat = load_event(fnum, ev)
    labels, e, tg = dat[algo], dat['E'], dat['truth']
    ok = (labels >= 0) & (tg >= 0)
    ulab = np.unique(labels[labels >= 0])
    li = np.searchsorted(ulab, labels[ok])
    M = np.zeros((len(ulab), int(tg.max()) + 1))
    np.add.at(M, (li, tg[ok]), e[ok])
    rows = []
    for k, lab in enumerate(ulab):
        et = M[k].sum()
        if et <= 0:
            continue
        dom = int(M[k].argmax())
        cmeta = dat['clusters'].get(int(lab), {}) if algo == 'shipped' else {}
        rows.append(dict(label=int(lab), type=cmeta.get('type', '?'),
                         e_mev=float(e[labels == lab].sum()),
                         purity=float(M[k, dom] / et),
                         foreign=float(et - M[k, dom]), dom=dom))
    return sorted(rows, key=lambda r: -r['foreign'])


def worst_tracks(n=5, fnum=None, ev=None, algo='shipped', tracks_only=True):
    '''Labels of the n clusters with most foreign energy (prints a table).'''
    fnum = FNUM if fnum is None else fnum
    ev = EVENT if ev is None else ev
    rows = _foreign_table(fnum, ev, algo)
    if tracks_only and algo == 'shipped':
        rows = [r for r in rows if r['type'] == 'track']
    rows = rows[:n]
    print(f' worst {algo} clusters by foreign energy (f{fnum} ev{ev}):')
    for r in rows:
        print(f"  label {r['label']:5d} {r['type'][:6]:6s} E {r['e_mev']:7.1f} MeV | "
              f"purity {r['purity']:.3f} | foreign {r['foreign']:7.1f} MeV")
    return [r['label'] for r in rows]


def _focus_data(labs, fnum, ev, algo):
    dat = load_event(fnum, ev)
    labels, e, tg, tpc = dat[algo], dat['E'], dat['truth'], dat['tpc']
    sel_tpcs = sorted(set(int(t) for lab in labs for t in np.unique(tpc[labels == lab])))
    print(f' focus {algo} labels {labs} | TPCs {sel_tpcs}')
    print(' tpc |  own MeV | foreign MeV')
    for lab in labs:
        m = labels == lab
        ok = m & (tg >= 0)
        row = np.bincount(tg[ok], weights=dat['E'][ok]) if ok.any() else np.array([0.0])
        dom = int(row.argmax())
        for t in sorted(np.unique(tpc[m]).tolist()):
            mt = m & (tpc == t)
            print(f'  {t:3d} | {dat["E"][mt & (tg == dom)].sum():8.1f} | '
                  f'{dat["E"][mt & (tg >= 0) & (tg != dom)].sum():8.1f}   (label {lab})')
    return dat, sel_tpcs


def focus2d(labs, fnum=None, ev=None, algo='shipped', proj='zx', s=3.0):
    '''2D projection of the given cluster labels only (+ context in their TPCs).'''
    fnum = FNUM if fnum is None else fnum
    ev = EVENT if ev is None else ev
    dat, sel_tpcs = _focus_data(labs, fnum, ev, algo)
    labels, e, tg, tpc, xyz = dat[algo], dat['E'], dat['truth'], dat['tpc'], dat['xyz']
    ax_i = {'zy': (2, 1), 'zx': (2, 0), 'xy': (0, 1)}[proj]
    fig, ax = plt.subplots(figsize=(15, 6))
    ctx = np.isin(tpc, sel_tpcs) & ~np.isin(labels, labs)
    ax.scatter(xyz[ctx, ax_i[0]], xyz[ctx, ax_i[1]], c='0.85', s=1.5, lw=0, label='context')
    for i, lab in enumerate(labs):
        m = labels == lab
        ok = m & (tg >= 0)
        row = np.bincount(tg[ok], weights=e[ok]) if ok.any() else np.array([0.0])
        dom = int(row.argmax())
        own = m & (tg == dom); foreign = m & (tg >= 0) & (tg != dom)
        ax.scatter(xyz[own, ax_i[0]], xyz[own, ax_i[1]], c=_BRIGHTS[i % len(_BRIGHTS)],
                   s=s, lw=0, label=f'lab {lab} own (g{dom})')
        if foreign.any():
            ax.scatter(xyz[foreign, ax_i[0]], xyz[foreign, ax_i[1]], c='red', marker='D',
                       s=4 * s, lw=0.3, edgecolors='k', label=f'lab {lab} FOREIGN')
    lbl = {0: 'x [cm]', 1: 'y [cm]', 2: 'z [cm]'}
    ax.set_xlabel(lbl[ax_i[0]]); ax.set_ylabel(lbl[ax_i[1]])
    ax.set_aspect('equal'); ax.grid(alpha=0.2); ax.legend(markerscale=3, fontsize=9)
    ax.set_title(f'{algo} labels {labs} - f{fnum} ev{ev}, TPCs {sel_tpcs}, proj {proj}')
    plt.show()


def focus(labs, fnum=None, ev=None, algo='shipped', size=2.4, show=True, save=True):
    '''3D view of the given cluster labels only (+ context in their TPCs).'''
    fnum = FNUM if fnum is None else fnum
    ev = EVENT if ev is None else ev
    dat, sel_tpcs = _focus_data(labs, fnum, ev, algo)
    labels, e, tg, tpc, xyz = dat[algo], dat['E'], dat['truth'], dat['tpc'], dat['xyz']
    tt = dat['truth_t0']
    fig = go.Figure()
    for i, lab in enumerate(labs):
        m = labels == lab
        ok = m & (tg >= 0)
        row = np.bincount(tg[ok], weights=e[ok]) if ok.any() else np.array([0.0])
        dom = int(row.argmax())
        own = m & (tg == dom); foreign = m & (tg >= 0) & (tg != dom); naked = m & (tg < 0)
        fig.add_trace(go.Scatter3d(x=xyz[own, 0], y=xyz[own, 1], z=xyz[own, 2],
            mode='markers', marker=dict(size=size, color=_BRIGHTS[i % len(_BRIGHTS)]),
            name=f'lab {lab} own (g{dom})',
            text=[f'lab {lab} g{g} t0={t:.0f} tpc{p}' for g, t, p in
                  zip(tg[own], tt[own], tpc[own])], hoverinfo='text'))
        if naked.any():
            fig.add_trace(go.Scatter3d(x=xyz[naked, 0], y=xyz[naked, 1], z=xyz[naked, 2],
                mode='markers', marker=dict(size=size * 0.8, color='rgba(90,90,90,0.7)'),
                name=f'lab {lab} no-truth', hoverinfo='skip'))
        if foreign.any():
            fig.add_trace(go.Scatter3d(x=xyz[foreign, 0], y=xyz[foreign, 1], z=xyz[foreign, 2],
                mode='markers', marker=dict(size=size * 2.1, color='red', symbol='diamond',
                                            line=dict(width=0.5, color='black')),
                name=f'lab {lab} FOREIGN',
                text=[f'lab {lab} g{g} t0={t:.0f} tpc{p}' for g, t, p in
                      zip(tg[foreign], tt[foreign], tpc[foreign])], hoverinfo='text'))
    ctx = np.isin(tpc, sel_tpcs) & ~np.isin(labels, labs)
    fig.add_trace(go.Scatter3d(x=xyz[ctx, 0], y=xyz[ctx, 1], z=xyz[ctx, 2], mode='markers',
        marker=dict(size=1.3, color='rgba(160,175,195,0.25)'), name='context', hoverinfo='skip'))
    fig.update_layout(autosize=True, height=640, legend=dict(orientation='h', y=-0.02),
                      title=f'{algo} labels {labs} - f{fnum} ev{ev}, TPCs {sel_tpcs}')
    fig.update_scenes(aspectmode='data', xaxis_title='x [cm]',
                      yaxis_title='y [cm]', zaxis_title='z [cm]')
    if save:
        path = os.path.join(HTML_DIR,
            f'focus_{algo}_f{fnum:07d}_ev{ev}_lab{"_".join(str(l) for l in labs)}.html')
        fig.write_html(path, include_plotlyjs=True,
                       default_width='100%', default_height='94vh')
        print(f'saved: {path}')
    if show:
        fig.show()
    return fig

worst_tracks(5, algo='tuned', tracks_only=False)
focus2d([133, 132], algo='tuned', proj='zx')
focus([133, 132], algo='tuned');"""))

cells.append(md("""## Merge-decision inspector: vertex_case(a, b) on PRE-MERGE labels

The wrong-merge catalogs (`vmfail*.html`, mining tables) quote **pre-merge**
labels (toolbox output before any vertex merging, cached in `prevm/`).
`vertex_case(a, b)` reconstructs the vertex-pinpointing decision for that pair:
both clusters, the local 10-cm end-line fits with their extensions, the
pinpointed vertex, its depth in the charge volume, and the v0.2 verdict -
plus the truth verdict."""))

cells.append(code(r"""import pickle as _pickle
PREVM_DIR = os.path.join(HERE, 'prevm')

def vertex_case(lab_a, lab_b, fnum=None, ev=None, size=2.6, show=True, save=True):
    '''Inspect one vertex-pinpointing decision on pre-merge (toolbox) labels.'''
    fnum = FNUM if fnum is None else fnum
    ev = EVENT if ev is None else ev
    with open(os.path.join(PREVM_DIR, f'flow{fnum:07d}_ev{ev}.pkl'), 'rb') as f:
        pre = _pickle.load(f)
    labels0 = np.asarray(pre['labels'])
    dat = load_event(fnum, ev)
    xyz, tg, tpcs, tt, E = dat['xyz'], dat['truth'], dat['tpc'], dat['truth_t0'], dat['E']
    vp = _file_load('clu_vp_nb', os.path.join(HERE, 'vertex_pinpoint.py'))
    LOCAL, EXT, EPS = 10.0, 5.0, 5.0
    pts = {l: xyz[labels0 == l] for l in (lab_a, lab_b)}
    anch = {l: vp._end_anchors(pts[l]) for l in (lab_a, lab_b)}
    best = None
    for pa in anch[lab_a]:
        fa = vp._local_end_line(pts[lab_a], pa, LOCAL)
        if fa is None:
            continue
        for pb in anch[lab_b]:
            fb = vp._local_end_line(pts[lab_b], pb, LOCAL)
            if fb is None:
                continue
            d, c1, c2 = vp._seg_seg_dist(pa - LOCAL * fa[1], pa + EPS * fa[1],
                                         pb - LOCAL * fb[1], pb + EPS * fb[1])
            if best is None or d < best[0]:
                best = (d, pa, fa[1], pb, fb[1], 0.5 * (c1 + c2))
    doca, pa, ua, pb, ub, v = best
    anchor_d = min(float(np.linalg.norm(u - w)) for u in anch[lab_a] for w in anch[lab_b])
    ang = float(np.degrees(np.arccos(np.clip(abs(ua @ ub), -1, 1))))
    depth = vp._depth_in_charge(v, vp._charge_boxes(xyz, tpcs))
    boundary = depth < 3.0
    backwall = v[2] > float(xyz[:, 2].max()) - 10.0
    margin = 0.75 if boundary else 1.5
    paired = anchor_d <= EPS
    merged = paired and doca < margin and not backwall
    info = {}
    for l in (lab_a, lab_b):
        m = (labels0 == l) & (tg >= 0)
        g = int(np.bincount(tg[m], weights=E[m]).argmax()) if m.any() else -1
        info[l] = (float(E[labels0 == l].sum()), g,
                   float(np.nanmean(tt[m & (tg == g)])) if g >= 0 else np.nan,
                   sorted(np.unique(tpcs[labels0 == l]).tolist()))
    print(f'pair {lab_a}+{lab_b}  f{fnum} ev{ev} (pre-merge labels)')
    for l in (lab_a, lab_b):
        e_, g, t0_, tp = info[l]
        print(f'  label {l:4d}: {e_:7.1f} MeV | truth g{g} (t0 {t0_:.0f}) | TPCs {tp}')
    print(f'  closest end anchors {anchor_d:.2f} cm (DBSCAN eps {EPS:g} -> '
          f'{"paired" if paired else "NOT paired"}) | local-end angle {ang:.1f} deg')
    print(f'  extended-line doca {doca:.2f} cm | vertex {np.round(v,1).tolist()} | '
          f'depth {depth:.2f} cm ({"BOUNDARY, margin 0.75" if boundary else "interior, margin 1.5"})'
          + (' | WITHIN 10 cm OF BACK WALL -> VETO' if backwall else ''))
    same = info[lab_a][1] == info[lab_b][1] and info[lab_a][1] >= 0
    why = 'MERGE' if merged else ('no merge (back-wall veto)' if backwall and paired and doca < margin
                                  else 'no merge')
    print(f'  v0.3 verdict: {why} | '
          f'truth: {"same interaction" if same else "DIFFERENT interactions"} '
          f'=> {"WRONG merge" if merged and not same else ("correct merge" if merged else ("missed true vertex" if same else "correct rejection"))}')
    fig = go.Figure()
    for l, col in ((lab_a, '#1f77b4'), (lab_b, '#ff7f0e')):
        m = labels0 == l
        fig.add_trace(go.Scatter3d(x=xyz[m, 0], y=xyz[m, 1], z=xyz[m, 2], mode='markers',
            marker=dict(size=size, color=col), name=f'pre-merge label {l}',
            text=[f'lab {l} t0={t:.0f} tpc{p}' for t, p in zip(tt[m], tpcs[m])],
            hoverinfo='text'))
    for p0, u, col in ((pa, ua, '#2ca02c'), (pb, ub, '#d62728')):
        seg = np.array([p0 - LOCAL * u, p0 + EPS * u])
        fig.add_trace(go.Scatter3d(x=seg[:, 0], y=seg[:, 1], z=seg[:, 2], mode='lines',
            line=dict(width=6, color=col), name='local end line + extension'))
    fig.add_trace(go.Scatter3d(x=[v[0]], y=[v[1]], z=[v[2]], mode='markers',
        marker=dict(size=9, color='black', symbol='x'), name='pinpointed vertex'))
    near = (np.linalg.norm(xyz - v, axis=1) < 40) & (labels0 != lab_a) & (labels0 != lab_b)
    fig.add_trace(go.Scatter3d(x=xyz[near, 0], y=xyz[near, 1], z=xyz[near, 2], mode='markers',
        marker=dict(size=1.4, color='rgba(160,175,195,0.3)'), name='context (<40 cm)',
        hoverinfo='skip'))
    fig.update_layout(autosize=True, height=640, legend=dict(orientation='h', y=-0.02),
        title=f'vertex_case f{fnum} ev{ev} {lab_a}+{lab_b}: doca {doca:.2f} cm, '
              f'angle {ang:.1f} deg, depth {depth:.1f} cm')
    fig.update_scenes(aspectmode='data', xaxis_title='x [cm]', yaxis_title='y [cm]',
                      zaxis_title='z [cm]')
    if save:
        path = os.path.join(HTML_DIR, f'vertexcase_f{fnum:07d}_ev{ev}_lab{lab_a}_{lab_b}.html')
        fig.write_html(path, include_plotlyjs=True,
                       default_width='100%', default_height='94vh')
        print(f'saved: {path}')
    if show:
        fig.show()
    return fig

# vertex_case(a, b [, fnum=..., ev=...])  # inspect any PRE-MERGE pair from the failure-book CSVs"""))

cells.append(md("""## Purity / completeness of the current event

Same definitions as `recluster.py` (energy-weighted; truth = 3-tick `truth_t0` groups)."""))

cells.append(code(r"""def eval_event(fnum=None, ev=None):
    fnum = FNUM if fnum is None else fnum
    ev = EVENT if ev is None else ev
    dat = load_event(fnum, ev)
    e, tg = dat['E'], dat['truth']
    for name in (('shipped', 'tuned') if PARAMS else ('shipped',)):
        labels = dat[name]
        ok = (labels >= 0) & (tg >= 0)
        pur_num = pur_den = 0.0
        for lab in np.unique(labels[ok]):
            w = np.bincount(tg[ok & (labels == lab)], weights=e[ok & (labels == lab)])
            pur_num += float(w.max()); pur_den += float(w.sum())
        com_num = com_den = 0.0
        for g in np.unique(tg[ok]):
            w = np.bincount(labels[ok & (tg == g)], weights=e[ok & (tg == g)])
            com_num += float(w.max()); com_den += float(w.sum())
        print(f'[{name:7s}] clusters {len(np.unique(labels[labels >= 0])):4d} | '
              f'noise hits {int(np.sum(labels < 0)):6d} | '
              f'purity {pur_num / max(pur_den, 1e-9):.4f} | '
              f'completeness {com_num / max(com_den, 1e-9):.4f}')

eval_event()"""))

cells.append(md("""### Notes
- **The loop**: copy `clustering_v055/clustering_defaults.json` to `my.json`, edit,
  set `PARAMS = 'my.json'` in the *Tune here* cell, re-run everything below it. The
  `tuned` panels + `eval_event()` show what changed; `show_mistakes(algo='tuned')`
  vs `algo='shipped'` shows over-merges you fixed or introduced. Full-set numbers:
  `python recluster.py pt/*.pt --params my.json`.
- Tuned labels are cached in `recluster_cache/` keyed by a hash of the params file -
  editing the json invalidates the cache automatically.
- Colors use a golden-ratio palette so every id gets a distinct color, but the same
  physical cluster has *different* colors across panels - compare **partitions**, not hues.
- Plotly figures also land as standalone HTML in `plots/html/` (open in any browser -
  much smoother than inline for whole events).
- Baseline to beat (file 1, all 13 events, shipped): purity **0.9880**,
  completeness **0.3887** (see `performance.md`).
- The original NERSC notebook (`event_display_compare.ipynb`) is kept untouched for
  reference; it needs NERSC-only modules (`fragcompare_lib`, `spine_frontend_v040`,
  SPINE npz dumps) and cannot run locally."""))

nb.cells = cells
out = r'F:\CLMatching_v0.5.6_release\event_display_local.ipynb'
nbf.write(nb, out)
print('wrote', out)
