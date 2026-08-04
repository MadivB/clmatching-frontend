"""fast_ransac v0.1: accelerated RANSAC scoring + connected components.

Two drop-in replacements for the hot spots of the per-TPC segment builder
(81-83% of chain wall time):

1. `_components_by_radius` -> cKDTree.query_pairs + scipy csgraph connected
   components (replaces a pure-Python union-find with 7M interpreter calls
   per event). Identical output contract: list of ascending index arrays
   sorted by (-size, first index).

2. `ransac_line_3d` -> same hypothesis generation (identical RNG stream) and
   the SAME legacy numpy scoring on CPU; optionally scores hypothesis counts
   on GPU (torch, float64). The winning hypothesis's inlier mask is ALWAYS
   recomputed on CPU with the legacy expression, so the GPU only accelerates
   the argmax search.

GPU POLICY (per Billy): backend 'auto' uses the GPU only if torch sees a
usable CUDA device; ANY GPU failure at ANY point (no device, CUDA OOM,
driver error) permanently drops this process to the CPU path with a single
warning - it must never break a run. 'cpu' forces the legacy-identical path.

API: enable(toolbox_v11_2_module, config) -> stats dict. Patches in place.
Config keys (defaults): fast_ransac_backend 'auto', fr_gpu_min_points 1500,
fr_gpu_mem_mb 384.
"""
from __future__ import annotations

import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

__all__ = ["enable", "components_by_radius_fast"]


def components_by_radius_fast(points, radius_cm):
    points = np.asarray(points, dtype=np.float64)
    n = int(points.shape[0])
    if n == 0:
        return []
    if n == 1:
        return [np.array([0], dtype=int)]
    pairs = cKDTree(points).query_pairs(float(radius_cm),
                                        output_type='ndarray')
    if len(pairs):
        adj = coo_matrix((np.ones(len(pairs), dtype=np.int8),
                          (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        _, lab = connected_components(adj, directed=False)
    else:
        lab = np.arange(n)
    order = np.argsort(lab, kind='stable')      # ascending index within comp
    labs_sorted = lab[order]
    cuts = np.flatnonzero(np.diff(labs_sorted)) + 1
    comps = [np.asarray(c, dtype=int) for c in np.split(order, cuts)]
    comps.sort(key=lambda arr: (-int(arr.size), int(arr[0])))
    return comps


class _Gpu:
    """Lazy torch/CUDA handle with permanent fallback on any failure."""

    def __init__(self, mode):
        self.mode = mode                 # 'auto' | 'gpu' | 'cpu'
        self.torch = None
        self.dead = (mode == 'cpu')
        self.warned = False

    def available(self):
        if self.dead:
            return False
        if self.torch is None:
            try:
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError('no CUDA device visible')
                # touch the device so occupancy/permission problems surface
                torch.zeros(8, dtype=torch.float64, device='cuda')
                self.torch = torch
            except Exception as exc:      # noqa: BLE001 - any failure -> CPU
                self._fail(exc)
                return False
        return True

    def _fail(self, exc):
        self.dead = True
        self.torch = None
        if not self.warned:
            self.warned = True
            print(f'[fast_ransac] GPU unavailable ({type(exc).__name__}: '
                  f'{exc}) - falling back to CPU for the rest of this run',
                  file=sys.stderr, flush=True)


def enable(tb, config=None):
    g = lambda k, d: getattr(config, k, d) if config is not None else d
    backend = str(g('fast_ransac_backend', 'auto'))
    gpu_min_n = int(g('fr_gpu_min_points', 1500))
    mem_mb = float(g('fr_gpu_mem_mb', 384.0))
    gpu = _Gpu('cpu' if backend == 'cpu' else backend)

    base = tb._base_toolbox
    tfr = base._track_fit_ransac
    legacy_auto_thresh = tfr._auto_dist_thresh
    legacy_refit = tfr._refit_line
    legacy_dists = tfr._line_distances

    def _cpu_best(points, points_sq, p0s, vs, dist_thresh):
        """The legacy scoring loop, verbatim (bit-identical path)."""
        best_mask, best_cnt = None, 0
        batch_size = 200
        for b in range(0, len(vs), batch_size):
            v_batch = vs[b:b + batch_size]
            p0_batch = p0s[b:b + batch_size]
            points_dot_v = points @ v_batch.T
            p0_dot_v = np.sum(p0_batch * v_batch, axis=1)
            dot_prods = points_dot_v - p0_dot_v
            points_dot_p0 = points @ p0_batch.T
            p0_sq = np.sum(p0_batch ** 2, axis=1)[None, :]
            sq_dist = points_sq - 2 * points_dot_p0 + p0_sq - dot_prods ** 2
            mask = sq_dist < dist_thresh ** 2
            cnts = np.sum(mask, axis=0)
            max_idx = np.argmax(cnts)
            if cnts[max_idx] > best_cnt:
                best_cnt = cnts[max_idx]
                best_mask = mask[:, max_idx]
        return best_mask, int(best_cnt)

    def _gpu_winner(points, p0s, vs, dist_thresh):
        """Hypothesis counts on GPU; returns (winner index, count) with the
        legacy first-global-max tie-breaking. Raises on any GPU trouble."""
        torch = gpu.torch
        with torch.no_grad():
            tp = torch.from_numpy(points).cuda()
            tps = (tp * tp).sum(1, keepdim=True)
            n = points.shape[0]
            h_all = len(vs)
            chunk = max(64, int(mem_mb * 1e6 / (n * 8 * 3)))
            best_cnt, best_j = 0, -1
            thr2 = float(dist_thresh) ** 2
            for b in range(0, h_all, chunk):
                tv = torch.from_numpy(vs[b:b + chunk]).cuda()
                tp0 = torch.from_numpy(p0s[b:b + chunk]).cuda()
                dot = tp @ tv.T - (tp0 * tv).sum(1)
                sq = tps - 2 * (tp @ tp0.T) + (tp0 * tp0).sum(1) - dot ** 2
                cnts = (sq < thr2).sum(0)
                j = int(torch.argmax(cnts).item())
                c = int(cnts[j].item())
                if c > best_cnt:
                    best_cnt, best_j = c, b + j
            return best_j, best_cnt

    def _mask_for(points, points_sq, p0s, vs, j, dist_thresh):
        """Inlier mask for hypothesis j, computed by replaying the legacy
        batched expression on the exact 200-wide batch containing j (same
        BLAS shapes -> same rounding as the legacy full scan)."""
        b0 = (j // 200) * 200
        v_batch = vs[b0:b0 + 200]
        p0_batch = p0s[b0:b0 + 200]
        points_dot_v = points @ v_batch.T
        p0_dot_v = np.sum(p0_batch * v_batch, axis=1)
        dot_prods = points_dot_v - p0_dot_v
        points_dot_p0 = points @ p0_batch.T
        p0_sq = np.sum(p0_batch ** 2, axis=1)[None, :]
        sq_dist = points_sq - 2 * points_dot_p0 + p0_sq - dot_prods ** 2
        return (sq_dist < dist_thresh ** 2)[:, j - b0]

    def fast_ransac_line_3d(points, iters=1200, dist_thresh=None,
                            min_inliers=35, k_for_scale=8, seed=None):
        N = len(points)
        if N < 2:
            return None
        rng = np.random.default_rng(seed)
        if dist_thresh is None:
            dist_thresh = legacy_auto_thresh(points, k_for_scale=k_for_scale,
                                             mult=1.5)
        idx = rng.integers(0, N, size=(iters, 2))
        diff = points[idx[:, 1]] - points[idx[:, 0]]
        nv = np.linalg.norm(diff, axis=1, keepdims=True)
        valid = (nv[:, 0] > 1e-12)
        p0s = points[idx[valid, 0]]
        vs = diff[valid] / nv[valid]
        points_sq = np.sum(points ** 2, axis=1)[:, None]

        best_mask, best_cnt = None, 0
        if N >= gpu_min_n and gpu.available():
            try:
                j, cnt = _gpu_winner(points, p0s, vs, dist_thresh)
                if j >= 0:
                    m = _mask_for(points, points_sq, p0s, vs, j, dist_thresh)
                    best_mask, best_cnt = m, int(m.sum())
            except Exception as exc:      # noqa: BLE001 - OOM, driver, ...
                gpu._fail(exc)
                best_mask, best_cnt = None, 0
        if best_mask is None:
            best_mask, best_cnt = _cpu_best(points, points_sq, p0s, vs,
                                            dist_thresh)

        if best_mask is None or best_cnt < min_inliers:
            return None
        in_points = points[best_mask]
        c, v, (pA, pB) = legacy_refit(in_points)
        d_all = legacy_dists(points, c, v)
        in_ref = d_all < dist_thresh
        rms = (float(np.sqrt(np.mean(d_all[in_ref] ** 2)))
               if in_ref.any() else np.nan)
        return {
            "point": c,
            "direction": v,
            "endpoints": (pA, pB),
            "inlier_mask": in_ref,
            "dist_thresh": float(dist_thresh),
            "n_inliers": int(in_ref.sum()),
            "n_total": int(N),
            "rms_dist": rms,
        }

    tfr.ransac_line_3d = fast_ransac_line_3d
    base._components_by_radius = components_by_radius_fast
    return {"backend": backend, "gpu_dead_at_start": gpu.dead}
