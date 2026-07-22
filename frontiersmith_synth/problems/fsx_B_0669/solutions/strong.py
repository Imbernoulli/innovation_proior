# TIER: strong
"""Insight: decompose the schedule into (1) the golden-angle divergence,
which is asymptotically optimal for the *bulk* self-similar growth region,
and (2) a boundary-aware local correction near the rim/transition, where the
growth law changes (or simply runs out of future primordia to average
against). Starting from the golden-angle baseline, run coordinate-descent
local search *only* over the scored rim window, evaluated against the exact
same rim Voronoi-uniformity objective the checker uses -- so every accepted
move is a real, verified improvement (never worse than the textbook recipe).
"""
import sys, math
import numpy as np
from scipy.spatial import Voronoi

GOLDEN = 137.50776405003785
EPS = 0.03
GHOST_MARGIN = 1.05
RING_MULT = 3.5
RING_M = 64


def build_radii(N, R, p_bulk, trans_frac, p_rim):
    K0 = max(1, min(N, round(trans_frac * N)))
    r = np.zeros(N + 1)
    for k in range(1, K0 + 1):
        r[k] = R * (k / N) ** p_bulk
    rK0 = r[K0]
    for k in range(K0 + 1, N + 1):
        t = (k - K0) / (N - K0)
        r[k] = rK0 + (R - rK0) * (t ** p_rim)
    r[N] = R
    return r, K0


def dmins(r, alpha, N):
    d = np.zeros(N + 1)
    for k in range(1, N + 1):
        d[k] = alpha * (r[k] - r[k - 1])
    return d


def positions_abs(theta_all, r, N):
    theta = np.asarray(theta_all) % 360.0
    xs = r[1:N + 1] * np.cos(np.radians(theta))
    ys = r[1:N + 1] * np.sin(np.radians(theta))
    return xs, ys


def feasible(xs, ys, d, N):
    for k in range(2, N + 1):
        dx = xs[:k - 1] - xs[k - 1]
        dy = ys[:k - 1] - ys[k - 1]
        dist = np.sqrt(dx * dx + dy * dy)
        if dist.min() < d[k] - 1e-6:
            return False
    return True


def voronoi_areas(xs, ys, R, N):
    rad = np.sqrt(xs ** 2 + ys ** 2)
    reflect_r = 2 * R * GHOST_MARGIN - rad
    theta = np.arctan2(ys, xs)
    gx = reflect_r * np.cos(theta)
    gy = reflect_r * np.sin(theta)
    ang = np.linspace(0, 2 * np.pi, RING_M, endpoint=False)
    ring_r = RING_MULT * R
    rx = ring_r * np.cos(ang)
    ry = ring_r * np.sin(ang)
    allpts = np.vstack([np.column_stack([xs, ys]),
                         np.column_stack([gx, gy]),
                         np.column_stack([rx, ry])])
    try:
        vor = Voronoi(allpts)
    except Exception:
        return None
    areas = np.zeros(N)
    for i in range(N):
        region = vor.regions[vor.point_region[i]]
        if -1 in region or len(region) == 0:
            areas[i] = np.nan
            continue
        poly = vor.vertices[region]
        x = poly[:, 0]; y = poly[:, 1]
        areas[i] = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return areas


def real_metric(theta_all, r, d, R, N, score_start_idx):
    xs, ys = positions_abs(theta_all, r, N)
    if not feasible(xs, ys, d, N):
        return -1e9
    areas = voronoi_areas(xs, ys, R, N)
    if areas is None:
        return -1e9
    sub = areas[score_start_idx:]
    if np.isnan(sub).any() or sub.mean() <= 0:
        return -1e9
    cv2 = sub.var() / (sub.mean() ** 2)
    return -math.log10(cv2 + EPS)


def solve(N, R, alpha, p_bulk, trans_frac, p_rim, score_frac):
    r, K0 = build_radii(N, R, p_bulk, trans_frac, p_rim)
    d = dmins(r, alpha, N)
    score_start_idx = max(0, min(N - 1, int((1.0 - score_frac) * N)))

    theta = np.array([(k * GOLDEN) % 360.0 for k in range(1, N + 1)])
    best_metric = real_metric(theta, r, d, R, N, score_start_idx)

    buf = max(2, (N - score_start_idx) // 3)
    idxs = list(range(max(0, score_start_idx - buf), N))
    window, step, sweeps = 30.0, 3.0, 2
    offsets = np.arange(-window, window + 1e-9, step)

    for _ in range(sweeps):
        improved_any = False
        for k0 in idxs:
            cur = theta[k0]
            best_local, best_local_metric = cur, best_metric
            for off in offsets:
                cand = (cur + off) % 360.0
                trial = theta.copy()
                trial[k0] = cand
                m = real_metric(trial, r, d, R, N, score_start_idx)
                if m > best_local_metric + 1e-9:
                    best_local_metric = m
                    best_local = cand
            if best_local_metric > best_metric + 1e-9:
                theta[k0] = best_local
                best_metric = best_local_metric
                improved_any = True
        if not improved_any:
            break
    return theta


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); R = float(next(it)); alpha = float(next(it))
    p_bulk = float(next(it)); trans_frac = float(next(it)); p_rim = float(next(it))
    score_frac = float(next(it))

    theta = solve(N, R, alpha, p_bulk, trans_frac, p_rim, score_frac)
    print("\n".join("%.6f" % v for v in theta))


if __name__ == "__main__":
    main()
