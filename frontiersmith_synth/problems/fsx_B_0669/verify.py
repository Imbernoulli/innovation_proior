#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for the phyllotaxis
angle-schedule problem. Prints 'Ratio: <float in [0,1]>' on its last line.
"""
import sys, math

try:
    import numpy as np
    from scipy.spatial import Voronoi
except Exception:
    print("Ratio: 0.0 (missing numpy/scipy)")
    sys.exit(0)

EPS = 0.03          # floor on cv^2 -> caps the quality metric F at -log10(EPS)
GHOST_MARGIN = 1.05  # reflection-ghost radius multiplier for the disk boundary
RING_MULT = 3.5      # far outer ghost ring radius multiplier
RING_M = 64           # number of far outer ghost points


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


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
    return r


def dmins(r, alpha, N):
    d = np.zeros(N + 1)
    for k in range(1, N + 1):
        d[k] = alpha * (r[k] - r[k - 1])
    return d


def positions_abs(theta_all, r, N):
    theta = np.array(theta_all) % 360.0
    xs = r[1:N + 1] * np.cos(np.radians(theta))
    ys = r[1:N + 1] * np.sin(np.radians(theta))
    return xs, ys


def feasible(xs, ys, d, N):
    for k in range(2, N + 1):
        dx = xs[:k - 1] - xs[k - 1]
        dy = ys[:k - 1] - ys[k - 1]
        dist = np.sqrt(dx * dx + dy * dy)
        if dist.min() < d[k] - 1e-6:
            return False, k
    return True, -1


def voronoi_areas(xs, ys, R, N):
    """Bound every real primordium's Voronoi cell using (a) a per-point mirror
    ghost reflected outward across the disk rim (so rim cells see a fair
    'what's beyond the edge' proxy) and (b) a coarse far outer ring (numerical
    closure only). Returns area per real point (NaN if still unbounded)."""
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


def quality(theta_all, r, d, R, N, score_start_idx):
    xs, ys = positions_abs(theta_all, r, N)
    feas, badk = feasible(xs, ys, d, N)
    if not feas:
        return None, badk
    areas = voronoi_areas(xs, ys, R, N)
    if areas is None:
        return None, -2
    sub = areas[score_start_idx:]
    if np.isnan(sub).any() or sub.mean() <= 0:
        return None, -3
    cv2 = sub.var() / (sub.mean() ** 2)
    return -math.log10(cv2 + EPS), -1


def main():
    inp = open(sys.argv[1]).read().split()
    outp = open(sys.argv[2]).read().split()

    try:
        it = iter(inp)
        N = int(next(it)); R = float(next(it)); alpha = float(next(it))
        p_bulk = float(next(it)); trans_frac = float(next(it)); p_rim = float(next(it))
        score_frac = float(next(it))
    except Exception:
        fail("bad input")

    if len(outp) != N:
        fail(f"expected {N} numbers, got {len(outp)}")

    theta_raw = []
    for tok in outp:
        try:
            v = float(tok)
        except Exception:
            fail(f"non-numeric token {tok!r}")
        if not math.isfinite(v):
            fail("non-finite value")
        theta_raw.append(v)

    r = build_radii(N, R, p_bulk, trans_frac, p_rim)
    d = dmins(r, alpha, N)
    score_start_idx = int((1.0 - score_frac) * N)
    score_start_idx = max(0, min(N - 1, score_start_idx))

    F, reason = quality(theta_raw, r, d, R, N, score_start_idx)
    if F is None:
        reasons = {-2: "voronoi failure", -3: "unbounded/degenerate rim cells"}
        if reason >= 2:
            fail(f"lateral-inhibition violated at primordium {reason}")
        fail(reasons.get(reason, "infeasible"))

    # ---- internal baseline B: naive constant-angular-step "spokes" ----
    theta_b = [(k * (360.0 / N)) % 360.0 for k in range(1, N + 1)]
    Fb, _ = quality(theta_b, r, d, R, N, score_start_idx)
    Fb = Fb if (Fb is not None and Fb > 1e-9) else 1e-9

    ratio = min(1000.0, 100.0 * F / Fb) / 1000.0
    ratio = max(0.0, min(1.0, ratio))
    print("Ratio: %.6f" % ratio)


if __name__ == "__main__":
    main()
