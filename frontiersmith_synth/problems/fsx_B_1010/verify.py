#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- checker for fsx_B_1010
   Family: fractal-dimension-lacunarity-match (format C, minimize distance).
"""
import sys, math

MAX_TOKENS = 20000


def read_instance(path):
    toks = open(path).read().split()
    n = int(toks[0]); k = int(toks[1])
    Dstar = float(toks[2]); Lamstar = float(toks[3])
    wD = float(toks[4]); wL = float(toks[5])
    return n, k, Dstar, Lamstar, wD, wL


def parse_masks(text, n, k):
    """Return (masks, reason). masks = list of k sets of (r,c) pairs, or None on failure."""
    toks = text.split()
    if len(toks) == 0:
        return None, "empty output"
    if len(toks) > MAX_TOKENS:
        return None, "too many tokens"
    try:
        vals = [int(t) for t in toks]
    except ValueError:
        return None, "non-integer token (nan/inf/garbage)"
    ptr = 0
    if ptr >= len(vals):
        return None, "missing k"
    k_out = vals[ptr]; ptr += 1
    if k_out != k:
        return None, "k mismatch"
    masks = []
    for lvl in range(k):
        if ptr >= len(vals):
            return None, "truncated (missing N_i)"
        Ni = vals[ptr]; ptr += 1
        if Ni < 1 or Ni > n * n - 1:
            return None, f"N_{lvl+1}={Ni} out of range [1,{n*n-1}]"
        need = 2 * Ni
        if ptr + need > len(vals):
            return None, "truncated (missing coordinate pairs)"
        cells = set()
        for _ in range(Ni):
            r = vals[ptr]; c = vals[ptr + 1]; ptr += 2
            if r < 0 or r >= n or c < 0 or c >= n:
                return None, "coordinate out of range"
            if (r, c) in cells:
                return None, "duplicate cell within a level"
            cells.add((r, c))
        masks.append(cells)
    if ptr != len(vals):
        return None, "trailing garbage after expected tokens"
    return masks, "ok"


def mask_matrix(cells, n):
    import numpy as np
    M = np.zeros((n, n), dtype=np.uint8)
    for (r, c) in cells:
        M[r, c] = 1
    return M


def measured_dimension(sizes, n):
    """Least-squares slope of ln(prod_{1..j} N_i) vs j*ln(n), j=1..k."""
    k = len(sizes)
    logn = math.log(n)
    xs = []
    ys = []
    acc = 0.0
    for j in range(1, k + 1):
        acc += math.log(sizes[j - 1])
        xs.append(j * logn)
        ys.append(acc)
    m = len(xs)
    sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    denom = m * sxx - sx * sx
    if abs(denom) < 1e-12:
        return ys[0] / xs[0] if xs[0] != 0 else 0.0
    slope = (m * sxy - sx * sy) / denom
    return slope


def measured_lacunarity(masks, n, k):
    """Build the n^k x n^k occupancy grid via Kronecker expansion of the k level masks,
    then gliding-box lacunarity Lam = <mass^2>/<mass>^2 at box size r = n^max(1,k//2)."""
    import numpy as np
    occ = mask_matrix(masks[0], n)
    for lvl in range(1, k):
        occ = np.kron(occ, mask_matrix(masks[lvl], n))
    L = occ.shape[0]
    m = max(1, k // 2)
    r = n ** m
    if r >= L:
        r = L // n if L // n >= 1 else 1
    P = np.zeros((L + 1, L + 1), dtype=np.int64)
    csum = np.cumsum(np.cumsum(occ.astype(np.int64), axis=0), axis=1)
    P[1:, 1:] = csum
    sums = P[r:, r:] - P[:-r, r:] - P[r:, :-r] + P[:-r, :-r]
    sums = sums.astype(np.float64)
    m1 = sums.mean()
    m2 = (sums * sums).mean()
    if m1 <= 1e-12:
        return 1.0
    return m2 / (m1 * m1)


def baseline_masks(n, k, Dstar):
    """Checker's own naive fixed-arrangement construction: same cardinality every
    level, TRUNCATED (floor) toward n^Dstar rather than rounded, cells filled in
    row-major (canonical) order -- a compact/clustered pattern that ignores
    Lam* and does not even round the cardinality carefully."""
    Nflat = int(n ** Dstar)  # floor for positive floats
    Nflat = max(1, min(n * n - 1, Nflat))
    order = [(r, c) for r in range(n) for c in range(n)]  # row-major
    cells = set(order[:Nflat])
    return [set(cells) for _ in range(k)]


def evaluate(masks, n, k, Dstar, Lamstar, wD, wL):
    """Objective (minimize): weighted L1 distance between the measured (D, Lam) of the
    grown attractor and the target (Dstar, Lamstar). Lacunarity is matched on a LOG
    scale (ln Lam) -- standard practice in fractal analysis (log-log lacunarity plots),
    and it keeps the objective well-conditioned across the huge dynamic range Lam can
    take (near 1 for homogeneous patterns, to 100s for sparse/clustered ones)."""
    sizes = [len(m) for m in masks]
    D_meas = measured_dimension(sizes, n)
    Lam_meas = measured_lacunarity(masks, n, k)
    F = wD * abs(D_meas - Dstar) + wL * abs(math.log(Lam_meas) - math.log(Lamstar))
    return F, D_meas, Lam_meas


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0")
        return 0
    inf, outf = sys.argv[1], sys.argv[2]
    n, k, Dstar, Lamstar, wD, wL = read_instance(inf)

    text = open(outf).read()
    masks, reason = parse_masks(text, n, k)
    if masks is None:
        print(f"infeasible: {reason}")
        print("Ratio: 0.0")
        return 0

    F, D_meas, Lam_meas = evaluate(masks, n, k, Dstar, Lamstar, wD, wL)
    if not (math.isfinite(F) and math.isfinite(D_meas) and math.isfinite(Lam_meas)):
        print("non-finite objective")
        print("Ratio: 0.0")
        return 0

    base_masks = baseline_masks(n, k, Dstar)
    B, Db, Lb = evaluate(base_masks, n, k, Dstar, Lamstar, wD, wL)
    B = max(B, 1e-6)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print(f"D_meas={D_meas:.4f} Lam_meas={Lam_meas:.4f} F={F:.4f} baseline={B:.4f}")
    print("Ratio: %.6f" % (sc / 1000.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
