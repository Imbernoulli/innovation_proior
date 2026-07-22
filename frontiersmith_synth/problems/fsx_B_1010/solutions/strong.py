# TIER: strong
"""The insight: on a digit-selection carpet built from k per-level masks
(each a subset of an n x n grid), the box-counting dimension depends ONLY on
the per-level cardinalities N_1..N_k (their product determines the box
count at every scale) -- it does NOT depend on which cells are chosen.
Lacunarity, on the other hand, depends entirely on the spatial arrangement
of the chosen cells and is nearly insensitive to small cardinality tweaks.
This decouples a coupled-looking 2-target matching problem into two
independent 1-D searches: (1) pick per-level cardinalities to fit D*
(allowing a MIX of two cardinalities across levels to hit fractional
targets no single integer N can reach exactly), (2) independently pick,
for each level, a placement pattern along a compact<->dispersed spectrum
to fit Lam*. We evaluate a handful of candidates from each axis (crossed)
with the same objective the checker uses and keep the best."""
import sys, math


def clustered_cells(n, N):
    order = [(r, c) for r in range(n) for c in range(n)]
    return order[:N]


def dispersed_cells(n, N):
    """Farthest-point (max-min-distance) greedy placement: spreads N cells
    across the n x n grid as evenly as possible."""
    cells = [(r, c) for r in range(n) for c in range(n)]
    chosen = [(0, 0)]
    remaining = [c for c in cells if c != (0, 0)]
    while len(chosen) < N:
        best = max(remaining,
                    key=lambda p: min((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 for q in chosen))
        chosen.append(best)
        remaining.remove(best)
    return chosen[:N]


def measured_dimension(sizes, n):
    k = len(sizes)
    logn = math.log(n)
    xs, ys, acc = [], [], 0.0
    for j in range(1, k + 1):
        acc += math.log(sizes[j - 1])
        xs.append(j * logn); ys.append(acc)
    m = len(xs)
    sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    denom = m * sxx - sx * sx
    if abs(denom) < 1e-12:
        return ys[0] / xs[0] if xs[0] != 0 else 0.0
    return (m * sxy - sx * sy) / denom


def measured_lacunarity(cell_lists, n, k):
    import numpy as np

    def mat(cells):
        M = np.zeros((n, n), dtype=np.uint8)
        for (r, c) in cells:
            M[r, c] = 1
        return M

    occ = mat(cell_lists[0])
    for lvl in range(1, k):
        occ = np.kron(occ, mat(cell_lists[lvl]))
    L = occ.shape[0]
    m = max(1, k // 2)
    r = n ** m
    if r >= L:
        r = max(1, L // n)
    P = np.zeros((L + 1, L + 1), dtype=np.int64)
    P[1:, 1:] = np.cumsum(np.cumsum(occ.astype(np.int64), axis=0), axis=1)
    sums = (P[r:, r:] - P[:-r, r:] - P[r:, :-r] + P[:-r, :-r]).astype(np.float64)
    m1 = sums.mean(); m2 = (sums * sums).mean()
    return m2 / (m1 * m1) if m1 > 1e-12 else 1.0


def objective(cell_lists, n, k, Dstar, Lamstar, wD, wL):
    sizes = [len(c) for c in cell_lists]
    D_meas = measured_dimension(sizes, n)
    Lam_meas = measured_lacunarity(cell_lists, n, k)
    return wD * abs(D_meas - Dstar) + wL * abs(math.log(Lam_meas) - math.log(Lamstar))


def main():
    data = sys.stdin.read().split()
    n = int(data[0]); k = int(data[1])
    Dstar = float(data[2]); Lamstar = float(data[3])
    wD = float(data[4]); wL = float(data[5])

    base_est = n ** Dstar
    lo = max(1, int(math.floor(base_est)))
    hi = min(n * n - 1, int(math.ceil(base_est)))
    nearest = max(1, min(n * n - 1, int(round(base_est))))

    size_candidates = {tuple([nearest] * k), tuple([lo] * k), tuple([hi] * k)}
    if lo != hi:
        for num_hi in range(0, k + 1):
            size_candidates.add(tuple([hi] * num_hi + [lo] * (k - num_hi)))

    best_F, best_cells = None, None
    for sizes in size_candidates:
        for mode in ("clustered", "dispersed", "mixed"):
            cell_lists = []
            for i, N in enumerate(sizes):
                if mode == "clustered":
                    cell_lists.append(clustered_cells(n, N))
                elif mode == "dispersed":
                    cell_lists.append(dispersed_cells(n, N))
                else:
                    cell_lists.append(clustered_cells(n, N) if i % 2 == 0 else dispersed_cells(n, N))
            F = objective(cell_lists, n, k, Dstar, Lamstar, wD, wL)
            if best_F is None or F < best_F:
                best_F, best_cells = F, cell_lists

    out = [str(k)]
    for cells in best_cells:
        parts = [str(len(cells))]
        for (r, c) in cells:
            parts.append(str(r)); parts.append(str(c))
        out.append(" ".join(parts))
    print("\n".join(out))


if __name__ == "__main__":
    main()
