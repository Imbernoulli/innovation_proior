# TIER: strong
# Two-stage deterministic search: (1) pick the best seed among rank-1 Korobov lattices
# (generating vector (1,a,a^2 mod m) with a shift sweep) and the Hammersley set, scored by
# the EXACT 3D star discrepancy; (2) refine it with seeded coordinate-descent hill climbing
# that resamples single coordinates and keeps only strict improvements. Which seed wins and
# how far refinement gets both vary per instance size.
import sys
import random
import numpy as np

def star_discrepancy_3d(P, m):
    P = np.asarray(P, dtype=np.float64)
    grids, ic, io = [], [], []
    for a in range(3):
        col = P[:, a]
        g = np.unique(np.concatenate([col, np.array([1.0])]))
        grids.append(g)
        ic.append(np.searchsorted(g, col, side="left"))
        io.append(np.searchsorted(g, col, side="right"))
    L = [len(g) for g in grids]
    Hc = np.zeros(tuple(L), dtype=np.int64)
    np.add.at(Hc, (ic[0], ic[1], ic[2]), 1)
    Nc = Hc.cumsum(0).cumsum(1).cumsum(2)
    Ho = np.zeros(tuple(L), dtype=np.int64)
    valid = (io[0] < L[0]) & (io[1] < L[1]) & (io[2] < L[2])
    ioc = [np.clip(io[a], 0, L[a] - 1) for a in range(3)]
    np.add.at(Ho, (ioc[0][valid], ioc[1][valid], ioc[2][valid]), 1)
    No = Ho.cumsum(0).cumsum(1).cumsum(2)
    V = np.multiply.outer(np.multiply.outer(grids[0], grids[1]), grids[2])
    return float(max((Nc / m - V).max(), (V - No / m).max()))

def radinv(i, b):
    f, r = 1.0, 0.0
    while i > 0:
        f /= b; r += f * (i % b); i //= b
    return r

def main():
    tok = sys.stdin.read().split()
    m = int(tok[0])
    cands = []

    # Hammersley candidate
    ham = []
    for i in range(m):
        y = min(1.0 - 1e-9, max(1e-9, radinv(i, 2)))
        z = min(1.0 - 1e-9, max(1e-9, radinv(i, 3)))
        ham.append(((i + 0.5) / m, y, z))
    cands.append(ham)

    # Korobov rank-1 lattices with a small deterministic shift sweep
    shifts = [0.5 / m, 0.0]
    for a in range(1, m):
        a2 = (a * a) % m
        for s in shifts:
            pts = []
            for i in range(m):
                x = ((i % m) + s) % 1.0
                if x >= 1.0: x -= 1.0
                y = (((i * a) % m) / m + s) % 1.0
                zc = (((i * a2) % m) / m + s) % 1.0
                pts.append((x, y, zc))
            cands.append(pts)

    best = None
    best_d = 1e18
    for c in cands:
        d = star_discrepancy_3d(c, m)
        if d < best_d:
            best_d = d; best = c

    # Stage 2: seeded coordinate-descent hill climbing on the best seed.
    rng = random.Random(20240624)
    cur = [list(p) for p in best]
    cur_d = best_d
    budget = min(1500, 45 * m)
    for _ in range(budget):
        j = rng.randrange(m)
        a = rng.randrange(3)
        old = cur[j][a]
        cur[j][a] = rng.random()
        d = star_discrepancy_3d(cur, m)
        if d < cur_d - 1e-12:
            cur_d = d
        else:
            cur[j][a] = old
    if cur_d < best_d:
        best = cur

    out = []
    for (x, y, z) in best:
        x = min(1.0, max(0.0, x)); y = min(1.0, max(0.0, y)); z = min(1.0, max(0.0, z))
        out.append("%.10f %.10f %.10f" % (x, y, z))
    sys.stdout.write("\n".join(out) + "\n")

main()
