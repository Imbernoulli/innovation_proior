# TIER: strong
# Best-of low-discrepancy search: build Hammersley sets under several prime-base
# assignments, apply a deterministic grid of Cranley-Patterson digital shifts,
# and emit the point set with the smallest EXACT star discrepancy.
import sys
import math
import itertools

TOL = 1e-12
PRIMES = [2, 3, 5, 7, 11]

def radinv(i, b):
    f = 1.0
    r = 0.0
    while i > 0:
        f /= b
        r += f * (i % b)
        i //= b
    return r

def frac(x):
    return x - math.floor(x)

def star_discrepancy(pts, d):
    n = len(pts)
    axes = []
    for k in range(d):
        vals = sorted(set(p[k] for p in pts) | {1.0})
        axes.append(vals)
    best = 0.0
    for corner in itertools.product(*axes):
        vol = 1.0
        for c in corner:
            vol *= c
        closed = 0
        opencnt = 0
        for p in pts:
            le = True
            lt = True
            for k in range(d):
                if p[k] > corner[k] + TOL:
                    le = False
                    lt = False
                    break
                if not (p[k] < corner[k] - TOL):
                    lt = False
            if le:
                closed += 1
            if lt:
                opencnt += 1
        over = closed / n - vol
        under = vol - opencnt / n
        if over > best:
            best = over
        if under > best:
            best = under
    return best

def hammersley(M, d, bases):
    pts = []
    for i in range(M):
        p = [(i + 0.5) / M]
        for k in range(d - 1):
            p.append(radinv(i, bases[k]))
        pts.append(p)
    return pts

def main():
    d, M = map(int, sys.stdin.read().split()[:2])

    # candidate prime-base permutations for the trailing (d-1) dimensions
    if d >= 2:
        base_choices = [
            [2, 3, 5, 7],
            [3, 2, 5, 7],
            [5, 3, 2, 7],
            [7, 5, 3, 2],
        ]
    else:
        base_choices = [[2]]
    # coarser shift grid in higher dimension to stay well inside the time limit
    nshift = 4 if d <= 2 else 2
    shifts = [j / nshift for j in range(nshift)]

    best_pts = None
    best_val = float("inf")
    for bc in base_choices:
        base_pts = hammersley(M, d, bc)
        for combo in itertools.product(shifts, repeat=d):
            pts = [tuple(frac(base_pts[i][k] + combo[k]) for k in range(d))
                   for i in range(M)]
            v = star_discrepancy(pts, d)
            if v < best_val:
                best_val = v
                best_pts = pts

    out = []
    for p in best_pts:
        out.append(" ".join("%.12f" % c for c in p))
    sys.stdout.write("\n".join(out) + "\n")

main()
