#!/usr/bin/env python3
"""
gen.py <testId>   -- prints ONE "sculpture" instance to stdout.

Instance format:
    line 1:   V
    next V:   "x y z"   (the target voxel set, sorted lexicographically)

The target is built internally as an iterated-function-system (IFS) fractal
(recursively: keep a fixed subset K of the 27 offsets of a 3x3x3 grid, apply
L times) or a deliberately unstructured / composite variant, so that the
instances span "cleanly self-similar" through "no exploitable structure"
(control) through "self-similar but only visible across the WHOLE structure,
not row by row" (the trap). Everything is seeded from testId only.
"""
import sys
import random

BASE = 3
ALL27 = [(a, b, c) for a in range(BASE) for b in range(BASE) for c in range(BASE)]


def ifs_points(L, K):
    """K: list of (a,b,c) offsets in [0,BASE). Returns the set of integer
    points after L recursive levels, starting from a single unit voxel."""
    pts = {(0, 0, 0)}
    scale = 1
    for _ in range(L):
        new_pts = set()
        for (a, b, c) in K:
            oa, ob, oc = a * scale, b * scale, c * scale
            for (x, y, z) in pts:
                new_pts.add((x + oa, y + ob, z + oc))
        pts = new_pts
        scale *= BASE
    return pts


def pick_K(rng, m, allowed=None):
    pool = list(allowed) if allowed is not None else list(ALL27)
    rng.shuffle(pool)
    return pool[:m]


def pick_K_symmetric(rng, m):
    """K symmetric under x -> (BASE-1-x): for every chosen (a,b,c) with a!=1
    we also keep its mirror (BASE-1-a,b,c). Gives a shape whose EVERY level is
    mirror-symmetric about the x mid-plane, so reflect() is genuinely useful."""
    a1 = [(b, c) for b in range(BASE) for c in range(BASE)]          # a==1 slice (self-mirrored)
    a0 = [(b, c) for b in range(BASE) for c in range(BASE)]          # a==0 slice (mirror is a==2)
    rng.shuffle(a1)
    rng.shuffle(a0)
    n1 = m % 2
    n0 = (m - n1) // 2
    n1 = min(n1, len(a1))
    n0 = min(n0, len(a0))
    K = [(1, b, c) for (b, c) in a1[:n1]]
    for (b, c) in a0[:n0]:
        K.append((0, b, c))
        K.append((BASE - 1, b, c))
    return K


def translate(pts, dx, dy, dz):
    return {(x + dx, y + dy, z + dz) for (x, y, z) in pts}


def gen_case(tid):
    rng = random.Random(1000003 * tid + 17)

    if tid == 1:
        # solid 3x3x3 cube: warm-up, tiny.
        pts = ifs_points(1, ALL27)
    elif tid == 2:
        # a 2D self-similar layer (c always 0, i.e. never grows along z)
        # replicated identically across 5 z-layers: a REPEATED, non-fractal
        # motif that a "cache repeated rows/layers" greedy handles well.
        K2d = pick_K(rng, 9, allowed=[(a, b, 0) for a in range(BASE) for b in range(BASE)])
        layer = ifs_points(2, K2d)
        pts = set()
        for z in range(5):
            pts |= translate(layer, 0, 0, z)
    elif tid == 3:
        K = pick_K(rng, 13, allowed=[o for o in ALL27 if o != (1, 1, 1)])
        pts = ifs_points(2, K)
    elif tid == 4:
        K = pick_K(rng, 10, allowed=[o for o in ALL27 if o != (1, 1, 1)])
        pts = ifs_points(3, K)
    elif tid == 5:
        # composite: two DIFFERENT fractals, far apart (no accidental overlap).
        K1 = pick_K(rng, 11, allowed=[o for o in ALL27 if o != (1, 1, 1)])
        K2 = pick_K(rng, 11, allowed=[o for o in ALL27 if o != (1, 1, 1)])
        A = ifs_points(2, K1)
        B = ifs_points(2, K2)
        pts = A | translate(B, 5000, 0, 0)
    elif tid == 6:
        K = pick_K(rng, 16, allowed=[o for o in ALL27 if o != (1, 1, 1)])
        pts = ifs_points(3, K)
    elif tid == 7:
        # unstructured control: no self-similarity to exploit. Random subset
        # of a bounding box, no repeated motif planted.
        W = 22
        universe = [(x, y, z) for x in range(W) for y in range(W) for z in range(2)]
        rng.shuffle(universe)
        pts = set(universe[:800])
    elif tid == 8:
        # mirror-symmetric fractal: reflect() genuinely helps here.
        K = pick_K_symmetric(rng, 14)
        pts = ifs_points(3, K)
    elif tid == 9:
        # sparse "corners" fractal (few kept offsets -> deep, sparse, hard for
        # any row/run-based heuristic since occupied cells are scattered).
        corners = [(a, b, c) for a in (0, 2) for b in (0, 2) for c in (0, 2)]  # 8 of them
        pts = ifs_points(4, corners)
    else:  # tid == 10: largest, mirror-symmetric, deep -- the biggest trap.
        K = pick_K_symmetric(rng, 12)
        pts = ifs_points(4, K)

    return pts


def main():
    if len(sys.argv) != 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    tid = int(sys.argv[1])
    pts = gen_case(tid)
    out = sys.stdout
    out.write("%d\n" % len(pts))
    for (x, y, z) in sorted(pts):
        out.write("%d %d %d\n" % (x, y, z))


if __name__ == "__main__":
    main()
