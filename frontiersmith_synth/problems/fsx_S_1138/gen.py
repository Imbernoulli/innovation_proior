import sys, random

# gen.py <testId> -- prints ONE paper-snowflake hole-target instance to stdout.
#
# Construction: pick a fold depth d and a small "core" pattern P of shape c x c
# (c = N / 2^d) with a handful of True cells.  Replicate P onto the full N x N
# sheet via the EXACT recursive "fold right-half-onto-left-half" / "fold
# bottom-half-onto-top-half" map (same map the solver must discover), so the
# resulting target is by construction the orbit of P under the depth-d fold
# subgroup -- perfectly foldable.  Then flip a handful of individual "defect"
# cells (trap tests, e>0): these are single original cells whose value breaks
# the pattern's GLOBAL bilateral symmetry (poisoning naive whole-sheet folding)
# while leaving the vast majority of the orbit structure intact.
#
# testId 1..10: difficulty ladder, N grows 4..64.  Tests 3,5,7,9,10 plant
# defects (trap cases); 1,2,4,6,8 are clean (no defects).

# (k, d, fillcount, e, seed) : N=2^k, fold depth d (per axis), number of True
# core cells, number of defect toggles, RNG seed.
SPECS = {
    1:  (2, 1, 2,  0, 1001),
    2:  (3, 1, 3,  0, 1002),
    3:  (3, 1, 3,  2, 1003),
    4:  (4, 1, 5,  0, 1004),
    5:  (4, 1, 5,  3, 1005),
    6:  (4, 2, 4,  0, 1006),
    7:  (5, 1, 8,  4, 1007),
    8:  (5, 2, 4,  0, 1008),
    9:  (6, 1, 12, 6, 1009),
    10: (6, 2, 5,  8, 1010),
}


def fold_reduce_indices(N, d):
    """pos[i] = coordinate that original index i lands on after d applications
    of 'fold the far half onto the near half' (exact bisection), starting from
    width N.  This is the SAME map used by FOLD_X / FOLD_Y in the checker."""
    pos = list(range(N))
    Wc = N
    for _ in range(d):
        newWc = Wc // 2
        for i in range(N):
            if pos[i] >= newWc:
                pos[i] = Wc - 1 - pos[i]
        Wc = newWc
    return pos


def main():
    tid = int(sys.argv[1])
    k, d, fillcount, e, seed = SPECS[tid]
    N = 1 << k
    c = N >> d
    rng = random.Random(seed)

    cells = [(a, b) for a in range(c) for b in range(c)]
    rng.shuffle(cells)
    chosen = set(cells[:max(1, fillcount)])

    px = fold_reduce_indices(N, d)

    target = [[(px[i], px[j]) in chosen for j in range(N)] for i in range(N)]

    defects = set()
    while len(defects) < e:
        i = rng.randrange(N)
        j = rng.randrange(N)
        defects.add((i, j))
    for (i, j) in defects:
        target[i][j] = not target[i][j]

    holes = [(i, j) for i in range(N) for j in range(N) if target[i][j]]
    if not holes:
        holes = [(0, 0)]

    out = [f"{N}", f"{len(holes)}"]
    for (i, j) in holes:
        out.append(f"{i} {j}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
