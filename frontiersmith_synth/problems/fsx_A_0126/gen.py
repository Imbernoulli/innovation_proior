#!/usr/bin/env python3
"""gen.py <testId> -- print ONE geothermal-grid instance to stdout.

testId 1..6 is a difficulty ladder in N (grid side).  A deterministic subset of
cells is 'cemented' (fixed) to the baseline block-grid's value at that cell, so the
baseline construction is always a feasible output.  Seeded by testId only.
"""
import sys, random


def build_baseline(N):
    """Block-diagonal 0/1 grid; |det| = 2^(N//3)."""
    M = [[0] * N for _ in range(N)]
    pos = 0
    blk = [[1, 1, 0], [0, 1, 1], [1, 0, 1]]  # det = 2
    while pos + 3 <= N:
        for a in range(3):
            for b in range(3):
                M[pos + a][pos + b] = blk[a][b]
        pos += 3
    r = N - pos
    if r == 1:
        M[pos][pos] = 1
    elif r == 2:
        M[pos][pos] = 1
        M[pos][pos + 1] = 1
        M[pos + 1][pos] = 1
        M[pos + 1][pos + 1] = 0
    return M


def main():
    testId = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    Ns = {1: 8, 2: 10, 3: 12, 4: 16, 5: 20, 6: 22}
    N = Ns.get(testId, 12)
    rng = random.Random(90000 + testId)

    base = build_baseline(N)
    # Cement ~20% of the cells to their baseline value.
    K = max(1, round(0.20 * N * N))
    cells = [(i, j) for i in range(N) for j in range(N)]
    rng.shuffle(cells)
    fixed = cells[:K]

    out = ["%d %d" % (N, K)]
    for (i, j) in fixed:
        out.append("%d %d %d" % (i, j, base[i][j]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
