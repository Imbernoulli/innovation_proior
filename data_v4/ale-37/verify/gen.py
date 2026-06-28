#!/usr/bin/env python3
"""Instance generator for "Quadratic Assignment Placement" (QAP, ALE-Bench).

Usage:
    python3 gen.py <seed>

Writes one instance to stdout:

    n
    f_{0,0} f_{0,1} ... f_{0,n-1}
    ...                                 (n rows: the FLOW matrix f)
    f_{n-1,0} ... f_{n-1,n-1}
    d_{0,0} d_{0,1} ... d_{0,n-1}
    ...                                 (n rows: the DISTANCE matrix d)
    d_{n-1,0} ... d_{n-1,n-1}

`n` is the number of facilities = number of locations. `f[i][j]` is the flow
between facilities i and j; `d[k][l]` is the distance between locations k and l.
A solution places facility i on location p[i] (a permutation) and pays
    cost = sum_i sum_j f[i][j] * d[p[i]][p[j]].

Structure (this is what makes the problem non-trivial AND makes the identity
permutation a beatable-but-positive baseline):

  * DISTANCES are EUCLIDEAN on a random 2-D layout: n location points are drawn
    uniformly in a square, and d[k][l] = round(euclidean(point_k, point_l)).
    The matrix is symmetric with a zero diagonal -- a real geometric distance
    matrix, the kind QAPLIB's "nug"/"tai*a" families use.

  * FLOWS are CLUSTERED: facilities are split into a few groups; within a group
    flows are high, across groups low, plus noise. This gives the instance real
    structure (group-mates want to be near each other), so a good permutation
    that co-locates heavily-communicating facilities at nearby locations scores
    well below the identity arrangement -- but the coupling between the flow
    clustering and the geometric layout makes finding that permutation NP-hard.

`n` is chosen deterministically from the seed in a range where exhaustive search
is hopeless (n! is astronomical) yet a strong local search has room to run.
"""
import sys
import math
import random


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py <seed>\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    rng = random.Random(0x37A1_0000 ^ (seed * 2654435761 & 0xFFFFFFFF))

    # problem size: large enough that n! is astronomical, small enough that a
    # strong O(n^2)-per-sweep tabu search gets thousands of sweeps in the budget.
    n = rng.randint(40, 80)

    # ---- DISTANCE matrix: euclidean on a random 2-D point layout ----
    span = 100.0
    pts = [(rng.uniform(0, span), rng.uniform(0, span)) for _ in range(n)]
    d = [[0] * n for _ in range(n)]
    for k in range(n):
        xk, yk = pts[k]
        for l in range(n):
            xl, yl = pts[l]
            dist = math.hypot(xk - xl, yk - yl)
            d[k][l] = int(round(dist))
        d[k][k] = 0

    # ---- FLOW matrix: clustered facilities, symmetric, zero diagonal ----
    num_groups = rng.randint(3, 6)
    group = [rng.randrange(num_groups) for _ in range(n)]
    f = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if group[i] == group[j]:
                base = rng.randint(20, 100)      # strong intra-group flow
            else:
                # most cross-group pairs have little or no flow
                if rng.random() < 0.30:
                    base = rng.randint(1, 15)
                else:
                    base = 0
            f[i][j] = base
            f[j][i] = base
        f[i][i] = 0

    out = [str(n)]
    for i in range(n):
        out.append(" ".join(str(f[i][j]) for j in range(n)))
    for k in range(n):
        out.append(" ".join(str(d[k][l]) for l in range(n)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
