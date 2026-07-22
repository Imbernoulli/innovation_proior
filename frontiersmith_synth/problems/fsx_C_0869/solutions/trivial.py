# TIER: trivial
"""Naive per-requirement sizing: for every check, assume EVERY member of that check
must individually carry a fair (1/count) share of the budget, inflated by a fixed
safety scale (SCALE=3) that a cautious engineer might apply without ever asking
"how many chains does this feature actually serve". Reproduces the checker's own
internal baseline construction (by design, this scores ~0.1)."""
import sys

TOL = [64, 32, 16, 8, 4, 2, 1]
GMAX = 6
SCALE = 3


def main():
    data = sys.stdin.read().split()
    idx = 0
    m = int(data[idx]); idx += 1
    C = int(data[idx]); idx += 1
    constraints = [[] for _ in range(m)]
    for _ in range(C):
        p1 = int(data[idx]); idx += 1
        p2 = int(data[idx]); idx += 1
        h = int(data[idx]); idx += 1
        sp = int(data[idx]); idx += 1
        sb = int(data[idx]); idx += 1
        constraints[p1].append((sp, 3))
        constraints[p1].append((sb, 2))
        constraints[p2].append((sp, 3))
        constraints[p2].append((sb, 2))
        constraints[h].append((sp, 3))

    grade = [0] * m
    for f in range(m):
        g = 0
        for spec, cnt in constraints[f]:
            while g < GMAX and TOL[g] * cnt * SCALE > spec:
                g += 1
        grade[f] = g

    sys.stdout.write(" ".join(map(str, grade)) + "\n")


if __name__ == "__main__":
    main()
