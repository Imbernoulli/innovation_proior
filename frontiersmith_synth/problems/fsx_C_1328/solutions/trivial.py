# TIER: trivial
"""Do the least possible: install the single mutation with the best individual
stability delta among those that individually respect the activity floor. This
exactly reproduces the checker's own internal baseline B."""
import sys


def main():
    toks = sys.stdin.read().split()
    p = 0

    def nxt():
        nonlocal p
        v = toks[p]
        p += 1
        return v

    n = int(nxt()); K = int(nxt()); C = int(nxt()); R = int(nxt())
    A0 = float(nxt()); ActMin = float(nxt()); alpha = float(nxt())
    dstab = [0.0] * n
    dact = [0.0] * n
    dist = [0] * n
    for i in range(n):
        dstab[i] = float(nxt())
        dact[i] = float(nxt())
        dist[i] = int(nxt())
    # epistasis table is irrelevant to a single-mutation choice; ignore it.

    best = -1
    best_val = -1.0
    for i in range(n):
        cc = 1 if dist[i] <= R else 0
        crowd = alpha * max(0, cc - C) ** 2
        if (A0 + dact[i] - crowd) >= ActMin - 1e-9 and dstab[i] > best_val:
            best_val = dstab[i]
            best = i

    if best == -1:
        print(0)
        print()
    else:
        print(1)
        print(best)


if __name__ == "__main__":
    main()
