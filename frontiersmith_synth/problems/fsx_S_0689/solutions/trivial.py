# TIER: trivial
# Uniform coupling at the midpoint of the allowed range -- no tuning, no shape.
import sys


def main():
    d = sys.stdin.read().split()
    it = iter(d)
    N = int(next(it))
    T = float(next(it))
    J_LO = float(next(it)); J_HI = float(next(it))
    D = int(next(it))
    for _ in range(D):
        next(it); next(it)
    K = int(next(it))
    frozen_edges = set()
    for _ in range(K):
        e = int(next(it)); next(it)
        frozen_edges.add(e)

    free_edges = sorted(e for e in range(1, N) if e not in frozen_edges)
    mid = 0.5 * (J_LO + J_HI)
    print(len(free_edges))
    print(" ".join("%.6f" % mid for _ in free_edges))


if __name__ == "__main__":
    main()
