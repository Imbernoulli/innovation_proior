# TIER: trivial
"""Reproduces the checker's own internal baseline: just the full length-1 layer
(the `a` single-character strings). Always feasible, always cheap, never insightful."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    a = int(next(it))
    Lmax = int(next(it))
    T = int(next(it))
    weight = [int(next(it)) for _ in range(Lmax)]
    cap = [int(next(it)) for _ in range(Lmax)]
    cap1 = cap[0]

    k = min(cap1, a, T)
    out = [str(d) for d in range(k)]
    print(len(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
