# TIER: trivial
#!/usr/bin/env python3
"""Spread the whole height budget evenly over every cell -- exactly the
checker's own reference construction, so this scores ~0.1."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    Budget = int(next(it))
    K = int(next(it))
    for _ in range(N):
        next(it)
    for _ in range(N):
        next(it)
    for _ in range(K):
        next(it)
        next(it)
        next(it)

    h = [Budget // N] * N
    rem = Budget - sum(h)
    for i in range(rem):
        h[i] += 1
    print(" ".join(map(str, h)))


if __name__ == "__main__":
    main()
