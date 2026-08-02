# TIER: trivial
"""Uniform stiffness: ignore the target spectrum entirely, just take the
midpoint of every edge's allowed weight range. This is exactly the checker's
own internal baseline construction."""
import sys


def main():
    toks = sys.stdin.read().split()
    p = 0
    n = int(toks[p]); p += 1
    m = int(toks[p]); p += 1
    bounds = []
    for _ in range(m):
        p += 2  # u, v
        lo = float(toks[p]); p += 1
        hi = float(toks[p]); p += 1
        bounds.append((lo, hi))
    w = [(lo + hi) / 2.0 for (lo, hi) in bounds]
    print(" ".join("%.6f" % x for x in w))


if __name__ == "__main__":
    main()
