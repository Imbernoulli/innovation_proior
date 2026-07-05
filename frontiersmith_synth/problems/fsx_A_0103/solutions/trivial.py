# TIER: trivial
"""Reproduces the checker's internal grid baseline: small equal domes on a coarse
grid, skipping cells near the antenna. Scores ~0.1 by construction."""
import sys
import math

CX, CY = 0.5, 0.5


def main():
    toks = sys.stdin.read().split()
    N = int(toks[0])
    r0 = float(toks[1])
    G = int(math.ceil(math.sqrt(N))) + 2
    rb = 0.3 / G
    out = []
    for i in range(G):
        for j in range(G):
            if len(out) >= N:
                break
            x = (i + 0.5) / G
            y = (j + 0.5) / G
            if math.hypot(x - CX, y - CY) >= r0 + rb + 1e-12:
                out.append((x, y, rb))
        if len(out) >= N:
            break
    for (x, y, r) in out:
        print("%.9f %.9f %.9f" % (x, y, r))


if __name__ == "__main__":
    main()
