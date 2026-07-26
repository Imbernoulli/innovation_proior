# TIER: invalid
"""Emits an infeasible artifact: every edge weight is set to cap+1, which
violates the 1 <= w_e <= cap_e range for every single edge. Must score 0."""
import sys


def main():
    toks = sys.stdin.read().split()
    p = 0
    n = int(toks[p]); p += 1
    m = int(toks[p]); p += 1
    T = int(toks[p]); p += 1
    caps = []
    for _ in range(m):
        u = int(toks[p]); v = int(toks[p + 1]); cap = int(toks[p + 2]); p += 3
        caps.append(cap)
    out = [str(cap + 1) for cap in caps]
    sys.stdout.write(" ".join(out) + "\n")


if __name__ == "__main__":
    main()
