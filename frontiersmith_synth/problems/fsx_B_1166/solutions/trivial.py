# TIER: trivial
"""Trivial baseline: assume the contamination is a diffuse area source and release the
entire mass budget UNIFORMLY over every grid cell. No use of the well readings at all.
This is exactly the checker's own internal baseline construction, so it scores ~0.1."""
import sys


def main():
    toks = sys.stdin.read().split()
    pos = 0
    test_id = int(toks[pos]); pos += 1
    N = int(toks[pos]); pos += 1
    K = int(toks[pos]); pos += 1
    MT = int(toks[pos]); pos += 1
    S_MAX = int(toks[pos]); pos += 1
    # skip D vx vy
    pos += 3
    # skip times
    pos += MT
    B_mass = float(toks[pos]); pos += 1

    M = N * N
    rate = B_mass / M
    sys.stdout.write(" ".join("%.8f" % rate for _ in range(M)) + "\n")


if __name__ == "__main__":
    main()
