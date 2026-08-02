# TIER: trivial
"""Reproduces the checker's own internal baseline exactly: the uniform
SOFTEST adhesive type (index 0, the first library row) on every segment.
Ignores the whole stiffness/strength library beyond its first row and never
considers grading at all."""
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); M = int(next(it))
    # Csub dAlpha C dT_1..dT_C k_0 s_0 .. k_{M-1} s_{M-1} -- all irrelevant here
    print(" ".join(["0"] * N))


if __name__ == "__main__":
    main()
