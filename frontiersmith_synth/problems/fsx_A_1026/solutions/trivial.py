# TIER: trivial
"""Reproduces the checker's own internal baseline: the all-zero refractive
lattice (spend none of the budget). Rays go straight; this is the trivial
feasible construction the checker normalizes against."""
import sys


def main():
    toks = sys.stdin.read().split()
    W = int(toks[0]); H = int(toks[1])
    row = " ".join("0" for _ in range(W))
    out = "\n".join(row for _ in range(H))
    print(out)


if __name__ == "__main__":
    main()
