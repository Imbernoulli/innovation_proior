# TIER: invalid
"""Deliberately infeasible: claims codeword 0 uses catalog entry 0 (word length >= 2)
but lists only a single cell for it, so the checker's partition parse runs out of
tokens and must reject with Ratio: 0.0."""
import sys


def main():
    toks = sys.stdin.read().split()
    N, M, LMAX = int(toks[0]), int(toks[1]), int(toks[2])
    print("1")
    print("0 0")


if __name__ == "__main__":
    main()
