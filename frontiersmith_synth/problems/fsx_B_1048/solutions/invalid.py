# TIER: invalid
"""Deliberately infeasible: emits a cut point outside [1,G-1]. Must score 0.0."""
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    K = int(next(it)); G = int(next(it))
    bad_cut = G + 5
    sys.stdout.write(f"1 {bad_cut}\nS S\n")


if __name__ == "__main__":
    main()
