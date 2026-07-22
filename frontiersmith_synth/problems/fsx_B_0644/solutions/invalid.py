# TIER: invalid
"""Blast every edge to an upgrade level far past its cap -- infeasible output."""
import sys


def main():
    toks = sys.stdin.read().split()
    idx = 0
    n = int(toks[idx]); idx += 1
    m = int(toks[idx]); idx += 1
    print("\n".join(["99"] * m))


if __name__ == "__main__":
    main()
