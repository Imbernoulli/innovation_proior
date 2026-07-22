# TIER: invalid
"""Emits an infeasible loading: every cell at (cap+1), which is out of range
and also almost certainly blows the budget."""
import sys


def main():
    toks = sys.stdin.read().split()
    p = 0
    N = int(toks[p]); p += 1
    _k = int(toks[p]); p += 1
    cap = int(toks[p]); p += 1
    Nc = N * N
    loads = [cap + 1] * Nc
    print(" ".join(map(str, loads)))


if __name__ == "__main__":
    main()
