# TIER: invalid
"""Keeps every fusion edge, including all the redundant hinge candidates at
each boundary. Because m > n-1 by construction, this always contains a
cycle, so the checker must reject it as infeasible (Ratio: 0.0)."""
import sys


def main():
    toks = sys.stdin.read().split()
    m = int(toks[1])
    print(m)
    print(" ".join(str(i) for i in range(m)))


if __name__ == "__main__":
    main()
