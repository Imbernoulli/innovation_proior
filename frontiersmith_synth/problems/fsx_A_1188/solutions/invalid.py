# TIER: invalid
"""
Deliberately infeasible: claims to have spent Q+1 queries (over budget) and
lists Q+1 copies of node 1 (also a duplicate id) -- must score Ratio: 0.0.
"""
import sys


def main():
    toks = sys.stdin.read().split()
    Q = int(toks[3])
    m = Q + 1
    print(m)
    print(" ".join(["1"] * m))


if __name__ == "__main__":
    main()
