# TIER: invalid
"""Garbage: every weight set to -1, which violates every edge's lower bound
(all lower bounds are >= 1). Must score 0."""
import sys


def main():
    toks = sys.stdin.read().split()
    m = int(toks[1])
    print(" ".join(["-1.0"] * m))


if __name__ == "__main__":
    main()
