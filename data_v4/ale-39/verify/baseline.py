#!/usr/bin/env python3
"""Trivial baseline: 'all-zero codes' solution.

Outputs code 0 for every one of the S*K gains. With code 0 each gain maps to its box
lower bound (Kp = 0, Kd = 0, Kv = -1.2 in the standard boxes), i.e. an almost-passive
controller. This is always FEASIBLE (all codes in [0, Q]) but tracks the reference
poorly -- the solution the heuristic solver must beat.

Usage: python3 baseline.py <instance_file>   (writes S*K codes to stdout)
"""
import sys


def main():
    with open(sys.argv[1]) as f:
        toks = f.read().split()
    it = iter(toks)
    S = int(next(it)); K = int(next(it)); _Q = int(next(it))
    out = []
    for _s in range(S):
        out.append(" ".join("0" for _ in range(K)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
