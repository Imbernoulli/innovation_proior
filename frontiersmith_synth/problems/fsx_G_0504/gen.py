#!/usr/bin/env python3
"""Deterministic generator for fsx_G_0504.

gen.py <testId> prints one instance. The difficulty ladder tightens and enlarges the
constant-weight cover-free packing problem from small to larger cases. All instance data
is determined by testId only.
"""
import sys


LADDER = [
    (20, 10, 5, 80),
    (22, 10, 5, 90),
    (24, 11, 6, 100),
    (26, 11, 6, 110),
    (28, 12, 6, 120),
    (30, 12, 6, 130),
    (32, 13, 7, 140),
    (34, 13, 7, 150),
    (36, 14, 7, 160),
    (38, 14, 7, 170),
]


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    try:
        t = int(sys.argv[1])
    except ValueError:
        t = 1
    if t < 1:
        t = 1
    if t > len(LADDER):
        t = len(LADDER)
    n, w, lam, cap = LADDER[t - 1]
    print("%d %d %d 2 %d %d" % (n, w, lam, cap, t))


if __name__ == "__main__":
    main()
