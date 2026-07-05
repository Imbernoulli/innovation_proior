#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE instance to stdout.

Instance format:
    a single integer n  (number of reservoirs in the chain)

Difficulty ladder (testId 1..10): n grows from small to large. The exact
O(n^2) autoconvolution stays cheap even at the largest size.
"""
import sys

SIZES = [12, 20, 32, 48, 68, 92, 120, 152, 190, 240]


def main():
    t = int(sys.argv[1])
    if t < 1:
        t = 1
    if t > len(SIZES):
        t = len(SIZES)
    print(SIZES[t - 1])


if __name__ == "__main__":
    main()
