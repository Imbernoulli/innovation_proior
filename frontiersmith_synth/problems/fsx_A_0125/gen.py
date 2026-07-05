#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE instance to stdout.

Instance format:
    a single integer n  (number of wind-tunnel sensors)

Difficulty ladder (testId 1..10): n grows from 12 to 30. Medium scale, so the
O(n^2) exact autoconvolution in the checker stays cheap and deterministic.
"""
import sys


def main():
    t = int(sys.argv[1])
    # n = 12, 14, 16, ..., 30  for t = 1..10
    n = 10 + 2 * t
    print(n)


if __name__ == "__main__":
    main()
