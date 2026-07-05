#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE instance to stdout.

Instance format:
    a single integer n  (number of corridor segments)

Difficulty ladder (testId 1..10): n grows from small to larger, all "small"
scale so the O(n^2) exact autoconvolution stays cheap.
"""
import sys


def main():
    t = int(sys.argv[1])
    # n = 8, 10, 12, ..., 26  for t = 1..10
    n = 6 + 2 * t
    print(n)


if __name__ == "__main__":
    main()
