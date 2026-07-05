#!/usr/bin/env python3
"""gen.py <testId>  ->  prints one instance (a single integer n) to stdout.

Difficulty ladder: testId 1..10 maps to increasing n (more wiring cells ->
larger self-convolution, harder to flatten the crosstalk peak). Deterministic:
depends only on testId.
"""
import sys


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if t < 1:
        t = 1
    n = 6 + 3 * (t - 1)   # t=1 -> 6, t=10 -> 33
    print(n)


if __name__ == "__main__":
    main()
