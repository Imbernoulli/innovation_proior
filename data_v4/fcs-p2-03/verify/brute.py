#!/usr/bin/env python3
"""Independent brute-force oracle for the maximum product subarray problem.

Reads the same stdin format as sol.cpp:
    n
    a[0] a[1] ... a[n-1]
and prints the maximum product over all contiguous (non-empty) subarrays.

This deliberately uses the most naive O(n^2) double loop with Python's exact
big integers, so it shares no logic with the O(n) min/max DP being verified.
"""
import sys


def solve(n, a):
    best = None
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= a[j]
            if best is None or prod > best:
                best = prod
    return best


def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    a = [int(data[idx + k]) for k in range(n)]
    print(solve(n, a))


if __name__ == "__main__":
    main()
