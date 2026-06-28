#!/usr/bin/env python3
"""Independent brute-force oracle.

Reads the same stdin format as sol.cpp:
    n T
    a[0] a[1] ... a[n-1]
and prints the number of subsets of {a[0..n-1]} (positions distinct) whose sum
equals T, taken modulo 1e9+7.

Method: literal enumeration of all 2^n subsets for small n.  This shares no code
or algorithmic idea with the O(n*T) DP in sol.cpp -- it is a direct definition.
"""
import sys
from itertools import combinations

MOD = 1000000007


def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    T = int(data[idx]); idx += 1
    a = [int(data[idx + i]) for i in range(n)]

    count = 0
    # Enumerate every subset by choosing a bitmask over positions.
    for mask in range(1 << n):
        s = 0
        m = mask
        i = 0
        while m:
            if m & 1:
                s += a[i]
            m >>= 1
            i += 1
        if s == T:
            count += 1
    print(count % MOD)


if __name__ == "__main__":
    main()
