#!/usr/bin/env python3
"""Independent brute-force oracle for the huge-capacity 0/1 knapsack problem.

Reads the same stdin format as sol.cpp:
    n C
    w_i v_i   (n lines)
and prints the maximum total value of a subset whose total weight <= C.

This oracle uses plain exhaustive subset enumeration (2^n) -- only valid for
small n (the generator keeps n small). It shares NO logic with the
meet-in-the-middle solution, so it is a genuinely independent check.
"""
import sys
from itertools import combinations


def solve(n, C, w, v):
    best = 0
    # Enumerate every subset by choosing how many items, then which ones.
    idx = list(range(n))
    for r in range(0, n + 1):
        for combo in combinations(idx, r):
            sw = 0
            sv = 0
            for i in combo:
                sw += w[i]
                sv += v[i]
            if sw <= C and sv > best:
                best = sv
    return best


def main():
    data = sys.stdin.read().split()
    if not data:
        return
    pos = 0
    n = int(data[pos]); pos += 1
    C = int(data[pos]); pos += 1
    w = []
    v = []
    for _ in range(n):
        w.append(int(data[pos])); pos += 1
        v.append(int(data[pos])); pos += 1
    print(solve(n, C, w, v))


if __name__ == "__main__":
    main()
