#!/usr/bin/env python3
"""Independent brute-force oracle for the least-rotation problem.

Reads a single string token from stdin. Outputs the smallest 0-based index k
such that the rotation s[k:]+s[:k] is lexicographically minimal among all
rotations. Empty input -> 0.

This is the obviously-correct O(n^2)-ish method: build every rotation and pick
the lexicographically smallest, breaking ties by smallest index. (Python string
comparison is exact lexicographic comparison by code point / byte.)
"""
import sys


def least_rotation_index(s: str) -> int:
    n = len(s)
    if n == 0:
        return 0
    best_idx = 0
    best_rot = s  # rotation starting at index 0
    for k in range(1, n):
        rot = s[k:] + s[:k]
        if rot < best_rot:
            best_rot = rot
            best_idx = k
        # strictly-less keeps the smallest index on ties automatically
    return best_idx


def main() -> None:
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    s = data[0]
    print(least_rotation_index(s))


if __name__ == "__main__":
    main()
