#!/usr/bin/env python3
"""Brute-force oracle for fcs-gx-02 (Lexicographically-Smallest via deletions).

Reads the same stdin format as sol.cpp:
    s
    k
Deletes exactly k characters (when k < len(s)) and prints the lexicographically
smallest string of length len(s)-k that is a subsequence of s. When k >= len(s)
the result is the empty string. Method: enumerate every combination of k
positions to delete (equivalently, every length-(n-k) subsequence) and take the
minimum. Exponential -- only valid for small n (n <= ~16).
"""
import sys
from itertools import combinations


def solve(s: str, k: int) -> str:
    n = len(s)
    if k < 0:
        k = 0
    if k >= n:
        return ""
    keep = n - k
    best = None
    # Choose which 'keep' positions to retain, in increasing index order.
    for idx in combinations(range(n), keep):
        cand = "".join(s[i] for i in idx)
        if best is None or cand < best:
            best = cand
    return best if best is not None else ""


def main() -> None:
    data = sys.stdin.read().split()
    if not data:
        return
    s = data[0]
    k = int(data[1]) if len(data) > 1 else 0
    sys.stdout.write(solve(s, k) + "\n")


if __name__ == "__main__":
    main()
