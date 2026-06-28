#!/usr/bin/env python3
"""Independent brute oracle for the wildcard matching problem.

Reads two tokens from stdin: pattern p and string s.
The literal token "-" denotes an empty string (so empty tokens can be passed).
Outputs YES if p matches s under wildcard rules ('?' = exactly one char,
'*' = any sequence including empty), else NO.

This oracle uses a deliberately different implementation from sol.cpp:
plain top-down recursion with memoization over (i, j) index pairs, written
from the matching definition directly. For the small sizes used in the
differential test this is also exact (and for tiny sizes we could even use
raw exponential recursion, but memoization keeps it robust).
"""
import sys
from functools import lru_cache


def solve(p: str, s: str) -> str:
    n, m = len(p), len(s)
    sys.setrecursionlimit(10000)

    @lru_cache(maxsize=None)
    def match(i: int, j: int) -> bool:
        # Does p[i:] match s[j:] ?
        if i == n:
            return j == m
        c = p[i]
        if c == '*':
            # '*' matches empty (advance i) or one more char of s (advance j).
            if match(i + 1, j):
                return True
            if j < m and match(i, j + 1):
                return True
            return False
        else:
            if j < m and (c == '?' or c == s[j]):
                return match(i + 1, j + 1)
            return False

    ans = match(0, 0)
    match.cache_clear()
    return "YES" if ans else "NO"


def main():
    data = sys.stdin.read().split()
    if not data:
        return
    p = data[0]
    s = data[1] if len(data) > 1 else ""
    if p == "-":
        p = ""
    if s == "-":
        s = ""
    print(solve(p, s))


if __name__ == "__main__":
    main()
