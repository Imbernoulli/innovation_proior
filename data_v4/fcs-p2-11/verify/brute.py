#!/usr/bin/env python3
"""Independent brute-force oracle for the word-break (segmentable?) problem.

Input format (stdin):
  line/token 1: n               number of dictionary words (0 <= n)
  next n tokens: the dictionary words
  next token: s                 the string to segment

Output: YES if s can be partitioned into a sequence of dictionary words
(each word may be reused any number of times, empty string s is always YES),
otherwise NO.

This oracle uses a *recursive* memoized search over split points, written
independently from the C++ DP, so a logic bug in either is unlikely to be
shared. For the small sizes used in differential testing this is plenty fast.
"""
import sys
from functools import lru_cache


def solve(tokens):
    it = iter(tokens)
    n = int(next(it))
    dict_words = set()
    for _ in range(n):
        dict_words.add(next(it))
    # s is the next token; if absent (no s given) treat as empty string.
    try:
        s = next(it)
    except StopIteration:
        s = ""
    m = len(s)

    sys.setrecursionlimit(100000)

    @lru_cache(maxsize=None)
    def can(start):
        # Can s[start:] be segmented?
        if start == m:
            return True
        for end in range(start + 1, m + 1):
            if s[start:end] in dict_words and can(end):
                return True
        return False

    return "YES" if can(0) else "NO"


def main():
    data = sys.stdin.read().split()
    sys.stdout.write(solve(data) + "\n")


if __name__ == "__main__":
    main()
