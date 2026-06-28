#!/usr/bin/env python3
"""Fully exhaustive minimax (no DP, no memoization) for the Stone Game.

Used only on tiny n as an ultra-independent cross-check of brute.py / sol.cpp.
Recursively explores the full game tree. State: deque (left, right pointers)
plus whose turn. Each player maximizes their OWN collected total.

We return a pair (first_mover_total, second_mover_total) for the subarray
a[i..j], where 'first_mover' is whoever is about to move. The overall answer
is the first element of solve(0, n-1).
"""
import sys
from functools import lru_cache


def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    a = tuple(int(x) for x in data[1:1 + n])
    if n == 0:
        print(0)
        return

    @lru_cache(maxsize=None)
    def solve(i, j):
        # returns (mover_total, other_total) for a[i..j], mover plays first
        if i > j:
            return (0, 0)
        if i == j:
            return (a[i], 0)
        # take left: opponent then plays first on (i+1, j)
        ol, sl = solve(i + 1, j)  # ol = opponent's take, sl = our subsequent take
        left_val = a[i] + sl
        left_opp = ol
        # take right
        orr, sr = solve(i, j - 1)
        right_val = a[j] + sr
        right_opp = orr
        # mover picks the option maximizing mover_total
        if left_val >= right_val:
            return (left_val, left_opp)
        else:
            return (right_val, right_opp)

    print(solve(0, n - 1)[0])


if __name__ == "__main__":
    main()
