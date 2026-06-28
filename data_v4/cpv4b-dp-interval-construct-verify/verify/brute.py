#!/usr/bin/env python3
"""
INDEPENDENT brute force for the SAME problem.

Problem (restated): read n. Output n integers x_0 < x_1 < ... < x_{n-1} with
x_0 = 0 and x_{n-1} <= M = 8*n*n, such that all pairwise differences
x_j - x_i (i < j) are pairwise DISTINCT (a Sidon set / perfect ruler).

This brute uses a completely different method from sol.cpp: a depth-first
backtracking search that places marks one at a time, always keeping the
difference multiset collision-free, and returns the lexicographically smallest
valid ruler. It is obviously correct (it literally checks the Sidon property at
every step) but exponential, so it is only usable for small n.

Prints the marks space-separated on one line.
"""
import sys

def solve(n):
    M = 8 * n * n
    if n == 1:
        return [0]
    marks = [0]
    used = set()  # current pairwise differences

    def dfs():
        if len(marks) == n:
            return True
        last = marks[-1]
        # next mark must be > last and <= M; try smallest first for lex-smallest result
        for cand in range(last + 1, M + 1):
            new_diffs = []
            ok = True
            for m in marks:
                d = cand - m
                if d in used or d in new_diffs:
                    ok = False
                    break
                new_diffs.append(d)
            if not ok:
                continue
            for d in new_diffs:
                used.add(d)
            marks.append(cand)
            if dfs():
                return True
            marks.pop()
            for d in new_diffs:
                used.discard(d)
        return False

    if not dfs():
        return None  # should never happen within M = 8 n^2 for the n we brute
    return marks

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    res = solve(n)
    assert res is not None, "no valid ruler found within bound (bug in brute)"
    print(" ".join(map(str, res)))

if __name__ == "__main__":
    main()
