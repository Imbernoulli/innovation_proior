#!/usr/bin/env python3
"""Trivial baseline for Wall Painting (ALE-10).

Reads an instance on stdin and writes a feasible solution that paints the whole
canvas with the single most frequent target colour (one stroke). This is the
natural "do the obvious thing" reference the SA solver must beat.

If the budget T == 0, it emits Q = 0 (the bare all-0 canvas), which is still
feasible.
"""
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); C = int(next(it)); T = int(next(it))
    freq = [0] * C
    for _ in range(N * N):
        freq[int(next(it))] += 1
    base = max(range(C), key=lambda col: freq[col])
    if T >= 1 and base != 0:
        print(1)
        print(f"0 0 {N-1} {N-1} {base}")
    else:
        # base colour is 0 (already the canvas) or no budget: empty is optimal here
        print(0)


if __name__ == "__main__":
    main()
