#!/usr/bin/env python3
"""Independent brute oracle for the Stone Game.

Two players alternate, each turn taking the stone at either end of the row.
Both play optimally to maximize their OWN final total. Output the first
player's max total.

Independent formulation (does NOT reuse the diff/parity trick from sol.cpp):
gain[i][j] = the maximum total the player who moves FIRST on subarray a[i..j]
can collect, assuming both play optimally. The mover picks an end; the rest of
that subarray is then a fresh game in which the opponent moves first, so the
opponent collects gain[rest] and the mover collects (sum of rest) - gain[rest].

  gain[i][j] = max(
      a[i] + (subsum(i+1, j) - gain[i+1][j]),   # take left end
      a[j] + (subsum(i, j-1) - gain[i][j-1])     # take right end
  )
  base: gain[i][i] = a[i]

We compute it with explicit interval-length iteration (no recursion limit
issues) and prefix sums.
"""
import sys


def solve(a):
    n = len(a)
    if n == 0:
        return 0
    # prefix[k] = sum of a[0..k-1]
    prefix = [0] * (n + 1)
    for k in range(n):
        prefix[k + 1] = prefix[k] + a[k]

    def subsum(i, j):  # inclusive
        return prefix[j + 1] - prefix[i]

    gain = [[0] * n for _ in range(n)]
    for i in range(n):
        gain[i][i] = a[i]

    for length in range(2, n + 1):
        for i in range(0, n - length + 1):
            j = i + length - 1
            take_left = a[i] + (subsum(i + 1, j) - gain[i + 1][j])
            take_right = a[j] + (subsum(i, j - 1) - gain[i][j - 1])
            gain[i][j] = max(take_left, take_right)
    return gain[0][n - 1]


def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    a = [int(data[idx + k]) for k in range(n)]
    print(solve(a))


if __name__ == "__main__":
    main()
