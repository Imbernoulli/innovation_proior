#!/usr/bin/env python3
"""Independent brute oracle for the 2xN domino-tiling-count problem.

For small N we count tilings directly by a different route than the
shipped matrix-power solution:

  - For very small N (<= 12) we *enumerate* all domino tilings of the
    2xN board by recursive column-filling, to confirm the count is the
    Fibonacci number with no assumptions about a recurrence.
  - For larger (but still small) N we use a plain O(N) integer DP
    T(k) = T(k-1) + T(k-2) over big integers, then reduce mod p.

Both are independent of the matrix exponentiation under test.
"""
import sys


def count_tilings_enumerate(n):
    """Count exact tilings of a 2 x n board with 1x2 dominoes by brute search.

    We fill cells column-major. State: a frontier over the current set of
    cells. Simpler: process cells row by row using a bitmask DP across two
    rows, but to be maximally independent we do explicit backtracking on a
    2 x n grid placing horizontal or vertical dominoes.
    """
    if n == 0:
        return 1
    grid = [[False, False] for _ in range(n)]  # grid[col][row]
    count = 0

    def first_empty():
        for col in range(n):
            for row in range(2):
                if not grid[col][row]:
                    return col, row
        return None

    def backtrack():
        nonlocal count
        cell = first_empty()
        if cell is None:
            count += 1
            return
        col, row = cell
        # vertical domino (covers both rows of this column) -- only if row==0 and row1 free
        if row == 0 and not grid[col][1]:
            grid[col][0] = grid[col][1] = True
            backtrack()
            grid[col][0] = grid[col][1] = False
        # horizontal domino: this cell and the cell to its right, same row
        if col + 1 < n and not grid[col + 1][row]:
            grid[col][row] = grid[col + 1][row] = True
            backtrack()
            grid[col][row] = grid[col + 1][row] = False

    backtrack()
    return count


def count_tilings_dp(n):
    """O(n) big-integer DP. T(0)=1, T(1)=1, T(k)=T(k-1)+T(k-2)."""
    if n == 0:
        return 1
    a, b = 1, 1  # T(0), T(1)
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def tilings(n):
    # Cross-check the two methods on the overlap range to be safe.
    if n <= 12:
        e = count_tilings_enumerate(n)
        d = count_tilings_dp(n)
        assert e == d, f"enumerate vs dp disagree at n={n}: {e} vs {d}"
        return e
    return count_tilings_dp(n)


def main():
    data = sys.stdin.read().split()
    idx = 0
    q = int(data[idx]); idx += 1
    out = []
    for _ in range(q):
        n = int(data[idx]); p = int(data[idx + 1]); idx += 2
        val = tilings(n) % p
        out.append(str(val))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    main()
