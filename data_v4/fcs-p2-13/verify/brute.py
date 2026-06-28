#!/usr/bin/env python3
"""Independent brute-force oracle for the largest all-ones square problem.

Reads the same stdin format as sol.cpp:
    first line / tokens: H W
    then H*W integers (each 0 or 1), row-major.
Prints the AREA (side*side) of the largest all-ones square submatrix.

Method: completely independent of the DP. For every possible square side
length k from min(H,W) down to 1, and every possible top-left corner, check
whether the entire k*k block is all ones using a 2D prefix-sum of the grid.
The first k for which such a square exists gives the answer; if none, area 0.

This is O(H*W*min(H,W)) in the worst case (well, O(H*W) checks per k via
prefix sums), which is far too slow for 1500x1500 but exact and obviously
correct on the small random/edge cases used for differential testing.
"""
import sys


def solve(tokens, idx):
    H = tokens[idx]; W = tokens[idx + 1]
    idx += 2
    grid = [[0] * W for _ in range(H)]
    for i in range(H):
        for j in range(W):
            grid[i][j] = tokens[idx]
            idx += 1

    # 2D prefix sums: P[i][j] = sum of grid[0..i-1][0..j-1].
    P = [[0] * (W + 1) for _ in range(H + 1)]
    for i in range(H):
        row = grid[i]
        Pi = P[i]
        Pi1 = P[i + 1]
        for j in range(W):
            Pi1[j + 1] = Pi1[j] + Pi[j + 1] - Pi[j] + row[j]

    def block_sum(r, c, k):
        # sum of the k*k block with top-left (r, c)
        return (P[r + k][c + k] - P[r][c + k]
                - P[r + k][c] + P[r][c])

    best_side = 0
    kmax = min(H, W)
    for k in range(kmax, 0, -1):
        found = False
        for r in range(0, H - k + 1):
            for c in range(0, W - k + 1):
                if block_sum(r, c, k) == k * k:
                    found = True
                    break
            if found:
                break
        if found:
            best_side = k
            break

    return best_side * best_side, idx


def main():
    data = sys.stdin.read().split()
    tokens = list(map(int, data))
    if not tokens:
        return
    out = []
    idx = 0
    # Single test case per file (matches sol.cpp contract).
    area, idx = solve(tokens, idx)
    out.append(str(area))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
