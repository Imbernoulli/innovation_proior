# TIER: invalid
"""Deliberately infeasible: two separate blobs of open cells with no shared edge between them
-- the induced graph is disconnected, which the checker must reject with Ratio: 0.0."""
import sys


def main():
    W, H, n, D = map(int, sys.stdin.read().split())
    half = n // 2
    rest = n - half
    cells = []
    for i in range(half):
        cells.append((1, 1 + i))
    # jump far away (guaranteed not adjacent to the first blob) to force disconnection
    far_r = min(H - 1, 20 + (H // 2))
    for i in range(rest):
        cells.append((far_r, 1 + i))
    print("\n".join(f"{r} {c}" for r, c in cells))


if __name__ == "__main__":
    main()
