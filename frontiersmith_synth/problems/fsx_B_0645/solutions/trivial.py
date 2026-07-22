# TIER: trivial
"""Reproduce the checker's own reference floorplan exactly: a compact near-square block of
n rooms. This is a valid, connected, low-diameter layout (always feasible since D is always
at least this block's own diameter) but it has the best (highest) conductance / fastest
mixing of any connected shape on n cells -- by design it scores ~0.1 (F == B)."""
import sys


def compact_rect(n, r0, c0):
    best = None
    for rows in range(1, n + 1):
        cols = -(-n // rows)
        if best is None or rows + cols < best[0] + best[1]:
            best = (rows, cols)
    rows, cols = best
    cells = []
    cnt = 0
    for i in range(rows):
        for j in range(cols):
            if cnt >= n:
                break
            cells.append((r0 + i, c0 + j))
            cnt += 1
        if cnt >= n:
            break
    return cells


def main():
    W, H, n, D = map(int, sys.stdin.read().split())
    cells = compact_rect(n, 1, 1)
    print("\n".join(f"{r} {c}" for r, c in cells))


if __name__ == "__main__":
    main()
