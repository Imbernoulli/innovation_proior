#!/usr/bin/env python3
"""Trivial baseline: greedy best-first region growth (the normalization reference).

Seed at the best single cell, then repeatedly add the frontier cell with the best
(most positive) weight while it is positive and the budget allows. No removals, no
backtracking -- this is exactly the one-shot growth the SA solver is meant to beat.

Reads the instance on stdin, writes a feasible solution (K, then K lines "r c").
"""
import sys
import heapq


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    H = int(next(it)); W = int(next(it)); B = int(next(it))
    grid = [[0] * W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            grid[r][c] = int(next(it))

    # best single cell
    best = (None, None)
    bestv = None
    for r in range(H):
        for c in range(W):
            if bestv is None or grid[r][c] > bestv:
                bestv = grid[r][c]
                best = (r, c)
    sr, sc = best
    inset = [[False] * W for _ in range(H)]
    inset[sr][sc] = True
    chosen = [(sr, sc)]

    # max-heap on weight (negate for heapq)
    heap = []
    seenfront = set()

    def push(r, c):
        if 0 <= r < H and 0 <= c < W and not inset[r][c] and (r, c) not in seenfront:
            seenfront.add((r, c))
            heapq.heappush(heap, (-grid[r][c], r, c))

    for (dr, dc) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        push(sr + dr, sc + dc)

    while heap and len(chosen) < B:
        negw, r, c = heapq.heappop(heap)
        if inset[r][c]:
            continue
        w = -negw
        if w <= 0:
            break
        inset[r][c] = True
        chosen.append((r, c))
        for (dr, dc) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            push(r + dr, c + dc)

    out = [str(len(chosen))]
    for (r, c) in chosen:
        out.append(f"{r} {c}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
