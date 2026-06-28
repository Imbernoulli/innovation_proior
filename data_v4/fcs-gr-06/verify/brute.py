#!/usr/bin/env python3
"""
Independent oracle for fcs-gr-06.

Problem: grid of '.', '#', 'S', 'T'. Move 4-directionally. You may break up to K
walls. Stepping onto a free cell costs 0 breaks, stepping onto a '#' costs 1 break
(and consumes budget). Output the minimum number of walls broken to reach T from S,
or -1 if T is unreachable using at most K breaks.

This oracle is Dijkstra over the SAME layered state graph (cell x breaks_used),
which is the cross-check approach named in the candidate. It is implemented
completely independently of the 0-1 BFS solution (heap-based, no deque tricks).
"""
import sys
import heapq


def solve(data_tokens):
    it = iter(data_tokens)
    R = int(next(it)); C = int(next(it)); K = int(next(it))
    grid = [next(it) for _ in range(R)]

    sr = sc = tr = tc = -1
    for i in range(R):
        for j in range(C):
            if grid[i][j] == 'S':
                sr, sc = i, j
            elif grid[i][j] == 'T':
                tr, tc = i, j

    INF = float('inf')
    # dist[r][c][k]
    dist = [[[INF] * (K + 1) for _ in range(C)] for _ in range(R)]
    dist[sr][sc][0] = 0
    pq = [(0, sr, sc, 0)]
    while pq:
        d, r, c, k = heapq.heappop(pq)
        if d > dist[r][c][k]:
            continue
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C:
                ch = grid[nr][nc]
                if ch == '#':
                    if k == K:
                        continue
                    nk = k + 1
                    w = 1
                else:
                    nk = k
                    w = 0
                nd = d + w
                if nd < dist[nr][nc][nk]:
                    dist[nr][nc][nk] = nd
                    heapq.heappush(pq, (nd, nr, nc, nk))

    best = min(dist[tr][tc][k] for k in range(K + 1))
    return -1 if best == INF else best


def main():
    data = sys.stdin.read().split()
    print(solve(data))


if __name__ == "__main__":
    main()
