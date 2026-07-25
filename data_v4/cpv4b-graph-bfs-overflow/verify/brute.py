import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    R = int(data[idx]); idx += 1
    C = int(data[idx]); idx += 1
    grid = []
    for i in range(R):
        grid.append(data[idx]); idx += 1

    INF = float('inf')
    dist = [[INF] * C for _ in range(R)]
    q = deque()
    for i in range(R):
        for j in range(C):
            if grid[i][j] == '*':
                dist[i][j] = 0
                q.append((i, j))

    # Independent BFS implementation (per-cell coordinates, list-of-lists dist)
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= R or nc < 0 or nc >= C:
                continue
            if grid[nr][nc] == '#':
                continue
            if dist[nr][nc] != INF:
                continue
            dist[nr][nc] = dist[r][c] + 1
            q.append((nr, nc))

    total = 0
    for i in range(R):
        for j in range(C):
            if grid[i][j] == '#':
                continue
            if dist[i][j] == INF:
                continue
            total += dist[i][j]

    print(total)

main()
