import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    idx = 0
    H = int(data[idx]); idx += 1
    W = int(data[idx]); idx += 1
    L = int(data[idx]); idx += 1
    R = int(data[idx]); idx += 1
    g = []
    for i in range(H):
        g.append(data[idx]); idx += 1

    INF = float('inf')
    # Independent method: for EVERY open cell, run a single-source BFS and take the
    # distance to the nearest station by minimizing over all stations. We do it the
    # other way that is obviously correct: for each open cell compute its own BFS to
    # all cells, then the nearest station distance = min over stations of that BFS.
    # Equivalent but computed without the multi-source trick.

    def bfs_from(si, sj):
        d = [[INF]*W for _ in range(H)]
        d[si][sj] = 0
        q = deque([(si, sj)])
        while q:
            x, y = q.popleft()
            for ddx, ddy in ((-1,0),(1,0),(0,-1),(0,1)):
                nx, ny = x+ddx, y+ddy
                if 0 <= nx < H and 0 <= ny < W and g[nx][ny] != '#' and d[nx][ny] == INF:
                    d[nx][ny] = d[x][y] + 1
                    q.append((nx, ny))
        return d

    stations = [(i, j) for i in range(H) for j in range(W) if g[i][j] == 'S']

    # nearest station distance for each open cell, computed by BFS from each open cell
    count = 0
    for i in range(H):
        for j in range(W):
            if g[i][j] == '#':
                continue
            d = bfs_from(i, j)
            best = INF
            for (si, sj) in stations:
                if d[si][sj] < best:
                    best = d[si][sj]
            if best == INF:
                continue
            if L <= best <= R:
                count += 1

    print(count)

main()
