import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    R = int(next(it)); C = int(next(it)); L = int(next(it)); U = int(next(it))
    g = [next(it) for _ in range(R)]

    INF = float('inf')
    # Independent approach: run a separate BFS from EACH source, take the
    # element-wise minimum distance over all sources.
    best = [[INF] * C for _ in range(R)]
    sources = [(r, c) for r in range(R) for c in range(C) if g[r][c] == '*']

    for (sr, sc) in sources:
        d = [[INF] * C for _ in range(R)]
        d[sr][sc] = 0
        q = deque([(sr, sc)])
        while q:
            r, c = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < R and 0 <= nc < C and g[nr][nc] != '#' and d[nr][nc] == INF:
                    d[nr][nc] = d[r][c] + 1
                    q.append((nr, nc))
        for r in range(R):
            for c in range(C):
                if d[r][c] < best[r][c]:
                    best[r][c] = d[r][c]

    cnt = 0
    for r in range(R):
        for c in range(C):
            if g[r][c] == '#':
                continue
            dd = best[r][c]
            if dd == INF:
                continue
            if L <= dd <= U:   # inclusive on both ends
                cnt += 1
    print(cnt)

main()
