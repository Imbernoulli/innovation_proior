import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    n = int(data[idx]); idx += 1
    m = int(data[idx]); idx += 1
    h = []
    for i in range(n):
        row = []
        for j in range(m):
            row.append(int(data[idx])); idx += 1
        h.append(row)

    INF = float('inf')
    dist = [[INF] * m for _ in range(n)]
    dist[0][0] = 0

    # Obviously-correct independent method: Bellman-Ford-style repeated relaxation
    # over all directed edges until no distance improves. Edge (r,c)->(nr,nc) has
    # weight 1 if h[nr][nc] > h[r][c] (boost up) else 0 (glide level/down).
    dr = [-1, 1, 0, 0]
    dc = [0, 0, -1, 1]
    changed = True
    while changed:
        changed = False
        for r in range(n):
            for c in range(m):
                if dist[r][c] == INF:
                    continue
                d = dist[r][c]
                for k in range(4):
                    nr = r + dr[k]
                    nc = c + dc[k]
                    if nr < 0 or nr >= n or nc < 0 or nc >= m:
                        continue
                    w = 1 if h[nr][nc] > h[r][c] else 0
                    if d + w < dist[nr][nc]:
                        dist[nr][nc] = d + w
                        changed = True

    print(dist[n-1][m-1])

main()
