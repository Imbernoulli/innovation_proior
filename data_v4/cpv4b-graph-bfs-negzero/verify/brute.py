import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    idx = 0
    R = int(data[idx]); idx += 1
    C = int(data[idx]); idx += 1
    N = R * C
    t = []
    for _ in range(N):
        t.append(int(data[idx])); idx += 1

    INF = float('inf')
    # Independent solver: iterative relaxation (Bellman-Ford-like) over the grid.
    #   source   : t <  0  -> dist 0
    #   conductor: t <= 0  -> can hold/propagate frost
    #   wall     : t >  0  -> never frosts (stays INF), never propagates
    dist = [INF] * N
    for i in range(N):
        if t[i] < 0:
            dist[i] = 0

    def neighbors(i):
        r, c = divmod(i, C)
        res = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C:
                res.append(nr * C + nc)
        return res

    changed = True
    while changed:
        changed = False
        for i in range(N):
            if t[i] > 0:
                continue  # wall: stays INF, cannot receive frost
            best = dist[i]
            for j in neighbors(i):
                # frost may flow from any frosted conductor neighbor
                if t[j] <= 0 and dist[j] + 1 < best:
                    best = dist[j] + 1
            if best < dist[i]:
                dist[i] = best
                changed = True

    ans = -1
    for i in range(N):
        if t[i] <= 0 and dist[i] != INF:
            ans = max(ans, dist[i])
    print(ans)

main()
