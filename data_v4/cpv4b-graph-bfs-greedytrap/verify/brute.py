import sys

# INDEPENDENT brute force via Bellman-Ford-style fixpoint relaxation over the
# state graph (no BFS, no priority queue). State = (row, col, used) where used
# is the number of blast charges spent (0..K). Every move has weight 1. We
# initialise dist[start]=0, then repeatedly relax ALL edges until no distance
# improves. The fixpoint of single-source shortest path on a graph with unit
# (non-negative) weights is the true shortest distance, independent of the
# order edges are processed -- so this does not share BFS's frontier logic.

def solve(data):
    it = iter(data.split())
    R = int(next(it)); C = int(next(it)); K = int(next(it))
    g = [next(it) for _ in range(R)]

    sr = sc = tr = tc = -1
    for r in range(R):
        for c in range(C):
            if g[r][c] == 'S':
                sr, sc = r, c
            elif g[r][c] == 'T':
                tr, tc = r, c

    INF = float('inf')
    # dist over (r, c, used)
    dist = {}
    start = (sr, sc, 0)
    dist[start] = 0

    moves = ((-1, 0), (1, 0), (0, -1), (0, 1))

    changed = True
    while changed:
        changed = False
        # relax every reachable state's outgoing edges
        for (r, c, used), d in list(dist.items()):
            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= R or nc < 0 or nc >= C:
                    continue
                nused = used
                if g[nr][nc] == '#':
                    if used + 1 > K:
                        continue
                    nused = used + 1
                ns = (nr, nc, nused)
                nd = d + 1
                if nd < dist.get(ns, INF):
                    dist[ns] = nd
                    changed = True

    best = INF
    for used in range(K + 1):
        v = dist.get((tr, tc, used), INF)
        if v < best:
            best = v
    return best if best != INF else -1


def main():
    data = sys.stdin.read()
    print(solve(data))


if __name__ == "__main__":
    main()
