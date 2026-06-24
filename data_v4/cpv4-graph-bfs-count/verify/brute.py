import sys
from collections import deque

MOD = 1000000007

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    m = int(data[idx]); idx += 1
    s = int(data[idx]); idx += 1
    t = int(data[idx]); idx += 1
    adj = [[] for _ in range(n + 1)]
    edges = []
    for _ in range(m):
        u = int(data[idx]); idx += 1
        v = int(data[idx]); idx += 1
        if u == v:
            continue
        adj[u].append(v)
        adj[v].append(u)
        edges.append((u, v))

    # First find the shortest distance from s to t by plain BFS.
    dist = [None] * (n + 1)
    dist[s] = 0
    dq = deque([s])
    while dq:
        u = dq.popleft()
        for w in adj[u]:
            if dist[w] is None:
                dist[w] = dist[u] + 1
                dq.append(w)
    if dist[t] is None:
        print(0)
        return
    D = dist[t]

    # Independent method: enumerate ALL simple-ish walks of length exactly D from s to t
    # by depth-limited DFS that never revisits a node within the current path.
    # Any shortest path is a simple path, and any simple path of length D from s to t
    # is a shortest path, so counting length-D simple paths == number of shortest paths.
    count = 0
    visited = [False] * (n + 1)

    def dfs(u, depth):
        nonlocal count
        if depth == D:
            if u == t:
                count += 1
            return
        # prune: cannot finish if remaining steps < graph distance to t
        # (use precomputed dist from t for a sound prune; keeps brute tractable)
        for w in adj[u]:
            if not visited[w]:
                # prune using distance-from-t lower bound
                if distT[w] is not None and depth + 1 + distT[w] <= D:
                    visited[w] = True
                    dfs(w, depth + 1)
                    visited[w] = False

    # distance from t for pruning
    distT = [None] * (n + 1)
    distT[t] = 0
    dq = deque([t])
    while dq:
        u = dq.popleft()
        for w in adj[u]:
            if distT[w] is None:
                distT[w] = distT[u] + 1
                dq.append(w)

    visited[s] = True
    dfs(s, 0)
    print(count % MOD)

if __name__ == "__main__":
    main()
