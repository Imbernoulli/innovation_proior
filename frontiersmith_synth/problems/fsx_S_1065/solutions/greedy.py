# TIER: greedy
"""The obvious "textbook" fix-up over the trivial order: a single breadth-first
traversal of the brace graph (starting from the lowest-degree pier, expanding to
lower-degree neighbours first -- i.e. Cuthill-McKee-style level ordering), which is
the standard go-to bandwidth/fill heuristic many people reach for first. It never
revisits a decision once made and never re-examines how braces it has already
crossed have changed the live degree of anything -- it is fill-oblivious. Because
the true separator hierarchy is hidden behind a full relabeling plus a few
long-range noise braces, a single non-adaptive sweep like this drifts into the
hierarchy at essentially an arbitrary point and racks up large avoidable fill-in,
even though it "looks" like a structure-aware graph algorithm.
"""
import sys
from collections import deque


def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    m = int(data[idx]); idx += 1
    adj = [set() for _ in range(n)]
    for _ in range(m):
        u = int(data[idx]) - 1; idx += 1
        v = int(data[idx]) - 1; idx += 1
        adj[u].add(v)
        adj[v].add(u)

    visited = [False] * n
    order = []
    verts_by_deg = sorted(range(n), key=lambda x: len(adj[x]))
    for s0 in verts_by_deg:
        if visited[s0]:
            continue
        dq = deque([s0])
        visited[s0] = True
        while dq:
            u = dq.popleft()
            order.append(u)
            nbrs = sorted((w for w in adj[u] if not visited[w]), key=lambda x: len(adj[x]))
            for w in nbrs:
                if not visited[w]:
                    visited[w] = True
                    dq.append(w)

    print(" ".join(str(v + 1) for v in order))


if __name__ == "__main__":
    main()
