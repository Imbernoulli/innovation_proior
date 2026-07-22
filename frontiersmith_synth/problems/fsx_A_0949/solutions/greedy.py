# TIER: greedy
import sys
from collections import deque


def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    N = int(next(it)); M = int(next(it)); K = int(next(it)); S = int(next(it))
    sources = [int(next(it)) for _ in range(S)]
    theta = [0.0] * (N + 1)
    amp = [1.0] * (N + 1)
    for i in range(1, N + 1):
        theta[i] = float(next(it))
        amp[i] = float(next(it))
    adj = [[] for _ in range(N + 1)]
    out_w = [0.0] * (N + 1)
    for _ in range(M):
        u = int(next(it)); v = int(next(it)); w = float(next(it))
        adj[u].append(v)
        out_w[u] += w

    # "Obvious" move: build the firebreak right next to the fire. Rank nodes
    # within hop-distance 1 of any source by their raw outgoing weight
    # (a natural, degree-style notion of "how dangerous is this node"),
    # and remove the top-K. Amplification is not accounted for, and nothing
    # beyond the immediate neighbourhood is ever considered.
    src_set = set(sources)
    dist = [None] * (N + 1)
    q = deque()
    for s in sources:
        dist[s] = 0
        q.append(s)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] is None:
                dist[v] = dist[u] + 1
                q.append(v)

    RADIUS = 1
    cand = [i for i in range(1, N + 1)
            if i not in src_set and dist[i] is not None and dist[i] <= RADIUS]
    cand.sort(key=lambda i: (-out_w[i], i))
    chosen = cand[:K]

    if len(chosen) < K:
        rest = [i for i in range(1, N + 1) if i not in src_set and i not in chosen]
        rest.sort(key=lambda i: (-out_w[i], i))
        chosen += rest[:K - len(chosen)]

    print(len(chosen))
    print(" ".join(str(x) for x in chosen))


if __name__ == "__main__":
    main()
