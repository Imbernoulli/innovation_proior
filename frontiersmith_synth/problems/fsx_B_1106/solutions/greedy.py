# TIER: greedy
# The obvious "chain it into one string" improvement over spamming: walk
# transitions in FIXED (state, symbol) table order; whenever the current
# position hasn't got the next-in-order transition available, BFS to that
# transition's source state (reusing whatever the walk already passed over)
# and take it. One continuous string, but it is blind to graph structure --
# it just follows state-id order, so an adversarial relabelling of the states
# can force long detours back and forth across the automaton that a
# graph-distance-aware (Chinese-postman) planner would never take.
import sys
from collections import deque


def main():
    data = sys.stdin.read().split("\n")
    head = data[0].split()
    n, k, s0 = int(head[0]), int(head[1]), int(head[2])
    symbols = data[1].split()
    trans = []
    for i in range(n):
        trans.append([int(x) for x in data[2 + i].split()])

    def bfs_path(src, dst):
        if src == dst:
            return []
        dist = [-1] * n
        parent = [None] * n
        dist[src] = 0
        dq = deque([src])
        while dq:
            u = dq.popleft()
            for s in range(k):
                v = trans[u][s]
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = (u, s)
                    dq.append(v)
        seq = []
        cur = dst
        while cur != src:
            u, s = parent[cur]
            seq.append(s)
            cur = u
        seq.reverse()
        return seq

    covered = [[False] * k for _ in range(n)]
    order = [(i, j) for i in range(n) for j in range(k)]
    cur = s0
    out_chars = []
    for (u, s) in order:
        if covered[u][s]:
            continue
        for sym_idx in bfs_path(cur, u):
            covered[cur][sym_idx] = True
            out_chars.append(symbols[sym_idx])
            cur = trans[cur][sym_idx]
        covered[u][s] = True
        out_chars.append(symbols[s])
        cur = trans[u][s]

    sys.stdout.write("1\n" + "".join(out_chars) + "\n")


if __name__ == "__main__":
    main()
