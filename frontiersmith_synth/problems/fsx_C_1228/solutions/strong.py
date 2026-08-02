# TIER: strong
# The insight: correctness is a property of CYCLES in the read/write
# dependency graph, not of any single transaction's conflict count. So:
#   1. Build the static rw-hazard graph (i->j iff R_i and W_j share a key).
#   2. Repeatedly find ANY directed cycle still standing; promote the
#      CHEAPEST (min-weight) member of that cycle all the way to
#      SERIALIZABLE (the only level that removes a node from every hazard
#      edge it touches). This is a minimum-feedback-vertex-set heuristic:
#      only the transactions that actually sit on a dangerous cycle ever
#      pay the SERIALIZABLE tax -- everyone else stays cheap.
#   3. Everything not promoted defaults to READ COMMITTED (fastest); then a
#      second, independent pass covers the lost-update (ww) pairs by
#      bumping the cheaper endpoint of any still-both-RC pair to SNAPSHOT
#      (SNAPSHOT alone is enough to stop lost updates -- no need to go all
#      the way to SERIALIZABLE for that hazard class).
import sys


def find_cycle(nodes, adj):
    color = {v: 0 for v in nodes}
    stack = []
    result = [None]

    def dfs(u):
        color[u] = 1
        stack.append(u)
        for v in adj.get(u, ()):
            if result[0] is not None:
                return
            if color.get(v, 2) == 0:
                dfs(v)
            elif color.get(v) == 1:
                idx = stack.index(v)
                result[0] = list(stack[idx:])
                return
        if result[0] is None:
            stack.pop()
            color[u] = 2

    for s in nodes:
        if color[s] == 0:
            dfs(s)
        if result[0] is not None:
            return result[0]
    return None


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); K = int(next(it))
    weight = [0] * N
    R = [set() for _ in range(N)]
    W = [set() for _ in range(N)]
    for i in range(N):
        weight[i] = int(next(it))
        nr = int(next(it))
        for _ in range(nr):
            R[i].add(int(next(it)))
        nw = int(next(it))
        for _ in range(nw):
            W[i].add(int(next(it)))

    rw_edges = []
    for i in range(N):
        for j in range(N):
            if i != j and (R[i] & W[j]):
                rw_edges.append((i, j))
    ww_pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            if W[i] & W[j]:
                ww_pairs.append((i, j))

    promoted = set()
    remaining = set(range(N))
    while True:
        adj = {}
        for (i, j) in rw_edges:
            if i in remaining and j in remaining:
                adj.setdefault(i, []).append(j)
        cyc = find_cycle(sorted(remaining), adj)
        if cyc is None:
            break
        victim = min(cyc, key=lambda v: (weight[v], v))
        promoted.add(victim)
        remaining.discard(victim)

    lvl = [2 if i in promoted else 0 for i in range(N)]

    for (i, j) in ww_pairs:
        if lvl[i] == 0 and lvl[j] == 0:
            # bump the cheaper (lower-weight) endpoint to SNAPSHOT
            victim = i if (weight[i], i) <= (weight[j], j) else j
            lvl[victim] = 1

    print(" ".join(str(x) for x in lvl))


if __name__ == "__main__":
    main()
