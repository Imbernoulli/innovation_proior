# TIER: strong
# Insight: the datum tree and the precise-slot budget are ONE coupled decision.
# Route every critical pair through the common deep trunk (attach leaves to the
# deep option, allowed[0]) so a handful of trunk edges lie on the maximum number
# of weighted pair-paths -- then spend the precise slots by criticality-weighted
# path-BETWEENNESS along the current bottleneck, not by chain length or edge size.
#
# This deliberately accepts LONGER reference chains than the shallow tree, because
# a single precise trunk edge then serves every pair at once. A "pick tree, then
# upgrade the longest chains" pipeline cannot see this coupling.
import sys

def main():
    tok = sys.stdin.read().split()
    it = iter(tok)
    n = int(next(it)); k = int(next(it))
    a = [int(next(it)) for _ in range(n)]
    p = [int(next(it)) for _ in range(n)]
    allowed = []
    for i in range(n):
        d = int(next(it)); allowed.append([int(next(it)) for _ in range(d)])
    C = int(next(it))
    pairs = [(int(next(it)), int(next(it)), int(next(it))) for _ in range(C)]

    # tree: route through the common deep trunk (first-listed = deepest option)
    par = [-1] + [allowed[i][0] for i in range(1, n)]
    depth = [0] * n
    for i in range(1, n):
        depth[i] = depth[par[i]] + 1

    # static path (nodes carrying an edge) for every pair
    def path_nodes(u, v):
        x, y = u, v; nodes = []
        while depth[x] > depth[y]:
            nodes.append(x); x = par[x]
        while depth[y] > depth[x]:
            nodes.append(y); y = par[y]
        while x != y:
            nodes.append(x); nodes.append(y); x = par[x]; y = par[y]
        return nodes

    paths = [path_nodes(u, v) for (u, v, w) in pairs]
    wts = [w for (u, v, w) in pairs]

    # weighted betweenness of each edge/feature
    bet = [0] * n
    for pi, nodes in enumerate(paths):
        w = wts[pi]
        for x in nodes:
            bet[x] += w

    eff = a[:]
    precise = []
    chosen = set()
    for _ in range(k):
        # current cost of every pair
        costs = [wts[pi] * sum(eff[x] for x in paths[pi]) for pi in range(C)]
        M = max(costs)
        if M <= 0:
            break
        # candidate edges: on any (near-)bottleneck pair, not yet precise
        cand = {}
        thresh = M * 0.999
        for pi in range(C):
            if costs[pi] >= thresh:
                for x in paths[pi]:
                    if x not in chosen:
                        cand[x] = True
        if not cand:
            break
        # pick the edge with the largest weighted-betweenness * error-drop
        best = max(cand, key=lambda x: ((a[x] - p[x]) * bet[x], -x))
        chosen.add(best)
        precise.append(best)
        eff[best] = p[best]

    print(" ".join(map(str, par)))
    print(str(len(precise)) + ((" " + " ".join(map(str, precise))) if precise else ""))

main()
