# TIER: strong
# Two insights the recipe misses:
#  (1) delta direction is a free variable -- the cheapest way to reach an
#      exemplar need not be "ascend to the nearest ancestor"; a recension can
#      just as legally point DOWN at one of its own derivatives. We compute
#      true shortest reconstruction cost with a multi-source Dijkstra over
#      the directed ascent/descent arc graph (reverse-graph trick: the
#      predecessor found while relaxing node v IS v's optimal pointer).
#  (2) the parchment budget is a knapsack, not a popularity contest -- new
#      exemplars are chosen by weighted-savings PER UNIT VELLUM COST
#      (a value-density / exchange-argument greedy over facility location),
#      re-evaluated against the current basin shape after every addition.
import sys
import heapq


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    s_budget = int(next(it))
    size = [0] * (n + 1)
    w = [0.0] * (n + 1)
    parent = [0] * (n + 1)
    up = [0] * (n + 1)
    down = [0] * (n + 1)
    size[1] = int(next(it))
    w[1] = float(next(it))
    rev_adj = [[] for _ in range(n + 1)]
    for i in range(2, n + 1):
        p = int(next(it))
        u = int(next(it))
        d = int(next(it))
        sz = int(next(it))
        wi = float(next(it))
        parent[i] = p
        up[i] = u
        down[i] = d
        size[i] = sz
        w[i] = wi
        # reverse graph: original arc i->p (cost up[i]) reverses to p->i;
        # original arc p->i (cost down[i]) reverses to i->p.
        rev_adj[p].append((i, u))
        rev_adj[i].append((p, d))

    def dijkstra(checkpoints_set):
        INF = float("inf")
        dist = [INF] * (n + 1)
        pred = [0] * (n + 1)
        pq = []
        for c in checkpoints_set:
            dist[c] = 0.0
            heapq.heappush(pq, (0.0, c))
        while pq:
            dv, v = heapq.heappop(pq)
            if dv > dist[v]:
                continue
            for u, c in rev_adj[v]:
                nd = dv + c
                if nd < dist[u] - 1e-12:
                    dist[u] = nd
                    pred[u] = v
                    heapq.heappush(pq, (nd, u))
        return dist, pred

    checkpoints = {1}
    remaining = s_budget - size[1]

    dist, pred = dijkstra(checkpoints)

    while True:
        best_v, best_score = -1, 0.0
        for v in range(2, n + 1):
            if v in checkpoints or size[v] > remaining:
                continue
            gain = w[v] * dist[v]  # direct benefit of v becoming self-served
            if gain <= 1e-12:
                continue
            score = gain / size[v]
            if score > best_score:
                best_score = score
                best_v = v
        if best_v == -1:
            break
        checkpoints.add(best_v)
        remaining -= size[best_v]
        dist, pred = dijkstra(checkpoints)

    out = [str(len(checkpoints)), " ".join(map(str, sorted(checkpoints)))]
    for v in range(1, n + 1):
        if v in checkpoints:
            continue
        out.append(f"{v} {pred[v]}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
