# TIER: strong
"""Backward-likelihood scoring instead of forward centrality.

Two structural facts computable straight from the input beat raw degree:

1. FEASIBILITY (necessary condition of the SI model): infection moves at most
   one hop per round, so a candidate c cannot be the source unless every
   infected node lies within T hops of c in the contact graph. Any candidate
   that fails this is immediately near-impossible.

2. FOOTPRINT DENSITY (the actual insight): among feasible candidates, a hub
   that merely relays the epidemic has a T-hop reach that fans out into many
   directions the outbreak never actually took -- its footprint is much
   bigger than the infected set, i.e. it "predicts" far more infections than
   are observed. A genuine index case's T-hop footprint tends to match the
   infected set tightly. We score candidates by how much of their reachable
   footprint is actually infected (density = |infected| / |reachable|),
   which penalizes hubs precisely because they are reachable from everywhere.

3. A secondary backward log-likelihood tie-break: under a naive per-edge
   model, a plausible source should have almost all of its immediate
   contacts already infected (they had all T rounds to convert), while a
   node still surrounded by many untouched susceptible contacts is a poor
   fit for being the *origin* (more like a still-spreading frontier node).
"""
import sys, math


def main():
    data = sys.stdin.read().split("\n")
    N, M = map(int, data[1].split())
    adj = [[] for _ in range(N)]      # (neighbour, prob)
    adj_h = [[] for _ in range(N)]    # plain hop adjacency
    ptr = 2
    for _ in range(M):
        u, v, w = map(int, data[ptr].split())
        ptr += 1
        p = w / 10.0
        adj[u].append((v, p)); adj[v].append((u, p))
        adj_h[u].append(v); adj_h[v].append(u)
    T = int(data[ptr]); ptr += 1
    K = int(data[ptr]); ptr += 1
    cand = list(map(int, data[ptr].split()))
    infected = set(cand)
    ninf = len(infected)

    def hop_ball(src, radius):
        dist = {src: 0}
        frontier = [src]
        d = 0
        while frontier and d < radius:
            nf = []
            for u in frontier:
                for v in adj_h[u]:
                    if v not in dist:
                        dist[v] = d + 1
                        nf.append(v)
            frontier = nf
            d += 1
        return dist.keys()

    def backward_loglik(c, shrink=2.0):
        s = 0.0
        deg = len(adj[c])
        if deg == 0:
            return 0.0
        for (v, p) in adj[c]:
            if v in infected:
                s += math.log(max(p, 1e-9))
            else:
                s += math.log(max(1.0 - p, 1e-9))
        return s / (deg + shrink)

    combo = {}
    for c in cand:
        ball = hop_ball(c, T)
        covered = sum(1 for v in ball if v in infected)
        feasible = (covered == ninf)
        if not feasible:
            combo[c] = None
            continue
        density = covered / len(ball)
        ll = backward_loglik(c)
        combo[c] = (density, ll)

    feasible_vals = [v for v in combo.values() if v is not None]
    if feasible_vals:
        max_density = max(v[0] for v in feasible_vals)
    else:
        max_density = 1.0

    POWER = 2.0
    BETA = 2.0
    weights = []
    for c in cand:
        v = combo[c]
        if v is None:
            weights.append(1e-9)
        else:
            density, ll = v
            w = (density / max_density) ** POWER * math.exp(BETA * ll)
            weights.append(max(w, 1e-9))

    out = sys.stdout
    out.write("%d\n" % K)
    out.write(" ".join("%.10g" % w for w in weights))
    out.write("\n")


if __name__ == "__main__":
    main()
