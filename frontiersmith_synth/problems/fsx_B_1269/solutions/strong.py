# TIER: strong
"""The insight: the per-hop withholding rate does NOT separate from the
compliance predicate, because the substance test and the timing test are
both properties of the PATH AS A WHOLE, not of any single hop. Ranking
routes by cheapest cumulative rate (the greedy approach) throws away
exactly the information needed to tell a compliant route from a non-
compliant one that looks identical hop-by-hop.

So instead of a per-hop shortest-path search, this exhaustively searches
over ROUTES (the graph is a small layered DAG, so full path enumeration is
cheap) and evaluates each candidate route's PATH-LEVEL feasibility
(timing window + aggregate substance-vs-benefit test, including the
instrument-mismatch halving that only shows up when you know both the
edge feeding INTO and the edge leaving a jurisdiction) before ranking
by net-of-tax value. This is a decomposition/exchange search over the
feasible-path set, not a shortest-path recipe."""
import sys, math


def main():
    toks = sys.stdin.read().split()
    ptr = 0
    n = int(toks[ptr]); ptr += 1
    m = int(toks[ptr]); ptr += 1
    V0 = float(toks[ptr]); ptr += 1
    baseline_rate_bp = float(toks[ptr]); ptr += 1
    gamma = float(toks[ptr]); ptr += 1
    T_min = int(toks[ptr]); ptr += 1
    T_max = int(toks[ptr]); ptr += 1
    substance = [int(toks[ptr + i]) for i in range(n)]; ptr += n

    adj_out = [[] for _ in range(n)]   # adj_out[u] = list of (v, rate_bp, hold, itype)
    for _ in range(m):
        u = int(toks[ptr]); v = int(toks[ptr + 1])
        rate_bp = int(toks[ptr + 2]); hold = int(toks[ptr + 3]); itype = int(toks[ptr + 4])
        ptr += 5
        adj_out[u].append((v, rate_bp, hold, itype))
    b = int(toks[ptr]); ptr += 1
    backbone = [int(toks[ptr + i]) for i in range(b)]; ptr += b

    target = n - 1
    best_net = -1.0
    best_path = backbone[:]   # always-feasible fallback

    def dfs(u, path, hops):
        nonlocal best_net, best_path
        if u == target:
            total_time = sum(h[1] for h in hops)
            if not (T_min <= total_time <= T_max):
                return
            benefit_bp = sum(max(0.0, baseline_rate_bp - h[0]) for h in hops)
            required = math.ceil(gamma * benefit_bp / 10000.0)
            S = 0
            interm = path[1:-1]
            for i, node in enumerate(interm):
                eff = substance[node]
                type_in = hops[i][2]
                type_out = hops[i + 1][2]
                if type_in != type_out:
                    eff = eff // 2
                S += eff
            if S < required:
                return
            net = V0
            for h in hops:
                net *= (1.0 - h[0] / 10000.0)
            if net > best_net:
                best_net = net
                best_path = path[:]
            return
        for (v, rate_bp, hold, itype) in adj_out[u]:
            path.append(v)
            hops.append((rate_bp, hold, itype))
            dfs(v, path, hops)
            path.pop()
            hops.pop()

    dfs(0, [0], [])

    print(len(best_path))
    print(" ".join(str(x) for x in best_path))


if __name__ == "__main__":
    main()
