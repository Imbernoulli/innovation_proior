# TIER: strong
# Insight: feasible deposit vectors that leak nothing to haulers live inside the
# no-arbitrage polytope { d : d_i - d_j <= dist(i,j) for all i,j }, where dist is the
# SHORTEST-PATH metric of the hauler corridor graph (not the raw edge list -- haulers
# relay bottles through intermediate cantons). We first solve each canton's own
# elasticity optimum in isolation (the "target" v_i), then take the McShane-Whitney
# Lipschitz extension of v w.r.t. the graph metric: d_i = min_j (v_j + dist(i,j)).
# This is the largest deposit vector that is provably arbitrage-free and never exceeds
# any single region's own elasticity target. Finally we run bounded coordinate ascent
# on top of it, letting each region ride back up to the tight polytope face against its
# neighbours whenever that raises the total replayed objective -- "saturating the faces".
import sys

INF = 10**9


def piecewise_rate(d, e):
    d0, d1, r1, d2, r2 = e['d0'], e['d1'], e['r1'], e['d2'], e['r2']
    if d <= d0:
        return 0
    if d <= d1:
        return (r1 * (d - d0)) // (d1 - d0)
    if d <= d2:
        return r1 + ((r2 - r1) * (d - d1)) // (d2 - d1)
    return r2


def region_value(d, e, V, Fpm):
    rate = piecewise_rate(d, e)
    ret = (e['pop'] * rate) // 1_000_000
    payout = ret * d
    fcost = (payout * Fpm) // 1000
    return ret * V - payout - fcost


def standalone_optimum(e, V, Fpm, D_MAX):
    best_d, best_v = 0, None
    for d in range(0, D_MAX + 1):
        val = region_value(d, e, V, Fpm)
        if best_v is None or val > best_v:
            best_v, best_d = val, d
    return best_d


def floyd_warshall(n, edges):
    dist = [[INF] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    for (u, v, c) in edges:
        if c < dist[u][v]:
            dist[u][v] = c
        if c < dist[v][u]:
            dist[v][u] = c
    for k in range(n):
        dk = dist[k]
        for i in range(n):
            dik = dist[i][k]
            if dik >= INF:
                continue
            di = dist[i]
            for j in range(n):
                nd = dik + dk[j]
                if nd < di[j]:
                    di[j] = nd
    return dist


def total_score(deposits, regions, n, V, Fpm, dist):
    returns = [0] * n
    for i in range(n):
        rate = piecewise_rate(deposits[i], regions[i])
        returns[i] = (regions[i]['pop'] * rate) // 1_000_000
    total = 0
    for i in range(n):
        best = deposits[i]
        best_j = -1
        for j in range(n):
            if j == i or dist[i][j] >= INF:
                continue
            cand = deposits[j] - dist[i][j]
            if cand > best:
                best = cand
                best_j = j
        eff = deposits[best_j] if best_j != -1 else deposits[i]
        pf = returns[i] * eff
        fcost = (pf * Fpm) // 1000
        total += returns[i] * V - pf - fcost
    return total


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    V = int(next(it))
    D_MAX = int(next(it))
    Fpm = int(next(it))
    regions = []
    for _ in range(n):
        name = next(it)
        pop = int(next(it))
        d0 = int(next(it))
        d1 = int(next(it))
        r1 = int(next(it))
        d2 = int(next(it))
        r2 = int(next(it))
        regions.append({'name': name, 'pop': pop, 'd0': d0, 'd1': d1, 'r1': r1, 'd2': d2, 'r2': r2})
    E = int(next(it))
    edges = []
    for _ in range(E):
        u = int(next(it))
        v = int(next(it))
        c = int(next(it))
        edges.append((u, v, c))

    dist = floyd_warshall(n, edges)

    # per-region unconstrained elasticity target
    v = [standalone_optimum(regions[i], V, Fpm, D_MAX) for i in range(n)]

    # McShane-Whitney Lipschitz extension w.r.t. the shortest-path metric: the largest
    # vector that is safely inside the no-arbitrage polytope and never overshoots any
    # region's own target.
    d_safe = [0] * n
    for i in range(n):
        best = v[i]  # j = i term, dist(i,i) = 0
        for j in range(n):
            if dist[i][j] >= INF:
                continue
            cand = v[j] + dist[i][j]
            if cand < best:
                best = cand
        d_safe[i] = max(0, min(D_MAX, best))

    def coordinate_ascent(start):
        # Bounded coordinate ascent: saturate polytope faces where it helps. Search the
        # FULL deposit range per region -- the fix for an arbitrage-triggered overpay
        # can be to raise the SOURCE region's own deposit past the trigger point, OR to
        # lower the DESTINATION region's deposit so it stops looking attractive; which
        # is cheaper depends on how many regions sit on each side, so we explore both
        # basins below by starting from two structurally different vectors.
        d = list(start)
        for _sweep in range(8):
            improved = False
            for i in range(n):
                best_val = total_score(d, regions, n, V, Fpm, dist)
                best_d = d[i]
                orig = d[i]
                for cand in range(0, D_MAX + 1):
                    d[i] = cand
                    val = total_score(d, regions, n, V, Fpm, dist)
                    if val > best_val:
                        best_val = val
                        best_d = cand
                d[i] = best_d
                if best_d != orig:
                    improved = True
            if not improved:
                break
        return d, total_score(d, regions, n, V, Fpm, dist)

    # Basin 1: start from the conservative Lipschitz-safe projection.
    d1, s1 = coordinate_ascent(d_safe)
    # Basin 2: start from the aggressive unconstrained per-region targets -- lets the
    # ascent discover the "raise the source instead of lowering the destinations" face.
    d2, s2 = coordinate_ascent(v)

    d_best = d1 if s1 >= s2 else d2
    print(" ".join(str(x) for x in d_best))


if __name__ == "__main__":
    main()
