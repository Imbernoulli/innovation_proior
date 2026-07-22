# TIER: strong
# Insight: the double-cover bonus on critical nodes makes the objective NOT submodular --
# a critical node's second reaching sensor can be worth far more than its first, so a
# myopic one-step-at-a-time greedy can permanently miss it (its first cover never looks
# locally attractive enough to beat other single sites, so greedy never "gets started" on
# it, and by the time the budget would otherwise be free the cheap alternatives are gone).
#
# Strategy: start from the same greedy plan, then explicitly co-optimize redundancy and
# reach -- for every critical node, evaluate the BUNDLE of its two best-reaching candidate
# sites (the pair that would unlock its double-cover bonus) against the currently weakest
# members of the current placement, and swap the whole bundle in at once whenever that
# joint move raises the true objective, even though neither site alone would. Repeat until
# no bundle swap helps, then polish with single-site exchanges.
import sys, math

INF = float("inf")


def floyd_warshall(N, edges):
    dist = [[INF] * N for _ in range(N)]
    for i in range(N):
        dist[i][i] = 0
    for (u, v, c) in edges:
        if c < dist[u][v]:
            dist[u][v] = c
            dist[v][u] = c
    for k in range(N):
        dk = dist[k]
        for i in range(N):
            dik = dist[i][k]
            if dik == INF:
                continue
            di = dist[i]
            for j in range(N):
                nd = dik + dk[j]
                if nd < di[j]:
                    di[j] = nd
    return dist


def objective(S, N, R, dist, w, crit, tau, beta):
    cov = [0] * N
    mass = [0.0] * N
    for s in S:
        ds = dist[s]
        for v in range(N):
            d = ds[v]
            if d <= R:
                cov[v] += 1
                mass[v] += (R - d)
    total = 0.0
    for v in range(N):
        if mass[v] > 0.0:
            total += w[v] * (1.0 - math.exp(-mass[v] / tau[v]))
        if crit[v] and cov[v] >= 2:
            total += beta[v]
    return total


def greedy_build(N, R, K, dist, w, crit, tau, beta):
    S = []
    remaining = set(range(N))
    cur = 0.0
    for _ in range(K):
        best_v, best_gain = None, -1.0
        for cand in sorted(remaining):
            g = objective(S + [cand], N, R, dist, w, crit, tau, beta) - cur
            if g > best_gain + 1e-12:
                best_gain, best_v = g, cand
        S.append(best_v)
        remaining.discard(best_v)
        cur = objective(S, N, R, dist, w, crit, tau, beta)
    return S


def bundle_improve(S, N, R, K, dist, w, crit, tau, beta, rounds):
    obj = lambda X: objective(X, N, R, dist, w, crit, tau, beta)
    for _ in range(rounds):
        cur = obj(S)
        Sset = set(S)
        best_delta, best_newS = 1e-9, None
        for v in range(N):
            if not crit[v] or beta[v] <= 0:
                continue
            reach_sites = [s for s in range(N) if dist[s][v] <= R]
            if len(reach_sites) < 2:
                continue
            reach_sites.sort(key=lambda s: -(R - dist[s][v]))
            c1, c2 = reach_sites[0], reach_sites[1]
            need = [x for x in (c1, c2) if x not in Sset]
            if not need:
                continue
            pool = [r for r in S if r not in (c1, c2)]
            pool.sort(key=lambda r: obj([x for x in S if x != r]), reverse=True)
            if len(pool) < len(need):
                continue
            remove_set = pool[:len(need)]
            newS = [x for x in S if x not in remove_set] + need
            if len(newS) != K:
                continue
            delta = obj(newS) - cur
            if delta > best_delta:
                best_delta, best_newS = delta, newS
        if best_newS is None:
            break
        S = best_newS
    return S


def single_exchange_polish(S, N, R, K, dist, w, crit, tau, beta, rounds):
    obj = lambda X: objective(X, N, R, dist, w, crit, tau, beta)
    for _ in range(rounds):
        cur = obj(S)
        Sset = set(S)
        best_delta, best_newS = 1e-9, None
        for r in S:
            base = [x for x in S if x != r]
            for c in range(N):
                if c in Sset:
                    continue
                newS = base + [c]
                delta = obj(newS) - cur
                if delta > best_delta:
                    best_delta, best_newS = delta, newS
        if best_newS is None:
            break
        S = best_newS
    return S


def main():
    toks = sys.stdin.read().split()
    p = 0
    N = int(toks[p]); p += 1
    M = int(toks[p]); p += 1
    K = int(toks[p]); p += 1
    R = int(toks[p]); p += 1
    w = [0] * N; crit = [0] * N; tau = [1] * N; beta = [0] * N
    for v in range(N):
        w[v] = int(toks[p]); p += 1
        crit[v] = int(toks[p]); p += 1
        tau[v] = int(toks[p]); p += 1
        beta[v] = int(toks[p]); p += 1
    edges = []
    for _ in range(M):
        u = int(toks[p]); p += 1
        v = int(toks[p]); p += 1
        c = int(toks[p]); p += 1
        edges.append((u, v, c))

    dist = floyd_warshall(N, edges)

    S = greedy_build(N, R, K, dist, w, crit, tau, beta)
    S = bundle_improve(S, N, R, K, dist, w, crit, tau, beta, rounds=N)
    S = single_exchange_polish(S, N, R, K, dist, w, crit, tau, beta, rounds=10)

    print(" ".join(str(x) for x in S))


if __name__ == "__main__":
    main()
