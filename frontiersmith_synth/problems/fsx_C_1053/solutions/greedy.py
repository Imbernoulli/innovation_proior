# TIER: greedy
# Textbook budgeted-coverage greedy: repeatedly add the not-yet-chosen site with the
# largest marginal increase to the total objective (recomputed honestly, bonus included),
# until K sites are placed. This is the standard reference algorithm for "maximize a
# coverage-style objective under a cardinality budget" and is optimal for submodular
# objectives -- but it is myopic: it never looks past the current step, so it can never
# see that trading two locally-weaker picks for a matched pair near a critical node would
# unlock a large bonus later.
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

    S = []
    remaining = set(range(N))
    cur = 0.0
    for _ in range(K):
        best_v = None
        best_gain = -1.0
        for cand in sorted(remaining):
            newF = objective(S + [cand], N, R, dist, w, crit, tau, beta)
            gain = newF - cur
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_v = cand
        S.append(best_v)
        remaining.discard(best_v)
        cur = objective(S, N, R, dist, w, crit, tau, beta)

    print(" ".join(str(x) for x in S))


if __name__ == "__main__":
    main()
