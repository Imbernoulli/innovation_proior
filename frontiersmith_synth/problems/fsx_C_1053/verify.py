#!/usr/bin/env python3
# Deterministic checker for "Saturating Double-Cover Placement" (format C, maximize).
# CLI: python3 verify.py <in> <out> <ans>   (ans is ignored)
# Prints "... Ratio: <r>" with r in [0,1], on the LAST line, and always exits 0.
import sys, math

INF = float("inf")


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def read_instance(path):
    try:
        toks = open(path).read().split()
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
    except Exception:
        return None
    return N, M, K, R, w, crit, tau, beta, edges


def floyd_warshall(N, edges):
    dist = [[INF] * N for _ in range(N)]
    for i in range(N):
        dist[i][i] = 0
    for (u, v, c) in edges:
        if 0 <= u < N and 0 <= v < N and c > 0:
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
    inst = read_instance(sys.argv[1])
    if inst is None:
        fail("bad instance")
    N, M, K, R, w, crit, tau, beta, edges = inst
    if N <= 0 or K <= 0 or K > N:
        fail("bad instance parameters")

    dist = floyd_warshall(N, edges)

    try:
        otoks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")

    if len(otoks) != K:
        fail("expected exactly %d integers (node ids), got %d" % (K, len(otoks)))

    S = []
    seen = set()
    for tok in otoks:
        # Reject non-integer tokens (this also rejects "nan"/"inf"/floats) without ever
        # widening to float, so an absurdly large-but-valid integer literal (e.g. hundreds
        # of digits) can't trigger an OverflowError on float() conversion -- it is instead
        # caught cleanly by the plain integer range check below.
        try:
            iv = int(tok)
        except Exception:
            fail("non-integer token %r" % tok)
        if iv < 0 or iv >= N:
            fail("node id out of range [0,%d): %r" % (N, tok))
        if iv in seen:
            fail("duplicate node id %d (sensors must be placed at distinct sites)" % iv)
        seen.add(iv)
        S.append(iv)

    F = objective(S, N, R, dist, w, crit, tau, beta)

    # internal trivial baseline: place all K sensors on the first K node ids (0..K-1)
    base_S = list(range(K))
    B = objective(base_S, N, R, dist, w, crit, tau, beta)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
