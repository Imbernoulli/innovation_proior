import sys, math

INF = 10**9


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def piecewise_rate(d, e):
    d0, d1, r1, d2, r2 = e['d0'], e['d1'], e['r1'], e['d2'], e['r2']
    if d <= d0:
        return 0
    if d <= d1:
        return (r1 * (d - d0)) // (d1 - d0)
    if d <= d2:
        return r1 + ((r2 - r1) * (d - d1)) // (d2 - d1)
    return r2


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


def score_vector(deposits, regions, edges, n, V, Fpm, dist):
    """Full econ replay. Returns raw objective F (int, may be negative)."""
    returns = [0] * n
    for i in range(n):
        rate = piecewise_rate(deposits[i], regions[i])
        returns[i] = (regions[i]['pop'] * rate) // 1_000_000

    eff_payout = list(deposits)
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
        eff_payout[i] = deposits[best_j] if best_j != -1 else deposits[i]

    total_value = 0
    total_payout_float = 0
    for i in range(n):
        total_value += returns[i] * V
        pf = returns[i] * eff_payout[i]
        fcost = (pf * Fpm) // 1000
        total_payout_float += pf + fcost
    return total_value - total_payout_float


def main():
    inp = open(sys.argv[1]).read().split()
    out_text = open(sys.argv[2]).read()

    try:
        it = iter(inp)
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
    except Exception:
        fail("bad input")

    dist = floyd_warshall(n, edges)

    # ---- internal baseline B: uniform deposit at a fixed simple level ----
    U = 90
    baseline_deposits = [U] * n
    B_raw = score_vector(baseline_deposits, regions, edges, n, V, Fpm, dist)
    B = max(1, B_raw)

    # ---- parse participant output ----
    toks = out_text.split()
    if len(toks) != n:
        fail("expected %d tokens, got %d" % (n, len(toks)))
    deposits = []
    for t in toks:
        try:
            x = int(t)
        except Exception:
            fail("non-integer token %r" % t)
        if not math.isfinite(x):
            fail("non-finite %r" % t)
        if x < 0 or x > D_MAX:
            fail("deposit out of range %r" % t)
        deposits.append(x)

    F_raw = score_vector(deposits, regions, edges, n, V, Fpm, dist)
    F = max(0, F_raw)

    sc = min(1000.0, 100.0 * F / max(1e-9, float(B)))
    print("F=%d B=%d Ratio: %.6f" % (F_raw, B, sc / 1000.0))


if __name__ == "__main__":
    main()
