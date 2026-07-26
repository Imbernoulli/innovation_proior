import sys, math

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def parse_instance(text):
    toks = text.split()
    p = 0
    def nxt():
        nonlocal p
        v = toks[p]; p += 1
        return v
    N = int(nxt()); A = int(nxt()); L = int(nxt()); K = int(nxt()); W = int(nxt())
    scenarios = []
    for _ in range(K):
        q = int(nxt())
        orders = []
        for _ in range(q):
            s = int(nxt())
            skus = [int(nxt()) for _ in range(s)]
            orders.append(skus)
        scenarios.append(orders)
    return N, A, L, K, W, scenarios


def scenario_totals(perm, N, A, L, K, W, scenarios):
    totals = []
    for orders in scenarios:
        tot = 0
        for skus in orders:
            aisles = set()
            max_depth = {}
            for sku in skus:
                slot = perm[sku]
                aisle = slot // L + 1
                depth = slot % L + 1
                aisles.add(aisle)
                if depth > max_depth.get(aisle, 0):
                    max_depth[aisle] = depth
            if not aisles:
                continue
            u_m = max(aisles)
            tot += 2 * W * u_m + 2 * sum(max_depth.values())
        totals.append(tot)
    return totals


def p90(values):
    v = sorted(values)
    K = len(v)
    idx = max(0, min(K - 1, math.ceil(0.9 * K) - 1))
    return v[idx]


def main():
    inp = open(sys.argv[1]).read()
    out_text = open(sys.argv[2]).read()

    try:
        N, A, L, K, W, scenarios = parse_instance(inp)
    except Exception:
        fail("bad input")

    # ---- internal baseline: identity slot assignment perm[i] = i ----
    ident = list(range(N))
    base_totals = scenario_totals(ident, N, A, L, K, W, scenarios)
    B = max(1e-9, float(p90(base_totals)))

    # ---- parse + validate participant artifact ----
    out = out_text.split()
    if not out:
        fail("empty output")
    try:
        decl_n = int(out[0])
    except Exception:
        fail("bad header")
    if decl_n != N:
        fail("declared N=%d != required %d" % (decl_n, N))
    if len(out) < 1 + N:
        fail("not enough tokens (need %d, got %d)" % (1 + N, len(out) - 1))
    if len(out) > 1 + N:
        fail("too many tokens (extra output after permutation)")

    perm = [None] * N
    seen = set()
    for i in range(N):
        tok = out[1 + i]
        try:
            v = float(tok)
        except Exception:
            fail("non-numeric token at position %d" % i)
        if not math.isfinite(v):
            fail("non-finite value at position %d" % i)
        if v != int(v):
            fail("non-integer value at position %d" % i)
        iv = int(v)
        if iv < 0 or iv >= N:
            fail("slot %d for SKU %d out of range [0,%d]" % (iv, i, N - 1))
        if iv in seen:
            fail("slot %d assigned twice (not a permutation)" % iv)
        seen.add(iv)
        perm[i] = iv

    F = float(p90(scenario_totals(perm, N, A, L, K, W, scenarios)))

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%.3f B=%.3f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
