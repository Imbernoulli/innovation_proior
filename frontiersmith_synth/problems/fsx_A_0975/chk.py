import sys, math

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def main():
    inf_path, ouf_path = sys.argv[1], sys.argv[2]
    try:
        toks = open(inf_path).read().split()
    except Exception:
        fail("bad input file")

    it = iter(toks)
    def rdi():
        return int(next(it))

    try:
        n = rdi(); m = rdi(); P = rdi()
        xs = [0.0] * (n + 1)
        ys = [0.0] * (n + 1)
        for v in range(1, n + 1):
            xs[v] = float(rdi()); ys[v] = float(rdi())
        eu = [0] * (m + 1)
        ev = [0] * (m + 1)
        edge_of = {}
        for e in range(1, m + 1):
            u = rdi(); v = rdi()
            eu[e] = u; ev[e] = v
            edge_of[(u, v)] = e
            edge_of[(v, u)] = e
        K = rdi()
        for _ in range(K):
            rdi(); rdi(); rdi(); rdi()
    except Exception:
        fail("malformed input (generator contract)")

    if n <= 0 or m <= 0 or P <= 0:
        fail("degenerate input")

    def dist(a, b):
        return math.hypot(xs[a] - xs[b], ys[a] - ys[b])

    # ---- internal baseline B: cut every edge as its own individual pierce,
    #      visiting edges in input order (a "do nothing clever" reference). ----
    B = float(P) * m
    for e in range(1, m):
        B += dist(ev[e], eu[e + 1])
    B = max(1.0, B)

    # ---- parse participant output ----
    try:
        out_toks = open(ouf_path).read().split()
    except Exception:
        fail("no output")
    if not out_toks:
        fail("empty output")

    oit = iter(out_toks)
    def ordi():
        return int(next(oit))
    def ordf():
        return float(next(oit))

    try:
        T = ordi()
    except Exception:
        fail("bad T")
    if T < 1 or T > m:
        fail("T out of range")

    used = [False] * (m + 1)
    n_used = 0
    trail_ends = []  # (first_vertex, last_vertex)
    for t in range(T):
        try:
            L = ordi()
        except Exception:
            fail("bad L")
        if L < 1 or L > m:
            fail("trail length out of range")
        try:
            vs = [ordi() for _ in range(L + 1)]
        except Exception:
            fail("bad trail vertices")
        for v in vs:
            if v < 1 or v > n:
                fail("vertex id out of range")
        for j in range(L):
            u, v = vs[j], vs[j + 1]
            e = edge_of.get((u, v))
            if e is None:
                fail("no such edge (%d,%d)" % (u, v))
            if used[e]:
                fail("edge %d cut more than once" % e)
            used[e] = True
            n_used += 1
        trail_ends.append((vs[0], vs[-1]))

    # trailing-token check
    try:
        next(oit)
        fail("trailing tokens after declared trails")
    except StopIteration:
        pass

    if n_used != m:
        fail("not all edges cut (%d of %d)" % (n_used, m))

    F = float(P) * T
    for t in range(1, T):
        F += dist(trail_ends[t - 1][1], trail_ends[t][0])
    F = max(1.0, F)

    if not (math.isfinite(F) and math.isfinite(B)):
        fail("non-finite objective")

    sc = min(1000.0, 100.0 * B / F)
    print("F=%.3f B=%.3f Ratio: %.6f" % (F, B, sc / 1000.0))

if __name__ == "__main__":
    main()
