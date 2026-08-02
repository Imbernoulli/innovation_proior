import sys, math

SPEED = {0: 4.0, 1: 2.0, 2: 1.0}


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def main():
    try:
        inp = open(sys.argv[1]).read().split()
    except Exception:
        fail("cannot read input")
    try:
        out_raw = open(sys.argv[2]).read()
    except Exception:
        fail("cannot read output")

    try:
        it = iter(inp)
        N = int(next(it)); K = int(next(it))
        if N <= 0 or K < 0:
            fail("bad header")
        weight = [0] * N
        R = [set() for _ in range(N)]
        W = [set() for _ in range(N)]
        for i in range(N):
            w = int(next(it))
            if w <= 0:
                fail("bad weight")
            weight[i] = w
            nr = int(next(it))
            if nr < 0:
                fail("bad nr")
            for _ in range(nr):
                k = int(next(it))
                if k < 0 or k >= K:
                    fail("read key out of range")
                R[i].add(k)
            nw = int(next(it))
            if nw < 0:
                fail("bad nw")
            for _ in range(nw):
                k = int(next(it))
                if k < 0 or k >= K:
                    fail("write key out of range")
                W[i].add(k)
        leftover = list(it)
        if leftover:
            fail("trailing garbage in input")
    except SystemExit:
        raise
    except Exception:
        fail("malformed input")

    # ---- static conflict structures (independent of chosen levels) ----
    rw_edges = []   # i -> j  meaning R_i and W_j share a key
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if R[i] & W[j]:
                rw_edges.append((i, j))
    ww_pairs = []   # unordered i<j sharing a write key
    for i in range(N):
        for j in range(i + 1, N):
            if W[i] & W[j]:
                ww_pairs.append((i, j))

    # ---- parse participant output: exactly N tokens, each in {0,1,2} ----
    out_toks = out_raw.split()
    if len(out_toks) != N:
        fail("expected exactly N level tokens, got %d" % len(out_toks))
    lvl = [0] * N
    for i, tok in enumerate(out_toks):
        try:
            x = float(tok)
        except Exception:
            fail("level token not a number")
        if not math.isfinite(x):
            fail("non-finite level")
        r = round(x)
        if abs(x - r) > 1e-6:
            fail("level not integral")
        if r not in (0, 1, 2):
            fail("level out of range {0,1,2}")
        lvl[i] = r

    # ---- feasibility: exposed rw-cycle check ----
    # exposed edge i->j survives only if both endpoints are at level <= 1
    adj = [[] for _ in range(N)]
    for (i, j) in rw_edges:
        if lvl[i] <= 1 and lvl[j] <= 1:
            adj[i].append(j)

    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * N
    has_cycle = [False]

    def dfs(u):
        color[u] = GRAY
        for v in adj[u]:
            if color[v] == GRAY:
                has_cycle[0] = True
                return
            if color[v] == WHITE:
                dfs(v)
                if has_cycle[0]:
                    return
        color[u] = BLACK

    for s in range(N):
        if color[s] == WHITE:
            dfs(s)
        if has_cycle[0]:
            fail("exposed read-write dependency cycle (write-skew) survives")

    # ---- feasibility: exposed ww lost-update check ----
    for (i, j) in ww_pairs:
        if lvl[i] == 0 and lvl[j] == 0:
            fail("lost-update: two READ COMMITTED transactions share a write key")

    # ---- objective ----
    F = sum(weight[i] * SPEED[lvl[i]] for i in range(N))

    # internal baseline B: the checker's own trivial reference -- run every
    # transaction fully SERIALIZABLE (always safe, ignores all structure).
    B = sum(weight[i] * SPEED[2] for i in range(N))
    B = max(B, 1e-9)

    sc = min(1000.0, 100.0 * F / B)
    print("F=%.4f B=%.4f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
