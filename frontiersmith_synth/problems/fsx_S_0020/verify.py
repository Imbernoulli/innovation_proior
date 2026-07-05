import sys

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def main():
    inp = open(sys.argv[1]).read().split()
    out = open(sys.argv[2]).read().split()

    it = iter(inp)
    try:
        n = int(next(it)); m = int(next(it))
    except Exception:
        fail("bad header")

    edges = []
    try:
        for _ in range(m):
            u = int(next(it)); v = int(next(it)); w = int(next(it))
            edges.append((u, v, w))
    except Exception:
        fail("bad edge data")

    # ---- checker baseline B: the index-block split (v <= n/2 -> 0, else 1) ----
    def side_block(v):
        return 0 if v <= n // 2 else 1
    B = 0
    for u, v, w in edges:
        if side_block(u) != side_block(v):
            B += w
    B = max(1, B)

    # ---- parse participant output ----
    if len(out) != n:
        fail("expected %d values, got %d" % (n, len(out)))
    s = [0] * (n + 1)  # 1-indexed
    c0 = c1 = 0
    for i, t in enumerate(out):
        if t == "0":
            s[i + 1] = 0; c0 += 1
        elif t == "1":
            s[i + 1] = 1; c1 += 1
        else:
            fail("value not 0/1: %r" % t)
    if abs(c0 - c1) > 1:
        fail("not a balanced bisection: count0=%d count1=%d" % (c0, c1))

    # ---- objective F ----
    F = 0
    for u, v, w in edges:
        if s[u] != s[v]:
            F += w

    sc = min(1000.0, 100.0 * F / max(1e-9, float(B)))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))

if __name__ == "__main__":
    main()
