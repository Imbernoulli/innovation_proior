import sys

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def main():
    inp = open(sys.argv[1]).read().split()
    out = open(sys.argv[2]).read().split()

    # ---- parse instance ----
    try:
        it = iter(inp)
        m = int(next(it))
        e = int(next(it))
        emb = set()
        for _ in range(e):
            w = int(next(it)); t = int(next(it))
            emb.add((w, t))
    except Exception:
        fail("bad input")

    # ---- internal baseline B: delivery window t=0, minus embargoed ----
    B = sum(1 for w in range(m) if (w, 0) not in emb)
    B = max(1, B)

    # ---- parse participant output ----
    try:
        k = int(out[0])
    except Exception:
        fail("parse count")
    toks = out[1:]
    if len(toks) != 2 * k:
        fail("token count mismatch")

    cells = []
    try:
        for j in range(k):
            w = int(toks[2 * j]); t = int(toks[2 * j + 1])
            cells.append((w, t))
    except Exception:
        fail("parse cells")

    seen = set()
    for (w, t) in cells:
        if not (0 <= w < m and 0 <= t < m):
            fail("out of range %d %d" % (w, t))
        if (w, t) in seen:
            fail("duplicate %d %d" % (w, t))
        if (w, t) in emb:
            fail("embargoed %d %d" % (w, t))
        seen.add((w, t))

    # ---- corner (spoilage cascade) check ----
    # A corner is apex (w,t), (w+d,t), (w,t+d), d != 0. Every corner has a unique
    # apex sharing a window with one leg and a warehouse with the other; enumerate
    # each active order as apex.
    by_row = {}   # t -> list of w  (same delivery window)
    for (w, t) in seen:
        by_row.setdefault(t, []).append(w)
    for (w, t) in seen:
        for w2 in by_row[t]:
            d = w2 - w
            if d == 0:
                continue
            if (w, t + d) in seen:
                fail("spoilage cascade (%d,%d) (%d,%d) (%d,%d)" % (w, t, w2, t, w, t + d))

    F = k
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))

if __name__ == "__main__":
    main()
