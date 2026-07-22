import sys


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def rect_cells(x, y, w, h):
    return frozenset((x + dx, y + dy) for dx in range(w) for dy in range(h))


def main():
    try:
        inp = open(sys.argv[1]).read().split()
    except Exception:
        fail("cannot read input")

    try:
        it = iter(inp)
        n = int(next(it))
        W = int(next(it))
        H = int(next(it))
        values = []
        footprints = []
        corridors = []
        for _ in range(n):
            val = int(next(it))
            fx = int(next(it)); fy = int(next(it)); fw = int(next(it)); fh = int(next(it))
            cx = int(next(it)); cy = int(next(it)); cw = int(next(it)); ch = int(next(it))
            if fw <= 0 or fh <= 0 or cw <= 0 or ch <= 0:
                fail("bad instance rect")
            if fx < 0 or fy < 0 or fx + fw > W or fy + fh > H:
                fail("bad instance footprint bounds")
            if cx < 0 or cy < 0 or cx + cw > W or cy + ch > H:
                fail("bad instance corridor bounds")
            values.append(val)
            footprints.append(rect_cells(fx, fy, fw, fh))
            corridors.append(rect_cells(cx, cy, cw, ch))
    except Exception:
        fail("bad input")

    if n <= 0:
        fail("empty instance")

    # ---- parse participant output: must be a permutation of 0..n-1 ----
    try:
        out_tokens = open(sys.argv[2]).read().split()
    except Exception:
        fail("cannot read output")

    if len(out_tokens) != n:
        fail("expected %d tokens, got %d" % (n, len(out_tokens)))

    order = []
    seen = set()
    for tok in out_tokens:
        try:
            v = int(tok)
        except Exception:
            fail("non-integer token %r" % tok)
        if v < 0 or v >= n:
            fail("index out of range: %d" % v)
        if v in seen:
            fail("duplicate index: %d" % v)
        seen.add(v)
        order.append(v)
    if len(seen) != n:
        fail("not a permutation")

    def simulate(seq):
        occupied = set()
        total = 0
        for idx in seq:
            fcells = footprints[idx]
            ccells = corridors[idx]
            if fcells & occupied:
                continue
            if ccells & occupied:
                continue
            occupied |= fcells
            total += values[idx]
        return total

    F = simulate(order)
    B = simulate(range(n))  # checker's own trivial baseline: install in raw index order
    B = max(1e-9, B)

    sc = min(1000.0, 100.0 * F / B)
    print("F=%d B=%d Ratio: %.6f" % (F, int(B), sc / 1000.0))


if __name__ == "__main__":
    main()
