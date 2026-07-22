import sys


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def main():
    try:
        inp = open(sys.argv[1]).read().split()
    except Exception:
        fail("cannot read input")

    try:
        it = iter(inp)
        n = int(next(it))
        m = int(next(it))
        s = int(next(it))
        r1 = [0] * n
        r2 = [0] * n
        f = [0] * n
        for i in range(n):
            a = int(next(it))
            b = int(next(it))
            w = int(next(it))
            if a < 1 or a > m or b < 1 or b > m or a == b or w <= 0:
                fail("malformed instance (should not happen)")
            r1[i], r2[i], f[i] = a, b, w
    except Exception:
        fail("cannot parse input")

    # ---- internal baseline B: naive first-fit in ledger (input) order ----
    def first_fit_cost(order):
        used_slot = set()   # (room, slot) taken
        annex_used = 0
        total = 0
        for i in order:
            placed = False
            for code, room, slot in ((1, r1[i], 1), (2, r1[i], 2), (3, r2[i], 1), (4, r2[i], 2)):
                if (room, slot) not in used_slot:
                    used_slot.add((room, slot))
                    total += code * f[i]
                    placed = True
                    break
            if not placed:
                if annex_used < s:
                    annex_used += 1
                total += 10 * f[i]   # counted regardless, for a well-defined internal scratch value
                placed = True
        return total

    B = first_fit_cost(range(n))
    B = max(1, B)

    # ---- parse participant output ----
    try:
        out = open(sys.argv[2]).read().split()
    except Exception:
        fail("cannot read output")

    if len(out) != n:
        fail("expected %d tokens, got %d" % (n, len(out)))

    codes = []
    for tok in out:
        try:
            c = int(tok)
        except Exception:
            fail("non-integer token %r" % tok)
        if c < 0 or c > 4:
            fail("code %d out of range [0,4]" % c)
        codes.append(c)

    used_slot = {}
    annex_used = 0
    F = 0
    for i, c in enumerate(codes):
        if c == 0:
            annex_used += 1
            if annex_used > s:
                fail("annex over capacity (%d > %d)" % (annex_used, s))
            F += 10 * f[i]
            continue
        room = r1[i] if c in (1, 2) else r2[i]
        slot = 1 if c in (1, 3) else 2
        key = (room, slot)
        if key in used_slot:
            fail("cot collision at room %d slot %d: regulars %d and %d" % (room, slot, used_slot[key], i))
        used_slot[key] = i
        F += c * f[i]

    if F <= 0:
        fail("non-positive total cost")

    sc = min(1000.0, 100.0 * B / max(1e-9, float(F)))
    print("B=%d F=%d Ratio: %.6f" % (B, F, sc / 1000.0))


if __name__ == "__main__":
    main()
