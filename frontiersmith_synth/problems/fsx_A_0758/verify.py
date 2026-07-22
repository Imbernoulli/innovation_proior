import sys, math


def fail(msg):
    print("INFEASIBLE: " + msg)
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    if len(sys.argv) < 3:
        fail("bad invocation")
    inf, outf = sys.argv[1], sys.argv[2]

    with open(inf) as f:
        toks = f.read().split()
    idx = [0]

    def nxt():
        v = toks[idx[0]]
        idx[0] += 1
        return v

    n = int(nxt()); k = int(nxt()); r = int(nxt())
    s = [int(nxt()) for _ in range(n)]
    cost = [0] * (k + 1)
    for v in range(1, k + 1):
        cost[v] = int(nxt())
    me = int(nxt())
    expand = {}
    for _ in range(me):
        v = int(nxt()); x = int(nxt()); y = int(nxt())
        expand[v] = (x, y)
    mc = int(nxt())
    collapse = {}
    for _ in range(mc):
        x = int(nxt()); y = int(nxt()); z = int(nxt())
        collapse[(x, y)] = z

    B = sum(cost[v] for v in s)
    if B <= 0:
        fail("bad instance baseline")

    # ---- read participant artifact ----
    try:
        raw = open(outf).read()
    except Exception:
        fail("cannot read output")

    lines_nb = [ln.strip() for ln in raw.splitlines() if ln.strip() != ""]
    if not lines_nb:
        fail("empty output (missing move count)")

    first = lines_nb[0].split()
    if len(first) != 1:
        fail("first line must contain exactly one integer (move count)")
    try:
        m = int(first[0])
    except ValueError:
        fail("move count is not an integer")
    if m < 0 or m > r:
        fail(f"move count {m} out of budget [0,{r}]")
    if len(lines_nb) - 1 != m:
        fail(f"expected {m} move lines, found {len(lines_nb) - 1}")

    cur = list(s)
    MAXLEN = n + r + 5

    for li in range(1, m + 1):
        parts = lines_nb[li].split()
        if len(parts) != 2:
            fail(f"move line {li} malformed (need '<op> <pos>')")
        op, pos_tok = parts[0], parts[1]
        try:
            pos = int(pos_tok)
        except ValueError:
            fail(f"move line {li}: position is not an integer")

        if op == "E":
            if not (1 <= pos <= len(cur)):
                fail(f"move {li}: expand position out of range")
            v = cur[pos - 1]
            if v not in expand:
                fail(f"move {li}: symbol {v} has no expand rule")
            x, y = expand[v]
            cur[pos - 1:pos] = [x, y]
        elif op == "C":
            if not (1 <= pos <= len(cur) - 1):
                fail(f"move {li}: collapse position out of range")
            x, y = cur[pos - 1], cur[pos]
            if (x, y) not in collapse:
                fail(f"move {li}: pair ({x},{y}) has no collapse rule")
            z = collapse[(x, y)]
            cur[pos - 1:pos + 1] = [z]
        else:
            fail(f"move {li}: unknown op '{op}' (expected E or C)")

        if len(cur) > MAXLEN:
            fail("string grew beyond the allowed bound")

    for v in cur:
        if not isinstance(v, int) or not (1 <= v <= k):
            fail("final string contains an out-of-range symbol")

    F = sum(cost[v] for v in cur)
    if not math.isfinite(F) or F <= 0:
        fail("non-finite or non-positive final cost")

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print(f"B={B} F={F} moves={m}")
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
