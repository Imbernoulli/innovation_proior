import sys


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def canonical_code(N):
    """Checker's own baseline rack: near-balanced code assigned purely by chime ID,
    no frequency/order information used. Standard 'x short + (N-x) long' construction
    with exactly N leaves and minimum possible max-depth ceil(log2 N)."""
    W = 0
    while (1 << W) < N:
        W += 1
    x = (1 << W) - N
    codes = [None] * (N + 1)
    for i in range(1, x + 1):
        codes[i] = format(i - 1, "0{}b".format(W - 1)) if W - 1 > 0 else ""
    for j in range(N - x):
        idx = x + 1 + j
        val = 2 * x + j
        codes[idx] = format(val, "0{}b".format(W))
    return codes, W


def walk_cost(codes, trace):
    total = 0
    prev = codes[trace[0]]
    for k in range(1, len(trace)):
        cur = codes[trace[k]]
        m = len(prev) if len(prev) < len(cur) else len(cur)
        lcp = 0
        while lcp < m and prev[lcp] == cur[lcp]:
            lcp += 1
        total += len(prev) + len(cur) - 2 * lcp
        prev = cur
    return total


def main():
    inp = open(sys.argv[1]).read().split()
    outp = open(sys.argv[2]).read().split()

    try:
        it = iter(inp)
        N = int(next(it))
        Dmax = int(next(it))
        L = int(next(it))
        trace = [int(next(it)) for _ in range(L)]
    except Exception:
        fail("bad input")

    for s in trace:
        if s < 1 or s > N:
            fail("bad input trace symbol")

    if len(outp) != N:
        fail("expected %d codewords, got %d" % (N, len(outp)))

    codes = [None] * (N + 1)
    for i in range(1, N + 1):
        tok = outp[i - 1]
        if not tok or any(ch not in "01" for ch in tok):
            fail("bad address for chime %d: %r" % (i, tok))
        if len(tok) > Dmax:
            fail("address for chime %d exceeds Dmax=%d" % (i, Dmax))
        codes[i] = tok

    # prefix-free check: sort lexicographically, only adjacent pairs need checking
    # (a proper-prefix relationship always survives as an adjacent pair after sorting).
    order = sorted(range(1, N + 1), key=lambda i: codes[i])
    for k in range(len(order) - 1):
        a, b = codes[order[k]], codes[order[k + 1]]
        if a == b:
            fail("duplicate address for chimes %d and %d" % (order[k], order[k + 1]))
        if b.startswith(a):
            fail("address of chime %d is a prefix of chime %d" % (order[k], order[k + 1]))

    F = walk_cost(codes, trace)

    base_codes, W = canonical_code(N)
    B = walk_cost(base_codes, trace)
    B = max(1, B)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("N=%d L=%d F=%d B=%d Ratio: %.6f" % (N, L, F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
