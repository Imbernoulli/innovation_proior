import sys, math

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def main():
    try:
        inp = open(sys.argv[1]).read().split()
    except Exception:
        fail("no input")
    out_raw = open(sys.argv[2]).read().split()

    # ---- parse instance ----
    try:
        it = iter(inp)
        K = int(next(it)); M = int(next(it)); P = int(next(it))
        C = int(next(it)); TPM = int(next(it))
        targets = [float(next(it)) for _ in range(K)]
        catalog = [int(next(it)) for _ in range(M)]
        corners = []
        for _ in range(C):
            corners.append([int(next(it)) for _ in range(M)])
    except Exception:
        fail("bad instance")
    t = TPM / 1000.0

    # ---- internal baseline B: realize every tap as 1/2 (two matched units of value[0]).
    #      corner-invariant, so B = worst nominal deviation of the all-half network. ----
    B = max(abs(0.5 - r) for r in targets)
    B = max(B, 1e-9)

    # ---- parse participant netlist: K lines each 2M nonneg integer counts
    #      (top[0..M-1] then bottom[0..M-1]) ----
    if len(out_raw) != K * 2 * M:
        fail("expected %d integers, got %d" % (K * 2 * M, len(out_raw)))
    counts = []
    idx = 0
    for tok in out_raw:
        try:
            x = int(tok)
        except Exception:
            fail("non-integer count %r" % tok)
        if x != x or x == float("inf"):     # defensive; int() already rejects nan/inf tokens
            fail("non-finite count")
        if x < 0 or x > P:
            fail("count %d out of [0,%d]" % (x, P))
        counts.append(x)

    total_parts = sum(counts)
    if total_parts > P:
        fail("part budget exceeded: %d > %d" % (total_parts, P))

    taps = []
    for k in range(K):
        base = k * 2 * M
        top = counts[base:base + M]
        bot = counts[base + M:base + 2 * M]
        if sum(top) < 1:
            fail("tap %d has empty top resistance" % k)
        if sum(bot) < 1:
            fail("tap %d has empty bottom resistance" % k)
        taps.append((top, bot))

    # ---- evaluate worst deviation over {nominal} U {C corner vectors} ----
    # precompute nominal weighted sums for speed; corners recompute (O(K*C*M)).
    eval_signs = [[0] * M] + corners
    F = 0.0
    for kk in range(K):
        top, bot = taps[kk]
        rk = targets[kk]
        # only iterate over catalog indices actually used (sparse) for speed
        used = [j for j in range(M) if top[j] or bot[j]]
        for signs in eval_signs:
            rt = 0.0
            rb = 0.0
            for j in used:
                f = catalog[j] * (1.0 + signs[j] * t)
                rt += top[j] * f
                rb += bot[j] * f
            denom = rt + rb
            if denom <= 0:
                fail("non-positive network at tap %d" % kk)
            ratio = rb / denom
            dev = abs(ratio - rk)
            if dev > F:
                F = dev
    if not math.isfinite(F):
        fail("non-finite objective")

    F = max(F, 1e-9)
    sc = min(1000.0, 100.0 * B / F)
    print("F=%.6f B=%.6f parts=%d Ratio: %.6f" % (F, B, total_parts, sc / 1000.0))

if __name__ == "__main__":
    main()
