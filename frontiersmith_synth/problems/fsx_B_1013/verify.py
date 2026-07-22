import sys, math


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def parse_int_token(tok):
    # Strict integer parsing -- rejects nan/inf/floats/garbage outright,
    # which is exactly what we want to treat as an infeasible submission.
    if tok is None:
        raise ValueError("missing token")
    t = tok.strip()
    if not (t.lstrip("+-").isdigit()):
        raise ValueError("not an integer token: %r" % tok)
    return int(t)


def compute_F(curves, r_in, r_out, K, S):
    """curves: list of (R, r, d, p). Returns F = 0.5*coverage + 0.5*entropy."""
    binwidth = (r_out - r_in) / K
    h = [0] * K
    P = 0
    for (R, r, d, p) in curves:
        g = math.gcd(R, r)
        w = R // g
        C = R - r
        for k in range(S):
            idx = (w * k + p) % S
            rho = C + d * math.cos(2.0 * math.pi * idx / S)
            pos = (rho - r_in) / binwidth
            b = int(pos)
            if b < 0:
                b = 0
            if b >= K:
                b = K - 1
            h[b] += 1
            P += 1
    coverage = sum(1 for b in h if b > 0) / K
    if P <= 0:
        entropy = 0.0
    else:
        ent = 0.0
        for b in h:
            if b > 0:
                pb = b / P
                ent -= pb * math.log(pb)
        denom = math.log(K) if K > 1 else 1.0
        entropy = ent / denom if denom > 0 else 0.0
    return 0.5 * coverage + 0.5 * entropy


def band_centers_and_caps(r_in, r_out, Q):
    """Partition the annulus into Q equal segments; each curve's center is
    its segment's midpoint and `cap` is the largest pen offset that keeps
    the curve's whole band inside its own segment (hence inside the
    annulus). Shared by the baseline and every reference solution so no
    curve is starved of offset room by sitting near r_in/r_out."""
    span = r_out - r_in
    seg = span // Q
    centers, caps = [], []
    for i in range(Q):
        lo = r_in + i * seg
        hi = r_in + (i + 1) * seg if i < Q - 1 else r_out
        c = (lo + hi) // 2
        cap = max(1, min(c - lo, hi - c, (hi - lo) // 2))
        centers.append(c)
        caps.append(cap)
    return centers, caps


def internal_baseline(r_in, r_out, K, Q, S, M):
    """Naive full-budget, evenly-spaced-center portfolio with a minimal,
    fixed pen offset that ignores the R/S gcd interaction entirely."""
    centers, _caps = band_centers_and_caps(r_in, r_out, Q)
    curves = []
    for C in centers:
        r = 2
        R = C + r
        d = 1
        p = 0
        curves.append((R, r, d, p))
    return compute_F(curves, r_in, r_out, K, S)


def main():
    try:
        inp_tokens = open(sys.argv[1]).read().split()
        r_in, r_out, K, Q, S, M = (int(x) for x in inp_tokens[:6])
    except Exception:
        fail("bad input file")

    B = internal_baseline(r_in, r_out, K, Q, S, M)
    B = max(1e-9, B)

    try:
        out_tokens = open(sys.argv[2]).read().split()
    except Exception:
        fail("cannot read output")

    if not out_tokens:
        fail("empty output")

    try:
        it = iter(out_tokens)
        q = parse_int_token(next(it, None))
    except Exception as e:
        fail("bad q: %s" % e)

    if not (1 <= q <= Q):
        fail("q=%d out of range [1,%d]" % (q, Q))

    curves = []
    try:
        for _ in range(q):
            R = parse_int_token(next(it, None))
            r = parse_int_token(next(it, None))
            d = parse_int_token(next(it, None))
            p = parse_int_token(next(it, None))
            curves.append((R, r, d, p))
    except Exception as e:
        fail("bad curve tokens: %s" % e)

    leftover = list(it)
    if leftover:
        fail("trailing garbage after %d curves: %r" % (q, leftover[:5]))

    for (R, r, d, p) in curves:
        if not (2 <= r < R <= M):
            fail("bad gear range R=%d r=%d M=%d" % (R, r, M))
        if not (1 <= d <= r):
            fail("bad pen offset d=%d r=%d" % (d, r))
        if not (0 <= p <= S - 1):
            fail("bad phase p=%d S=%d" % (p, S))
        C = R - r
        if not (r_in <= C - d and C + d <= r_out):
            fail("curve band [%d,%d] escapes annulus [%d,%d]" % (C - d, C + d, r_in, r_out))
        for v in (R, r, d, p):
            if not math.isfinite(v):
                fail("non-finite value")

    F = compute_F(curves, r_in, r_out, K, S)
    if not math.isfinite(F):
        fail("non-finite objective")

    sc = min(1000.0, 100.0 * F / B)
    sc = max(0.0, sc)
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
