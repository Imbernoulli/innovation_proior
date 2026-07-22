import sys

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def conv(a, b, n, p):
    """Cyclic convolution of length-n vectors a,b over F_p: res[k] = sum_i a[i]*b[(k-i)%n]."""
    res = [0] * n
    for idx in range(n):
        ai = a[idx]
        if ai == 0:
            continue
        for j in range(n):
            k = idx + j
            if k >= n:
                k -= n
            res[k] = (res[k] + ai * b[j]) % p
    return res

def poly_pow(c0, t, n, p):
    """Convolution power c0^{*t} via fast doubling (identity = delta at 0)."""
    result = [0] * n
    result[0] = 1 % p
    base = c0[:]
    while t > 0:
        if t & 1:
            result = conv(result, base, n, p)
        t >>= 1
        if t > 0:
            base = conv(base, base, n, p)
    return result

def main():
    inp = open(sys.argv[1]).read().split()
    out_raw = open(sys.argv[2]).read().split()

    try:
        it = iter(inp)
        n = int(next(it)); p = int(next(it)); r = int(next(it))
        s = int(next(it)); k = int(next(it))
        coeffs = [int(next(it)) for _ in range(2 * r + 1)]
        targets = []
        for _ in range(k):
            t = int(next(it)); pos = int(next(it)); val = int(next(it)); w = int(next(it))
            targets.append((t, pos, val, w))
    except Exception:
        fail("bad input")

    c0 = [0] * n
    for d in range(-r, r + 1):
        c0[d % n] = coeffs[d + r] % p

    # ---- parse & validate participant output strictly ----
    if len(out_raw) != n:
        fail("expected exactly %d integers, got %d" % (n, len(out_raw)))
    x0 = []
    for tok in out_raw:
        try:
            v = int(tok)
        except Exception:
            fail("non-integer token %r" % tok)
        if v < 0 or v > p - 1:
            fail("value %d out of range [0,%d]" % (v, p - 1))
        x0.append(v)
    nz = sum(1 for v in x0 if v != 0)
    if nz > s:
        fail("sparsity budget exceeded: %d nonzero cells > budget %d" % (nz, s))

    # ---- cache C_t = c0 convolved with itself t times, per distinct target time ----
    distinct_t = sorted(set(t for (t, _, _, _) in targets))
    Ct_cache = {t: poly_pow(c0, t, n, p) for t in distinct_t}

    # ---- evaluate the submitted seed against every target ----
    F = 0.0
    for (t, pos, val, w) in targets:
        Ct = Ct_cache[t]
        predicted = 0
        for m in range(n):
            if x0[m]:
                predicted = (predicted + Ct[(pos - m) % n] * x0[m]) % p
        if predicted == val:
            F += w

    # ---- internal baseline B: best single well-placed seed (the checker's own trivial
    # construction) -- try EVERY (target i, planting cell j) pair: the one value at cell j
    # that solves target i's equation exactly, then see how many targets (including
    # possibly others, by coincidence or structure) that single seed matches in total.
    # This is exhaustive over single-seed constructions: any single seed that matches at
    # least one target is, by definition, the solve of *some* target i at *some* cell j,
    # so the max over all (i,j) is the true best one-seed baseline. Always well-defined and
    # positive because the t=0 anchor target always yields a valid, non-degenerate
    # candidate at its own cell (self-coefficient exactly 1). ----
    B = 0
    for i, (ti, posi, vali, wi) in enumerate(targets):
        Ci = Ct_cache[ti]
        for j in range(n):
            a_j = Ci[(posi - j) % n] % p
            if a_j == 0:
                continue
            value = (vali * pow(a_j, p - 2, p)) % p
            matched = 0.0
            for (tj_, posj, valj, wj) in targets:
                Cj = Ct_cache[tj_]
                predicted = (Cj[(posj - j) % n] * value) % p
                if predicted == valj:
                    matched += wj
            if matched > B:
                B = matched
    B = max(1e-9, B)

    sc = min(1000.0, 100.0 * F / B)
    print("F=%.3f B=%.3f Ratio: %.6f" % (F, B, sc / 1000.0))

if __name__ == "__main__":
    main()
