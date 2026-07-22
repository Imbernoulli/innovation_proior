import sys, re

INT_RE = re.compile(r"^[+-]?\d+$")


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def divisors(n):
    ds = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            ds.append(i)
            if i != n // i:
                ds.append(n // i)
        i += 1
    return sorted(ds)


LAMBDA = 2  # fixed global penalty: F = correct - LAMBDA * wrong


def best_single_binomial(S, cand_divs, p):
    """Search a pool of candidate divisors d of (p-1); for each, find the value
    c* that x^d takes most often over x in S (a d-th power residue by
    construction), and score the IMPLIED binomial x^d - c* exactly: the full
    solution set of x^d = c* has exactly d elements in F_p (no enumeration
    needed), so correct = mode count, wrong = d - correct.
    Returns best F value found (>=0) among the pool, or 0 if pool is empty."""
    best = 0
    for d in cand_divs:
        counts = {}
        for x in S:
            v = pow(x, d, p)
            counts[v] = counts.get(v, 0) + 1
        if not counts:
            continue
        cnt = max(counts.values())
        F = cnt - LAMBDA * (d - cnt)
        if F > best:
            best = F
    return best


def main():
    try:
        inp = open(sys.argv[1]).read().split()
        it = iter(inp)
        p = int(next(it))
        T = int(next(it))
        m = int(next(it))
        S = [int(next(it)) for _ in range(m)]
    except Exception:
        fail("bad input")

    Sset = set(S)

    # ---- internal baseline B: best single binomial found via a NARROW window
    # search (small subgroup orders 12..25 only) -- a naive first guess that,
    # by design of the generator, can only ever recover the single SMALL
    # planted coset, never the bigger ones. ----
    window_divs = [d for d in divisors(p - 1) if 12 <= d <= 25]
    B = max(1, best_single_binomial(S, window_divs, p))

    # ---- parse participant output ----
    try:
        out = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")
    if not out:
        fail("empty output")
    if not INT_RE.match(out[0]):
        fail("bad term count token")
    k = int(out[0])
    if k < 0 or k > T:
        fail("term count %d out of [0,%d]" % (k, T))
    if len(out) != 1 + 2 * k:
        fail("expected %d tokens, got %d" % (1 + 2 * k, len(out)))

    terms = []
    seen_exp = set()
    for i in range(k):
        etok = out[1 + 2 * i]
        atok = out[2 + 2 * i]
        if not INT_RE.match(etok) or not INT_RE.match(atok):
            fail("non-integer term token")
        e = int(etok)
        a = int(atok)
        if e < 0 or e > p - 2:
            fail("exponent %d out of [0,%d]" % (e, p - 2))
        if a < 0 or a > p - 1:
            fail("coefficient %d out of [0,%d]" % (a, p - 1))
        if e in seen_exp:
            fail("duplicate exponent %d" % e)
        seen_exp.add(e)
        terms.append((e, a))

    # ---- evaluate f(x) = sum a_i * x^{e_i} mod p over ALL of F_p ----
    correct = 0
    wrong = 0
    for x in range(p):
        val = 0
        for e, a in terms:
            if a == 0:
                continue
            val = (val + a * pow(x, e, p)) % p
        if val == 0:
            if x in Sset:
                correct += 1
            else:
                wrong += 1

    F = correct - LAMBDA * wrong
    if F < 0:
        F = 0.0

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("correct=%d wrong=%d F=%.3f B=%d Ratio: %.6f" % (correct, wrong, F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
