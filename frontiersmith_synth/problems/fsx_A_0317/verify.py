import sys


def fail(msg):
    # Any feasibility violation -> hard zero.
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def sumset_size(A):
    """|A + A| with a=b allowed:  { a_i + a_j : i <= j }, computed exactly."""
    s = set()
    m = len(A)
    for i in range(m):
        ai = A[i]
        for j in range(i, m):
            s.add(ai + A[j])
    return len(s)


def _is_plain_int(tok):
    # accept optional single leading '-' ; reject nan/inf/floats/hex/junk
    t = tok[1:] if tok[:1] == "-" else tok
    return len(t) > 0 and t.isdigit()


def main():
    inst = open(sys.argv[1]).read().split()
    raw = open(sys.argv[2]).read().split()

    # ---- parse instance ----
    try:
        n = int(inst[0]); M = int(inst[1])
    except Exception:
        fail("bad instance")

    # ---- internal baseline B: the trivial arithmetic-progression apiary ----
    # A0 = {0,1,...,n-1} (always fits: M >= n-1). Its sumset is {0,1,...,2n-2},
    # so |A0+A0| = 2n-1. Build + measure it rather than trust the formula.
    A0 = list(range(n))
    B = sumset_size(A0)                      # == 2n-1, always positive

    # ---- parse & strictly validate participant artifact ----
    # Expect EXACTLY n integer tokens, all distinct, all in [0, M].
    if len(raw) != n:
        fail("expected exactly %d integers, got %d tokens" % (n, len(raw)))
    for tok in raw:
        if not _is_plain_int(tok):
            fail("non-integer / nan / inf token: %r" % tok)
    A = [int(tok) for tok in raw]

    seen = set()
    for p in A:
        if p < 0 or p > M:
            fail("post %d out of range [0,%d]" % (p, M))
        if p in seen:
            fail("duplicate post %d" % p)
        seen.add(p)

    # ---- objective: maximize distinct waggle-resonance channels |A+A| ----
    F = sumset_size(A)

    # maximization normalization (brief convention):
    #   trivial (AP) -> F=B -> Ratio 0.1 ; 10x-better caps at 1.0.
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
