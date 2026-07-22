import sys


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def popcount(x):
    return bin(x).count("1")


def sweep_patterns(L, Bmin, Bmax, G, pairs):
    """All published burst / composite-double-burst error patterns, as bitmasks."""
    pats = set()
    for b in range(Bmin, Bmax + 1):
        for off in range(L):
            p = 0
            for k in range(b):
                p |= 1 << ((off + k) % L)
            pats.add(p)
    for (b1, b2) in pairs:
        for off in range(L):
            p = 0
            for k in range(b1):
                p |= 1 << ((off + k) % L)
            start2 = off + b1 + G
            for k in range(b2):
                p |= 1 << ((start2 + k) % L)
            pats.add(p)
    return sorted(pats)


def worst_case_mean_margin(L, codewords, pats):
    """mean over codewords i of: min over swept patterns p of
       ( min over j!=i of popcount((c_i^c_j) ^ p)  -  popcount(p) )
       normalized by L. This is the mean *worst-case decode margin* metric."""
    K = len(codewords)
    diffs = [[0] * K for _ in range(K)]
    for a in range(K):
        for b in range(K):
            if a != b:
                diffs[a][b] = codewords[a] ^ codewords[b]
    total = 0.0
    for i in range(K):
        worst = None
        for p in pats:
            wp = popcount(p)
            nearest = min(popcount(diffs[i][j] ^ p) for j in range(K) if j != i)
            m = nearest - wp
            if worst is None or m < worst:
                worst = m
        total += worst
    return total / K / L


def bits_of(K):
    return max(1, (K - 1).bit_length())


def comb_codebook(L, K, modulus):
    """Baseline/insight construction: 'nbits' index bits, row r = the arithmetic
    progression {r, r+modulus, r+2*modulus, ...} (a spread comb, NOT a contiguous
    block). Codeword i = XOR of the rows selected by the set bits of i."""
    nbits = bits_of(K)
    rows = []
    for r in range(modulus):
        v = 0
        p = r
        while p < L:
            v |= 1 << p
            p += modulus
        rows.append(v)
    cws = []
    for i in range(K):
        v = 0
        for r in range(nbits):
            if (i >> r) & 1:
                v ^= rows[r]
        cws.append(v)
    return cws


BASELINE_EXTRA_MODULUS = 6  # fixed calibration constant, same on every test


def internal_baseline(L, K, Bmax):
    """The checker's own trivial-but-valid codebook: a comb code with a modulus
    a bit LARGER than the minimum needed to be burst-safe (max(nbits,Bmax+1)),
    i.e. it is spread out enough to never be exactly matched by a swept burst,
    but wastes some raw Hamming weight compared to the tightest safe spacing."""
    nbits = bits_of(K)
    m_strong = max(nbits, Bmax + 1)
    m_base = m_strong + BASELINE_EXTRA_MODULUS
    return comb_codebook(L, K, m_base)


def main():
    inp = open(sys.argv[1]).read().split()
    try:
        it = iter(inp)
        L = int(next(it))
        K = int(next(it))
        Bmin = int(next(it))
        Bmax = int(next(it))
        G = int(next(it))
        D = int(next(it))
        pairs = []
        for _ in range(D):
            b1 = int(next(it))
            b2 = int(next(it))
            pairs.append((b1, b2))
    except Exception:
        fail("bad input")

    if L <= 0 or K <= 0:
        fail("bad instance")

    # ---- internal baseline B (always well-defined and > 0 for our instances) ----
    base_cws = internal_baseline(L, K, Bmax)
    pats = sweep_patterns(L, Bmin, Bmax, G, pairs)
    B = worst_case_mean_margin(L, base_cws, pats)
    B = max(1e-9, B)

    # ---- parse & strictly validate participant output ----
    try:
        raw = open(sys.argv[2]).read()
    except Exception:
        fail("no output")
    lines = raw.splitlines()
    # drop trailing blank lines only (do not silently accept garbage in the middle)
    while lines and lines[-1].strip() == "":
        lines.pop()
    if len(lines) != K:
        fail("expected %d codewords, got %d lines" % (K, len(lines)))

    valid_chars = set("01")
    seen = set()
    cws = []
    for line in lines:
        s = line.strip()
        if len(s) != L:
            fail("codeword length %d != L=%d" % (len(s), L))
        if any(ch not in valid_chars for ch in s):
            fail("non-binary codeword %r" % s)
        if s in seen:
            fail("duplicate codeword %r" % s)
        seen.add(s)
        v = int(s, 2)
        cws.append(v)

    F = worst_case_mean_margin(L, cws, pats)

    sc = min(1000.0, 100.0 * F / B)
    ratio = sc / 1000.0
    ratio = max(0.0, min(1.0, ratio))
    print("F=%.6f B=%.6f Ratio: %.6f" % (F, B, ratio))


if __name__ == "__main__":
    main()
