import sys

# Deterministic scorer for hidden-block-orbit-cover (format C, maximize coverage).
# Reads the instance from <in> and the participant's k words from <out>.
# Applies each word to the seed set S; F = number of distinct points covered by the union.
# Baseline B = coverage of a trivial construction the checker builds itself: spam move 1
#   (an intra-district shuffle) at lengths 0,1,...,capped at L over k words. Ratio = 100*F/B.

def fail(msg):
    print("INVALID %s Ratio: 0.0" % msg)
    sys.exit(0)

def read_ints(path):
    with open(path) as f:
        data = f.read().split()
    return data

def main():
    inp = sys.argv[1]
    out = sys.argv[2]
    d = read_ints(inp)
    idx = 0
    n = int(d[idx]); idx += 1
    m = int(d[idx]); idx += 1
    k = int(d[idx]); idx += 1
    L = int(d[idx]); idx += 1
    t = int(d[idx]); idx += 1
    Gens = []
    for _ in range(m):
        G = [int(d[idx + i]) for i in range(n)]
        idx += n
        Gens.append(G)
    S = [int(d[idx + i]) for i in range(t)]
    idx += t

    # inverse generators
    Ginv = []
    for G in Gens:
        H = [0] * n
        for x in range(n):
            H[G[x]] = x
        Ginv.append(H)

    def apply_word(word, p):
        cur = p
        for tok in word:
            if tok > 0:
                cur = Gens[tok - 1][cur]
            else:
                cur = Ginv[-tok - 1][cur]
        return cur

    def coverage(words):
        cov = set()
        for w in words:
            for p in S:
                cov.add(apply_word(w, p))
        return len(cov)

    # ---- baseline B: spam move 1 at increasing (capped) lengths ----
    base_words = [[1] * min(i, L) for i in range(k)]
    B = coverage(base_words)
    if B <= 0:
        B = len(set(S))

    # ---- parse participant output strictly ----
    with open(out) as f:
        lines = f.read().split("\n")
    words = []
    for ln in lines:
        toks = ln.split()
        if not toks:
            continue  # blank line = skip (identity words may be given as a line "0"? -> handled below)
        w = []
        for tk in toks:
            try:
                v = int(tk)
            except Exception:
                fail("nonint token %r" % tk)
            if v == 0:
                continue  # 0 = no-op token (allows an explicit identity word)
            if v < -m or v > m:
                fail("token out of range %d" % v)
            w.append(v)
        words.append(w)

    if len(words) > k:
        fail("too many words %d > %d" % (len(words), k))
    for w in words:
        if len(w) > L:
            fail("word too long %d > %d" % (len(w), L))

    F = coverage(words)

    sc = min(1000.0, 100.0 * F / max(1e-9, float(B)))
    print("F=%d B=%d Ratio: %.6f" % (F, B, sc / 1000.0))

if __name__ == "__main__":
    main()
