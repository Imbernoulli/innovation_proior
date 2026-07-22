import sys, random, string

WORD_ALPHA = string.ascii_lowercase[:12]     # a..l  -- letters that build "words"
NOISE_ALPHA = string.ascii_lowercase[12:]    # m..z  -- letters used only as filler
D = 12                                       # number of corpora, fixed by the theme
PRIV_N = 6                                   # private (domain-only) words per corpus
POOL_N = 8                                   # size of the global cross-domain pool
MINLEN, MAXLEN = 4, 9                        # legal dictionary-entry length range
PRIV_WEIGHTS = [30, 18, 11, 7, 5, 3]
SHARED_WEIGHT = 10
NOISE_GAP_P = 0.35                           # prob. of a filler run between two words

# difficulty ladder: (big_len, small_len, K, M, shared_prob)
#   corpora 0,1,2 use big_len; corpora 3..11 use small_len.
#   K = total dictionary-character budget, M = max number of entries.
PROFILE = {
    1:  (700,  700,  240, 40, 0.40),
    2:  (900,  900,  260, 40, 0.40),
    3:  (700,  700,  260, 40, 0.35),
    4:  (900,  900,  260, 40, 0.35),
    5:  (1000, 400,  230, 40, 0.38),
    6:  (1200, 450,  240, 40, 0.38),
    7:  (2600, 340,  190, 40, 0.30),
    8:  (3000, 360,  180, 40, 0.28),
    9:  (3600, 420,  250, 40, 0.34),
    10: (4200, 440,  200, 40, 0.28),
}


def rand_word(rng, lo, hi):
    L = rng.randint(lo, hi)
    return "".join(rng.choice(WORD_ALPHA) for _ in range(L))


def build_doc(rng, words, weights, L, noise_gap_p):
    # Words are always emitted INTACT (never split mid-word); short filler runs
    # from a disjoint alphabet are spliced only BETWEEN words. This keeps every
    # exact word match uncorrupted while still leaving an irreducible fraction
    # of literal-only characters (so no dictionary can compress a corpus to
    # nothing -- the score never saturates).
    out = []
    total = 0
    while total < L:
        w = rng.choices(words, weights=weights)[0]
        if total + len(w) > L:
            w = w[: L - total]
        if not w:
            break
        out.append(w)
        total += len(w)
        if total >= L:
            break
        if rng.random() < noise_gap_p:
            remaining = L - total
            nl = min(remaining, rng.randint(1, 3))
            out.append("".join(rng.choice(NOISE_ALPHA) for _ in range(nl)))
            total += nl
    return "".join(out)[:L]


def gen_instance(test_id):
    rng = random.Random(31000 + 97 * test_id)
    big_len, small_len, K, M, shared_p = PROFILE[test_id]

    pool = [rand_word(rng, 5, 7) for _ in range(POOL_N)]
    docs = []
    for d in range(D):
        priv = [rand_word(rng, 5, 8) for _ in range(PRIV_N)]
        shared_here = [w for w in pool if rng.random() < shared_p]
        words = priv + shared_here
        weights = PRIV_WEIGHTS[:PRIV_N] + [SHARED_WEIGHT] * len(shared_here)
        L = big_len if d < 3 else small_len
        docs.append(build_doc(rng, words, weights, L, NOISE_GAP_P))
    return docs, K, M, MINLEN, MAXLEN


def main():
    test_id = int(sys.argv[1])
    docs, K, M, minlen, maxlen = gen_instance(test_id)
    out = [f"{len(docs)} {K} {M} {minlen} {maxlen}"]
    out.extend(docs)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
