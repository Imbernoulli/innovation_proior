#!/usr/bin/env python3
# Deterministic checker for "Long Square-Free Word Hitting a Target Letter Mix" (format C).
# CLI: python3 verify.py <in> <out> <ans>   (ans is ignored)
# Prints "... Ratio: <r>" with r in [0,1] on its own final line, exits 0.
import sys

ALPHABET = 4
WU = 0.25   # weight on the per-letter frequency term
WD = 0.75   # weight on the letter-transition term


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


# ---- the square-free morphism scaffold (0 -> 0 1 2, 1 -> 0 2, 2 -> 1), fixed point
# from letter 0.  Proven-by-construction square-free; used ONLY to build the checker's
# own target-blind reference word (the baseline B).
def _apply_morph(word):
    m = {0: (0, 1, 2), 1: (0, 2), 2: (1,)}
    out = []
    for c in word:
        out.extend(m[c])
    return out


def scaffold3(n):
    w = [0]
    while len(w) < n:
        w = _apply_morph(w)
    return w[:n]


def build_uniform_baseline(L):
    # Target-blind reference construction: insert separator letter 0 after every
    # 3rd filler drawn from the square-free scaffold (relabelled onto {1,2,3}).
    # This reaches exactly length L and realizes a flat 0.25/0.25/0.25/0.25 mix.
    raw = scaffold3(L + 10)
    remap = {0: 1, 1: 2, 2: 3}
    out = []
    cnt = 0
    fi = 0
    while len(out) < L:
        if cnt % 3 == 0:
            out.append(0)
            if len(out) >= L:
                break
        out.append(remap[raw[fi]])
        fi += 1
        cnt += 1
    return out[:L]


def is_square_free(word):
    """O(N^2) full validity scan (character-level, no slicing): for every period p,
    walk once and flag a run of >= p consecutive positions with word[i]==word[i+p],
    which is exactly a length-p block equal to the block p later (a square)."""
    n = len(word)
    for p in range(1, n // 2 + 1):
        run = 0
        for i in range(n - p):
            if word[i] == word[i + p]:
                run += 1
                if run >= p:
                    return False
            else:
                run = 0
    return True


def tv_distance(p, w):
    return 0.5 * sum(abs(a - b) for a, b in zip(p, w))


def quality(word, L, w, succ):
    n = len(word)
    if n == 0:
        return 0.0
    length_score = n / L
    counts = [0] * ALPHABET
    for c in word:
        counts[c] += 1
    freq = [c / n for c in counts]
    unigram_score = 1.0 - tv_distance(freq, w)
    if n >= 2:
        matches = sum(1 for i in range(n - 1) if word[i + 1] == succ[word[i]])
        succ_score = matches / (n - 1)
    else:
        succ_score = 0.0
    return length_score * (WU * unigram_score + WD * succ_score)


def main():
    try:
        itoks = open(sys.argv[1]).read().split()
        L = int(itoks[0])
        w = [int(itoks[1 + i]) / 10000.0 for i in range(4)]
        succ = [int(itoks[5 + i]) for i in range(4)]
    except Exception:
        fail("bad instance")

    try:
        otext = open(sys.argv[2]).read()
    except Exception:
        fail("no output")

    otoks = otext.split()
    if not otoks:
        fail("empty output")

    try:
        N = int(otoks[0])
    except Exception:
        fail("bad N")

    if N < 0 or N > L:
        fail("N out of range")

    if N == 0:
        word_str = ""
    else:
        if len(otoks) < 2:
            fail("missing word")
        word_str = otoks[1]
        if len(word_str) != N:
            fail("word length != N")

    word = []
    for ch in word_str:
        if ch not in "0123":
            fail("non-alphabet char %r" % ch)
        word.append(int(ch))

    if not is_square_free(word):
        fail("word contains a square factor")

    F = quality(word, L, w, succ)

    base = build_uniform_baseline(L)
    B = quality(base, L, w, succ)

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    ratio = sc / 1000.0
    print("N=%d F=%.6f B=%.6f Ratio: %.6f" % (N, F, B, ratio))


if __name__ == "__main__":
    main()
