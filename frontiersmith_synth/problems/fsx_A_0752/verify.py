#!/usr/bin/env python3
# Deterministic checker for "RLL-Constrained Huffman Code" (format C, minimize).
# CLI: python3 verify.py <in> <out> <ans>   (ans is ignored)
# Prints "... Ratio: <r>" with r in [0,1] on its own final line.
import sys

MAX_LEN = 96      # generous upper bound on any sane codeword length for our sizes
MAX_TOKEN_CHARS = set("01")


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def count_legal(L, d):
    """Number of binary strings of length L with no run of identical bits > d."""
    if L <= 0:
        return 1
    g = [0] * (d + 1)   # g[r] = count of legal strings (current length) with trailing run r
    g[1] = 2
    for _ in range(2, L + 1):
        s = sum(g[1:d + 1])
        newg = [0] * (d + 1)
        for r in range(1, d):
            newg[r + 1] += g[r]
        newg[1] += s
        g = newg
    return sum(g[1:d + 1])


def min_len_for(n, d):
    L = 0
    while count_legal(L, d) < n:
        L += 1
        if L > 4000:
            # should never happen for the sizes/constraints of this problem
            break
    return L


def max_run(s):
    best = 1
    cur = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def main():
    try:
        itoks = open(sys.argv[1]).read().split()
        n = int(itoks[0]); d = int(itoks[1])
        p = [int(x) for x in itoks[2:2 + n]]
        if len(p) != n or n < 1 or d < 1:
            fail("bad instance")
    except Exception:
        fail("bad instance")

    try:
        raw = open(sys.argv[2]).read()
    except Exception:
        fail("no output")

    otoks = raw.split()
    if len(otoks) != n:
        fail("expected %d codewords, got %d" % (n, len(otoks)))

    words = []
    for k, tok in enumerate(otoks):
        if len(tok) == 0 or len(tok) > MAX_LEN:
            fail("codeword %d bad length" % k)
        if any(c not in MAX_TOKEN_CHARS for c in tok):
            fail("codeword %d has non-binary characters" % k)
        if max_run(tok) > d:
            fail("codeword %d violates run-length limit (d=%d)" % (k, d))
        words.append(tok)

    # prefix-free check: sort, then only adjacent pairs can witness a violation.
    order = sorted(range(n), key=lambda i: words[i])
    sorted_words = [words[i] for i in order]
    for i in range(n - 1):
        a, b = sorted_words[i], sorted_words[i + 1]
        if b.startswith(a):
            fail("codeword set is not prefix-free (%r prefixes %r)" % (a, b))

    F = sum(p[i] * len(words[i]) for i in range(n))

    # internal trivial baseline: a single fixed length L for every symbol, where L
    # is the minimal length such that at least n legal (run<=d) codewords exist.
    L = min_len_for(n, d)
    B = L * sum(p)

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%d B=%d L=%d Ratio: %.6f" % (F, B, L, sc / 1000.0))


if __name__ == "__main__":
    main()
