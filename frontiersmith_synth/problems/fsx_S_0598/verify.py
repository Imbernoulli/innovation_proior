#!/usr/bin/env python3
"""Deterministic checker for fsx_S_0598.

Usage: python3 verify.py <in> <out> <ans>   (ans ignored)

Reads the string-rewriting system from <in>, the participant's set of words from
<out>, computes the true rewriting congruence over all words of length in [1,L]
(union-find over the one-step rewrite graph -- the exact bounded Thue congruence),
and scores the number of DISTINCT congruence classes the submitted words hit.

Feasibility (any violation -> Ratio 0.0): at most Nmax words, each a non-empty
string over the digit alphabet {0..k-1} of length <= L. Non-digit / out-of-range /
over-length / too-many tokens are all rejected (this also rejects nan/inf floods).
"""
import sys, itertools

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def all_words(k, L):
    al = [str(i) for i in range(k)]
    W = []
    for ln in range(1, L + 1):
        for t in itertools.product(al, repeat=ln):
            W.append("".join(t))
    return W

class DSU:
    def __init__(s, n): s.p = list(range(n))
    def find(s, x):
        while s.p[x] != x:
            s.p[x] = s.p[s.p[x]]; x = s.p[x]
        return x
    def union(s, a, b):
        ra, rb = s.find(a), s.find(b)
        if ra != rb: s.p[ra] = rb

def apply_all(w, l, r):
    out = []; i = 0
    while True:
        j = w.find(l, i)
        if j < 0: break
        out.append(w[:j] + r + w[j + len(l):]); i = j + 1
    return out

def main():
    try:
        toks = open(sys.argv[1]).read().split()
        it = iter(toks)
        k = int(next(it)); L = int(next(it)); Nmax = int(next(it))
        m = int(next(it))
        rules = []
        for _ in range(m):
            l = next(it); r = next(it)
            rules.append((l, r))
    except Exception:
        fail("bad input")

    if not (1 <= k <= 9 and 1 <= L <= 8):
        fail("bad params")
    alphabet = set(str(i) for i in range(k))

    # ---- true congruence over U = words of length in [1,L] ----
    W = all_words(k, L)
    idx = {w: i for i, w in enumerate(W)}
    dsu = DSU(len(W))
    for w in W:
        for (l, r) in rules:
            for w2 in apply_all(w, l, r):
                if 1 <= len(w2) <= L and w2 in idx:
                    dsu.union(idx[w], idx[w2])
    root = {w: dsu.find(idx[w]) for w in W}
    Ktot = len(set(root.values()))
    B = max(1, round(Ktot / 10.0))               # internal baseline density

    # ---- parse + strictly validate participant output ----
    try:
        out_tokens = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")
    if len(out_tokens) == 0:
        fail("empty output")
    if len(out_tokens) > Nmax:
        fail("more than Nmax words (%d > %d)" % (len(out_tokens), Nmax))

    hit = set()
    for tok in out_tokens:
        if len(tok) < 1 or len(tok) > L:
            fail("word of illegal length: %r" % tok[:20])
        for ch in tok:
            if ch not in alphabet:
                fail("illegal symbol in word: %r" % tok[:20])
        # every legal digit word of length <= L is in U
        hit.add(root[tok])

    F = len(hit)
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    ratio = max(0.0, sc / 1000.0)
    print("classes_hit=%d total_classes=%d baseline=%d Ratio: %.6f" % (F, Ktot, B, ratio))

if __name__ == "__main__":
    main()
