#!/usr/bin/env python3
"""verify.py <in> <out> <ans>   (deterministic scorer for the ternary-tagging cap set)

Instance (<in>): the tag length n (ambient dimension of F_3^n).
Artifact (<out>): a line with the count m, then m ternary tag strings of length n.

Feasibility (STRICT): m matches, every tag has length n over {0,1,2}, all tags
distinct, and NO three distinct tags a,b,c are collinear (a+b+c == 0 in F_3^n,
i.e. coordinatewise sum divisible by 3). Any violation -> Ratio: 0.0.

Objective: F = |cap| = m  (MAXIMIZE).
Baseline: B = 2^n, the cardinality of the {0,1}^n cap (a provably-valid cap the
checker builds itself). Normalize: sc = min(1000, 100*F/B); Ratio = sc/1000.
So the {0,1}^n baseline scores 0.1 and a 10x-denser cap would cap at 1.0
(unreachable: the true maximum cap density is far below that for these n).
"""
import sys


def fail(reason):
    print("reject: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def read_int_token(path):
    with open(path) as f:
        for tok in f.read().split():
            try:
                return int(tok)
            except ValueError:
                continue
    return None


def main():
    if len(sys.argv) < 3:
        fail("usage")
    in_path, out_path = sys.argv[1], sys.argv[2]

    n = read_int_token(in_path)
    if n is None or n < 1 or n > 40:
        fail("bad instance n")

    # ---- parse participant artifact (bounded, strict) ----
    try:
        with open(out_path) as f:
            toks = f.read().split()
    except Exception:
        fail("cannot read output")
    if not toks:
        fail("empty output")

    # first token = claimed count m (must be a plain finite integer)
    m_tok = toks[0]
    try:
        m = int(m_tok)
    except ValueError:
        fail("count not an integer (got %r)" % m_tok[:32])
    # explicitly reject non-finite garbage that slipped through
    if m_tok.lower() in ("nan", "inf", "-inf", "+inf"):
        fail("non-finite count")

    space = 3 ** n
    if m < 1 or m > space:
        fail("count out of range")

    tags = toks[1:]
    if len(tags) < m:
        fail("declared %d tags but only %d present" % (m, len(tags)))
    tags = tags[:m]

    # validate each tag: length n, chars in {0,1,2}
    allowed = set("012")
    pts = []
    for tg in tags:
        if len(tg) != n:
            fail("tag has length %d != n=%d" % (len(tg), n))
        if any(c not in allowed for c in tg):
            fail("tag has illegal char")
        pts.append(tuple(ord(c) - 48 for c in tg))

    # distinctness
    keyset = set()
    pow3 = [3 ** k for k in range(n)]

    def key(p):
        s = 0
        for k in range(n):
            s += p[k] * pow3[k]
        return s

    for p in pts:
        kk = key(p)
        if kk in keyset:
            fail("duplicate tag")
        keyset.add(kk)

    # ---- cap validity: no collinear triple ----
    _cap_check(pts, n, keyset, pow3)

    # ---- score ----
    F = float(m)
    B = float(2 ** n)          # {0,1}^n cap cardinality (provably a cap)
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    ratio = sc / 1000.0
    print("valid cap: |S|=%d  baseline 2^%d=%d" % (m, n, 2 ** n))
    print("Ratio: %.6f" % ratio)
    sys.exit(0)


def _cap_check(pts, n, keyset, pow3):
    m = len(pts)
    try:
        import numpy as np
    except Exception:
        np = None
    if np is not None and m >= 2:
        P = np.array(pts, dtype=np.int16)                # (m,n)
        p3 = np.array(pow3, dtype=np.int64)
        keys = (P.astype(np.int64) * p3).sum(axis=1)     # (m,)
        order = np.argsort(keys)
        ksort = keys[order]
        for i in range(m - 1):
            J = P[i + 1:]                                 # (k,n)
            C = (-(P[i].astype(np.int64) + J.astype(np.int64))) % 3
            ck = (C * p3).sum(axis=1)                     # (k,)
            idx = np.searchsorted(ksort, ck)
            idx = np.clip(idx, 0, m - 1)
            if np.any(ksort[idx] == ck):
                fail("collinear triple (a+b+c=0) present")
        return
    # pure-python fallback
    for i in range(m):
        a = pts[i]
        for j in range(i + 1, m):
            b = pts[j]
            c = tuple((-(a[k] + b[k])) % 3 for k in range(n))
            if key_of(c, pow3, n) in keyset:
                fail("collinear triple (a+b+c=0) present")


def key_of(p, pow3, n):
    s = 0
    for k in range(n):
        s += p[k] * pow3[k]
    return s


if __name__ == "__main__":
    main()
