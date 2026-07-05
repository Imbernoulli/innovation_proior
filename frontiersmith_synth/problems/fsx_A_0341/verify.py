#!/usr/bin/env python3
"""
Deterministic checker for the deep-sea cable interference-free backbone problem.

Usage: verify.py <in> <out> <ans>     (ans is an ignored placeholder)

Reads the instance (n, weights) from <in> and the participant backbone from
<out>.  A submission is a set of routes (vectors in F_3^n).  It is FEASIBLE iff
  * every route is a valid length-n trit string,
  * all routes are distinct,
  * NO three distinct routes a,b,c satisfy a+b+c == 0 (mod 3) coordinate-wise
    (no interfering / collinear triple)  -- i.e. the set is a cap set.
The objective F is the total throughput (sum of weights) of the deployed routes.

Scoring (maximization): the checker builds its own trivial feasible baseline B
(weight-BLIND greedy in canonical order, which just deploys the 0/1 sub-cube),
then reports  sc = min(1000, 100*F/B);  Ratio = sc/1000.
Any feasibility violation -> Ratio: 0.0.
"""
import sys


def read_instance(path):
    with open(path, "r") as f:
        toks = f.read().split()
    n = int(toks[0])
    N = 3 ** n
    w = list(map(int, toks[1:1 + N]))
    if len(w) != N:
        # malformed instance -- should never happen from our gen
        raise ValueError("bad instance")
    return n, w


def idx_of(coords, n):
    r = 0
    for c in coords:
        r = r * 3 + c
    return r


def coords_of(i, n):
    c = [0] * n
    for k in range(n - 1, -1, -1):
        c[k] = i % 3
        i //= 3
    return c


def blind_baseline(n, w):
    """Weight-blind greedy cap in canonical (base-3) order -> the trivial ref B."""
    N = 3 ** n
    coords = [coords_of(i, n) for i in range(N)]
    forbidden = set()
    chosen = set()
    tot = 0
    for i in range(N):
        if i in forbidden or i in chosen:
            continue
        ci = coords[i]
        for q in chosen:
            cq = coords[q]
            r = 0
            for k in range(n):
                r = r * 3 + ((-(ci[k] + cq[k])) % 3)
            forbidden.add(r)
        chosen.add(i)
        tot += w[i]
    return tot


def fail(reason):
    print("reason: " + reason)
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    inf, outf = sys.argv[1], sys.argv[2]
    n, w = read_instance(inf)
    N = 3 ** n

    # ---- parse participant output strictly ----
    try:
        with open(outf, "r") as f:
            raw = f.read()
    except Exception:
        fail("no output")
    toks = raw.split()
    if not toks:
        fail("empty output")
    # first token = claimed count k
    try:
        k = int(toks[0])
    except Exception:
        fail("count not an integer")
    if k < 0 or k > N:
        fail("count out of range")
    body = toks[1:]
    if len(body) != k:
        fail("declared count %d != %d route tokens" % (k, len(body)))

    # ---- validate each route -> index ----
    sel = []
    seen = set()
    for s in body:
        if len(s) != n:
            fail("route '%s' has wrong length" % s)
        idx = 0
        for ch in s:
            if ch not in "012":
                fail("route '%s' has a non-trit char" % s)
            idx = idx * 3 + (ord(ch) - 48)
        if idx in seen:
            fail("duplicate route '%s'" % s)
        seen.add(idx)
        sel.append(idx)

    # ---- feasibility: cap-set (no interfering triple), streaming forbidden set ----
    coords = {i: coords_of(i, n) for i in sel}
    forbidden = set()
    added = []
    for i in sel:
        if i in forbidden:
            fail("interfering triple: route completes a collinear triple")
        ci = coords[i]
        for q in added:
            cq = coords[q]
            r = 0
            for x in range(n):
                r = r * 3 + ((-(ci[x] + cq[x])) % 3)
            forbidden.add(r)
        added.append(i)

    # ---- objective ----
    F = sum(w[i] for i in sel)

    # ---- baseline + normalized score ----
    B = blind_baseline(n, w)
    sc = min(1000.0, 100.0 * F / max(1e-9, float(B)))
    print("n=%d selected=%d throughput=%d baseline=%d" % (n, len(sel), F, B))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
