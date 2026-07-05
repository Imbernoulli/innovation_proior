#!/usr/bin/env python3
"""verify.py <in> <out> <ans>  (ans ignored)

Deterministic scorer for the orbital-debris cleanup (cap-set-avoiding-forbidden) problem.

Feasibility (any violation -> Ratio: 0.0):
  * output tokens are integers only (rejects nan/inf/garbage), count a multiple of n,
    and not absurdly large;
  * every signature has n digits in {0,1,2}, no duplicates, none protected;
  * NO resonance triple: no three distinct collinear signatures (a+b+c == 0 mod 3).

Score: F = |S|; B = size of the internal reference manifest (first-coordinate = 0 and all
other coordinates in {0,1}, minus protected). maximization ratio, capped:
  sc = min(1000, 100*F/max(1e-9,B));  print Ratio sc/1000.
"""
import sys


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    n = int(next(it))
    m = int(next(it))
    pw = [3 ** i for i in range(n)]
    forb = set()
    for _ in range(m):
        code = 0
        for i in range(n):
            d = int(next(it))
            code += d * pw[i]
        forb.add(code)
    return n, m, forb, pw


def baseline_size(n, forb, pw):
    """Reference manifest: signatures with d[0]==0 and d[1..n-1] in {0,1}, minus protected.
    This is a genuine cap (a subset of the {0,1}-only cap restricted to one hyperplane)."""
    cnt = 0
    # enumerate the (n-1)-bit subcube, d[0] fixed to 0
    for mask in range(1 << (n - 1)):
        code = 0
        for j in range(n - 1):
            if (mask >> j) & 1:
                code += pw[j + 1]  # digit j+1 = 1
        if code not in forb:
            cnt += 1
    return cnt


def third(a, b, n, pw):
    """int code of the unique c with a+b+c == 0 (mod 3) coordinatewise."""
    c = 0
    aa, bb = a, b
    for i in range(n):
        da = aa % 3
        db = bb % 3
        aa //= 3
        bb //= 3
        c += ((-da - db) % 3) * pw[i]
    return c


def fail(reason):
    print("infeasible: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def main():
    inf, outf = sys.argv[1], sys.argv[2]
    n, m, forb, pw = read_instance(inf)
    N = 3 ** n

    with open(outf) as f:
        raw = f.read().split()

    # bounded / integer-only parse (rejects nan, inf, floats, junk)
    if len(raw) > n * N + 10:
        fail("output too large")
    vals = []
    for t in raw:
        try:
            x = int(t)
        except ValueError:
            fail("non-integer token %r" % t)
        vals.append(x)
    if len(vals) % n != 0:
        fail("token count %d not a multiple of n=%d" % (len(vals), n))

    codes = []
    seen = set()
    for k in range(0, len(vals), n):
        code = 0
        for i in range(n):
            d = vals[k + i]
            if d < 0 or d > 2:
                fail("digit %d out of range" % d)
            code += d * pw[i]
        if code in seen:
            fail("duplicate signature")
        seen.add(code)
        if code in forb:
            fail("protected signature selected")
        codes.append(code)

    S = set(codes)
    # resonance-triple (cap) check: for every unordered pair, the third collinear point
    # must be absent.  O(|S|^2).
    L = codes
    for i in range(len(L)):
        a = L[i]
        for j in range(i + 1, len(L)):
            b = L[j]
            c = third(a, b, n, pw)
            if c in S and c != a and c != b:
                fail("resonance triple detected")

    F = len(S)
    B = baseline_size(n, forb, pw)
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%d B=%d n=%d" % (F, B, n))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
