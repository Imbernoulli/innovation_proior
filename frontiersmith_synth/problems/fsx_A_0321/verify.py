#!/usr/bin/env python3
# verify.py <in> <out> <ans>   -- deterministic scorer for the vineyard problem.
#
# Reads the instance from <in> and the participant's chosen emitter plots from
# <out>.  Strictly validates the "irrigation cap-set" feasibility, then scores
#   F = total water yield of the chosen plots
# against an internal trivial baseline B (the {0,1}^n structural cap set).
#   sc = min(1000, 100 * F / max(1e-9, B));  Ratio = sc/1000.
# Trivial construction -> ~0.10; a genuinely better ordering climbs higher.
import sys


def read_instance(path):
    toks = open(path).read().split()
    it = iter(toks)
    n = int(next(it))
    m = int(next(it))
    blocked = set()
    for _ in range(m):
        blocked.add(next(it))
    N = 3 ** n
    weights = []
    for _ in range(N):
        weights.append(int(next(it)))
    return n, blocked, weights


def idx_of(s, n):
    v = 0
    for c in s:
        v = v * 3 + (ord(c) - 48)
    return v


def str_of(idx, n):
    d = []
    for _ in range(n):
        d.append(idx % 3)
        idx //= 3
    return "".join(str(x) for x in reversed(d))


def third(x, y, n):
    return "".join(str((-(int(a) + int(b))) % 3) for a, b in zip(x, y))


def fail(reason):
    print("INFEASIBLE (%s)  Ratio: 0.000000" % reason)
    sys.exit(0)


def main():
    inf, outf = sys.argv[1], sys.argv[2]
    n, blocked, weights = read_instance(inf)
    N = 3 ** n

    # ---- parse participant output strictly ----
    raw = open(outf).read().split()
    # bound: no more plots than the whole vineyard
    if len(raw) > N:
        fail("too many plots")
    S = []
    seen = set()
    validchars = set("012")
    for tok in raw:
        if len(tok) != n or any(c not in validchars for c in tok):
            fail("malformed plot token")
        if tok in seen:
            fail("duplicate plot")
        if tok in blocked:
            fail("rocky (blocked) plot used")
        seen.add(tok)
        S.append(tok)

    # ---- feasibility: no three chosen plots collinear in F_3^n ----
    # a line is {x,y,z} distinct with x+y+z == 0 (mod 3) componentwise.
    Sset = seen
    k = len(S)
    for i in range(k):
        xi = S[i]
        for j in range(i + 1, k):
            z = third(xi, S[j], n)
            if z in Sset and z != xi and z != S[j]:
                fail("three collinear emitters")

    # ---- objective ----
    F = 0
    for tok in S:
        F += weights[idx_of(tok, n)]

    # ---- internal trivial baseline B: {0,1}^n structural cap set ----
    B = 0
    for mask in range(1 << n):
        s = "".join("1" if (mask >> (n - 1 - b)) & 1 else "0" for b in range(n))
        if s in blocked:
            continue
        B += weights[idx_of(s, n)]

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("chosen=%d yield=%d baseline=%d Ratio: %.6f" % (k, F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
