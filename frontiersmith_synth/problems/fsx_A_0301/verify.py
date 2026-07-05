#!/usr/bin/env python3
# verify.py <in> <out> <ans>  (ans ignored) -- deterministic scorer for the
# weighted line-free shelf activation (cap set) problem. Prints "Ratio: <x>".
import sys, math

def read_ints(path):
    with open(path) as f:
        return f.read().split()

def fail(reason):
    print("infeasible: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)

def main():
    inf, outf = sys.argv[1], sys.argv[2]

    # ---- parse instance ----
    itoks = read_ints(inf)
    n = int(itoks[0])
    N = 3 ** n
    W = [int(x) for x in itoks[1:1 + N]]
    if len(W) != N:
        fail("bad instance")

    pow3 = [3 ** k for k in range(n)]
    def digits(i):
        d = []
        x = i
        for _ in range(n):
            d.append(x % 3); x //= 3
        return d

    # ---- internal baseline B: binary sub-cube {0,1}^n (always line-free) ----
    B = 0
    for i in range(N):
        if all(x != 2 for x in digits(i)):
            B += W[i]
    if B <= 0:
        B = 1

    # ---- parse participant output (bounded, strict) ----
    otoks = read_ints(outf)
    if not otoks:
        fail("empty output")
    # reject any non-finite / non-integer token up-front
    for tk in otoks:
        try:
            fv = float(tk)
        except ValueError:
            fail("non-numeric token")
        if not math.isfinite(fv):
            fail("non-finite token")
    try:
        k = int(otoks[0])
    except ValueError:
        fail("bad count")
    if k < 0 or k > N:
        fail("count out of range")
    if len(otoks) - 1 < k:
        fail("fewer indices than declared count")
    idxs = []
    for tk in otoks[1:1 + k]:
        try:
            v = int(tk)
        except ValueError:
            fail("non-integer index")
        idxs.append(v)

    seen = set()
    for v in idxs:
        if v < 0 or v >= N:
            fail("index out of range")
        if v in seen:
            fail("duplicate index")
        seen.add(v)

    # ---- feasibility: no hazard line (cap set) ----
    # a,b distinct in S ; c = completion(a,b) ; if c in S -> line present.
    D = {v: digits(v) for v in seen}
    S = idxs
    Sset = seen
    for ia in range(len(S)):
        a = S[ia]; da = D[a]
        for ib in range(ia + 1, len(S)):
            b = S[ib]; db = D[b]
            c = 0
            for kk in range(n):
                c += ((3 - (da[kk] + db[kk]) % 3) % 3) * pow3[kk]
            if c in Sset and c != a and c != b:
                fail("hazard line present: %d %d %d" % (a, b, c))

    # ---- objective + score ----
    F = sum(W[v] for v in S)
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%d B=%d k=%d" % (F, B, k))
    print("Ratio: %.6f" % (sc / 1000.0))

if __name__ == "__main__":
    main()
