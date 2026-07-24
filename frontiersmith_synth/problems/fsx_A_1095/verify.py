#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for fsx_A_1095
(The Sighing Caravanserai: buoyancy-vent circulation loop).

Replays the submitted set of cut vents through an exact rational
stack-effect network model (warmth -> pressure -> steady flows -> upwind
freshness with a 7/8 credit per bay-to-bay passage) and scores the total
fresh-air throughflow F delivered to occupied bays.

Internal baseline B = F achieved by cutting only vents 1 and 2 (the
great-hall floor mouth and its roof lantern, always the first two listed),
so the minimal hall loop scores exactly 0.1 and 10x better caps at 1.0.
All arithmetic is exact (fractions.Fraction); no randomness, no timing.
"""
import sys
from fractions import Fraction

LAMBDA = Fraction(7, 8)


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def solve_linear(A, b):
    """Exact Gaussian elimination with deterministic partial pivoting
    (largest |pivot|, lowest row index wins ties). None if singular."""
    n = len(A)
    Mx = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = -1
        best = None
        for r in range(col, n):
            v = Mx[r][col]
            if v != 0:
                a = abs(v)
                if best is None or a > best:
                    best = a
                    piv = r
        if piv < 0:
            return None
        if piv != col:
            Mx[col], Mx[piv] = Mx[piv], Mx[col]
        pv = Mx[col][col]
        for r in range(col + 1, n):
            f = Mx[r][col]
            if f == 0:
                continue
            qq = f / pv
            for c in range(col, n + 1):
                if Mx[col][c] != 0:
                    Mx[r][c] -= qq * Mx[col][c]
    x = [Fraction(0)] * n
    for r in range(n - 1, -1, -1):
        s = Mx[r][n]
        for c in range(r + 1, n):
            if Mx[r][c] != 0:
                s -= Mx[r][c] * x[c]
        if Mx[r][r] == 0:
            return None
        x[r] = s / Mx[r][r]
    return x


def evaluate(inst, cut):
    """Exact fresh-air throughflow F for the set `cut` of 0-based vent indices."""
    W = inst['W']
    k0 = inst['k0']
    q = inst['q']
    occ = inst['occ']
    doorways = inst['doorways']
    cands = inst['cands']
    openj = list(cut)
    if not openj:
        return Fraction(0)
    G = [int(k0)] * W
    for (c, d, z, g) in doorways:
        G[c] += g
        G[d] += g
    for j in openj:
        c, z, g = cands[j]
        G[c] += g
    t = [Fraction(q[c], G[c]) for c in range(W)]
    # pressure references: A pi = b
    A = [[Fraction(0)] * W for _ in range(W)]
    b = [Fraction(0)] * W
    for (c, d, z, g) in doorways:
        A[c][c] += g
        A[d][d] += g
        A[c][d] -= g
        A[d][c] -= g
        rhs = g * (t[c] - t[d]) * z
        b[c] -= rhs
        b[d] += rhs
    for j in openj:
        c, z, g = cands[j]
        A[c][c] += g
        b[c] -= g * t[c] * z
    pi = solve_linear(A, b)
    if pi is None:
        return Fraction(0)
    dflow = [(c, d, g * (pi[c] - pi[d] + (t[c] - t[d]) * z))
             for (c, d, z, g) in doorways]
    oflow = [(cands[j][0],
              cands[j][2] * (pi[cands[j][0]] + t[cands[j][0]] * cands[j][1]))
             for j in openj]
    # freshness transport (upwind, exterior freshness 1, decay LAMBDA per
    # bay-to-bay passage). Row c: In_c*phi_c - LAMBDA*sum_{d->c} f*phi_d = src_c
    In = [Fraction(0)] * W
    A2 = [[Fraction(0)] * W for _ in range(W)]
    b2 = [Fraction(0)] * W
    for (c, d, f) in dflow:
        if f > 0:
            In[d] += f
            A2[d][c] -= LAMBDA * f
        elif f < 0:
            In[c] += -f
            A2[c][d] -= LAMBDA * (-f)
    for (c, f) in oflow:
        if f < 0:
            In[c] += -f
            b2[c] += -f
    for c in range(W):
        if In[c] == 0:
            A2[c][c] = Fraction(1)   # phi_c = 0 (b2[c] already 0)
        else:
            A2[c][c] += In[c]
    phi = solve_linear(A2, b2)
    if phi is None:
        return Fraction(0)
    F = Fraction(0)
    for c in range(W):
        if occ[c]:
            F += phi[c] * In[c]
    return F


def main():
    try:
        inp = open(sys.argv[1]).read().split()
        out = open(sys.argv[2]).read().split()
    except Exception:
        fail("io")

    # ---- parse instance ----
    try:
        it = iter(inp)
        W = int(next(it))
        B = int(next(it))
        k0 = int(next(it))
        h = [int(next(it)) for _ in range(W)]
        q = [int(next(it)) for _ in range(W)]
        occ = [int(next(it)) for _ in range(W)]
        D = int(next(it))
        doorways = []
        for _ in range(D):
            c = int(next(it)) - 1
            d = int(next(it)) - 1
            z = int(next(it))
            g = int(next(it))
            doorways.append((c, d, z, g))
        M = int(next(it))
        cands = []
        for _ in range(M):
            c = int(next(it)) - 1
            z = int(next(it))
            g = int(next(it))
            cands.append((c, z, g))
    except Exception:
        fail("bad input")
    if not (1 <= W <= 16 and 1 <= k0 <= 64 and 0 <= B <= M <= 256):
        fail("bad input bounds")
    if any(x <= 0 or x > 1000 for x in h):
        fail("bad heights")
    if any(x < 0 or x > 10 ** 6 for x in q):
        fail("bad heat")
    if any(x not in (0, 1) for x in occ):
        fail("bad occupancy")
    if any(c < 0 or c >= W or d < 0 or d >= W or c == d or z < 0 or g <= 0
           for (c, d, z, g) in doorways):
        fail("bad doorway")
    if any(c < 0 or c >= W or z < 0 or z > h[c] or g <= 0 for (c, z, g) in cands):
        fail("bad vent")
    if M < 2:
        fail("degenerate instance")

    inst = dict(W=W, B=B, k0=k0, h=h, q=q, occ=occ,
                doorways=doorways, cands=cands)

    F0 = evaluate(inst, [0, 1])          # internal baseline: vents 1 and 2 only
    if F0 <= 0:
        fail("degenerate instance")

    # ---- parse participant output ----
    if not out:
        fail("empty output")
    try:
        k = int(out[0])
    except Exception:
        fail("bad vent count")
    if k < 0 or k > B:
        fail("vent count out of range")
    if len(out) != 1 + k:
        fail("token count mismatch")
    idxs = []
    seen = set()
    for tok in out[1:]:
        try:
            j = int(tok)
        except Exception:
            fail("bad vent index %r" % tok)
        if j < 1 or j > M:
            fail("vent index %d out of range" % j)
        if j in seen:
            fail("duplicate vent index %d" % j)
        seen.add(j)
        idxs.append(j - 1)

    F = evaluate(inst, idxs)
    sc = min(Fraction(1000), Fraction(100) * F / F0)
    print("F=%.9g B=%.9g Ratio: %.6f" % (float(F), float(F0), float(sc) / 1000.0))


if __name__ == "__main__":
    main()
