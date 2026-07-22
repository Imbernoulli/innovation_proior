#!/usr/bin/env python3
"""Deterministic scorer for the monastery-insulation problem (format C).

Reads the instance and the participant's layer allocation, checks feasibility
strictly, solves the steady-state thermal network EXACTLY over the rationals,
computes total heat loss to outside H (minimize), and normalizes against the
do-nothing baseline B (all-zero allocation).

    minimization:  sc = min(1000, 100 * B / H);  Ratio = sc/1000
"""
import sys
from fractions import Fraction as Fr

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def read_instance(path):
    toks = open(path).read().split()
    it = iter(toks)
    N = int(next(it)); M = int(next(it)); K = int(next(it))
    Tstar = Fr(next(it)); Tout = Fr(next(it))
    occ = [int(next(it)) for _ in range(N)]
    walls = []
    for _ in range(M):
        a = int(next(it)); b = int(next(it))
        R0 = Fr(next(it)); rho = Fr(next(it)); kmax = int(next(it))
        walls.append((a, b, R0, rho, kmax))
    return N, M, K, Tstar, Tout, occ, walls

def heat_loss(N, Tstar, Tout, occ, walls, klist):
    """Exact steady-state loss to outside for a given (already validated) klist."""
    # node potential: occupied rooms fixed Tstar, outside(0) fixed Tout, others unknown
    fixedT = {0: Tout}
    for rid in range(1, N + 1):
        if occ[rid - 1]:
            fixedT[rid] = Tstar
    # index the floating rooms
    floaters = [rid for rid in range(1, N + 1) if rid not in fixedT]
    idx = {rid: p for p, rid in enumerate(floaters)}
    nf = len(floaters)

    g = []
    for (a, b, R0, rho, kmax), k in zip(walls, klist):
        R = R0 + Fr(k) * rho
        g.append(Fr(1) / R)

    # assemble A T = bvec over floating nodes
    A = [[Fr(0)] * nf for _ in range(nf)]
    bvec = [Fr(0)] * nf
    for (a, b, R0, rho, kmax), ge in zip(walls, g):
        for (i, j) in ((a, b), (b, a)):
            if i in idx:                      # equation for floating node i
                p = idx[i]
                A[p][p] += ge
                if j in idx:
                    A[p][idx[j]] -= ge
                else:
                    bvec[p] += ge * fixedT[j]

    T = solve(A, bvec, nf)
    Tof = dict(fixedT)
    for rid in floaters:
        Tof[rid] = T[idx[rid]]

    # total loss to outside = sum over walls touching node 0
    H = Fr(0)
    for (a, b, R0, rho, kmax), ge in zip(walls, g):
        if a == 0 or b == 0:
            room = b if a == 0 else a
            H += ge * (Tof[room] - Tout)
    return H

def solve(A, bvec, n):
    if n == 0:
        return []
    # Fraction Gaussian elimination with partial pivot (deterministic)
    M = [row[:] + [bvec[r]] for r, row in enumerate(A)]
    for c in range(n):
        piv = -1
        for r in range(c, n):
            if M[r][c] != 0:
                piv = r
                break
        if piv < 0:
            continue
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        M[c] = [x / pv for x in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [xr - f * xc for xr, xc in zip(M[r], M[c])]
    return [M[r][n] for r in range(n)]

def main():
    N, M, K, Tstar, Tout, occ, walls = read_instance(sys.argv[1])

    raw = open(sys.argv[2]).read().split()
    if len(raw) != M:
        fail("expected %d integers, got %d" % (M, len(raw)))
    klist = []
    for tok in raw:
        # reject floats / nan / inf / garbage: strict integer only
        if not (tok.lstrip("+-").isdigit()):
            fail("non-integer token %r" % tok)
        klist.append(int(tok))

    total = 0
    for k, (a, b, R0, rho, kmax) in zip(klist, walls):
        if k < 0 or k > kmax:
            fail("layer out of [0,%d]" % kmax)
        total += k
    if total > K:
        fail("budget exceeded (%d > %d)" % (total, K))

    B = heat_loss(N, Tstar, Tout, occ, walls, [0] * M)   # do-nothing baseline
    H = heat_loss(N, Tstar, Tout, occ, walls, klist)

    Bf = float(B); Hf = float(H)
    sc = min(1000.0, 100.0 * Bf / max(1e-9, Hf))
    print("loss=%.6f baseline=%.6f Ratio: %.6f" % (Hf, Bf, sc / 1000.0))

if __name__ == "__main__":
    main()
