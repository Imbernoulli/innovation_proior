#!/usr/bin/env python3
"""
counter.py <in> <out> <ans>   --  deterministic scorer for fsx_A_0348 (format D, eval_form=flops).

Instance (in) : a 3-D integer tensor  T  of size I x J x K.
Artifact (out): a rank-R decomposition
        line 1 : R
        next R lines : each  I + J + K  rational tokens =  a_r (len I) , b_r (len J) , c_r (len K)
    representing  T_hat[i][j][k] = sum_{r} a_r[i] * b_r[j] * c_r[k].

Scoring (minimise R = number of scalar multiplications / rank-1 terms):
    1. Feasibility: parse strictly, reject non-finite / malformed / out-of-range R,
       then verify EXACT reconstruction  T_hat == T  over the rationals.
       Any violation  ->  Ratio: 0.0
    2. Objective  F = R.   Internal baseline  B = I*J   (the mode-3 "fiber" decomposition
       that the checker can always build itself).
    3. Ratio = min(1.0, 0.1 * B / F).   (trivial fiber decomposition R=I*J -> 0.1)

Exact rational arithmetic only; fully deterministic.
"""
import sys
from fractions import Fraction


def fail(reason):
    print("reject: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def read_ints(path):
    with open(path) as f:
        return f.read().split()


def parse_frac(tok):
    # Fraction() raises ValueError on 'nan'/'inf'/garbage and on floats like '1e5' -> caught by caller.
    return Fraction(tok)


def main():
    if len(sys.argv) < 3:
        fail("bad args")
    in_toks = read_ints(sys.argv[1])
    try:
        idx = 0
        I = int(in_toks[idx]); idx += 1
        J = int(in_toks[idx]); idx += 1
        K = int(in_toks[idx]); idx += 1
    except (ValueError, IndexError):
        fail("bad instance header")
    T = [[[0] * K for _ in range(J)] for _ in range(I)]
    try:
        for i in range(I):
            for j in range(J):
                for k in range(K):
                    T[i][j][k] = int(in_toks[idx]); idx += 1
    except (ValueError, IndexError):
        fail("bad instance body")

    # ---- participant artifact ----
    with open(sys.argv[2]) as f:
        toks = f.read().split()
    if not toks:
        fail("empty output")
    try:
        R = int(toks[0])
    except ValueError:
        fail("R not an integer")
    if R < 1:
        fail("R < 1")
    CAP = 4 * I * J * K + 10
    if R > CAP:
        fail("R=%d exceeds cap %d" % (R, CAP))

    L = I + J + K
    need = 1 + R * L
    if len(toks) < need:
        fail("not enough tokens: have %d need %d" % (len(toks), need))

    # parse the R terms (exact rationals, reject non-finite)
    terms = []
    p = 1
    for _ in range(R):
        try:
            vals = [parse_frac(toks[p + q]) for q in range(L)]
        except (ValueError, ZeroDivisionError):
            fail("non-finite/garbage token in a term")
        p += L
        a = vals[:I]
        b = vals[I:I + J]
        c = vals[I + J:]
        terms.append((a, b, c))

    # ---- exact reconstruction check ----
    Th = [[[Fraction(0) for _ in range(K)] for _ in range(J)] for _ in range(I)]
    for (a, b, c) in terms:
        for i in range(I):
            ai = a[i]
            if ai == 0:
                continue
            for j in range(J):
                aibj = ai * b[j]
                if aibj == 0:
                    continue
                rowh = Th[i][j]
                for k in range(K):
                    rowh[k] += aibj * c[k]
    for i in range(I):
        for j in range(J):
            for k in range(K):
                if Th[i][j][k] != T[i][j][k]:
                    fail("reconstruction mismatch at (%d,%d,%d)" % (i, j, k))

    # ---- score ----
    F = R
    B = I * J
    sc = min(1.0, 0.1 * B / max(1, F))
    print("valid: R=%d  baseline_B=%d  (I=%d J=%d K=%d)" % (R, B, I, J, K))
    print("Ratio: %.6f" % sc)


if __name__ == "__main__":
    main()
