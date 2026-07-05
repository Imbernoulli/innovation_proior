#!/usr/bin/env python3
"""counter.py <in> <out> <ans>   (ans ignored)

Format-D deterministic op-count checker (eval_form = flops) for the
thermal-coupling tensor CP-decomposition problem.

Instance: a dense I x J x K integer tensor T (thermal-coupling coefficients).

A participant submits a rank-R CP decomposition: R "cooling modes", each mode a
rank-1 outer product (a_r in Q^I) (x) (b_r in Q^J) (x) (c_r in Q^K).  The
decomposition is VALID iff, over the rationals,

    sum_{r=1..R} a_r[i] * b_r[j] * c_r[k] == T[i][j][k]   for all (i,j,k).

Verification is EXACT (Fraction arithmetic).  The score counts R = the number of
scalar multiplications the resulting controller performs.  Baseline B = nnz(T)
(the schoolbook decomposition with one rank-1 term per non-zero entry, which the
checker can always build itself).  ratio = min(1, 0.1 * B / R)  -> lower R better.
"""
import sys
from fractions import Fraction


def main():
    def fail(msg):
        print("Ratio: 0.0 (%s)" % msg)
        sys.exit(0)

    try:
        inp = open(sys.argv[1]).read().split()
        out = open(sys.argv[2]).read().split()
    except Exception as e:
        fail("io error: %s" % e)

    if len(inp) < 3:
        fail("no instance")
    try:
        I, J, K = int(inp[0]), int(inp[1]), int(inp[2])
    except Exception:
        fail("bad dims")
    if I <= 0 or J <= 0 or K <= 0:
        fail("nonpositive dims")

    need_in = 3 + I * J * K
    if len(inp) < need_in:
        fail("truncated instance")
    T = [[[0] * K for _ in range(J)] for _ in range(I)]
    idx = 3
    nnz = 0
    for i in range(I):
        for j in range(J):
            for k in range(K):
                v = int(inp[idx]); idx += 1
                T[i][j][k] = v
                if v != 0:
                    nnz += 1
    B = nnz
    if B <= 0:
        fail("degenerate instance (zero tensor)")

    # ---- parse participant decomposition (strict, bounded) ----
    if not out:
        fail("empty output")
    try:
        R = int(out[0])
    except Exception:
        fail("R not an integer")
    if R <= 0:
        fail("R <= 0")
    if R > 2 * I * J * K:
        fail("R=%d absurdly large" % R)

    per = I + J + K
    need_out = 1 + R * per
    if len(out) != need_out:
        fail("token count %d != expected %d" % (len(out), need_out))

    # reconstruct exactly over the rationals
    Th = [[[Fraction(0) for _ in range(K)] for _ in range(J)] for _ in range(I)]
    pos = 1
    try:
        for _ in range(R):
            a = [Fraction(out[pos + c]) for c in range(I)]; pos += I
            b = [Fraction(out[pos + c]) for c in range(J)]; pos += J
            c = [Fraction(out[pos + c]) for c in range(K)]; pos += K
            for i in range(I):
                if a[i] == 0:
                    continue
                ai = a[i]
                Ti = Th[i]
                for j in range(J):
                    if b[j] == 0:
                        continue
                    aibj = ai * b[j]
                    row = Ti[j]
                    for k in range(K):
                        if c[k] != 0:
                            row[k] += aibj * c[k]
    except Exception as e:
        fail("parse error: %s" % e)

    for i in range(I):
        for j in range(J):
            for k in range(K):
                if Th[i][j][k] != T[i][j][k]:
                    fail("reconstruction mismatch at (%d,%d,%d)" % (i, j, k))

    F = R
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("R=%d nnz=%d Ratio: %.6f" % (R, B, sc / 1000.0))


if __name__ == "__main__":
    main()
