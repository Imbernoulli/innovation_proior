#!/usr/bin/env python3
# Deterministic checker for "Grid Phase-Decoupling Matrix" (format C, maximize |det|).
# CLI: python3 verify.py <in> <out> <ans>   (ans is ignored)
# Prints "... Ratio: <r>" with r in [0,1].
#
# The participant emits an N x N matrix of phase signs (+1 / -1) with the first row and
# first column pinned to +1 (reference synchronisation).  Quality = |det(M)| computed
# EXACTLY with Bareiss integer elimination (no floating point in the determinant).  The
# score is a LOG-normalised position between an internal trivial baseline and the
# Hadamard determinant bound N^(N/2), so the objective is graded, not pass/fail.
import sys, math


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def bareiss_absdet(M):
    """Exact |det| of an integer matrix via fraction-free Bareiss elimination."""
    n = len(M)
    A = [row[:] for row in M]
    prev = 1
    for k in range(n - 1):
        if A[k][k] == 0:
            piv = -1
            for r in range(k + 1, n):
                if A[r][k] != 0:
                    piv = r
                    break
            if piv < 0:
                return 0
            A[k], A[piv] = A[piv], A[k]
        akk = A[k][k]
        for i in range(k + 1, n):
            aik = A[i][k]
            Ai = A[i]
            Ak = A[k]
            for j in range(k + 1, n):
                Ai[j] = (Ai[j] * akk - aik * Ak[j]) // prev
        prev = akk
    return abs(A[n - 1][n - 1])


def normalize(M):
    """Flip row/column signs so row 0 and column 0 are all +1 (preserves |det|)."""
    n = len(M)
    for j in range(n):
        if M[0][j] == -1:
            for i in range(n):
                M[i][j] = -M[i][j]
    for i in range(1, n):
        if M[i][0] == -1:
            for j in range(n):
                M[i][j] = -M[i][j]
    return M


def baseline_matrix(n):
    """Normalised triangular +/-1 matrix; |det| = 2^(n-1)."""
    T = [[1 if j >= i else -1 for j in range(n)] for i in range(n)]
    return normalize(T)


def main():
    try:
        N = int(open(sys.argv[1]).read().split()[0])
    except Exception:
        fail("bad instance")

    try:
        toks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")

    if len(toks) != N * N:
        fail("expected %d entries, got %d" % (N * N, len(toks)))

    M = [[0] * N for _ in range(N)]
    for idx, t in enumerate(toks):
        try:
            v = int(t)
        except Exception:
            fail("non-integer entry")
        if v != 1 and v != -1:
            fail("entry not in {-1,+1}")
        M[idx // N][idx % N] = v

    # feasibility: reference synchronisation constraints
    for j in range(N):
        if M[0][j] != 1:
            fail("first row not all +1")
    for i in range(N):
        if M[i][0] != 1:
            fail("first column not all +1")

    F = bareiss_absdet(M)
    if F <= 0:
        fail("singular matrix (|det|=0)")

    B = bareiss_absdet(baseline_matrix(N))     # internal trivial baseline, positive
    Lbase = math.log(B)
    Lcap = (N / 2.0) * math.log(N)             # Hadamard determinant bound N^(N/2)
    Lf = math.log(F)

    if Lcap <= Lbase:
        # degenerate (won't happen for N>=4), guard anyway
        ratio = 1.0 if Lf >= Lbase else 0.0
    else:
        ratio = 0.1 + 0.9 * (Lf - Lbase) / (Lcap - Lbase)
        if ratio < 0.0:
            ratio = 0.0
        if ratio > 1.0:
            ratio = 1.0

    print("|det|=%d B=%d Lf=%.4f Lbase=%.4f Lcap=%.4f Ratio: %.6f"
          % (F, B, Lf, Lbase, Lcap, ratio))


if __name__ == "__main__":
    main()
