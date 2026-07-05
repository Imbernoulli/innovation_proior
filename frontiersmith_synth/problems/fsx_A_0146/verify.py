#!/usr/bin/env python3
"""verify.py <in> <out> <ans>   (ans ignored) -- deterministic scorer.

Reads the dig-grid instance and the participant's completed +/-1 grid, validates
it strictly, computes the EXACT integer determinant via Bareiss elimination, and
scores the completion by where its log|det| lands between the checker's own
minimal 'triangular' baseline completion and the (unreachable) Hadamard upper
bound N^(N/2):

    p     = (log|det| - log|det_baseline|) / (log(Hadamard) - log|det_baseline|)
    Ratio = clamp(0.10 + 0.90 * p , 0, 1)

The trivial baseline completion reproduces det_baseline -> p = 0 -> Ratio = 0.10.
The Hadamard bound is unreachable for these odd orders, so there is permanent
headroom above any real construction. Any feasibility violation -> Ratio 0.0.
"""
import sys, math


def det_bareiss(M):
    n = len(M)
    A = [row[:] for row in M]
    sign = 1
    prev = 1
    for k in range(n - 1):
        if A[k][k] == 0:
            piv = -1
            for i in range(k + 1, n):
                if A[i][k] != 0:
                    piv = i
                    break
            if piv == -1:
                return 0
            A[k], A[piv] = A[piv], A[k]
            sign = -sign
        akk = A[k][k]
        for i in range(k + 1, n):
            aik = A[i][k]
            Ai = A[i]
            Ak = A[k]
            for j in range(k + 1, n):
                Ai[j] = (Ai[j] * akk - aik * Ak[j]) // prev
        prev = akk
    return sign * A[n - 1][n - 1]


def fail(reason):
    print("reason: " + reason)
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    if not toks:
        fail("empty instance")
    N = int(toks[0])
    vals = list(map(int, toks[1:1 + N * N]))
    if len(vals) != N * N:
        fail("bad instance")
    G = [vals[i * N:(i + 1) * N] for i in range(N)]
    fixed = {}
    for i in range(N):
        for j in range(N):
            if G[i][j] != 0:
                fixed[(i, j)] = G[i][j]
    return N, fixed


def read_output(path, N):
    try:
        with open(path) as f:
            toks = f.read().split()
    except Exception:
        fail("cannot read output")
    # need EXACTLY N*N tokens (no trailing junk allowed)
    if len(toks) != N * N:
        fail("expected %d entries, got %d" % (N * N, len(toks)))
    vals = []
    for t in toks:
        try:
            v = int(t)
        except ValueError:
            fail("non-integer entry '%s'" % t[:16])
        if v != 1 and v != -1:
            fail("entry not in {-1,+1}: %d" % v)
        vals.append(v)
    return [vals[i * N:(i + 1) * N] for i in range(N)]


def baseline_matrix(fixed, N):
    M = [[1 if j <= i else -1 for j in range(N)] for i in range(N)]
    for (i, j), v in fixed.items():
        M[i][j] = v
    return M


def main():
    inf, outf = sys.argv[1], sys.argv[2]
    N, fixed = read_instance(inf)
    M = read_output(outf, N)

    # feasibility: excavated cells must be respected
    for (i, j), v in fixed.items():
        if M[i][j] != v:
            fail("cell (%d,%d) violates excavated artifact" % (i, j))

    Fdet = abs(det_bareiss(M))
    Bdet = abs(det_bareiss(baseline_matrix(fixed, N)))
    if Bdet <= 0:
        fail("degenerate baseline (should not happen)")

    Ls = math.log(Fdet) if Fdet > 0 else 0.0
    Lb = math.log(Bdet)
    Lmax = (N / 2.0) * math.log(N)          # log of Hadamard bound N^(N/2)
    denom = Lmax - Lb
    if denom <= 1e-9:
        denom = 1e-9
    p = (Ls - Lb) / denom
    ratio = 0.10 + 0.90 * p
    if ratio < 0.0:
        ratio = 0.0
    if ratio > 1.0:
        ratio = 1.0
    print("|det|=%d baseline=%d p=%.4f" % (Fdet, Bdet, p))
    print("Ratio: %.6f" % ratio)


if __name__ == "__main__":
    main()
