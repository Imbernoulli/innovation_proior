#!/usr/bin/env python3
"""verify.py <in> <out> <ans>  -- deterministic exact-integer scorer.

Reads N from <in>, an N x N +/-1 matrix from <out>. Validates feasibility strictly, then computes
the EXACT integer determinant via Bareiss fraction-free elimination. Normalizes against an internal
baseline B = |det(A0)| where A0 is a lightly-perturbed Legendre quadratic-residue circulant (the same
matrix the trivial reference emits). Prints `Ratio: <x>` on the final line; exits 0.

On ANY feasibility violation prints `Ratio: 0.0` and exits 0.
"""
import sys


def bareiss_det(M):
    """Exact integer determinant, fraction-free (Bareiss). M is a list of lists of ints."""
    n = len(M)
    A = [row[:] for row in M]
    sign = 1
    prev = 1
    for k in range(n - 1):
        if A[k][k] == 0:
            sw = None
            for i in range(k + 1, n):
                if A[i][k] != 0:
                    sw = i
                    break
            if sw is None:
                return 0
            A[k], A[sw] = A[sw], A[k]
            sign = -sign
        akk = A[k][k]
        for i in range(k + 1, n):
            Aik = A[i][k]
            Ai = A[i]
            Ak = A[k]
            for j in range(k + 1, n):
                Ai[j] = (Ai[j] * akk - Aik * Ak[j]) // prev
        prev = akk
    return sign * A[n - 1][n - 1]


def legendre(a, p):
    a %= p
    if a == 0:
        return 0
    r = pow(a, (p - 1) // 2, p)
    return 1 if r == 1 else -1


def baseline(p):
    """Lightly-perturbed Legendre quadratic-residue circulant (deterministic). MUST match trivial.py."""
    M = [[1 if i == j else (legendre((j - i) % p, p) or 1) for j in range(p)] for i in range(p)]
    for t in range(2):
        i = (2 * t + 1) % p
        j = (5 * t + 3) % p
        M[i][j] = -M[i][j]
    return M


def emit(sc, reason=""):
    if reason:
        sys.stdout.write("info: %s\n" % reason)
    sys.stdout.write("Ratio: %.6f\n" % sc)
    sys.exit(0)


def main():
    if len(sys.argv) < 3:
        emit(0.0, "usage")
    with open(sys.argv[1]) as f:
        itxt = f.read().split()
    if not itxt:
        emit(0.0, "no N in input")
    try:
        N = int(itxt[0])
    except ValueError:
        emit(0.0, "bad N")
    if N < 1 or N > 200:
        emit(0.0, "N out of range")

    with open(sys.argv[2]) as f:
        otxt = f.read().split()

    # strict token count
    if len(otxt) != N * N:
        emit(0.0, "expected %d tokens, got %d" % (N * N, len(otxt)))

    # strict parse: every token exactly -1 or +1 (rejects nan/inf/huge/non-integer)
    vals = []
    for tok in otxt:
        try:
            v = int(tok)
        except ValueError:
            emit(0.0, "non-integer token %r" % tok)
        if v != -1 and v != 1:
            emit(0.0, "entry %d not in {-1,1}" % v)
        vals.append(v)

    A = [vals[r * N:(r + 1) * N] for r in range(N)]

    F = abs(bareiss_det(A))
    B = abs(bareiss_det(baseline(N)))
    if B <= 0:
        B = 1

    sc = min(1000.0, 100.0 * F / max(1e-9, float(B)))
    emit(sc / 1000.0, "|det|=%d baseline=%d" % (F, B))


if __name__ == "__main__":
    main()
