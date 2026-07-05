#!/usr/bin/env python3
# counter.py <in> <out> <ans>   (ans ignored) -- deterministic Format-D scorer.
#
# Instance: a 3-tensor T of shape (P, m, n) of integers, given as P slices.
# Participant artifact: a BILINEAR decomposition of T into R rank-1 product terms
#     T[p][i][j] == sum_{r=1..R} c_r[p] * a_r[i] * b_r[j]     (exact, over rationals)
# The "cost" is R = the number of scalar MULTIPLICATIONS (the essential products
# (a_r . x)(b_r . y) of a bilinear algorithm). Fewer is better.
#
# Checker: (1) parse strictly, reject non-finite / malformed / R out of [1, m*n];
#          (2) verify EXACT reconstruction (any mismatch -> Ratio: 0.0);
#          (3) score  ratio = min(1, 0.1 * (m*n) / R)   [baseline B = naive m*n scheme].
import sys
from fractions import Fraction

def fail(msg):
    print("reject: " + msg)
    print("Ratio: 0.0")
    sys.exit(0)

def read_ints(path):
    with open(path) as f:
        toks = f.read().split()
    return toks

def main():
    if len(sys.argv) < 3:
        fail("usage")
    in_toks = read_ints(sys.argv[1])
    it = iter(in_toks)
    try:
        P = int(next(it)); m = int(next(it)); n = int(next(it))
    except StopIteration:
        fail("bad header")
    if not (1 <= P <= 64 and 1 <= m <= 64 and 1 <= n <= 64):
        fail("bad dims")
    T = []
    try:
        for _ in range(P):
            S = [[int(next(it)) for _ in range(n)] for _ in range(m)]
            T.append(S)
    except (StopIteration, ValueError):
        fail("bad tensor")

    baseline = m * n                      # naive "compute every x_i*y_j" scheme

    # ---- parse participant output strictly ----
    otoks = read_ints(sys.argv[2])
    if not otoks:
        fail("empty output")
    # first token = R
    try:
        R = int(otoks[0])
    except ValueError:
        fail("R not an integer")
    if not (1 <= R <= baseline):
        fail("R out of range [1, m*n]")
    need = 1 + R * (m + n + P)
    if len(otoks) != need:
        fail("token count %d != expected %d" % (len(otoks), need))

    # parse each term's a (m), b (n), c (P) as exact rationals; reject non-finite/garbage
    def frac(tok):
        low = tok.lower()
        if low in ("nan", "inf", "+inf", "-inf", "infinity", "-infinity"):
            raise ValueError("nonfinite")
        return Fraction(tok)   # rejects floats-with-'e'? Fraction accepts decimals like 1.5 and ints/fractions

    idx = 1
    terms = []
    try:
        for _ in range(R):
            a = [frac(otoks[idx + k]) for k in range(m)]; idx += m
            b = [frac(otoks[idx + k]) for k in range(n)]; idx += n
            c = [frac(otoks[idx + k]) for k in range(P)]; idx += P
            terms.append((a, b, c))
    except (ValueError, ZeroDivisionError, IndexError):
        fail("bad term token")

    # ---- verify EXACT reconstruction T[p][i][j] == sum_r c_r[p]*a_r[i]*b_r[j] ----
    # Precompute for each term the outer product a_r[i]*b_r[j] lazily per (i,j).
    for p in range(P):
        for i in range(m):
            for j in range(n):
                s = Fraction(0)
                for (a, b, c) in terms:
                    cp = c[p]
                    if cp != 0:
                        ai = a[i]
                        if ai != 0:
                            bj = b[j]
                            if bj != 0:
                                s += cp * ai * bj
                if s != T[p][i][j]:
                    fail("reconstruction mismatch at p=%d i=%d j=%d" % (p, i, j))

    F = R
    sc = min(1000.0, 100.0 * baseline / max(1e-9, F))
    print("valid decomposition: R=%d terms, baseline=%d" % (R, baseline))
    print("Ratio: %.6f" % (sc / 1000.0))

if __name__ == "__main__":
    main()
