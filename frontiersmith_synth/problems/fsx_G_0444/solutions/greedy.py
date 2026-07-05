# TIER: greedy
# Slice the structure tensor along the SIGNAL axis (fix each signal sample i).
# Slice i is the s x q matrix M_i[k][j] = T[k][i][j]; factor it into rank(M_i)
# rank-1 bilinear terms by exact skeleton (Gaussian) elimination.  Each factor
# shares one multiplier across all output taps for that fixed signal sample.
# Total multipliers = sum_i rank(M_i).  This is a real, valid Winograd algorithm
# found from T alone and it beats the per-fiber baseline -- but the signal axis is
# the largest mode, so this fixed choice is strictly wasteful.
import sys
from fractions import Fraction

def read_T():
    data = sys.stdin.read().split()
    it = iter(data)
    p = int(next(it)); q = int(next(it)); s = int(next(it))
    T = [[[0] * q for _ in range(p)] for _ in range(s)]
    for i in range(p):
        for j in range(q):
            for k in range(s):
                T[k][i][j] = int(next(it))
    return p, q, s, T

def skeleton(M, nr, nc):
    """Factor nr x nc matrix into rank-1 terms (col length nr, row length nc)."""
    A = [[Fraction(M[i][j]) for j in range(nc)] for i in range(nr)]
    terms = []
    while True:
        pi = pj = -1
        for i in range(nr):
            for j in range(nc):
                if A[i][j] != 0:
                    pi, pj = i, j; break
            if pi != -1:
                break
        if pi == -1:
            break
        piv = A[pi][pj]
        col = [A[i][pj] for i in range(nr)]
        row = [A[pi][j] / piv for j in range(nc)]
        terms.append((col, row))
        for i in range(nr):
            ci = col[i]
            if ci == 0:
                continue
            for j in range(nc):
                if row[j] != 0:
                    A[i][j] -= ci * row[j]
    return terms

def unit(n, idx):
    e = [Fraction(0)] * n; e[idx] = Fraction(1); return e

def main():
    p, q, s, T = read_T()
    terms = []
    for i in range(p):
        M = [[T[k][i][j] for j in range(q)] for k in range(s)]   # s x q
        for (col, row) in skeleton(M, s, q):
            # col over output axis -> w ; row over filter axis -> v ; u = e_i
            u = unit(p, i); v = row; w = col
            terms.append((u, v, w))

    out = [str(len(terms))]
    for (u, v, w) in terms:
        out.append(" ".join(str(x) for x in list(u) + list(v) + list(w)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
