# TIER: greedy
# Per-slice exact rank factorization: factor each output matrix M_p = sum_t a_t b_t^T
# (rank(M_p) products) independently. R = sum_p rank(M_p) -- beats the naive m*n scheme
# because every planted slice is low rank, but wastes products by never sharing across slices.
import sys
from fractions import Fraction

def rref(M):
    A = [row[:] for row in M]
    rows = len(A); cols = len(A[0]) if rows else 0
    piv = []; r = 0
    for c in range(cols):
        sel = None
        for i in range(r, rows):
            if A[i][c] != 0:
                sel = i; break
        if sel is None:
            continue
        A[r], A[sel] = A[sel], A[r]
        pv = A[r][c]
        A[r] = [x / pv for x in A[r]]
        for i in range(rows):
            if i != r and A[i][c] != 0:
                f = A[i][c]
                A[i] = [x - f * y for x, y in zip(A[i], A[r])]
        piv.append(c); r += 1
        if r == rows:
            break
    return A[:r], piv, r

def rank_factor(S):
    # S: m x n Fraction matrix -> list of (a[m], b[n]) with S = sum a b^T, len = rank
    m = len(S); B, piv, r = rref(S)
    return [([S[i][piv[t]] for i in range(m)], B[t][:]) for t in range(r)]

def fmt(x):
    return str(x.numerator) if x.denominator == 1 else "%d/%d" % (x.numerator, x.denominator)

def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    P = int(next(it)); m = int(next(it)); n = int(next(it))
    T = [[[Fraction(int(next(it))) for _ in range(n)] for _ in range(m)] for _ in range(P)]

    terms = []  # (a[m], b[n], c[P])
    for p in range(P):
        for (a, b) in rank_factor(T[p]):
            c = [Fraction(1) if q == p else Fraction(0) for q in range(P)]
            terms.append((a, b, c))

    out = [str(len(terms))]
    for (a, b, c) in terms:
        out.append(" ".join(fmt(x) for x in a))
        out.append(" ".join(fmt(x) for x in b))
        out.append(" ".join(fmt(x) for x in c))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
