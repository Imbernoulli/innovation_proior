# TIER: strong
# Slice along the mode that minimizes the sum of exact (rational) slice ranks,
# then rank-1-factor each slice exactly:  M = C @ Rr  (C = pivot columns of M,
# Rr = nonzero rows of RREF(M)).  Each rank-1 piece of a slice becomes one CP
# term with a unit vector on the sliced mode.  R = sum_k rank(slice_k).
import sys
from fractions import Fraction

def read_tensor():
    tok = sys.stdin.read().split()
    it = iter(tok)
    a = int(next(it)); b = int(next(it)); c = int(next(it))
    T = [[[0] * c for _ in range(b)] for _ in range(a)]
    for k in range(c):
        for i in range(a):
            for j in range(b):
                T[i][j][k] = int(next(it))
    return a, b, c, T

def rref(M):
    # returns (reduced matrix as Fractions, list of pivot columns)
    R = len(M); C = len(M[0]) if R else 0
    A = [[Fraction(x) for x in row] for row in M]
    piv_cols = []
    r = 0
    for col in range(C):
        piv = None
        for rr in range(r, R):
            if A[rr][col] != 0:
                piv = rr; break
        if piv is None:
            continue
        A[r], A[piv] = A[piv], A[r]
        pv = A[r][col]
        A[r] = [x / pv for x in A[r]]
        for rr in range(R):
            if rr != r and A[rr][col] != 0:
                f = A[rr][col]
                A[rr] = [A[rr][cc] - f * A[r][cc] for cc in range(C)]
        piv_cols.append(col)
        r += 1
        if r == R:
            break
    return A, piv_cols

def rank1_factor(M):
    # M (list of rows, Fractions/ints) -> list of (col_vec, row_vec) with
    # M = sum col_vec (x) row_vec, length = rank(M).
    Rr, piv = rref(M)
    r = len(piv)
    Rrows = [Rr[t] for t in range(r)]                 # r x cols
    Ccols = [[Fraction(M[i][p]) for i in range(len(M))] for p in piv]  # r vecs of len rows
    return [(Ccols[t], Rrows[t]) for t in range(r)]

def as_str(x):
    x = Fraction(x)
    if x.denominator == 1:
        return str(x.numerator)
    return "%d/%d" % (x.numerator, x.denominator)

def main():
    a, b, c, T = read_tensor()

    # slice-rank cost along each mode
    # mode c: slices M_k (a x b)
    slices_c = [[[T[i][j][k] for j in range(b)] for i in range(a)] for k in range(c)]
    # mode a: slices N_i (b x c)
    slices_a = [[[T[i][j][k] for k in range(c)] for j in range(b)] for i in range(a)]
    # mode b: slices P_j (a x c)
    slices_b = [[[T[i][j][k] for k in range(c)] for i in range(a)] for j in range(b)]

    def total_rank(slices):
        return sum(len(rref(S)[1]) for S in slices)

    cost = [("c", total_rank(slices_c)),
            ("a", total_rank(slices_a)),
            ("b", total_rank(slices_b))]
    mode = min(cost, key=lambda x: x[1])[0]

    terms = []  # (u len a, v len b, w len c)
    if mode == "c":
        for k in range(c):
            for (col, row) in rank1_factor(slices_c[k]):  # col len a, row len b
                w = [Fraction(1 if r == k else 0) for r in range(c)]
                terms.append((col, row, w))
    elif mode == "a":
        for i in range(a):
            for (col, row) in rank1_factor(slices_a[i]):  # col len b, row len c
                u = [Fraction(1 if r == i else 0) for r in range(a)]
                terms.append((u, col, row))
    else:  # mode b
        for j in range(b):
            for (col, row) in rank1_factor(slices_b[j]):  # col len a, row len c
                v = [Fraction(1 if r == j else 0) for r in range(b)]
                terms.append((col, v, row))

    out = [str(len(terms))]
    for (u, v, w) in terms:
        out.append(" ".join(as_str(x) for x in u))
        out.append(" ".join(as_str(x) for x in v))
        out.append(" ".join(as_str(x) for x in w))
    sys.stdout.write("\n".join(out) + "\n")

main()
