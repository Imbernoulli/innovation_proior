# TIER: greedy
import sys
from fractions import Fraction


def rank_factor(A):
    # A: list of m rows, each a list of n numbers. Returns list of rank-1 terms
    # (u_col len m, v_row len n) with A == sum_s u_s (outer) v_s, exactly over Q.
    m = len(A)
    if m == 0:
        return []
    n = len(A[0])
    Aorig = [[Fraction(A[i][j]) for j in range(n)] for i in range(m)]
    M = [[Fraction(A[i][j]) for j in range(n)] for i in range(m)]
    pivots = []
    row = 0
    for col in range(n):
        pr = None
        for r in range(row, m):
            if M[r][col] != 0:
                pr = r
                break
        if pr is None:
            continue
        M[row], M[pr] = M[pr], M[row]
        pv = M[row][col]
        M[row] = [x / pv for x in M[row]]
        for r in range(m):
            if r != row and M[r][col] != 0:
                f = M[r][col]
                M[r] = [M[r][c] - f * M[row][c] for c in range(n)]
        pivots.append(col)
        row += 1
        if row == m:
            break
    terms = []
    for s in range(len(pivots)):
        col = pivots[s]
        u = [Aorig[i][col] for i in range(m)]
        v = [M[s][c] for c in range(n)]
        terms.append((u, v))
    return terms


def read_T():
    tok = sys.stdin.read().split()
    idx = 0
    n1 = int(tok[idx]); n2 = int(tok[idx + 1]); n3 = int(tok[idx + 2]); idx += 3
    T = [[[0] * n3 for _ in range(n2)] for _ in range(n1)]
    for k in range(n3):
        for i in range(n1):
            for j in range(n2):
                T[i][j][k] = int(tok[idx]); idx += 1
    return n1, n2, n3, T


def fmt(x):
    x = Fraction(x)
    return str(x.numerator) if x.denominator == 1 else "%d/%d" % (x.numerator, x.denominator)


def main():
    n1, n2, n3, T = read_T()
    # mode-1 factorization: for each i factor the (n2 x n3) slice T[i,:,:].
    prods = []
    for i in range(n1):
        S = [[T[i][j][k] for k in range(n3)] for j in range(n2)]
        for (vc, wr) in rank_factor(S):
            u = [0] * n1
            u[i] = 1
            prods.append(([Fraction(x) for x in u], vc, wr))
    lines = [str(len(prods))]
    for u, v, w in prods:
        lines.append(" ".join(fmt(x) for x in list(u) + list(v) + list(w)))
    sys.stdout.write("\n".join(lines) + "\n")


main()
