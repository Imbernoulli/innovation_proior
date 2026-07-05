# TIER: strong
import sys
from fractions import Fraction


def rank_factor(A):
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


def zeros(n):
    return [Fraction(0)] * n


def one_hot(n, i):
    v = [Fraction(0)] * n
    v[i] = Fraction(1)
    return v


def mode1(n1, n2, n3, T):
    prods = []
    for i in range(n1):
        S = [[T[i][j][k] for k in range(n3)] for j in range(n2)]
        for (vc, wr) in rank_factor(S):
            prods.append((one_hot(n1, i), vc, wr))
    return prods


def mode2(n1, n2, n3, T):
    prods = []
    for j in range(n2):
        S = [[T[i][j][k] for k in range(n3)] for i in range(n1)]
        for (uc, wr) in rank_factor(S):
            prods.append((uc, one_hot(n2, j), wr))
    return prods


def mode3(n1, n2, n3, T):
    prods = []
    for k in range(n3):
        S = [[T[i][j][k] for j in range(n2)] for i in range(n1)]
        for (uc, vr) in rank_factor(S):
            prods.append((uc, vr, one_hot(n3, k)))
    return prods


def trivial(n1, n2, n3, T):
    prods = []
    for i in range(n1):
        for j in range(n2):
            if any(T[i][j][k] != 0 for k in range(n3)):
                prods.append((one_hot(n1, i), one_hot(n2, j),
                              [Fraction(T[i][j][k]) for k in range(n3)]))
    return prods


def fmt(x):
    x = Fraction(x)
    return str(x.numerator) if x.denominator == 1 else "%d/%d" % (x.numerator, x.denominator)


def main():
    n1, n2, n3, T = read_T()
    cands = [
        mode1(n1, n2, n3, T),
        mode2(n1, n2, n3, T),
        mode3(n1, n2, n3, T),
        trivial(n1, n2, n3, T),
    ]
    best = min(cands, key=len)
    lines = [str(len(best))]
    for u, v, w in best:
        lines.append(" ".join(fmt(x) for x in list(u) + list(v) + list(w)))
    sys.stdout.write("\n".join(lines) + "\n")


main()
