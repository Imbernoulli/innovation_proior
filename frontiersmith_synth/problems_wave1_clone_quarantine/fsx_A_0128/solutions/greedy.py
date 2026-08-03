# TIER: greedy
# Slice the tensor along the fixed tidal-state axis (mode 3). Each frontal slice
# T[:,:,k] is a matrix; its exact rank factorization gives rank(T[:,:,k]) rank-1
# channels.  Total channels = sum_k rank(slice_k)  <<  number of nonzeros.
import sys
from fractions import Fraction

def rank_factor(M, rows, cols):
    mat = [[Fraction(M[i][j]) for j in range(cols)] for i in range(rows)]
    pivot_cols = []
    r = 0
    for col in range(cols):
        piv = None
        for rr in range(r, rows):
            if mat[rr][col] != 0:
                piv = rr; break
        if piv is None:
            continue
        mat[r], mat[piv] = mat[piv], mat[r]
        pv = mat[r][col]
        mat[r] = [x / pv for x in mat[r]]
        for rr in range(rows):
            if rr != r and mat[rr][col] != 0:
                f = mat[rr][col]
                mat[rr] = [mat[rr][t] - f * mat[r][t] for t in range(cols)]
        pivot_cols.append(col)
        r += 1
        if r == rows:
            break
    terms = []
    for idx, pc in enumerate(pivot_cols):
        u = [Fraction(M[i][pc]) for i in range(rows)]
        v = list(mat[idx])
        terms.append((u, v))
    return terms

def fmt(x):
    x = Fraction(x)
    return str(x.numerator) if x.denominator == 1 else "%d/%d" % (x.numerator, x.denominator)

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    a = int(next(it)); b = int(next(it)); c = int(next(it))
    T = [[[0] * c for _ in range(b)] for _ in range(a)]
    for k in range(c):
        for i in range(a):
            for j in range(b):
                T[i][j][k] = int(next(it))
    channels = []
    for k in range(c):
        M = [[T[i][j][k] for j in range(b)] for i in range(a)]
        for (u, v) in rank_factor(M, a, b):
            w = [Fraction(0)] * c
            w[k] = Fraction(1)
            channels.append(list(u) + list(v) + w)
    outp = [str(len(channels))]
    for ch in channels:
        outp.append(" ".join(fmt(x) for x in ch))
    sys.stdout.write("\n".join(outp) + "\n")

if __name__ == "__main__":
    main()
