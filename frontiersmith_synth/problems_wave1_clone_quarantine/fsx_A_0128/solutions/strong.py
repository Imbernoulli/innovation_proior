# TIER: strong
# Try slicing along EACH of the three tensor axes; each choice yields a valid
# channel list of size sum_slices rank(slice). Emit the smallest of the three.
# Beats the fixed-axis greedy whenever another axis factors more compactly.
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

def zeros(n):
    return [Fraction(0)] * n

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    a = int(next(it)); b = int(next(it)); c = int(next(it))
    T = [[[0] * c for _ in range(b)] for _ in range(a)]
    for k in range(c):
        for i in range(a):
            for j in range(b):
                T[i][j][k] = int(next(it))

    # mode 3: slice by k -> u(a) v(b) e_k(c)
    ch3 = []
    for k in range(c):
        M = [[T[i][j][k] for j in range(b)] for i in range(a)]
        for (u, v) in rank_factor(M, a, b):
            w = zeros(c); w[k] = Fraction(1)
            ch3.append(list(u) + list(v) + w)

    # mode 1: slice by i -> e_i(a) v(b) w(c)
    ch1 = []
    for i in range(a):
        M = [[T[i][j][k] for k in range(c)] for j in range(b)]
        for (v, w) in rank_factor(M, b, c):
            u = zeros(a); u[i] = Fraction(1)
            ch1.append(u + list(v) + list(w))

    # mode 2: slice by j -> u(a) e_j(b) w(c)
    ch2 = []
    for j in range(b):
        M = [[T[i][j][k] for k in range(c)] for i in range(a)]
        for (u, w) in rank_factor(M, a, c):
            v = zeros(b); v[j] = Fraction(1)
            ch2.append(list(u) + v + list(w))

    best = min([ch1, ch2, ch3], key=len)
    outp = [str(len(best))]
    for ch in best:
        outp.append(" ".join(fmt(x) for x in ch))
    sys.stdout.write("\n".join(outp) + "\n")

if __name__ == "__main__":
    main()
