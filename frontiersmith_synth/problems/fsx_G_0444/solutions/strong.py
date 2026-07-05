# TIER: strong
# Try slicing the structure tensor along EACH of the three modes (output / signal /
# filter); every choice yields a valid Winograd algorithm of size (sum of slice
# ranks).  Keep the fewest-multiplier one.  Because the instances satisfy s < q < p,
# slicing along a small mode (output/filter) shares products far better than the
# greedy signal-axis choice -- so this strictly beats greedy, while the planted
# overcomplete rank sits below every slice bound, leaving the true optimum open.
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

def decompose(p, q, s, T, axis):
    terms = []
    if axis == 0:      # fix output k : slice is p x q  (u = col, v = row, w = e_k)
        for k in range(s):
            M = [[T[k][i][j] for j in range(q)] for i in range(p)]
            for (col, row) in skeleton(M, p, q):
                terms.append((col, row, unit(s, k)))
    elif axis == 1:    # fix signal i : slice is s x q  (w = col, v = row, u = e_i)
        for i in range(p):
            M = [[T[k][i][j] for j in range(q)] for k in range(s)]
            for (col, row) in skeleton(M, s, q):
                terms.append((unit(p, i), row, col))
    else:              # fix filter j : slice is s x p  (w = col, u = row, v = e_j)
        for j in range(q):
            M = [[T[k][i][j] for i in range(p)] for k in range(s)]
            for (col, row) in skeleton(M, s, p):
                terms.append((row, unit(q, j), col))
    return terms

def main():
    p, q, s, T = read_T()
    best = None
    for axis in (0, 1, 2):
        cand = decompose(p, q, s, T, axis)
        if best is None or len(cand) < len(best):
            best = cand

    out = [str(len(best))]
    for (u, v, w) in best:
        out.append(" ".join(str(x) for x in list(u) + list(v) + list(w)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
