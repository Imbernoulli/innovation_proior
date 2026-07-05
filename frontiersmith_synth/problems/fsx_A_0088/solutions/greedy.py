# TIER: greedy
# Slice the tensor along the fixed TIME axis (mode-3): each time slot k gives an
# a x b matrix; factor it into rank(M_k) rank-1 stages via exact skeleton
# (Gaussian) elimination.  Total rank = sum_k rank(M_k).  This is a real, valid
# decomposition found from the tensor alone -- and strictly beats per-entry.
import sys
from fractions import Fraction

def read_tensor():
    data = sys.stdin.read().split()
    it = iter(data)
    a = int(next(it)); b = int(next(it)); c = int(next(it))
    G = [[[0] * c for _ in range(b)] for _ in range(a)]
    for i in range(a):
        for j in range(b):
            for k in range(c):
                G[i][j][k] = int(next(it))
    return a, b, c, G

def skeleton(M, nr, nc):
    """Factor matrix M (nr x nc, Fraction) into rank-1 terms (col, row)."""
    A = [[Fraction(M[i][j]) for j in range(nc)] for i in range(nr)]
    terms = []
    while True:
        pi = pj = -1
        for i in range(nr):
            for j in range(nc):
                if A[i][j] != 0:
                    pi, pj = i, j
                    break
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

def main():
    a, b, c, G = read_tensor()
    stages = []
    for k in range(c):
        M = [[G[i][j][k] for j in range(b)] for i in range(a)]  # a x b
        for (col, row) in skeleton(M, a, b):
            w = [Fraction(0)] * c; w[k] = Fraction(1)
            stages.append((col, row, w))

    outp = [str(len(stages))]
    for (u, v, w) in stages:
        outp.append(" ".join(str(x) for x in list(u) + list(v) + list(w)))
    sys.stdout.write("\n".join(outp) + "\n")

if __name__ == "__main__":
    main()
