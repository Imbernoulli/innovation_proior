# TIER: strong
# Try slicing along EACH of the three axes; each choice yields a valid
# decomposition of size (sum of slice ranks).  Pick the axis giving the fewest
# stages.  Because instances always have c > b, slicing along the time axis
# (greedy's fixed choice) is strictly suboptimal, so this beats greedy while
# still leaving the true CP-rank ceiling open.
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

def unit(n, idx):
    e = [Fraction(0)] * n; e[idx] = Fraction(1); return e

def decompose_axis(a, b, c, G, axis):
    """Return list of (u,v,w) stages slicing along `axis` (0=antenna,1=chan,2=time)."""
    stages = []
    if axis == 0:  # fix i, slice is b x c
        for i in range(a):
            M = [[G[i][j][k] for k in range(c)] for j in range(b)]  # b x c
            for (col, row) in skeleton(M, b, c):
                stages.append((unit(a, i), col, row))
    elif axis == 1:  # fix j, slice is a x c
        for j in range(b):
            M = [[G[i][j][k] for k in range(c)] for i in range(a)]  # a x c
            for (col, row) in skeleton(M, a, c):
                stages.append((col, unit(b, j), row))
    else:  # fix k, slice is a x b
        for k in range(c):
            M = [[G[i][j][k] for j in range(b)] for i in range(a)]  # a x b
            for (col, row) in skeleton(M, a, b):
                stages.append((col, row, unit(c, k)))
    return stages

def main():
    a, b, c, G = read_tensor()
    best = None
    for axis in (0, 1, 2):
        st = decompose_axis(a, b, c, G, axis)
        if best is None or len(st) < len(best):
            best = st

    outp = [str(len(best))]
    for (u, v, w) in best:
        outp.append(" ".join(str(x) for x in list(u) + list(v) + list(w)))
    sys.stdout.write("\n".join(outp) + "\n")

if __name__ == "__main__":
    main()
