# TIER: strong
"""Mode-wise rank exploitation.  A valid bilinear split of T can be built by fixing
one tensor mode and integer/rational rank-1 peeling each of its matrix slices:
  - output mode: sum_k rank(T[k,:,:])   (each slice p x q, term d = e_k)
  - x  mode:     sum_i rank(T[:,i,:])   (slice r x q, product x_i * (c.y), d = column)
  - y  mode:     sum_j rank(T[:,:,j])   (slice r x p, product (a.x) * y_j, d = column)
Each mode gives a valid split; the strong solver builds all three and emits the
smallest.  This captures the planted low linear-rank structure and uses far fewer
scalar products than the schoolbook / support splits -- but it is only an upper
bound on the (unknown) true bilinear rank, so headroom remains."""
import sys
from fractions import Fraction as Fr


def peel(M):
    """Exact rank-1 peel of integer matrix M over the rationals; returns list of
    (colvec, rowvec) with len == rank(M).  M[i][j] == sum_t col_t[i]*row_t[j]."""
    R = len(M); C = len(M[0]) if R else 0
    A = [[Fr(x) for x in row] for row in M]
    terms = []
    while True:
        piv = None
        for i in range(R):
            for j in range(C):
                if A[i][j] != 0:
                    piv = (i, j); break
            if piv:
                break
        if not piv:
            break
        a, b = piv
        pv = A[a][b]
        col = [A[i][b] / pv for i in range(R)]
        row = [A[a][j] for j in range(C)]
        for i in range(R):
            ci = col[i]
            if ci == 0:
                continue
            for j in range(C):
                if row[j] != 0:
                    A[i][j] -= ci * row[j]
        terms.append((col, row))
    return terms


def fstr(x):
    x = Fr(x)
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    p = int(next(it)); q = int(next(it)); r = int(next(it))
    T = [[[0] * q for _ in range(p)] for _ in range(r)]
    for k in range(r):
        for i in range(p):
            for j in range(q):
                T[k][i][j] = int(next(it))

    cands = []

    # output mode
    t = []
    for k in range(r):
        for col, row in peel(T[k]):          # col over i, row over j
            d = [Fr(0)] * r; d[k] = Fr(1)
            t.append((col, row, d))
    cands.append(t)

    # x mode
    t = []
    for i in range(p):
        Mi = [[T[k][i][j] for j in range(q)] for k in range(r)]
        for col, row in peel(Mi):            # col over k, row over j
            a = [Fr(0)] * p; a[i] = Fr(1)
            t.append((a, row, col))
    cands.append(t)

    # y mode
    t = []
    for j in range(q):
        Nj = [[T[k][i][j] for i in range(p)] for k in range(r)]
        for col, row in peel(Nj):            # col over k, row over i
            c = [Fr(0)] * q; c[j] = Fr(1)
            t.append((row, c, col))
    cands.append(t)

    terms = min(cands, key=len)
    if not terms:  # all-zero tensor
        terms = [([Fr(0)] * p, [Fr(0)] * q, [Fr(0)] * r)]

    lines = []
    for a, c, d in terms:
        lines.append(" ".join(fstr(v) for v in list(a) + list(c) + list(d)))
    sys.stdout.write(str(len(terms)) + "\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
