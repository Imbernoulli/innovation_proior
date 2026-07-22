#!/usr/bin/env python3
"""
gen.py <testId>  -- emit ONE instance of the shared-subplan query-compiler problem.

Theme: one compiled plan serving eight analysts' queries.  K=8 analyst queries are
homogeneous quadratic forms over n shared input variables.  Each query is secretly
built as  Q_q(x) = sum_{a<=b} S_q[a][b] * L_a(x) * L_b(x)  where L_1..L_F are a small
hidden LIBRARY of sparse linear forms shared across ALL queries.  The instance ships
only the EXPANDED monomial coefficients of each query -- the shared factor structure
is hidden and must be re-discovered.

Deterministic: everything is seeded from testId only.
"""
import sys
from fractions import Fraction

K = 8

# testId -> (n, F, support-per-form, nonzeros-per-query upper-tri)
PARAMS = {
    1:  (8,  3, 3, 3),
    2:  (9,  3, 3, 4),
    3:  (10, 4, 3, 4),
    4:  (12, 4, 4, 5),
    5:  (13, 4, 4, 5),
    6:  (14, 5, 4, 6),
    7:  (15, 5, 4, 6),
    8:  (16, 5, 5, 7),
    9:  (18, 6, 5, 7),
    10: (20, 6, 5, 8),
}


def rank(rows):
    """Exact rational rank of a list of row vectors (list of Fractions)."""
    mat = [list(r) for r in rows]
    if not mat:
        return 0
    R = len(mat)
    C = len(mat[0])
    row = 0
    rk = 0
    for col in range(C):
        piv = None
        for i in range(row, R):
            if mat[i][col] != 0:
                piv = i
                break
        if piv is None:
            continue
        mat[row], mat[piv] = mat[piv], mat[row]
        pv = mat[row][col]
        mat[row] = [v / pv for v in mat[row]]
        for i in range(R):
            if i != row and mat[i][col] != 0:
                f = mat[i][col]
                mat[i] = [a - f * b for a, b in zip(mat[i], mat[row])]
        row += 1
        rk += 1
        if row == R:
            break
    return rk


def query_monos(G0, S, n, F):
    """Expand Q = sum_{a<=b} S[a][b]*L_a*L_b into a dict {(i,j): coef}, i<=j, nonzero."""
    mono = {}
    for a in range(F):
        for b in range(a, F):
            s = S[a][b]
            if s == 0:
                continue
            ga = G0[a]
            gb = G0[b]
            for i in range(n):
                if ga[i] == 0:
                    continue
                for j in range(n):
                    if gb[j] == 0:
                        continue
                    ii, jj = (i, j) if i <= j else (j, i)
                    mono[(ii, jj)] = mono.get((ii, jj), 0) + s * ga[i] * gb[j]
    return {k: v for k, v in mono.items() if v != 0}


def sym_matrix(mono, n):
    """Symmetric rational matrix M with x^T M x == sum mono[(i,j)] x_i x_j."""
    M = [[Fraction(0)] * n for _ in range(n)]
    for (i, j), c in mono.items():
        if i == j:
            M[i][i] = Fraction(c)
        else:
            M[i][j] = Fraction(c, 2)
            M[j][i] = Fraction(c, 2)
    return M


def main():
    t = int(sys.argv[1])
    n, F, sup, sden = PARAMS[t]
    import random
    rng = random.Random(1234567 + t * 99991)

    def make_forms():
        while True:
            G0 = [[0] * n for _ in range(F)]
            for f in range(F):
                vs = rng.sample(range(n), sup)
                for i in vs:
                    G0[f][i] = rng.choice([-3, -2, -1, 1, 2, 3])
            if rank([[Fraction(x) for x in row] for row in G0]) == F:
                return G0

    def make_S():
        S = [[0] * F for _ in range(F)]
        pairs = [(a, b) for a in range(F) for b in range(a, F)]
        for (a, b) in rng.sample(pairs, min(sden, len(pairs))):
            c = rng.choice([-3, -2, -1, 1, 2, 3])
            S[a][b] = c
            S[b][a] = c
        return S

    # Build forms + a set of K queries whose union column space has full rank F
    # (so the shared factor library is genuinely F-dimensional) and every query
    # is a nonzero quadratic.
    G0 = make_forms()
    for _attempt in range(400):
        queries = []
        ok = True
        for _q in range(K):
            for _try in range(50):
                S = make_S()
                mono = query_monos(G0, S, n, F)
                if mono:
                    break
            if not mono:
                ok = False
                break
            queries.append(mono)
        if not ok:
            continue
        # union column space rank across all query matrices must be F
        allrows = []
        for mono in queries:
            M = sym_matrix(mono, n)
            allrows.extend(M)
        if rank(allrows) == F:
            break
    else:
        # fall back: regenerate forms once more (extremely rare)
        G0 = make_forms()
        queries = []
        for _q in range(K):
            while True:
                S = make_S()
                mono = query_monos(G0, S, n, F)
                if mono:
                    break
            queries.append(mono)

    out = ["%d %d" % (n, K)]
    for mono in queries:
        items = sorted(mono.items())
        out.append(str(len(items)))
        for (i, j), c in items:
            out.append("%d %d %d" % (i, j, c))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
