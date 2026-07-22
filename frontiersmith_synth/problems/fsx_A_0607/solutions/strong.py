# TIER: strong
# INSIGHT: the eight analyst queries are not independent quadratics -- they all
# factor through ONE small shared library of linear forms.  As symmetric matrices
# every query lives in a common F-dimensional column space (F small).  Recover a
# basis of that space (F linear forms shared across ALL queries), compute those F
# forms + their O(F^2) pairwise products ONCE, then read every query off as a cheap
# combination of the shared products.  This beats per-query / per-monomial sharing
# because F(F+1)/2 shared products << the (large) number of expanded monomials.
import sys
from fractions import Fraction


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); K = int(next(it))
    queries = []
    for _ in range(K):
        Tq = int(next(it))
        terms = []
        for _t in range(Tq):
            i = int(next(it)); j = int(next(it)); c = int(next(it))
            terms.append((i, j, c))
        queries.append(terms)

    # symmetric rational matrix per query: x^T M x == Q(x)
    def build_M(terms):
        M = [[Fraction(0)] * n for _ in range(n)]
        for (i, j, c) in terms:
            if i == j:
                M[i][i] = Fraction(c)
            else:
                M[i][j] = Fraction(c, 2)
                M[j][i] = Fraction(c, 2)
        return M

    Ms = [build_M(t) for t in queries]

    # ---- common row/column space basis via incremental Gaussian elimination ----
    basis = []          # chosen original vectors (rows), spanning the shared space
    redrows = []        # row-reduced echelon copies for independence testing

    def add_if_independent(vec):
        r = list(vec)
        for er in redrows:
            # find pivot col of er
            pc = next(k for k in range(n) if er[k] != 0)
            if r[pc] != 0:
                f = r[pc] / er[pc]
                r = [a - f * b for a, b in zip(r, er)]
        pcs = [k for k in range(n) if r[k] != 0]
        if not pcs:
            return
        pc = pcs[0]
        r = [a / r[pc] for a in r]
        redrows.append(r)
        basis.append(list(vec))

    for M in Ms:
        for row in M:
            if any(v != 0 for v in row):
                add_if_independent(row)

    F = len(basis)
    if F == 0:
        # degenerate: fall back to trivial-style naive
        _naive(queries)
        return
    Bmat = basis                      # F x n  (rows are shared linear forms)

    # ---- linear algebra helpers ----
    def matmul(A, Bm):
        ra, ca = len(A), len(A[0])
        rb, cb = len(Bm), len(Bm[0])
        assert ca == rb
        out = [[Fraction(0)] * cb for _ in range(ra)]
        for i in range(ra):
            Ai = A[i]
            oi = out[i]
            for k in range(ca):
                a = Ai[k]
                if a == 0:
                    continue
                Bk = Bm[k]
                for j in range(cb):
                    oi[j] += a * Bk[j]
        return out

    def transpose(A):
        return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]

    def inverse(A):
        m = len(A)
        aug = [[Fraction(A[i][j]) for j in range(m)] + [Fraction(int(i == j)) for j in range(m)]
               for i in range(m)]
        for col in range(m):
            piv = next((r for r in range(col, m) if aug[r][col] != 0), None)
            if piv is None:
                return None
            aug[col], aug[piv] = aug[piv], aug[col]
            pv = aug[col][col]
            aug[col] = [v / pv for v in aug[col]]
            for r in range(m):
                if r != col and aug[r][col] != 0:
                    f = aug[r][col]
                    aug[r] = [a - f * b for a, b in zip(aug[r], aug[col])]
        return [row[m:] for row in aug]

    Bt = transpose(Bmat)               # n x F
    BBt = matmul(Bmat, Bt)             # F x F, invertible (full row rank)
    BBt_inv = inverse(BBt)
    if BBt_inv is None:
        _naive(queries)
        return

    # Sred_q = (B B^T)^-1 B M_q B^T (B B^T)^-1   solves  B^T Sred_q B = M_q
    Sreds = []
    for M in Ms:
        tmp = matmul(matmul(Bmat, M), Bt)     # F x F
        Sred = matmul(matmul(BBt_inv, tmp), BBt_inv)
        Sreds.append(Sred)

    # ---- emit the straight-line program ----
    ins = []

    def emit(op, a, b):
        ins.append((op, a, b))
        return "r%d" % (len(ins) - 1)

    def frac(c):
        c = Fraction(c)
        return str(c.numerator) if c.denominator == 1 else "%d/%d" % (c.numerator, c.denominator)

    def lincomb(pairs):
        pairs = [(Fraction(c), o) for c, o in pairs if c != 0]
        if not pairs:
            return "0"

        def scaled(c, o):
            if c == 1:
                return ("+", o)
            if c == -1:
                return ("-", o)
            return ("+", emit("*", frac(c), o))
        s0, first = scaled(*pairs[0])
        acc = first if s0 == "+" else emit("-", "0", first)
        for c, o in pairs[1:]:
            s, r = scaled(c, o)
            acc = emit("+" if s == "+" else "-", acc, r)
        return acc

    # 1) shared linear forms f_k = sum_i Bmat[k][i] x_i
    freg = []
    for k in range(F):
        pairs = [(Bmat[k][i], "x%d" % i) for i in range(n) if Bmat[k][i] != 0]
        freg.append(lincomb(pairs))

    # 2) shared pairwise products f_a*f_b, only those actually used
    used = set()
    for Sred in Sreds:
        for a in range(F):
            for b in range(a, F):
                if Sred[a][b] != 0:
                    used.add((a, b))
    prodreg = {}
    for (a, b) in sorted(used):
        prodreg[(a, b)] = emit("*", freg[a], freg[b])

    # 3) each query = combination of the shared products.
    #    coefficient of product (a,b): Sred[a][a] for a==b, else Sred[a][b]+Sred[b][a]=2*Sred[a][b].
    outs = []
    for Sred in Sreds:
        pairs = []
        for (a, b) in sorted(used):
            if a == b:
                coef = Sred[a][a]
            else:
                coef = Sred[a][b] + Sred[b][a]
            if coef != 0:
                pairs.append((coef, prodreg[(a, b)]))
        outs.append(lincomb(pairs))

    toks = [str(len(ins))]
    for op, a, b in ins:
        toks += [op, a, b]
    toks += outs
    sys.stdout.write(" ".join(toks) + "\n")


def _naive(queries):
    ins = []

    def emit(op, a, b):
        ins.append((op, a, b))
        return "r%d" % (len(ins) - 1)

    outs = []
    for terms in queries:
        acc = None
        for (i, j, c) in terms:
            prod = emit("*", "x%d" % i, "x%d" % j)
            tm = emit("*", str(c), prod)
            acc = tm if acc is None else emit("+", acc, tm)
        outs.append(acc if acc is not None else "0")
    toks = [str(len(ins))]
    for op, a, b in ins:
        toks += [op, a, b]
    toks += outs
    sys.stdout.write(" ".join(toks) + "\n")


if __name__ == "__main__":
    main()
