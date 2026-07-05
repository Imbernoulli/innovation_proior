# TIER: strong
# Cross-slice sharing: detect that some output matrices are exact linear combinations of
# others (mode-3 dependence). Factor ONLY a maximal independent subset (basis) of slices;
# express every other slice as a rational combination of the basis, folding those weights
# into the combiner c_r. R = sum over basis slices of rank(slice) -- strictly fewer products
# than the per-slice greedy whenever any slice is dependent. Still far from the (unknown)
# true tensor rank of this planted overcomplete form.
import sys
from fractions import Fraction

def rref(M):
    A = [row[:] for row in M]
    rows = len(A); cols = len(A[0]) if rows else 0
    piv = []; r = 0
    for c in range(cols):
        sel = None
        for i in range(r, rows):
            if A[i][c] != 0:
                sel = i; break
        if sel is None:
            continue
        A[r], A[sel] = A[sel], A[r]
        pv = A[r][c]
        A[r] = [x / pv for x in A[r]]
        for i in range(rows):
            if i != r and A[i][c] != 0:
                f = A[i][c]
                A[i] = [x - f * y for x, y in zip(A[i], A[r])]
        piv.append(c); r += 1
        if r == rows:
            break
    return A[:r], piv, r

def rank_factor(S):
    m = len(S); B, piv, r = rref(S)
    return [([S[i][piv[t]] for i in range(m)], B[t][:]) for t in range(r)]

def solve_in_basis(basis, v):
    # express vector v as combo of independent basis vectors; None if not in span.
    if not basis:
        return [] if all(x == 0 for x in v) else None
    L = len(v); k = len(basis)
    Aug = [[basis[j][i] for j in range(k)] + [v[i]] for i in range(L)]
    B, piv, r = rref(Aug)
    if piv and piv[-1] == k:      # pivot in RHS column -> inconsistent
        return None
    x = [Fraction(0)] * k
    for t, pc in enumerate(piv):
        if pc < k:
            x[pc] = B[t][k]
    for i in range(L):            # verify exactness
        s = Fraction(0)
        for j in range(k):
            if x[j] != 0:
                s += x[j] * basis[j][i]
        if s != v[i]:
            return None
    return x

def fmt(x):
    return str(x.numerator) if x.denominator == 1 else "%d/%d" % (x.numerator, x.denominator)

def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    P = int(next(it)); m = int(next(it)); n = int(next(it))
    T = [[[Fraction(int(next(it))) for _ in range(n)] for _ in range(m)] for _ in range(P)]

    flats = [[T[p][i][j] for i in range(m) for j in range(n)] for p in range(P)]
    basis = []          # flattened basis slices
    basis_p = []        # their slice indices
    coords = [None] * P # coords[p] = coordinates of slice p in the basis
    for p in range(P):
        x = solve_in_basis(basis, flats[p])
        if x is None:
            # independent -> new basis slice; its own coordinate is a fresh unit
            basis.append(flats[p]); basis_p.append(p)
            coords[p] = None  # filled below
    # rebuild coordinates now that the full basis is known
    for p in range(P):
        coords[p] = solve_in_basis(basis, flats[p])

    terms = []
    for bk, p_basis in enumerate(basis_p):
        for (a, b) in rank_factor(T[p_basis]):
            c = [coords[q][bk] for q in range(P)]
            terms.append((a, b, c))

    out = [str(len(terms))]
    for (a, b, c) in terms:
        out.append(" ".join(fmt(x) for x in a))
        out.append(" ".join(fmt(x) for x in b))
        out.append(" ".join(fmt(x) for x in c))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
