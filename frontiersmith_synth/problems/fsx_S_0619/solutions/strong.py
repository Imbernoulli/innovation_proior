# TIER: strong
# Insight: multiplication mod pairwise-COPRIME moduli (CRT over F_p[x]) plus
# evaluation at structured points.  Cover the 2d+1 dimensions of the product with
#   * every LINEAR modulus x-alpha (a genuine field point): cost 1 product each,
#     but only p of them exist over F_p, and
#   * IRREDUCIBLE QUADRATIC moduli (points in F_{p^2}): cost 3 products, cover 2.
# When p >= 2d+1 this reaches exactly 2d+1 products (the rank bound).  When p is
# small there are not enough linear points, so we splice in quadratic moduli --
# something a field-blind Karatsuba coder can never do.
import sys


def inv(a, p):
    return pow(a % p, p - 2, p)


def mat_inv(M, p):
    n = len(M)
    A = [[M[i][j] % p for j in range(n)] + [1 if j == i else 0 for j in range(n)]
         for i in range(n)]
    for col in range(n):
        piv = None
        for rr in range(col, n):
            if A[rr][col] % p != 0:
                piv = rr; break
        A[col], A[piv] = A[piv], A[col]
        iv = inv(A[col][col], p)
        A[col] = [(x * iv) % p for x in A[col]]
        for rr in range(n):
            if rr != col and A[rr][col] % p != 0:
                f = A[rr][col] % p
                A[rr] = [(A[rr][t] - f * A[col][t]) % p for t in range(2 * n)]
    return [[A[i][j + n] % p for j in range(n)] for i in range(n)]


def is_square(x, p):
    x %= p
    if x == 0:
        return True
    return pow(x, (p - 1) // 2, p) == 1


def irr_quads(p, count):
    """Monic irreducible x^2 + Bx + C, returned as (c1, c0) with x^2 == c1 x + c0."""
    res = []
    for B in range(p):
        for C in range(p):
            if not is_square((B * B - 4 * C) % p, p):
                res.append(((-B) % p, (-C) % p))
                if len(res) >= count:
                    return res
    return res


def xpow_mod_quad(c1, c0, p, kmax):
    """x^k mod (x^2 - c1 x - c0) as (coef_1, coef_x) for k = 0..kmax."""
    seq = [(1, 0)]
    if kmax >= 1:
        seq.append((0, 1))
    while len(seq) <= kmax:
        a0, a1 = seq[-1]
        seq.append(((a1 * c0) % p, (a0 + a1 * c1) % p))
    return seq[:kmax + 1]


def main():
    p, d = map(int, sys.stdin.read().split()[:2])
    n = d + 1
    T = 2 * d + 1  # number of product coefficients

    if p >= T:
        pts = list(range(T))
        quads = []
    else:
        pts = list(range(p))
        Rrem = T - p
        nq = (Rrem + 1) // 2
        quads = irr_quads(p, nq)

    D = len(pts) + 2 * len(quads)  # total covered dimension (>= T)

    U = []
    V = []
    Grows = []  # residue-coefficient linear forms over product columns

    # linear moduli
    for a in pts:
        col = len(U)
        row = [pow(a, i, p) for i in range(n)]
        U.append(row[:]); V.append(row[:])
        Grows.append({col: 1})

    # quadratic moduli (Karatsuba inside F_p[x]/(M): 3 products)
    for (c1, c0) in quads:
        seq = xpow_mod_quad(c1, c0, p, d)
        r0 = [seq[k][0] % p for k in range(n)]
        r1 = [seq[k][1] % p for k in range(n)]
        rs = [(r0[k] + r1[k]) % p for k in range(n)]
        cP1 = len(U); U.append(r0[:]); V.append(r0[:])
        cP2 = len(U); U.append(r1[:]); V.append(r1[:])
        cP3 = len(U); U.append(rs[:]); V.append(rs[:])
        # coeff0 = P1 + c0*P2 ; coeff1 = -P1 + (c1-1)*P2 + P3
        Grows.append({cP1: 1, cP2: c0 % p})
        Grows.append({cP1: (-1) % p, cP2: (c1 - 1) % p, cP3: 1})

    Rtot = len(U)

    # G : D x Rtot  (residues as combinations of products)
    G = [[0] * Rtot for _ in range(D)]
    for i, gr in enumerate(Grows):
        for kk, vv in gr.items():
            G[i][kk] = vv % p

    # E : D x D  reduction matrix, SAME row order (linear residues, then quad coeff0/coeff1)
    E = []
    for a in pts:
        E.append([pow(a, k, p) for k in range(D)])
    for (c1, c0) in quads:
        seq = xpow_mod_quad(c1, c0, p, D - 1)
        E.append([seq[k][0] % p for k in range(D)])
        E.append([seq[k][1] % p for k in range(D)])
    Einv = mat_inv(E, p)

    # W = (Einv * G)[first T rows]
    W = []
    for m in range(T):
        rowm = Einv[m]
        wrow = [0] * Rtot
        for k in range(D):
            rk = rowm[k]
            if rk:
                Gk = G[k]
                for c in range(Rtot):
                    if Gk[c]:
                        wrow[c] = (wrow[c] + rk * Gk[c]) % p
        W.append(wrow)

    out = [str(Rtot)]
    out += [' '.join(str(x % p) for x in row) for row in U]
    out += [' '.join(str(x % p) for x in row) for row in V]
    out += [' '.join(str(x % p) for x in row) for row in W]
    sys.stdout.write('\n'.join(out) + '\n')


main()
