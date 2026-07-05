# TIER: strong
# Shop across ALL THREE slicing modes: decompose the tensor by slicing along mode 1, 2, and 3,
# exactly factor every slice over the rationals, and keep whichever mode yields the FEWEST rank-1
# terms.  R_strong = min_m sum_{s} rank(slice_m,s)  <=  R_greedy  (mode-1 only).
# Still a slice heuristic -- it does NOT reach the (unknown, overcomplete) true tensor rank, so
# it leaves headroom below itself.
import sys
from fractions import Fraction


def rref(M):
    R = [row[:] for row in M]
    rows = len(R); cols = len(R[0]) if rows else 0
    pivcols = []; r = 0
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if R[i][c] != 0:
                piv = i; break
        if piv is None:
            continue
        R[r], R[piv] = R[piv], R[r]
        pv = R[r][c]
        R[r] = [x / pv for x in R[r]]
        for i in range(rows):
            if i != r and R[i][c] != 0:
                f = R[i][c]
                R[i] = [a - f * b for a, b in zip(R[i], R[r])]
        pivcols.append(c); r += 1
        if r == rows:
            break
    return R, pivcols


def factor(M):
    p = len(M); q = len(M[0]) if p else 0
    Rr, pivcols = rref(M)
    out = []
    for s, pc in enumerate(pivcols):
        u = [M[i][pc] for i in range(p)]
        v = Rr[s][:]
        out.append((u, v))
    return out


def unit(n, i):
    e = [Fraction(0)] * n; e[i] = Fraction(1); return e


def decompose(T, I, J, K, mode):
    """Return list of full (a,b,c) terms, slicing along `mode` (1,2,3)."""
    terms = []
    if mode == 1:                     # slices T[i,:,:] are J x K
        for i in range(I):
            M = [[Fraction(T[i][j][k]) for k in range(K)] for j in range(J)]
            for (u, v) in factor(M):  # u len J, v len K
                terms.append((unit(I, i), u, v))
    elif mode == 2:                   # slices T[:,j,:] are I x K
        for j in range(J):
            M = [[Fraction(T[i][j][k]) for k in range(K)] for i in range(I)]
            for (u, v) in factor(M):  # u len I, v len K
                terms.append((u, unit(J, j), v))
    else:                             # mode 3: slices T[:,:,k] are I x J
        for k in range(K):
            M = [[Fraction(T[i][j][k]) for j in range(J)] for i in range(I)]
            for (u, v) in factor(M):  # u len I, v len J
                terms.append((u, v, unit(K, k)))
    return terms


def main():
    toks = sys.stdin.read().split()
    idx = 0
    I = int(toks[idx]); idx += 1
    J = int(toks[idx]); idx += 1
    K = int(toks[idx]); idx += 1
    T = [[[0] * K for _ in range(J)] for _ in range(I)]
    for i in range(I):
        for j in range(J):
            for k in range(K):
                T[i][j][k] = int(toks[idx]); idx += 1

    best = None
    for mode in (1, 2, 3):
        cand = decompose(T, I, J, K, mode)
        if best is None or len(cand) < len(best):
            best = cand

    def tok(x):
        return str(x.numerator) if x.denominator == 1 else "%d/%d" % (x.numerator, x.denominator)

    out = [str(len(best))]
    for (a, b, c) in best:
        out.append(" ".join(tok(x) for x in (list(a) + list(b) + list(c))))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
