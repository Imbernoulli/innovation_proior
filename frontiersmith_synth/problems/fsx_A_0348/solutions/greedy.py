# TIER: greedy
# Slice the tensor along mode 1 (the i-axis).  Each slice T[i,:,:] is a J x K matrix;
# factor it EXACTLY over the rationals into rank(slice) rank-1 terms and lift back with e_i.
#   R_greedy = sum_i rank(T[i,:,:])   <=  I*J = trivial baseline.
# A fixed single-mode heuristic -- does not shop across modes.
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
    # M: p x q Fraction matrix -> list of (u len p, v len q) with sum u(x)v == M, count = rank.
    p = len(M); q = len(M[0]) if p else 0
    Rr, pivcols = rref(M)
    out = []
    for s, pc in enumerate(pivcols):
        u = [M[i][pc] for i in range(p)]   # original pivot column
        v = Rr[s][:]                        # reduced row s = coordinates
        out.append((u, v))
    return out


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

    terms = []
    for i in range(I):
        M = [[Fraction(T[i][j][k]) for k in range(K)] for j in range(J)]  # J x K
        for (u, v) in factor(M):   # u len J, v len K
            a = [Fraction(0)] * I; a[i] = Fraction(1)
            terms.append((a, u, v))

    def tok(x):
        return str(x.numerator) if x.denominator == 1 else "%d/%d" % (x.numerator, x.denominator)

    out = [str(len(terms))]
    for (a, b, c) in terms:
        out.append(" ".join(tok(x) for x in (list(a) + list(b) + list(c))))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
