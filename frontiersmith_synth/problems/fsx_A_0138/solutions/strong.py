# TIER: strong
# Try slicing along ALL THREE axes; for each axis sum the exact slice ranks and
# keep the cheapest decomposition.  tensor-rank <= min_axis sum_slice rank(slice).
# This strictly beats the fixed-axis greedy whenever the last axis is not the
# optimal one to slice.  Still far above the (unknown, overcomplete) true rank,
# so ample head-room remains.
import sys
from fractions import Fraction


def rank_factor(M, nr, nc):
    Rm = [[Fraction(M[i][j]) for j in range(nc)] for i in range(nr)]
    pivots = []
    r = 0
    for col in range(nc):
        piv = None
        for i in range(r, nr):
            if Rm[i][col] != 0:
                piv = i; break
        if piv is None:
            continue
        Rm[r], Rm[piv] = Rm[piv], Rm[r]
        pv = Rm[r][col]
        Rm[r] = [x / pv for x in Rm[r]]
        for i in range(nr):
            if i != r and Rm[i][col] != 0:
                f = Rm[i][col]
                Rm[i] = [a - f * b for a, b in zip(Rm[i], Rm[r])]
        pivots.append(col)
        r += 1
        if r == nr:
            break
    terms = []
    for s in range(r):
        u = [Fraction(M[i][pivots[s]]) for i in range(nr)]
        v = Rm[s][:]
        terms.append((u, v))
    return terms


def decomp_axis(T, I, J, K, axis):
    """Return list of full (a in Q^I, b in Q^J, c in Q^K) rank-1 terms obtained
    by slicing along `axis` (0=I, 1=J, 2=K)."""
    terms = []
    if axis == 2:
        for k in range(K):
            M = [[T[i][j][k] for j in range(J)] for i in range(I)]
            for u, v in rank_factor(M, I, J):
                c = [Fraction(0)] * K; c[k] = Fraction(1)
                terms.append((list(u), list(v), c))
    elif axis == 0:
        for i in range(I):
            M = [[T[i][j][k] for k in range(K)] for j in range(J)]  # J x K
            for u, v in rank_factor(M, J, K):
                a = [Fraction(0)] * I; a[i] = Fraction(1)
                terms.append((a, list(u), list(v)))
    else:  # axis == 1
        for j in range(J):
            M = [[T[i][j][k] for k in range(K)] for i in range(I)]  # I x K
            for u, v in rank_factor(M, I, K):
                b = [Fraction(0)] * J; b[j] = Fraction(1)
                terms.append((list(u), b, list(v)))
    return terms


def emit(terms):
    out = [str(len(terms))]
    for a, b, c in terms:
        out.append(" ".join(str(x) for x in a))
        out.append(" ".join(str(x) for x in b))
        out.append(" ".join(str(x) for x in c))
    sys.stdout.write("\n".join(out) + "\n")


def main():
    inp = sys.stdin.read().split()
    I, J, K = int(inp[0]), int(inp[1]), int(inp[2])
    idx = 3
    T = [[[0] * K for _ in range(J)] for _ in range(I)]
    for i in range(I):
        for j in range(J):
            for k in range(K):
                T[i][j][k] = int(inp[idx]); idx += 1

    best = None
    for axis in (0, 1, 2):
        d = decomp_axis(T, I, J, K, axis)
        if best is None or len(d) < len(best):
            best = d
    emit(best)


if __name__ == "__main__":
    main()
