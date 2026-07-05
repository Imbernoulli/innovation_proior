# TIER: greedy
# Slice the tensor along a FIXED axis (the last / K axis) and give each I x J
# slice an exact rank factorisation (column-row / CR decomposition over Q).
# Total rank = sum_k rank(slice_k) = K * min(I,J) generically -> well below nnz,
# but worse than the strong solver which picks the cheapest slicing axis.
import sys
from fractions import Fraction


def rank_factor(M, nr, nc):
    """M (nr x nc, ints).  Return list of (u,v): u in Q^nr, v in Q^nc with
    sum_s u_s v_s^T == M exactly, len == rank(M)."""
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

    terms = []
    for k in range(K):
        M = [[T[i][j][k] for j in range(J)] for i in range(I)]  # I x J slice
        for u, v in rank_factor(M, I, J):
            a = list(u)
            b = list(v)
            c = [Fraction(0)] * K; c[k] = Fraction(1)
            terms.append((a, b, c))
    emit(terms)


if __name__ == "__main__":
    main()
