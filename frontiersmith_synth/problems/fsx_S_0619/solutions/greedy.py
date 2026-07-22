# TIER: greedy
# The obvious "better than schoolbook" move a strong coder writes: recursive
# Karatsuba splitting.  It is field-INDEPENDENT -- it never looks at p -- so it
# leaves products on the table exactly where the field structure could be exploited.
import sys


def kara(na, nb):
    """Return integer bilinear scheme (U r x na, V r x nb, W nc x r), nc=na+nb-1."""
    nc = na + nb - 1
    if na == 1 or nb == 1:
        r = na * nb
        U = []
        V = []
        W = [[0] * r for _ in range(nc)]
        idx = 0
        for i in range(na):
            for j in range(nb):
                u = [0] * na; u[i] = 1; U.append(u)
                v = [0] * nb; v[j] = 1; V.append(v)
                W[i + j][idx] = 1
                idx += 1
        return U, V, W

    n = na  # our use: na == nb
    k = (n + 1) // 2
    U0, V0, W0 = kara(k, k)          # A_lo * B_lo
    U1, V1, W1 = kara(k, k)          # (A_lo+A_hi)*(B_lo+B_hi)
    U2, V2, W2 = kara(n - k, n - k)  # A_hi * B_hi
    r0, r1, r2 = len(U0), len(U1), len(U2)
    R = r0 + r1 + r2
    nc = 2 * n - 1
    U = [[0] * n for _ in range(R)]
    V = [[0] * n for _ in range(R)]
    W = [[0] * R for _ in range(nc)]

    # M0: low block, shift 0 (and subtracted in the middle at shift k)
    for c in range(r0):
        for j in range(k):
            U[c][j] = U0[c][j]; V[c][j] = V0[c][j]
        for t in range(2 * k - 1):
            W[t][c] += W0[t][c]
            W[k + t][c] -= W0[t][c]
    # M1: (lo+hi) operand, shift k
    for c in range(r1):
        gc = r0 + c
        for j in range(k):
            U[gc][j] += U1[c][j]
            V[gc][j] += V1[c][j]
            if j + k < n:
                U[gc][j + k] += U1[c][j]
                V[gc][j + k] += V1[c][j]
        for t in range(2 * k - 1):
            W[k + t][gc] += W1[t][c]
    # M2: high block, shift 2k (and subtracted in the middle at shift k)
    for c in range(r2):
        gc = r0 + r1 + c
        for j in range(n - k):
            U[gc][k + j] += U2[c][j]; V[gc][k + j] += V2[c][j]
        for t in range(2 * (n - k) - 1):
            W[2 * k + t][gc] += W2[t][c]
            W[k + t][gc] -= W2[t][c]
    return U, V, W


def main():
    p, d = map(int, sys.stdin.read().split()[:2])
    n = d + 1
    U, V, W = kara(n, n)
    r = len(U)
    out = [str(r)]
    out += [' '.join(str(x % p) for x in row) for row in U]
    out += [' '.join(str(x % p) for x in row) for row in V]
    out += [' '.join(str(x % p) for x in row) for row in W]
    sys.stdout.write('\n'.join(out) + '\n')


main()
