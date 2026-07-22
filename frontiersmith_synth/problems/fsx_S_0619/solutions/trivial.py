# TIER: trivial
# Schoolbook: one product a_i*b_j for every (i,j). r = (d+1)^2 = the checker baseline.
import sys


def main():
    p, d = map(int, sys.stdin.read().split()[:2])
    n = d + 1
    nc = 2 * d + 1
    U = []
    V = []
    prods = []  # (i,j) for each product column
    for i in range(n):
        for j in range(n):
            u = [0] * n; u[i] = 1
            v = [0] * n; v[j] = 1
            U.append(u); V.append(v)
            prods.append((i, j))
    r = len(prods)
    W = [[0] * r for _ in range(nc)]
    for c, (i, j) in enumerate(prods):
        W[i + j][c] = 1
    out = [str(r)]
    out += [' '.join(map(str, row)) for row in U]
    out += [' '.join(map(str, row)) for row in V]
    out += [' '.join(map(str, row)) for row in W]
    sys.stdout.write('\n'.join(out) + '\n')


main()
