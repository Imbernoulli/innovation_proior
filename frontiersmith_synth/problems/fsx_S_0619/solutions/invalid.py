# TIER: invalid
# Emits a syntactically well-formed but WRONG scheme: a single product a_0*b_0
# routed only to c_0.  The bilinear identity fails for every other coefficient,
# so the checker must score 0.
import sys


def main():
    p, d = map(int, sys.stdin.read().split()[:2])
    n = d + 1
    nc = 2 * d + 1
    r = 1
    U = [[1] + [0] * (n - 1)]
    V = [[1] + [0] * (n - 1)]
    W = [[0] for _ in range(nc)]
    W[0][0] = 1
    out = [str(r)]
    out += [' '.join(map(str, row)) for row in U]
    out += [' '.join(map(str, row)) for row in V]
    out += [' '.join(map(str, row)) for row in W]
    sys.stdout.write('\n'.join(out) + '\n')


main()
