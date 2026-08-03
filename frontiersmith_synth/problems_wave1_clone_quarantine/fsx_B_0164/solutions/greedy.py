# TIER: greedy
# Additive first-order model with the right monotonicity but the wrong curvature:
#   D = a + b*(1/T) + c*V     (linear in V, so it under-shoots the super-linear congestion
# growth at the busier held-out yard).  Beats the trivial baseline, loses to the strong law.
import sys, math


def lstsq(rows, y):
    m = len(rows[0])
    A = [[0.0] * m for _ in range(m)]
    bvec = [0.0] * m
    for r, yy in zip(rows, y):
        for i in range(m):
            bvec[i] += r[i] * yy
            for j in range(m):
                A[i][j] += r[i] * r[j]
    M = [A[i][:] + [bvec[i]] for i in range(m)]
    for c in range(m):
        piv = max(range(c, m), key=lambda rr: abs(M[rr][c]))
        M[c], M[piv] = M[piv], M[c]
        for r in range(m):
            if r != c:
                f = M[r][c] / M[c][c]
                for k in range(c, m + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][m] / M[i][i] for i in range(m)]


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    idx = 1
    rows, y = [], []
    for _ in range(n):
        t = float(toks[idx]); v = float(toks[idx + 1]); d = float(toks[idx + 2])
        idx += 3
        rows.append([1.0, 1.0 / t, v])
        y.append(d)
    a, b, c = lstsq(rows, y)
    print("%.10g + %.10g / T + %.10g * V" % (a, b, c))


if __name__ == "__main__":
    main()
