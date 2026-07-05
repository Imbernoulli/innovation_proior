# TIER: trivial
# Reproduce the grader's internal baseline: a single product power-law fitted by
# log-linear regression, D = k * T^c1 * V^c2.  Wrong functional shape (no floor, no
# additive split) -> extrapolates about as well as the baseline -> Ratio ~= 0.10.
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
    rows, ylog = [], []
    for _ in range(n):
        t = float(toks[idx]); v = float(toks[idx + 1]); d = float(toks[idx + 2])
        idx += 3
        rows.append([1.0, math.log(t), math.log(v)])
        ylog.append(math.log(d))
    c0, c1, c2 = lstsq(rows, ylog)
    k = math.exp(c0)
    print("%.10g * T ** %.10g * V ** %.10g" % (k, c1, c2))


if __name__ == "__main__":
    main()
