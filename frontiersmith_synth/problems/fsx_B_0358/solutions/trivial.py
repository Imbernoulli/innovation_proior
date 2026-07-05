# TIER: trivial
# Reproduce the grader's internal baseline: a single product power-law fitted by
# log-linear regression, D = k * A^c1 * H^c2.  Separable, single fixed exponents, no
# pooled super-linear driver -> extrapolates about as well as the baseline -> Ratio ~= 0.10.
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
        a = float(toks[idx]); h = float(toks[idx + 1]); d = float(toks[idx + 2])
        idx += 3
        rows.append([1.0, math.log(a), math.log(h)])
        ylog.append(math.log(d))
    c0, c1, c2 = lstsq(rows, ylog)
    k = math.exp(c0)
    print("%.10g * A ** %.10g * H ** %.10g" % (k, c1, c2))


if __name__ == "__main__":
    main()
