# TIER: trivial
# Reproduce the grader's internal baseline: a crude SATURATING logistic with the plateau
# fixed at K0 = 1.05*max(train), sigmoid slope/midpoint by a logit-linear fit.  It saturates
# (so it does not blow up) but its plateau under-shoots the true stationary density ->
# extrapolates about as well as the baseline -> Ratio ~= 0.10.
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
    pts = []
    for _ in range(n):
        t = float(toks[idx]); nn = float(toks[idx + 1])
        idx += 2
        pts.append((t, nn))
    nmax = max(nn for _, nn in pts)
    K0 = 1.05 * nmax
    rows, z = [], []
    for (t, nn) in pts:
        p = min(max(nn / K0, 1e-6), 1.0 - 1e-6)
        rows.append([1.0, t])
        z.append(math.log(p / (1.0 - p)))
    a0, a1 = lstsq(rows, z)
    print("%.10g / (1 + exp(-(%.10g + %.10g * t)))" % (K0, a0, a1))


if __name__ == "__main__":
    main()
