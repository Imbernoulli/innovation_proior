# TIER: greedy
# The obvious recipe: treat this as generic symbolic regression.  Assume a single
# free power-law monomial  F = k * rho^p1 * V^p2 * D^p3 * mu^p4  and recover the
# five parameters by ordinary least squares in log space (log F linear in the log
# inputs).  This fits the training notebook well -- but in that notebook rho and
# mu barely move (one fluid, one tunnel), so their columns are nearly flat and
# their exponents are NOT identifiable: the regression pins them to noise.  On the
# held-out grid (a different fluid, a different scale) rho and mu shift by orders
# of magnitude and those bogus exponents blow the prediction up.  The recipe never
# sees that the units already fix p1=1, p2=2, p3=2, p4=0.
import sys, math


def solve(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        d = M[c][c]
        if abs(d) < 1e-18:
            d = 1e-18
        for r in range(n):
            if r == c:
                continue
            f = M[r][c] / d
            for k in range(c, n + 1):
                M[r][k] -= f * M[c][k]
    return [M[i][n] / (M[i][i] if abs(M[i][i]) > 1e-18 else 1e-18) for i in range(n)]


def main():
    data = sys.stdin.read().split()
    if not data:
        print("0.0"); return
    n = int(data[0])
    vals = data[2:]
    feats = []  # [1, ln rho, ln V, ln D, ln mu]
    y = []
    for i in range(n):
        rho = float(vals[5 * i]); V = float(vals[5 * i + 1])
        D = float(vals[5 * i + 2]); mu = float(vals[5 * i + 3])
        F = float(vals[5 * i + 4])
        feats.append([1.0, math.log(rho), math.log(V), math.log(D), math.log(mu)])
        y.append(math.log(F))
    m = 5
    A = [[0.0] * m for _ in range(m)]
    b = [0.0] * m
    for x, yy in zip(feats, y):
        for r in range(m):
            b[r] += x[r] * yy
            for c in range(m):
                A[r][c] += x[r] * x[c]
    coef = solve(A, b)
    c0, p1, p2, p3, p4 = coef
    k = math.exp(c0)
    print("%.10g * rho**(%.10g) * V**(%.10g) * D**(%.10g) * mu**(%.10g)"
          % (k, p1, p2, p3, p4))


if __name__ == "__main__":
    main()
