# TIER: greedy
# The obvious recipe: this looks like a smooth curve-fitting problem, so fit a
# generic smooth surface.  Regress m directly (ordinary least squares) on a
# quadratic-in-T, linear-in-log(h) basis:
#     m ~ c0 + c1*T + c2*T^2 + c3*L + c4*L*T + c5*L*T^2      (L = log h)
# This matches the training band well -- it is narrow and stays far from Tc.
# But the true law has a NON-ANALYTIC cusp |Tc-T|^beta at the transition and
# FOLDS onto |T-Tc| on the far side; a polynomial in T has neither feature, so
# extrapolating close to Tc (and past it, where T > Tc was never seen) sends
# the prediction far from the truth.  The recipe never notices there even IS
# a critical point to search for.
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
    feats = []  # [1, T, T^2, L, L*T, L*T^2]
    y = []
    for i in range(n):
        T = float(vals[3 * i]); h = float(vals[3 * i + 1]); m = float(vals[3 * i + 2])
        L = math.log(h)
        feats.append([1.0, T, T * T, L, L * T, L * T * T])
        y.append(m)
    k = 6
    A = [[0.0] * k for _ in range(k)]
    b = [0.0] * k
    for x, yy in zip(feats, y):
        for r in range(k):
            b[r] += x[r] * yy
            for c in range(k):
                A[r][c] += x[r] * x[c]
    c0, c1, c2, c3, c4, c5 = solve(A, b)
    print("%.10g + %.10g*T + %.10g*T**2 + %.10g*log(h) + %.10g*log(h)*T + %.10g*log(h)*T**2"
          % (c0, c1, c2, c3, c4, c5))


if __name__ == "__main__":
    main()
