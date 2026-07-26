# TIER: strong
# The insight: the school's motion decomposes into (offset, steady current,
# seasonal eddy, rip-current growth) -- but the eddy's period is unknown, so
# a *single* linear regression over {1, t, t^3} alone leaves the periodic
# component in the residual, and vice versa. Grid-search the eddy period;
# for each candidate period the remaining model IS linear in its
# coefficients (offset, drift, cubic growth, sin/cos eddy amplitude), so fit
# all five jointly via a normal-equations solve and keep the period that
# minimises training residual. Working in NORMALISED time u=t/T_train keeps
# the (1, u, u^3, sin, cos) design matrix well conditioned. The resulting
# smooth closed-form law both extrapolates the drift correctly AND changes
# gradually, so the checker's hysteresis-band finger relocates near the free
# budget instead of paying repeated fees.
import sys, math


def solve5(A, b):
    """Gaussian elimination with partial pivoting on a 5x5 system A x = b."""
    n = 5
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-12:
            M[piv][c] += 1e-9
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        for r in range(c + 1, n):
            f = M[r][c] / pv
            if f == 0.0:
                continue
            for k in range(c, n + 1):
                M[r][k] -= f * M[c][k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / M[i][i]
    return x


def fit(us, ys, w):
    """Fit y ~ a + b*u + c*u^3 + d*sin(w*u) + e*cos(w*u); return (coeffs, RSS)."""
    n = len(us)
    basis = []
    for u in us:
        su, cu = math.sin(w * u), math.cos(w * u)
        basis.append((1.0, u, u ** 3, su, cu))
    A = [[0.0] * 5 for _ in range(5)]
    bb = [0.0] * 5
    for row, y in zip(basis, ys):
        for i in range(5):
            bb[i] += row[i] * y
            for j in range(5):
                A[i][j] += row[i] * row[j]
    coeffs = solve5(A, bb)
    rss = 0.0
    for row, y in zip(basis, ys):
        pred = sum(coeffs[k] * row[k] for k in range(5))
        rss += (pred - y) ** 2
    return coeffs, rss


def main():
    data = sys.stdin.read().split()
    T_train = int(data[0])
    idx = 5
    ts, ys = [], []
    for _ in range(T_train):
        i = int(data[idx]); obs = int(data[idx + 1]); idx += 2
        ts.append(float(i)); ys.append(float(obs))
    us = [t / T_train for t in ts]

    def w_of(P):
        return 2.0 * math.pi * T_train / P

    best = None  # (rss, P, coeffs, w)
    P = 20.0
    while P <= 90.0 + 1e-9:
        w = w_of(P)
        coeffs, rss = fit(us, ys, w)
        if best is None or rss < best[0]:
            best = (rss, P, coeffs, w)
        P += 1.0
    P0 = best[1]

    Plo, Phi = max(20.0, P0 - 1.5), min(90.0, P0 + 1.5)
    P = Plo
    while P <= Phi + 1e-9:
        w = w_of(P)
        coeffs, rss = fit(us, ys, w)
        if rss < best[0]:
            best = (rss, P, coeffs, w)
        P += 0.05

    _, P_hat, (a, b, c, d, e), w_hat = best

    T = T_train
    expr = ("EXPR %.8f + %.8f*(t/%d) + %.8f*(t/%d)**3 "
            "+ %.8f*sin(%.10f*(t/%d)) + %.8f*cos(%.10f*(t/%d))"
            % (a, b, T, c, T, d, w_hat, T, e, w_hat, T))
    print(expr)


if __name__ == "__main__":
    main()
