# TIER: strong
# The insight: the statement names THREE mechanisms, not two -- a constant
# standby loss, a temperature-modulated L^2 resistive loss, AND a core-
# saturation loss growing as L^p for some real exponent p. In the 20-60%
# commissioning band that third term is faint (a few percent near L=0.2,
# growing to ~15-20% by L=0.6) but it is NOT zero, and its curvature differs
# from L^2's. Rather than curve-fitting a flexible-but-wrong basis, grid-search
# the exponent p itself: for each candidate p, refit the full mechanistic
# basis {1, L^2, T*L^2, L^p} by least squares and score it by training
# residual. The p that best explains the faint in-band curvature carries a
# power-law term that keeps predicting correctly once L climbs past 0.6 into
# the overload band, where the same term can dominate -- exactly where the
# two-mechanism fit (greedy) or a generic flexible polynomial has no way to
# extrapolate the right shape.
import sys


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


def fit_for_p(L_list, T_list, y_list, p):
    feats = []
    for L, T in zip(L_list, T_list):
        feats.append([1.0, L * L, T * L * L, L ** p])
    m = 4
    A = [[0.0] * m for _ in range(m)]
    b = [0.0] * m
    for x, yy in zip(feats, y_list):
        for r in range(m):
            b[r] += x[r] * yy
            for c in range(m):
                A[r][c] += x[r] * x[c]
    coef = solve(A, b)
    rss = 0.0
    for x, yy in zip(feats, y_list):
        pred = sum(cc * xx for cc, xx in zip(coef, x))
        rss += (pred - yy) ** 2
    return coef, rss


def main():
    data = sys.stdin.read().split()
    if not data:
        print("0.0"); return
    n = int(data[0])
    vals = data[2:]
    Ls, Ts, ys = [], [], []
    for i in range(n):
        Ls.append(float(vals[3 * i]))
        Ts.append(float(vals[3 * i + 1]))
        ys.append(float(vals[3 * i + 2]))

    best = None
    p = 25  # candidate p = p/10, sweeps 2.5 .. 7.0 step 0.1
    while p <= 70:
        pv = p / 10.0
        coef, rss = fit_for_p(Ls, Ts, ys, pv)
        if best is None or rss < best[0] - 1e-9:
            best = (rss, pv, coef)
        p += 1
    rss, p_best, (c0, c1, c2, c3) = best
    print("%.10g + %.10g * L**2 + %.10g * T*L**2 + %.10g * L**%.6g"
          % (c0, c1, c2, c3, p_best))


if __name__ == "__main__":
    main()
