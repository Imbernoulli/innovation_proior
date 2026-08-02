# TIER: strong
# The insight: b (self-heating rate) and h (cooling coefficient) are GIVEN --
# they are not decoration.  At steady state, generation equals removal:
#   A*exp(b*T_i) = h*(T_i - Ta_i)   =>   A = h*(T_i-Ta_i)*exp(-b*T_i)
# so EVERY training row hands us a noisy estimate of the same hidden
# prefactor A.  We do not fit the temperature curve directly and extrapolate
# its shape (that is the greedy trap); instead we take a robust (median)
# estimate of A from the generation/removal balance, then ask directly: at
# what ambient does generation first reach the cooling ceiling Hmax (also
# given)?  That is exactly the elbow condition A*exp(b*(Ta+Hmax/h)) = Hmax,
# solved in closed form for Ta -- the true critical ambient, comparing
# generation and removal RATES rather than reading off the temperature
# trend.  Past that ambient we report the given Tfail directly instead of
# extrapolating a smooth curve.  Below it, a light local quadratic still
# captures the (genuinely smooth, bounded) pre-threshold branch.
import sys, math


def solve(M_in, rhs_in):
    n = len(M_in)
    M = [row[:] + [rhs_in[i]] for i, row in enumerate(M_in)]
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
        print("THRESH 1e9"); print("BELOW 0.0"); print("ABOVE 0.0"); return
    n = int(data[0])
    rest = data[2:]
    b = float(rest[0]); h = float(rest[1]); Hmax = float(rest[2]); Tfail = float(rest[3])
    rows_tok = rest[4:]
    Tas = [float(rows_tok[2 * i]) for i in range(n)]
    Ts = [float(rows_tok[2 * i + 1]) for i in range(n)]

    # robust estimate of the hidden self-heating prefactor A from the exact
    # steady-state balance, row by row.  Rows at LOW Ta have a tiny excess
    # (T-Ta) so their noise-to-signal ratio is terrible; rows near the
    # top of the observed range have a much bigger excess and give a far
    # more reliable read on A -- so restrict the (still robust, median)
    # estimate to the upper half of the training ambients by Ta.
    order = sorted(range(n), key=lambda i: Tas[i])
    top = order[len(order) // 2:] if len(order) >= 4 else order
    A_ests = []
    for i in top:
        Ta, T = Tas[i], Ts[i]
        dT = T - Ta
        if dT <= 1e-6:
            continue
        bt = max(-700.0, min(700.0, -b * T))
        A_ests.append(h * dT * math.exp(bt))
    if not A_ests:
        A_hat = 1e-6
    else:
        A_ests.sort()
        m = len(A_ests)
        A_hat = A_ests[m // 2] if m % 2 else 0.5 * (A_ests[m // 2 - 1] + A_ests[m // 2])
    A_hat = max(A_hat, 1e-12)

    # elbow condition: A*exp(b*(Ta+Hmax/h)) = Hmax  =>  Ta_crit closed form
    Ta_crit_hat = math.log(Hmax / A_hat) / b - Hmax / h
    # sanity floor: every training row is an OBSERVED stable steady state, so
    # the true threshold cannot lie at or below the highest ambient we saw
    Ta_crit_hat = max(Ta_crit_hat, max(Tas) + 0.5)

    # local smooth fit for the (genuinely smooth, bounded) pre-threshold branch
    feats = [[1.0, x, x * x] for x in Tas]
    mdim = 3
    Amat = [[0.0] * mdim for _ in range(mdim)]
    bvec = [0.0] * mdim
    for x, y in zip(feats, Ts):
        for r in range(mdim):
            bvec[r] += x[r] * y
            for c in range(mdim):
                Amat[r][c] += x[r] * x[c]
    c0, c1, c2 = solve(Amat, bvec)

    print("THRESH %.10g" % Ta_crit_hat)
    print("BELOW %.10g + %.10g*Ta + %.10g*Ta**2" % (c0, c1, c2))
    print("ABOVE %.10g" % Tfail)


if __name__ == "__main__":
    main()
