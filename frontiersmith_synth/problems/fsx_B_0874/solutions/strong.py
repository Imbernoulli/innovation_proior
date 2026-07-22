# TIER: strong
# The insight: a stationary wobble is only TRUSTWORTHY as "periodic" if the
# campaign actually contains enough of its cycles to resolve it -- so restrict
# the period search to periods no longer than ~0.42x the observed cycle span
# (requiring at least ~2.4 full cycles inside the window). Then compare that
# BEST resolvable periodic fit against the plain linear ephemeris: only if it
# clears a real significance bar (its SSE is well below the linear fit's) do
# we trust it as a genuine bounded wobble. If nothing in the resolvable range
# clears the bar, the leftover smooth trend is NOT explained by any
# candidate we're willing to call "periodic" -- so we instead fit it with an
# explicit, ever-GROWING quadratic (secular) term. This is exactly the
# growth-signature distinction the greedy periodogram never makes: bounded
# oscillation vs. unbounded drift, decided by whether a resolvable period
# exists, not by which curve merely fits best inside the training window
# (in the hard cases both fit about equally well there).
import sys, math


def lstsq(rows, y):
    m = len(rows[0])
    A = [[0.0] * m for _ in range(m)]
    b = [0.0] * m
    for x, yy in zip(rows, y):
        for r in range(m):
            b[r] += x[r] * yy
            for c in range(m):
                A[r][c] += x[r] * x[c]
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    n = m
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        d = M[c][c]
        if abs(d) < 1e-14:
            d = 1e-14
        for r in range(n):
            if r == c:
                continue
            f = M[r][c] / d
            for k in range(c, n + 1):
                M[r][k] -= f * M[c][k]
    return [M[i][n] / (M[i][i] if abs(M[i][i]) > 1e-14 else 1e-14) for i in range(n)]


def sse_of(rows, ys, coef):
    return sum((sum(c * x for c, x in zip(coef, row)) - y) ** 2 for row, y in zip(rows, ys))


def fit_lin(obs):
    rows = [[1.0, k] for k, _ in obs]
    ys = [v for _, v in obs]
    coef = lstsq(rows, ys)
    return coef, sse_of(rows, ys, coef)


def fit_quad(obs):
    rows = [[1.0, k, k * k] for k, _ in obs]
    ys = [v for _, v in obs]
    coef = lstsq(rows, ys)
    return coef, sse_of(rows, ys, coef)


def fit_per(obs, Mc):
    rows = [[1.0, k, math.sin(2 * math.pi * k / Mc), math.cos(2 * math.pi * k / Mc)]
             for k, _ in obs]
    ys = [v for _, v in obs]
    coef = lstsq(rows, ys)
    return coef, sse_of(rows, ys, coef)


def best_periodic(obs, Ktrainmax, hi_frac=0.42):
    Mgrid_lo = 3.0
    Mgrid_hi = max(6.0, hi_frac * Ktrainmax)
    ngrid = 90
    best = None
    for i in range(ngrid):
        Mc = Mgrid_lo + (Mgrid_hi - Mgrid_lo) * i / (ngrid - 1)
        coef, sse = fit_per(obs, Mc)
        if best is None or sse < best[0]:
            best = (sse, Mc, coef)
    return best


def solve(obs, Ktrainmax, sig_ratio=0.45):
    coef_lin, sse_lin = fit_lin(obs)
    sse_per, Mbest, coef_per = best_periodic(obs, Ktrainmax)
    if sse_per < sig_ratio * sse_lin:
        a, b, As, Ac = coef_per
        resid = [(k, t - (a + b * k + As * math.sin(2 * math.pi * k / Mbest)
                           + Ac * math.cos(2 * math.pi * k / Mbest)))
                  for k, t in obs]
        rows = [[1.0, k, k * k] for k, _ in resid]
        ys = [e for _, e in resid]
        coef_r = lstsq(rows, ys)
        sse_r = sse_of(rows, ys, coef_r)
        sse_r0 = sum(e * e for _, e in resid)
        if sse_r < 0.7 * sse_r0:
            a2, b2, c2 = coef_r
            return ("%.10g + %.10g*k + %.10g*k*k + %.10g*sin(2*pi*k/%.10g) + %.10g*cos(2*pi*k/%.10g)"
                     % (a + a2, b + b2, c2, As, Mbest, Ac, Mbest))
        else:
            return ("%.10g + %.10g*k + %.10g*sin(2*pi*k/%.10g) + %.10g*cos(2*pi*k/%.10g)"
                     % (a, b, As, Mbest, Ac, Mbest))
    else:
        a, b, c = fit_quad(obs)[0]
        return "%.10g + %.10g*k + %.10g*k*k" % (a, b, c)


def main():
    data = sys.stdin.read().split("\n")
    header = data[0].split()
    n = int(header[0])
    obs = []
    for ln in data[1:1 + n]:
        parts = ln.split()
        if len(parts) == 2:
            obs.append((int(parts[0]), float(parts[1])))
    if not obs:
        print("0.0")
        return
    Ktrainmax = max(k for k, _ in obs)
    print(solve(obs, Ktrainmax))


if __name__ == "__main__":
    main()
