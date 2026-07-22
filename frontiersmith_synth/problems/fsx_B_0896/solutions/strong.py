# TIER: strong
# The insight: hypothesize the hidden OFFSET directly. For a candidate offset
# multiplier k, form the shifted variable X = D - k*d (rejecting any k for
# which X is not positive on every training row) and check whether
# log(Q/rho) becomes LINEAR in log(X) -- i.e. RE-LINEARIZE after the shift,
# rather than fitting a flexible-but-wrong multiplicative basis directly.
# Grid-search k, and for each candidate fit a plain 1D linear regression of
# log(Q/rho) against log(X) (closed form: slope = exponent p, intercept =
# log C). Only the correct k makes this relationship clean -- its training
# RSS has a sharp minimum there, visibly lower than neighboring k. Because
# Q = C*rho*(D-k*d)^p is the TRUE functional form (not a local patch), the
# offset survives at the same relative importance the exponent p implies, so
# this law keeps predicting correctly on the held-out grid of much larger
# apertures and unseen grain sizes -- exactly where the multiplicative
# recipe (greedy) has no way to extrapolate the right shape.
import sys, math


def main():
    data = sys.stdin.read().split()
    if len(data) < 2:
        print("1.0"); return
    n = int(data[1])
    vals = data[2:]
    Ds, ds, rhos, qs = [], [], [], []
    for i in range(n):
        Ds.append(float(vals[4 * i]))
        ds.append(float(vals[4 * i + 1]))
        rhos.append(float(vals[4 * i + 2]))
        qs.append(float(vals[4 * i + 3]))

    best = None  # (rss, k, p, logC)
    kk = 10
    while kk <= 500:            # candidate k = kk/100, sweeps 0.10 .. 5.00
        kval = kk / 100.0
        Xs = [D - kval * d for D, d in zip(Ds, ds)]
        if min(Xs) <= 1e-9:
            kk += 2
            continue
        lx = [math.log(x) for x in Xs]
        ly = [math.log(q / rho) for q, rho in zip(qs, rhos)]
        m = len(lx)
        mx = sum(lx) / m; my = sum(ly) / m
        sxx = sum((x - mx) ** 2 for x in lx)
        sxy = sum((x - mx) * (y - my) for x, y in zip(lx, ly))
        if sxx < 1e-12:
            kk += 2
            continue
        p = sxy / sxx
        loga = my - p * mx
        rss = sum((loga + p * x - y) ** 2 for x, y in zip(lx, ly))
        if best is None or rss < best[0] - 1e-9:
            best = (rss, kval, p, loga)
        kk += 2

    # refine around the coarse minimum with a finer step
    _, k0, _, _ = best
    kk = int(round((k0 - 0.02) * 1000))
    kk_hi = int(round((k0 + 0.02) * 1000))
    while kk <= kk_hi:
        kval = kk / 1000.0
        if kval > 0.0:
            Xs = [D - kval * d for D, d in zip(Ds, ds)]
            if min(Xs) > 1e-9:
                lx = [math.log(x) for x in Xs]
                ly = [math.log(q / rho) for q, rho in zip(qs, rhos)]
                m = len(lx)
                mx = sum(lx) / m; my = sum(ly) / m
                sxx = sum((x - mx) ** 2 for x in lx)
                sxy = sum((x - mx) * (y - my) for x, y in zip(lx, ly))
                if sxx >= 1e-12:
                    p = sxy / sxx
                    loga = my - p * mx
                    rss = sum((loga + p * x - y) ** 2 for x, y in zip(lx, ly))
                    if rss < best[0] - 1e-9:
                        best = (rss, kval, p, loga)
        kk += 1

    rss, k_best, p_best, logC = best
    C = math.exp(logC)
    print("%.10g * rho * powv(D - %.10g * d, %.10g)" % (C, k_best, p_best))


if __name__ == "__main__":
    main()
