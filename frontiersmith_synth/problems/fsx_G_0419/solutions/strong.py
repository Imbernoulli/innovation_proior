# TIER: strong
# Discover the correct SATURATING family -- the Gompertz growth law
#     N(t) = K * exp( -b * exp( -c * t ) )
# For a trial (K, c) the law linearises: let u = log( log(K/N) ) = log(b) - c*t, so log(b) is a
# closed-form intercept.  A 2-D grid over the plateau K and the growth rate c, each scored by the
# RELATIVE (multiplicative-noise) train error, recovers the asymmetric S-curve; because it is the
# right family it extrapolates the late stationary plateau far better than a symmetric logistic.
# Irreducible measurement noise keeps the held-out error above zero (Ratio < 1).
import sys, math


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

    best = None
    Ksteps = 80
    Csteps = 80
    for ik in range(Ksteps + 1):
        K = nmax * (1.02 + (3.5 - 1.02) * ik / Ksteps)
        # precompute u_i = log(log(K/N_i)) ; requires 0 < N_i < K
        us = []
        ok = True
        for (t, nn) in pts:
            r = K / nn
            if r <= 1.0:
                ok = False
                break
            lr = math.log(r)          # = b*exp(-c t) > 0
            if lr <= 0.0:
                ok = False
                break
            us.append((t, nn, math.log(lr)))
        if not ok:
            continue
        for ic in range(Csteps + 1):
            c = 0.03 + (0.30 - 0.03) * ic / Csteps
            # u = log(b) - c*t  ->  intercept = mean(u + c*t) = log(b)
            logb = sum(u + c * t for (t, _n, u) in us) / len(us)
            b = math.exp(logb)
            if b <= 0.0:
                continue
            err = 0.0
            for (t, nn, _u) in us:
                pred = K * math.exp(-b * math.exp(-c * t))
                err += (pred - nn) ** 2             # absolute train error (robust for plateau)
            if best is None or err < best[0]:
                best = (err, K, b, c)

    if best is None:
        # degenerate fallback: flat plateau at the largest observation
        print("%.10g" % nmax)
        return
    _, K, b, c = best
    print("%.10g * exp(-%.10g * exp(-%.10g * t))" % (K, b, c))


if __name__ == "__main__":
    main()
