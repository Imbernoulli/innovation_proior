# TIER: greedy
"""
The obvious first move: treat each sieve size L as its own independent
logistic curve, fit its midpoint m_L and width s_L from the training rows at
that L, then extrapolate m_L and s_L to the held-out (much larger) L
assuming the standard leading-order finite-size correction "~ C/L" --
i.e. fit m_L and s_L each as a SEPARATE straight line in 1/L (ordinary
least squares) and read off the value of that line at 1/L=0 (or at the
held-out L).  This is the textbook first guess and looks excellent on the
training range.  It never notices that the midpoint's drift and the width's
shrinkage are tied together by the SAME exponent 1/nu -- unless 1/nu
happens to be close to 1, a straight line in 1/L is the WRONG shape, and
because it is fit independently for the midpoint and the width, the two
extrapolation errors compound rather than cancel, so predictions at
L=128/512 drift off in an uncontrolled way.

Output uses only +, -, *, / and absv (no fractional power) -- a plain
rational function of 1/L standing in for the true, cleaner power law; absv
and a small floor keep the extrapolated width from crossing zero (a
defensive clamp any careful implementer would add), not from being right.
"""
import sys, math

OFF, AMP = 0.1, 0.8


def logit(z):
    z = min(1 - 1e-6, max(1e-6, z))
    return math.log(z / (1 - z))


def linfit(xs, ys):
    """Ordinary least squares y = a*x + b."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    a = sxy / sxx if sxx > 1e-12 else 0.0
    b = my - a * mx
    return a, b


def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    _t = int(data[idx]); idx += 1
    rows = []
    for _ in range(n):
        L = int(data[idx]); p = float(data[idx + 1]); pih = float(data[idx + 2])
        idx += 3
        rows.append((L, p, pih))

    by_L = {}
    for L, p, pih in rows:
        by_L.setdefault(L, []).append((p, pih))

    mids, widths, invLs = [], [], []
    for L, pts in sorted(by_L.items()):
        xs = [p for p, _ in pts]
        ys = [logit((pih - OFF) / AMP) for _, pih in pts]
        # logit(z) = (p - m)/s  =>  logit = a*p + b  with a = 1/s, b = -m/s
        a, b = linfit(xs, ys)
        if abs(a) < 1e-6:
            a = 1e-6 if a >= 0 else -1e-6
        s_L = 1.0 / a
        m_L = -b / a
        if s_L <= 1e-4:
            s_L = 1e-4
        mids.append(m_L)
        widths.append(s_L)
        invLs.append(1.0 / L)

    # standard leading-order finite-size guess: m_L, s_L each linear in 1/L,
    # fit SEPARATELY (no shared exponent)
    am, bm = linfit(invLs, mids)      # m(L) ~ bm + am/L
    aw, bw = linfit(invLs, widths)    # s(L) ~ bw + aw/L

    # a careful implementer clamps the extrapolated width away from zero/sign
    # flip with absv + a floor -- still wrong in SHAPE, just not catastrophic
    expr = ("%.6f + %.6f * sig( (p - (%.6f + %.6f / L)) / (0.01 + absv(%.6f + %.6f / L)) )") % (
        OFF, AMP, bm, am, bw, aw)
    print(expr)


if __name__ == "__main__":
    main()
