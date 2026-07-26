# TIER: strong
# The insight: divide out the catalyst level. If the true law is
# rate = Vmax*C*S/(Km+S), then the SPECIFIC rate y = rate/C is independent of
# C -- it is the SAME saturating curve of S alone for every regime. Pooling
# y = rate/C across ALL catalyst regimes (multi-regime consistency) turns a
# handful of thin per-regime curves into one well-populated curve, which is
# exactly what makes the saturation constant Km identifiable despite noise.
#
# Fit that pooled curve with a 1-D profile search over Km (log-spaced): for
# each candidate Km, Vmax has a closed-form least-squares solution given
# x = S/(Km+S), y ~= Vmax*x  =>  Vmax = sum(x*y)/sum(x*x). Keep the (Vmax,Km)
# with lowest pooled squared error. Unlike a bilinear fit, this form is BOUNDED
# by Vmax*C as S -> infinity, so it generalises correctly deep into the
# held-out extrapolation and to catalyst levels never seen in training.
import sys


def main():
    data = sys.stdin.read().split()
    if len(data) < 3:
        print("1.0")
        return
    n_regimes = int(data[1])
    n_pts = int(data[2])
    vals = data[3:]
    n = n_regimes * n_pts
    pts = []
    for i in range(n):
        S = float(vals[3 * i])
        C = float(vals[3 * i + 1])
        r = float(vals[3 * i + 2])
        if C > 0:
            pts.append((S, r / C))

    if not pts:
        print("1.0")
        return

    Ss = [S for S, y in pts]
    smin, smax = min(Ss), max(Ss)
    lo = max(1e-3, smin * 0.02)
    hi = smax * 60.0

    best = None
    for i in range(140):
        frac = i / 139.0
        Km = lo * (hi / lo) ** frac
        sxx = sxy = 0.0
        for S, y in pts:
            x = S / (Km + S)
            sxx += x * x
            sxy += x * y
        if sxx < 1e-12:
            continue
        Vmax = sxy / sxx
        if Vmax <= 0:
            continue
        se = 0.0
        for S, y in pts:
            pred = Vmax * S / (Km + S)
            se += (pred - y) ** 2
        if best is None or se < best[0]:
            best = (se, Vmax, Km)

    if best is None:
        Vmax, Km = 1.0, 1.0
    else:
        Vmax, Km = best[1], best[2]

    print("%.8f * C * S / ( %.8f + S )" % (Vmax, Km))


if __name__ == "__main__":
    main()
