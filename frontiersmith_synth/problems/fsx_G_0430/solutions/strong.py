# TIER: strong
# Recover the true band-pass rational form
#     P(f) = K * f^2 / ( (f0^2 - f^2)^2 + (f0*f/Q)^2 )
# Grid-search the hidden (f0, Q); for each candidate the model is LINEAR in K,
# so solve K in closed form and keep the best training residual.  Emit the
# rational expression.  Matching the true roll-off makes it extrapolate far
# better out-of-band than the Lorentzian, but the coarse grid + measurement
# noise + irreducible held-out noise leave headroom below 1.0.
import sys


def main():
    data = sys.stdin.read().split("\n")
    n = int(data[0].split()[0])
    F = []; Y = []
    for ln in data[1:1 + n]:
        p = ln.split()
        if len(p) >= 2:
            F.append(float(p[0])); Y.append(float(p[1]))

    def frange(lo, hi, step):
        v = lo; out = []
        while v <= hi + 1e-9:
            out.append(v); v += step
        return out

    best = None
    for f0 in frange(0.70, 1.40, 0.025):
        A = f0 * f0
        for Q in frange(2.0, 7.0, 0.25):
            invQ = f0 / Q
            num = 0.0; den = 0.0
            g = []
            ok = True
            for fx in F:
                d = (A - fx * fx) ** 2 + (invQ * fx) ** 2
                if d <= 0:
                    ok = False; break
                gi = fx * fx / d
                g.append(gi)
            if not ok:
                continue
            for gi, yv in zip(g, Y):
                num += gi * yv; den += gi * gi
            if den <= 1e-18:
                continue
            K = num / den
            res = 0.0
            for gi, yv in zip(g, Y):
                e = K * gi - yv
                res += e * e
            if best is None or res < best[0]:
                best = (res, f0, Q, K)

    _, f0, Q, K = best
    A = f0 * f0
    invQ = f0 / Q
    # P = K*f^2 / ((A - f^2)^2 + (invQ*f)^2)
    print("%r * f**2 / ( ( %r - f**2 )**2 + ( %r * f )**2 )" % (K, A, invQ))


if __name__ == "__main__":
    main()
