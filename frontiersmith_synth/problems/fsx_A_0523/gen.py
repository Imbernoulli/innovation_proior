#!/usr/bin/env python3
# gen.py <testId>  -- prints ONE instance of the district-boiler dispatch problem.
# Deterministic: everything seeded from testId only.
#
# Instance:
#   line 1:  T K
#   line 2:  D[0..T-1]                          (integer heat demand per step)
#   next K lines (one per boiler): C pmin c a b x ramp minup mindown
#     C     capacity (max output)
#     pmin  minimum stable output when online
#     c     no-load fuel burned each step the boiler is online
#     a     specific-fuel scale
#     b     sweet-spot curvature
#     x     sweet-spot load fraction (efficiency peaks here)
#     ramp  max |output change| between two consecutive online steps
#     minup / mindown  minimum consecutive online / offline steps
#
# Fuel(o) = c + a*o*(1 + b*(o/C - x)^2)  when online (o>0), else 0.
import sys, math, random


def gen(testId):
    rng = random.Random(20260710 + testId * 7919)
    # "trap" cases: deep demand troughs + heavy no-load + strong sweet-spot curvature
    trap = testId in (4, 5, 6, 7, 9)

    if testId <= 2:
        K, T = 8, 300
    elif testId <= 5:
        K, T = 9, 450
    elif testId <= 8:
        K, T = 11, 650
    else:
        K, T = 12, 850

    caps = [rng.randint(30, 100) for _ in range(K)]
    xmins = [round(rng.uniform(0.30, 0.45), 3) for _ in range(K)]
    xstars = [round(rng.uniform(0.66, 0.74), 3) for _ in range(K)]
    if trap:
        betas = [round(rng.uniform(3.0, 4.5), 3) for _ in range(K)]
        nlfrac = rng.uniform(1.3, 1.9)
        dmin_frac = rng.uniform(0.22, 0.30)
    else:
        betas = [round(rng.uniform(1.4, 2.4), 3) for _ in range(K)]
        nlfrac = rng.uniform(0.95, 1.35)
        dmin_frac = rng.uniform(0.30, 0.38)
    s0s = [round(rng.uniform(0.8, 1.3), 3) for _ in range(K)]
    nls = [round(nlfrac * s0s[i] * caps[i], 3) for i in range(K)]
    pmins = [round(xmins[i] * caps[i], 3) for i in range(K)]

    totalcap = sum(caps)
    dmax = int(round(0.63 * totalcap))
    dmin = int(round(dmin_frac * dmax))

    maxstep_frac = 0.03
    margin = 0.025
    ramps = [round(caps[i] * (maxstep_frac + margin), 3) for i in range(K)]
    minups = [rng.randint(4, 8) for _ in range(K)]
    mindowns = [rng.randint(4, 8) for _ in range(K)]

    # ---- demand: a single broad winter hump (unimodal) so commitment stays
    #      min-up/min-down feasible; skew, width and peak position vary per case.
    peak_pos = rng.uniform(0.40, 0.60)
    width = rng.uniform(0.08, 0.15)
    skew = rng.uniform(-0.35, 0.35)
    raw = []
    for t in range(T):
        u = t / (T - 1)
        z = (u - peak_pos) / width
        z = z * (1.0 + skew * (u - peak_pos))  # gentle asymmetry
        v = math.exp(-0.5 * z * z)
        v += rng.uniform(-0.015, 0.015)
        raw.append(v)
    lo, hi = min(raw), max(raw)
    span = max(1e-9, hi - lo)
    D = [dmin + (dmax - dmin) * (r - lo) / span for r in raw]

    # slew limit so the aggregate never ramps faster than the boilers can follow
    ms = maxstep_frac * dmax
    for _ in range(4):
        for t in range(1, T):
            if D[t] - D[t - 1] > ms:
                D[t] = D[t - 1] + ms
        for t in range(T - 2, -1, -1):
            if D[t] - D[t + 1] > ms:
                D[t] = D[t + 1] + ms
    D = [int(round(x)) for x in D]
    D = [max(dmin, min(dmax, v)) for v in D]

    out = []
    out.append("%d %d" % (T, K))
    out.append(" ".join(str(v) for v in D))
    for i in range(K):
        out.append("%d %.3f %.3f %.3f %.3f %.3f %.3f %d %d" % (
            caps[i], pmins[i], nls[i], s0s[i], betas[i], xstars[i],
            ramps[i], minups[i], mindowns[i]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    gen(int(sys.argv[1]))
