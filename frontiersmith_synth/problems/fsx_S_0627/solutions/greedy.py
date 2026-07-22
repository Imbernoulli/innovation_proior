# TIER: greedy
# The obvious recipe an average strong coder writes first: treat this as REGRESSION.
# Spend the ENTIRE budget on a single UNIFORM sweep of the queryable window [0, QR], then
# fit by piecewise-linear interpolation and LINEARLY EXTRAPOLATE past QR into the tail.
# It ignores the stated point symmetry and never adapts to localize breakpoints -- so it
# resolves smooth interior it did not need and is blind on the un-probeable tail (QR, GR],
# which on trap instances carries steep, high-jump structure.
import sys, json

inst = json.load(sys.stdin)
phase = inst.get("phase")
QR = float(inst["QR"]); GR = float(inst["GR"]); G = int(inst["G"])
Q = int(inst["Q"])


def interp_extrap(xs, ys, x):
    n = len(xs)
    if n == 0:
        return 0.0
    if n == 1:
        return ys[0]
    if x <= xs[0]:
        # linear extrapolation from the first segment
        m = (ys[1] - ys[0]) / (xs[1] - xs[0]) if xs[1] != xs[0] else 0.0
        return ys[0] + m * (x - xs[0])
    if x >= xs[-1]:
        m = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) if xs[-1] != xs[-2] else 0.0
        return ys[-1] + m * (x - xs[-1])
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    if xs[hi] == xs[lo]:
        return ys[lo]
    t = (x - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] + t * (ys[hi] - ys[lo])


if phase == "query":
    # front-load the whole budget as a uniform grid in round 0; nothing afterwards
    if inst.get("round", 0) == 0:
        qs = [QR * i / (Q - 1) for i in range(Q)]
        print(json.dumps({"queries": qs}))
    else:
        print(json.dumps({"queries": []}))
else:
    hist = inst.get("history", [])
    pts = sorted((float(h[0]), float(h[1])) for h in hist if h[1] is not None)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    pred = [interp_extrap(xs, ys, GR * j / (G - 1)) for j in range(G)]
    print(json.dumps({"pred": pred}))
