# TIER: greedy
# The obvious recipe an average strong coder writes first: treat this as a 1-D REGRESSION
# problem on the "obvious" scalar T_lin (the energy-conservation weighted average).  Spend the
# ENTIRE budget in round 0 as a composition-space-filling sweep at a fixed, unremarkable
# B-fraction across the FULL achievable T_lin range -- ignoring the public transition-window
# hint entirely -- then sort readings by T_lin and linearly interpolate/extrapolate a single
# smooth curve.  It never notices that the plateau's width depends on the B-fraction used, and
# it spreads its budget over the whole range instead of committing it to the narrow band, so it
# is blind wherever the transition actually sits (which is exactly where the trap instances
# concentrate their held-out test mixes).
import sys, json

inst = json.load(sys.stdin)
phase = inst.get("phase")
mat = inst["materials"]
TA, TB0, TC = mat["TA"], mat["TB0"], mat["TC"]

XB_FIXED = 0.3


def compose(xB, t, M):
    mB = xB * M
    rem = (1.0 - xB) * M
    mA = rem * t
    mC = rem * (1.0 - t)
    return {"mA": mA, "mB": mB, "mC": mC}


def t_lin(mA, mB, mC):
    M = mA + mB + mC
    if M <= 0:
        return None
    return (mA * TA + mB * TB0 + mC * TC) / M


def interp_extrap(xs, ys, x):
    n = len(xs)
    if n == 0:
        return 0.0
    if n == 1:
        return ys[0]
    if x <= xs[0]:
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
    if inst.get("round", 0) == 0:
        n = min(inst["budget"]["max_experiments"], inst["budget_left"])
        exps = []
        for i in range(n):
            t = i / (n - 1) if n > 1 else 0.5
            exps.append(compose(XB_FIXED, t, 100.0))
        print(json.dumps({"experiments": exps}))
    else:
        print(json.dumps({"experiments": []}))
else:
    hist = inst.get("history", [])
    pts = sorted((t_lin(h["mA"], h["mB"], h["mC"]), h["T_f"]) for h in hist)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    preds = []
    for m in inst["test_mixes"]:
        z = t_lin(m["mA"], m["mB"], m["mC"])
        preds.append(interp_extrap(xs, ys, z))
    print(json.dumps({"predictions": preds}))
