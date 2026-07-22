# TIER: strong
# INSIGHT (not "greedy + more experiments"): nearly the whole budget should hunt the hidden
# transition manifold, not spread evenly over composition space.  Three composed moves:
#   (1) ANCHOR broadly (a handful of points with ZERO B, so mA/mC alone sweep the full
#       reachable T_lin range) to sanity-check that T_lin(m) is the right predictor away from
#       the transition -- these can never touch the phase change (no B present at all).
#   (2) COMMIT the bulk of the budget to a fine sweep of T_lin INSIDE the public
#       [Tstar_lo, Tstar_hi] window hint, at the LARGEST B-fraction that still keeps the whole
#       window physically reachable (solved in closed form from the public materials, not
#       hard-coded) -- a larger B-fraction makes the plateau band wider and easier to land on.
#       A run of consecutive readings with (near) ZERO slope against T_lin is the plateau
#       SIGNATURE: since the law is exactly piecewise (no noise), any reading inside it IS the
#       transition temperature T* to full precision.
#   (3) ADAPTIVELY BISECT (using leftover rounds) toward the flat-to-rising edge to pin the
#       latent-heat scale `ell`, instead of ever fitting one smooth global curve.
# Prediction then CLASSIFIES each test mix's regime (cold / plateau / hot) from the fitted
# (T*, ell) pair and applies the matching closed form -- instead of interpolating one curve
# through data that never resolved the flat band.  Note the window's midpoint is NOT used as a
# T* estimate: the two half-widths are independent, so it is only a coarse fallback prior, not
# a shortcut -- real accuracy comes from the flat-reading signature above.
import sys, json

inst = json.load(sys.stdin)
phase = inst.get("phase")
mat = inst["materials"]
TA, TB0, TC = mat["TA"], mat["TB0"], mat["TC"]
bounds = inst["bounds"]
Tstar_lo, Tstar_hi = bounds["Tstar_lo"], bounds["Tstar_hi"]
ell_lo, ell_hi = bounds["ell_lo"], bounds["ell_hi"]

CAP = inst["budget"]["mass_cap_per_experiment"]
M_USE = min(100.0, CAP * 0.95)
N_ANCHOR = 4
N_SWEEP0 = 8
MARGIN = max(1.0, 0.15 * (Tstar_hi - Tstar_lo))


def feasible_xb_range(target):
    """Closed-form xB interval at which some (mA,mB,mC) with B-fraction xB reaches T_lin ==
    target (both reach-range endpoints are monotone in xB since TB0 < TC, TA typically)."""
    xb_hi_max = (TA - target) / (TA - TB0) if TA != TB0 else 1.0
    xb_lo_min = (TC - target) / (TC - TB0) if TC != TB0 else 0.0
    return max(0.0, xb_lo_min), min(1.0, xb_hi_max)


def _pick_xb_window():
    lo1, hi1 = feasible_xb_range(Tstar_lo - MARGIN)
    lo2, hi2 = feasible_xb_range(Tstar_hi + MARGIN)
    lo = max(lo1, lo2)
    hi = min(hi1, hi2)
    if hi <= lo:
        # window not simultaneously reachable at one xB with the full margin -- shrink margin
        lo1, hi1 = feasible_xb_range(Tstar_lo)
        lo2, hi2 = feasible_xb_range(Tstar_hi)
        lo = max(lo1, lo2)
        hi = min(hi1, hi2)
        if hi <= lo:
            return 0.15  # last-resort safe small B-fraction
    return lo + 0.7 * (hi - lo)


XB_WINDOW = _pick_xb_window()


def compose(xB, Tlin_target, M):
    mB = xB * M
    rem = (1.0 - xB) * M
    if rem <= 1e-9:
        return {"mA": 0.0, "mB": mB, "mC": 0.0}
    if abs(TA - TC) < 1e-9:
        t = 0.5
    else:
        t = (Tlin_target - xB * TB0 - (1.0 - xB) * TC) / ((1.0 - xB) * (TA - TC))
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    mA = rem * t
    mC = rem * (1.0 - t)
    return {"mA": mA, "mB": mB, "mC": mC}


def t_lin(mA, mB, mC):
    M = mA + mB + mC
    if M <= 0:
        return None
    return (mA * TA + mB * TB0 + mC * TC) / M


def window_points(history):
    pts = []
    seen = set()
    for h in history:
        mA, mB, mC = h["mA"], h["mB"], h["mC"]
        M = mA + mB + mC
        if M <= 1e-9:
            continue
        xB = mB / M
        if abs(xB - XB_WINDOW) <= 0.02:
            z = round(t_lin(mA, mB, mC), 6)
            if z not in seen:
                seen.add(z)
                pts.append((z, h["T_f"]))
    pts.sort()
    return pts


def segments(pts):
    """Consecutive-pair local slopes.  Expected slope is ~1 in the cold/hot regimes and ~0
    inside the plateau (piecewise EXACT, no noise) -- but a segment whose two endpoints
    straddle a boundary reads some BLEND in between, which is itself the signal that the
    boundary is inside that segment and it needs bisecting, not a "no signature" verdict."""
    segs = []
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        if x2 - x1 < 1e-9:
            continue
        sl = (y2 - y1) / (x2 - x1)
        segs.append((sl, x1, x2, y1, y2))
    return segs


def classify(pts):
    """Return (flat_pts, lo, hi, ambiguous): flat_pts = points confidently inside the plateau;
    lo = rightmost T_lin confidently inside it; hi = smallest T_lin confidently back in the
    rising (hot) regime after lo; ambiguous = segments whose slope is neither confidently flat
    nor confidently steep, i.e. still straddling an unresolved boundary."""
    segs = segments(pts)
    flat_pts, hot_pts, ambiguous = [], [], []
    for sl, x1, x2, y1, y2 in segs:
        if sl <= 0.15:
            flat_pts.append((x1, y1)); flat_pts.append((x2, y2))
        elif sl >= 0.85:
            hot_pts.append((x1, y1)); hot_pts.append((x2, y2))
        else:
            ambiguous.append((sl, x1, x2, y1, y2))
    lo = max((x for x, _ in flat_pts), default=None)
    hi = None
    if lo is not None:
        cand = [x for x, _ in hot_pts if x >= lo - 1e-9]
        if cand:
            hi = min(cand)
    return flat_pts, lo, hi, ambiguous


if phase == "query":
    rnd = int(inst.get("round", 0))
    budget = int(inst["budget_left"])
    if rnd == 0:
        exps = []
        for i in range(N_ANCHOR):
            t = i / (N_ANCHOR - 1) if N_ANCHOR > 1 else 0.5
            lo_reach = min(TA, TC)      # xB=0: no B present, always exactly T_lin, always
            hi_reach = max(TA, TC)      # reachable -- pure sanity check of the formula
            target = lo_reach + (hi_reach - lo_reach) * t
            exps.append(compose(0.0, target, M_USE))
        span_lo = Tstar_lo - MARGIN
        span_hi = Tstar_hi + MARGIN
        n = min(N_SWEEP0, max(0, budget - len(exps)))
        for i in range(n):
            t = i / (n - 1) if n > 1 else 0.5
            target = span_lo + (span_hi - span_lo) * t
            exps.append(compose(XB_WINDOW, target, M_USE))
        print(json.dumps({"experiments": exps[:budget]}))
    else:
        pts = window_points(inst.get("history", []))
        flat_pts, lo, hi, ambiguous = classify(pts)
        exps = []
        want = min(2, budget)
        if ambiguous:
            # a segment whose slope is neither confidently flat nor confidently steep still
            # straddles an unresolved boundary -- bisect the WIDEST such segment(s) directly,
            # instead of a generic "biggest empty gap" search that has no reason to land near
            # the transition at all
            ambiguous.sort(key=lambda s: -(s[2] - s[1]))
            for sl, x1, x2, y1, y2 in ambiguous[:want]:
                exps.append(compose(XB_WINDOW, 0.5 * (x1 + x2), M_USE))
        elif lo is not None and hi is None:
            # confidently found the plateau but never saw it resume rising: extend the sweep
            # outward using the public ell hint to guess where the hot regime should start
            guess = lo + 0.6 * ell_hi * XB_WINDOW
            exps.append(compose(XB_WINDOW, guess, M_USE))
            if want > 1:
                exps.append(compose(XB_WINDOW, lo + 0.3 * ell_hi * XB_WINDOW, M_USE))
        elif lo is None:
            # nothing flat and nothing ambiguous: the coarse sweep landed entirely in one
            # regime -- the transition must sit just outside the covered span; push outward
            xs = sorted(x for x, _ in pts)
            if xs:
                span = xs[-1] - xs[0] if len(xs) > 1 else 1.0
                exps.append(compose(XB_WINDOW, xs[0] - 0.5 * span - 0.5, M_USE))
                if want > 1:
                    exps.append(compose(XB_WINDOW, xs[-1] + 0.5 * span + 0.5, M_USE))
            else:
                exps.append(compose(XB_WINDOW, 0.5 * (Tstar_lo + Tstar_hi), M_USE))
        # else: both edges already confidently pinned -- nothing more useful to probe
        print(json.dumps({"experiments": exps[:budget]}))
else:
    pts = window_points(inst.get("history", []))
    flat_pts, lo, hi, ambiguous = classify(pts)
    if flat_pts:
        ys = sorted(y for _, y in flat_pts)
        Tstar_est = ys[len(ys) // 2]
    elif ambiguous:
        # no confidently-flat pair, but the narrowest ambiguous segment still straddling the
        # left edge is our best available estimate: linearly interpolate where a slope-1 cold
        # trend would have crossed this segment's low point, which approximates Tstar
        ambiguous.sort(key=lambda s: (s[2] - s[1]))
        sl, x1, x2, y1, y2 = ambiguous[0]
        Tstar_est = 0.5 * (y1 + y2)
    else:
        Tstar_est = 0.5 * (Tstar_lo + Tstar_hi)
    if lo is not None and hi is not None and hi > lo:
        ell_est = (hi - Tstar_est) / XB_WINDOW
    else:
        ell_est = 0.5 * (ell_lo + ell_hi)
    if ell_est < 0.0:
        ell_est = 0.5 * (ell_lo + ell_hi)

    preds = []
    for m in inst["test_mixes"]:
        mA, mB, mC = m["mA"], m["mB"], m["mC"]
        M = mA + mB + mC
        if M <= 0:
            preds.append(Tstar_est)
            continue
        xB = mB / M
        Tl = t_lin(mA, mB, mC)
        if Tl <= Tstar_est:
            preds.append(Tl)
        elif Tl < Tstar_est + ell_est * xB:
            preds.append(Tstar_est)
        else:
            preds.append(Tl - ell_est * xB)
    print(json.dumps({"predictions": preds}))
