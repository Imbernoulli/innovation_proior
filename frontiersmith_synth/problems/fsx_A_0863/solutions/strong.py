# TIER: strong
# INSIGHT (not "greedy + a better gain"): re-derive an ONLINE estimate of the
# local buffer capacity from the two most recent (V, pH) readings (a secant
# slope = an inverted, locally-linearized titration curve) and take a damped
# NEWTON step toward the target -- but never trust that single estimate with
# an uncapped leap, because titrant can only be ADDED (an overshoot can never
# be undone, so "bracket and bisect" is not safe here). Three composed trust-
# region safeguards, all reconstructed fresh from `history` every call (no
# persistent state needed or used):
#   (1) CURVATURE CHECK -- compare the two most recent secant slopes; if they
#       disagree, that is direct evidence of a buffer-capacity regime change
#       nearby, so shrink the trusted step fraction of the remaining budget.
#   (2) GROWTH CAP -- never let a step exceed ~2.2x the step that produced the
#       most recent reading: don't let one lucky secant leap across ground
#       that has not actually been explored.
#   (3) FINAL-APPROACH THROTTLE -- shrink the allowed step further as the
#       (possibly stale) predicted pH gap narrows, since the target sits, by
#       construction, wherever the steepest part of the curve may be hiding.
import sys, json, math

inst = json.load(sys.stdin)
V_max = float(inst["V_max"])
pH = float(inst["pH"])
target = float(inst["target_pH"])
V = float(inst["V"])
hist = [(float(h[0]), float(h[1])) for h in inst.get("history", [])]
max_add = float(inst.get("max_add", V_max))

EPS = 0.05
remaining = V_max - V

if pH >= target - EPS or remaining <= 1e-9:
    add = 0.0
else:
    n = len(hist)
    if n < 2:
        add = min(V_max * 0.02, remaining, max_add)
        add = max(add, 1e-6)
    else:
        V1, pH1 = hist[-2]
        V2, pH2 = hist[-1]
        if (V2 - V1) < 1e-12 or (pH2 - pH1) <= 1e-9:
            slope = None
        else:
            slope = (pH2 - pH1) / (V2 - V1)

        # (1) curvature-based trust fraction
        trust = 0.12
        if n >= 3 and slope is not None:
            V0, pH0 = hist[-3]
            if (V1 - V0) > 1e-9 and (pH1 - pH0) > 1e-9:
                slope_prev = (pH1 - pH0) / (V1 - V0)
                if slope_prev > 1e-9:
                    ratio = slope / slope_prev
                    dev = abs(math.log(max(ratio, 1e-6)))
                    trust = 0.55 / (1.0 + 6.0 * dev)
                    trust = max(0.05, min(0.55, trust))

        if slope is None:
            raw = remaining * 0.25
        else:
            raw = (target - pH) / slope
        damp = 0.8
        step = damp * raw
        cap = remaining * trust

        # (2) growth cap relative to the step that produced the latest reading
        prev_step = V2 - V1
        if prev_step > 1e-9:
            cap = min(cap, prev_step * 2.2)

        # (3) final-approach throttle: get more cautious as the (possibly
        # stale) predicted gap narrows, not just when curvature says so
        gap0 = max(target - hist[0][1], 1e-6)
        gap = target - pH
        caution = max(0.06, min(1.0, (gap / gap0) / 0.35))
        cap = min(cap, remaining * trust * caution + remaining * 0.02)

        step = min(step, cap)
        step = max(step, remaining * 0.006)
        add = min(step, remaining, max_add)

print(json.dumps({"add": add}))
