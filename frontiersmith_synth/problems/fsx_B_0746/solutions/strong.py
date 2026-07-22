# TIER: strong
# Composes all three mechanisms: (1) the same embedded local-error estimate
# (RK4 step-doubling) as greedy to size h for ACCURACY; (2) a PI controller on
# that error estimate (uses both the current and the previous accepted step's
# error ratio) instead of greedy's bang-bang doubling/halving, so accepted
# step sizes converge smoothly instead of oscillating; and (3) a STIFFNESS
# INDICATOR: before committing to an explicit step, look ahead over the
# prospective interval [t, t+h] at the WORST-CASE K(t) it contains (available
# directly from the instance's K_segments) and compare the implied stability
# limit h_stab = STAB/K_max to the accuracy-driven h. Only while that
# indicator FIRES (h_acc > h_stab) do we flip to the stabilized (implicit)
# update, and then we stride through the ENTIRE flagged run in one big
# implicit step (unconditionally stable) instead of crawling through it with
# thousands of stability-limited explicit steps. This is what lets a stiff
# window cost a handful of evaluations instead of hundreds.
import sys, json, math


def K_at(t, segs):
    for seg in segs:
        if seg["t0"] - 1e-9 <= t < seg["t1"] + 1e-9:
            return seg["K"]
    return segs[-1]["K"]


def max_K_over(t0, t1, segs):
    m = K_at(t0, segs)
    for seg in segs:
        if seg["t1"] > t0 - 1e-9 and seg["t0"] < t1 + 1e-9:
            m = max(m, seg["K"])
    return max(m, K_at(t1, segs))


def E_at(t, ec):
    return ec["e0"] + ec["e1"] * t + ec["e2"] * math.sin(ec["w"] * t + ec["phase"])


def f(t, y, segs, ec):
    return -K_at(t, segs) * (y - E_at(t, ec))


def rk4_step(t, y, h, segs, ec):
    k1 = f(t, y, segs, ec)
    k2 = f(t + h / 2, y + h / 2 * k1, segs, ec)
    k3 = f(t + h / 2, y + h / 2 * k2, segs, ec)
    k4 = f(t + h, y + h * k3, segs, ec)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def be_step(t, y, h, segs, ec):
    t1 = t + h
    K1 = K_at(t1, segs)
    E1 = E_at(t1, ec)
    return (y + h * K1 * E1) / (1.0 + h * K1)


inst = json.load(sys.stdin)
T = inst["T"]
segs = inst["K_segments"]
ec = inst["E_coef"]
y0 = inst["y0"]
checkpoints = inst["checkpoints"]

TOL = 3e-3
STAB = 2.5
KP, KI = 0.3, 0.15
margin = 8.0 * abs(y0 - E_at(0.0, ec)) + 15.0

t = 0.0
y = y0
h = T / 60.0
prev_err_ratio = 1.0
boundaries = []

for target in checkpoints:
    while t < target - 1e-12:
        h_try = min(h, target - t)
        Kmax = max_K_over(t, t + h_try, segs)
        h_stab = STAB / Kmax if Kmax > 1e-9 else 1e18

        if h_try > h_stab:
            # stiffness indicator fires: switch to the stabilized update and
            # stride through the whole contiguous stiff run in one big step
            stride_end = t + h_try
            for seg in segs:
                if seg["t0"] <= t + 1e-9 < seg["t1"] and seg["K"] > 1e-9 and STAB / seg["K"] < h_try:
                    stride_end = max(stride_end, seg["t1"])
            h_im = min(target - t, max(stride_end - t, h_stab * 4.0, 0.25))
            y = be_step(t, y, h_im, segs, ec)
            t = t + h_im
            boundaries.append({"t1": t, "method": "implicit"})
            continue

        y_full = rk4_step(t, y, h_try, segs, ec)
        y_half1 = rk4_step(t, y, h_try / 2, segs, ec)
        y_half2 = rk4_step(t + h_try / 2, y_half1, h_try / 2, segs, ec)
        sane = (abs(y_full) < 1e4 and abs(y_half1) < 1e4 and abs(y_half2) < 1e4
                and abs(y_full - E_at(t + h_try, ec)) < margin
                and abs(y_half2 - E_at(t + h_try, ec)) < margin)
        err = (abs(y_half2 - y_full) if sane else 1e18) + 1e-14
        ratio = TOL / err

        if (err > TOL or not sane) and h_try > 1e-7:
            h = h_try * (max(0.2, min(0.9, 0.9 * ratio ** 0.2)) if sane else 0.3)
            continue

        t = t + h_try
        y = y_full  # match the evaluator's single-shot RK4 replay
        boundaries.append({"t1": t, "method": "explicit"})
        factor = max(0.3, min(4.0, (ratio ** KP) * (prev_err_ratio ** KI)))
        prev_err_ratio = ratio
        h = min(h_try * factor, T / 6.0)

    if abs(t - target) > 1e-9:
        t = target
        if boundaries and boundaries[-1]["t1"] != target:
            boundaries.append({"t1": target, "method": "explicit"})

print(json.dumps({"steps": boundaries}))
