# TIER: greedy
# The "obvious" adaptive integrator: classic embedded step-doubling (one RK4
# step of h vs two RK4 steps of h/2) to ESTIMATE local error and drive h up or
# down -- the embedded-local-error-estimate mechanism, and nothing else. It
# never switches update rule (explicit RK4 only) and never looks ahead at the
# stiffness profile, so during a stiff window it must shrink h all the way
# down to the raw stability limit and crawl through with many tiny steps. On
# generous ("lenient") instances that still fits the budget, giving a real
# (if modest) improvement over the uniform baseline. On tight ("trap")
# instances covering the stiff window this way blows the evaluation budget,
# so the schedule is rejected as infeasible -> 0 on those instances.
#
# IMPORTANT: the internal state used to keep adapting must be the SAME value
# the evaluator will independently recompute when it replays the reported
# schedule (one plain RK4 step per boundary) -- so we advance our own
# bookkeeping with the full-step estimate `y_full`, and use the half-step
# comparison ONLY to decide h, never to produce a "better" y to continue from.
import sys, json, math


def K_at(t, segs):
    for seg in segs:
        if seg["t0"] - 1e-9 <= t < seg["t1"] + 1e-9:
            return seg["K"]
    return segs[-1]["K"]


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


inst = json.load(sys.stdin)
T = inst["T"]
segs = inst["K_segments"]
ec = inst["E_coef"]
y0 = inst["y0"]
checkpoints = inst["checkpoints"]

TOL = 3e-3
margin = 8.0 * abs(y0 - E_at(0.0, ec)) + 15.0  # basic "did I diverge" sanity gate

t = 0.0
y = y0
h = T / 60.0
boundaries = []

for target in checkpoints:
    while t < target - 1e-12:
        h = min(h, target - t)
        err = None
        for _ in range(60):
            y_full = rk4_step(t, y, h, segs, ec)
            y_half1 = rk4_step(t, y, h / 2, segs, ec)
            y_half2 = rk4_step(t + h / 2, y_half1, h / 2, segs, ec)
            sane = (abs(y_full) < 1e4 and abs(y_half1) < 1e4 and abs(y_half2) < 1e4
                    and abs(y_full - E_at(t + h, ec)) < margin
                    and abs(y_half2 - E_at(t + h, ec)) < margin)
            err = abs(y_half2 - y_full) if sane else 1e18
            if (err <= TOL and sane) or h < 1e-7:
                break
            h = h / 2.0
        t = t + h
        y = y_full  # match the evaluator's single-shot RK4 replay
        boundaries.append({"t1": t, "method": "explicit"})
        if err < TOL / 8.0:
            h = h * 2.0
        if target - t > 1e-9:
            h = min(h, target - t)
        if t >= target - 1e-9:
            break
    if abs(t - target) > 1e-9:
        t = target
        boundaries[-1] = {"t1": t, "method": "explicit"}

print(json.dumps({"steps": boundaries}))
