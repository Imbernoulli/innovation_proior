#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_B_0746 -- "Boiler Startup: Step-Size Control for a
Relaxation Trajectory with a Wandering Stiff Transient".

Family: pi-controlled-step-integrator (embedded-local-error-estimate +
pi-stepsize-control + stiffness-switch). Skinned as a plant-control telemetry
task: a controlled variable y(t) (e.g. a boiler drum level / reactor
temperature deviation) obeys the scalar relaxation ODE

    dy/dt = f(t,y) = -K(t) * (y - E(t))

E(t) is a SMOOTH externally-driven setpoint trajectory (a slow drift plus one
sinusoidal disturbance -- always mild curvature). K(t) is the plant's own
relaxation RATE, given piecewise-constant over the whole horizon [0,T]: mostly
a small K_base, but with 1-2 short "stiff" windows (rate up to several hundred)
positioned anywhere along the path (early, in a few cases repeated later) --
the plant is briefly driven very hard, then returns to gentle tracking. THE
NOVELTY vs a textbook adaptive integrator: the stiffness is a genuine SWITCH in
regime, not a slow drift, and it can recur, so a strategy has to recognize it
locally (not just "the first bit is hard") and pay for it with a bounded
resource: every internal "step" the candidate schedules costs a fixed number
of DERIVATIVE-EVALUATION units against a hard budget `max_evals`.

The candidate does NOT report state values. It reports a STEP SCHEDULE: a
strictly increasing sequence of times covering [0,T] and, for each step, which
UPDATE RULE to use:
  - "explicit": one classical RK4 step (4th-order accurate but only
    CONDITIONALLY stable -- unstable once h*K(t) exceeds ~2.5 anywhere in the
    step, and the raw floating-point recurrence really does diverge then).
  - "implicit": one backward-Euler step evaluated with K(t),E(t) AT THE STEP'S
    END TIME (UNCONDITIONALLY stable for any h, since K>=0, but only 1st-order
    accurate, so it wastes budget if used where accuracy, not stability, is
    the bottleneck).
Explicit steps cost `cost_explicit` evaluation units; implicit steps cost the
(higher) `cost_implicit` units, reflecting the extra work of a stabilized
solve. The EVALUATOR -- not the candidate -- performs 100% of the actual
floating-point integration from this schedule, so there is no way to report a
cheap schedule while secretly computing an expensive, more-accurate one: the
schedule *is* the computation.

Feasibility: the schedule must exactly cover every published `checkpoints`
time as a step boundary (dense output at fixed report times), the final
boundary must equal T, and total spent evaluation units must not exceed
`max_evals`. Any violation -> INFEASIBLE -> score 0 on that instance.

Objective (minimize): RMSE of the evaluator's re-integrated trajectory against
a very high-accuracy reference (fixed fine-grid RK4, ~17.5k substeps) at the
checkpoints, CAPPED at ERR_CAP before the ratio is taken (bounds how much a
few catastrophically-diverged checkpoints can dominate the score -- once a
schedule is "clearly useless" further blow-up magnitude is not informative).

Trap: 6 of the 10 instances give a budget that is enough to cover the smooth
part generously plus only a FRACTION of what covering the stiff window with
EXPLICIT steps alone would cost (i.e. it comfortably affords a handful of big
implicit strides through the stiff window, but not the hundreds of tiny
explicit steps stability would otherwise force). A plain embedded-error
adaptive stepper that never switches method has no way to survive those
without either exceeding the budget or taking an oversized (and therefore
truly unstable) explicit step.

The candidate is UNTRUSTED model output: it runs in an ISOLATED subprocess via
`isorun`, sees ONLY the public instance on stdin, and returns ONLY its answer
on stdout, so it can never reach the evaluator's frames / scorer / baseline /
reference trajectory.

Scoring (deterministic; no wall-time):
  baseline b = the "trivial" construction's own (capped) objective: a UNIFORM,
               non-adaptive grid of explicit RK4 steps spending (almost) the
               full `max_evals` budget, still forced to land on every
               checkpoint exactly. Always FEASIBLE by construction.
  For a FEASIBLE answer with (capped) objective obj:  r = min(1, 0.1*b/obj)
  -> matching the trivial construction maps to exactly 0.1; a schedule with
     k times lower error maps to min(1, 0.1*k). Infeasible / malformed -> 0.

CLI:  python3 evaluator.py <candidate.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun

CAP = 1.0e6          # hard clip on the integrated state (guards overflow)
STAB = 2.5            # conservative explicit-RK4 stability bound: h*K <= STAB
ERR_CAP = 0.1          # cap on RMSE before the ratio is taken
N_CHECK = 14           # number of fixed report checkpoints
N_REF_MULT = 1250      # fine reference substeps per checkpoint interval


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    def uf():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (state >> 11) / float(1 << 53)

    nxt.uf = uf
    return nxt


# ----------------------------- dynamics -------------------------------------
def _K_at(t, segs):
    for seg in segs:
        if seg["t0"] - 1e-9 <= t < seg["t1"] + 1e-9:
            return seg["K"]
    return segs[-1]["K"]


def _max_K_over(t0, t1, segs):
    m = _K_at(t0, segs)
    for seg in segs:
        if seg["t1"] > t0 - 1e-9 and seg["t0"] < t1 + 1e-9:
            m = max(m, seg["K"])
    return max(m, _K_at(t1, segs))


def _E_at(t, ec):
    return ec["e0"] + ec["e1"] * t + ec["e2"] * math.sin(ec["w"] * t + ec["phase"])


def _f(t, y, segs, ec):
    return -_K_at(t, segs) * (y - _E_at(t, ec))


def _clip(y):
    if not math.isfinite(y):
        return CAP
    if y > CAP:
        return CAP
    if y < -CAP:
        return -CAP
    return y


def _rk4_step(t, y, h, segs, ec):
    k1 = _f(t, y, segs, ec)
    k2 = _f(t + h / 2, y + h / 2 * k1, segs, ec)
    k3 = _f(t + h / 2, y + h / 2 * k2, segs, ec)
    k4 = _f(t + h, y + h * k3, segs, ec)
    return _clip(y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4))


def _be_step(t, y, h, segs, ec):
    t1 = t + h
    K1 = _K_at(t1, segs)
    E1 = _E_at(t1, ec)
    return _clip((y + h * K1 * E1) / (1.0 + h * K1))


def _integrate(y0, boundaries, segs, ec):
    """boundaries: list of (t1, method), strictly increasing from t=0.
    Returns the list of states y at each boundary (evaluator-authoritative)."""
    t = 0.0
    y = y0
    out = []
    for (t1, method) in boundaries:
        h = t1 - t
        y = _rk4_step(t, y, h, segs, ec) if method == "explicit" else _be_step(t, y, h, segs, ec)
        t = t1
        out.append(y)
    return out


def _reference(y0, T, segs, ec, n_sub):
    h = T / n_sub
    t = 0.0
    y = y0
    out = [y0]
    for _ in range(n_sub):
        y = _rk4_step(t, y, h, segs, ec)
        t += h
        out.append(y)
    return out


def _rmse(traj, ref_cp):
    errs = [(a - b) ** 2 for a, b in zip(traj, ref_cp)]
    return math.sqrt(sum(errs) / len(errs))


# ----------------------------- instance family ------------------------------
def _build_K_segments(T, K_base, r, second_blip):
    segs_raw = []
    s0 = 0.1 + 0.3 * r.uf()
    w0 = 1.0 + 2.0 * r.uf()
    K0 = 80.0 + 350.0 * r.uf()
    segs_raw.append((s0, w0, K0))
    if second_blip:
        s1 = T * (0.5 + 0.3 * r.uf())
        w1 = 0.8 + 1.5 * r.uf()
        K1 = 80.0 + 300.0 * r.uf()
        segs_raw.append((s1, w1, K1))
    segs_raw.sort()

    K_segments = []
    cur = 0.0
    for (s, w, K) in segs_raw:
        if s > cur + 1e-6:
            K_segments.append({"t0": cur, "t1": s, "K": K_base})
        K_segments.append({"t0": s, "t1": s + w, "K": K})
        cur = s + w
    if cur < T - 1e-6:
        K_segments.append({"t0": cur, "t1": T, "K": K_base})
    return K_segments


def _make_one(seed, frac, second_blip, lenient):
    r = _rng(seed)
    T = round(12.0 + 8.0 * r.uf(), 6)
    K_base = round(0.4 + 0.4 * r.uf(), 6)
    ec = {
        "e0": round(r.uf() * 3.0, 6),
        "e1": round(0.02 + 0.13 * r.uf(), 6),
        "e2": round(1.0 + 2.0 * r.uf(), 6),
        "w": round(0.4 + 1.2 * r.uf(), 6),
        "phase": round(2 * math.pi * r.uf(), 6),
    }
    jump = 4.0 + 6.0 * r.uf()
    sign = 1.0 if r.uf() < 0.5 else -1.0
    y0 = round(_E_at(0.0, ec) + sign * jump, 6)

    K_segments = _build_K_segments(T, K_base, r, second_blip)

    checkpoints = [round(T * (k + 1) / N_CHECK, 6) for k in range(N_CHECK)]
    checkpoints[-1] = T

    cost_explicit = 4
    cost_implicit = 8
    stiff_cost = 0
    smooth_cost = 0
    for seg in K_segments:
        width = seg["t1"] - seg["t0"]
        if seg["K"] > K_base + 1e-9:
            n = max(1, math.ceil(width / (STAB / seg["K"])))
            stiff_cost += n * cost_explicit
        else:
            n = max(1, math.ceil(width / 0.8))
            smooth_cost += n * cost_explicit

    if lenient:
        max_evals = int(round((smooth_cost + stiff_cost) * frac)) + 300
    else:
        max_evals = int(round(smooth_cost + frac * stiff_cost))
    max_evals = max(max_evals, 40)

    public = {
        "T": T, "y0": y0, "K_segments": K_segments, "E_coef": ec,
        "checkpoints": checkpoints, "max_evals": max_evals,
        "cost_explicit": cost_explicit, "cost_implicit": cost_implicit,
    }

    n_sub = N_CHECK * N_REF_MULT
    ref = _reference(y0, T, K_segments, ec, n_sub)
    y_ref = [ref[k * N_REF_MULT] for k in range(1, N_CHECK + 1)]

    return {"public": public, "hidden": {"y_ref": y_ref}}


def make_instances():
    # (seed, frac, second_blip, lenient)
    # 6 TRAP instances (frac<1): budget covers the smooth part generously but
    # only a fraction of "explicit-through-the-stiff-window" -- forces a
    # method switch to survive. 4 LENIENT instances (frac>1, +300 slack):
    # budget generous enough that even a naive explicit-only adaptive
    # stepper fits, so the ladder isn't ALL zeros for the naive approach.
    specs = [
        (7601, 0.15, False, False), (7602, 0.18, False, False), (7603, 0.22, False, False),
        (7604, 0.28, True, False), (7605, 0.32, True, False), (7606, 0.20, True, False),
        (7607, 1.3, False, True), (7608, 1.4, False, True),
        (7609, 1.3, False, True), (7610, 1.4, False, True),
    ]
    return [_make_one(*s) for s in specs]


# ----------------------------- baseline (trivial construction) -------------
def _trivial_boundaries(pub):
    T = pub["T"]; max_evals = pub["max_evals"]; ce = pub["cost_explicit"]
    n_check = len(pub["checkpoints"])
    n_uniform = max(1, max_evals // ce - n_check - 2)
    h = T / n_uniform
    checkpoints = pub["checkpoints"]
    boundaries = []
    t = 0.0
    ci = 0
    while ci < len(checkpoints):
        nxt = t + h
        if nxt >= checkpoints[ci] - 1e-9:
            nxt = checkpoints[ci]
            ci += 1
        boundaries.append((nxt, "explicit"))
        t = nxt
    return boundaries


def baseline(inst):
    """Uniform, non-adaptive explicit-RK4 grid spending (almost) the full
    budget, forced to land on every checkpoint. Always feasible; the
    evaluator's own reference point for the score-0.1 normalization."""
    pub = inst["public"]
    b = _trivial_boundaries(pub)
    traj = _integrate(pub["y0"], b, pub["K_segments"], pub["E_coef"])
    rmse = _rmse(traj, inst["hidden"]["y_ref"])
    return min(rmse, ERR_CAP)


# ----------------------------- scoring --------------------------------------
def score(inst, answer):
    """Strictly validate the answer against the instance; return (ok, obj)."""
    pub = inst["public"]
    T = pub["T"]; checkpoints = pub["checkpoints"]; max_evals = pub["max_evals"]
    ce = pub["cost_explicit"]; ci = pub["cost_implicit"]

    if not isinstance(answer, dict):
        return False, None
    steps = answer.get("steps")
    if not isinstance(steps, list) or not steps or len(steps) > 100000:
        return False, None

    boundaries = []
    prev_t = 0.0
    total_cost = 0
    for st in steps:
        if not isinstance(st, dict):
            return False, None
        t1 = st.get("t1")
        method = st.get("method")
        if not isinstance(t1, (int, float)) or isinstance(t1, bool) or not math.isfinite(t1):
            return False, None
        t1 = float(t1)
        if method not in ("explicit", "implicit"):
            return False, None
        if not (t1 > prev_t + 1e-12):
            return False, None
        if t1 > T + 1e-6:
            return False, None
        boundaries.append((t1, method))
        total_cost += ce if method == "explicit" else ci
        prev_t = t1

    if total_cost > max_evals:
        return False, None
    if abs(prev_t - T) > 1e-6:
        return False, None

    boundary_ts = [b[0] for b in boundaries]
    bi = 0
    matched = []
    for c in checkpoints:
        while bi < len(boundary_ts) and boundary_ts[bi] < c - 1e-6:
            bi += 1
        if bi >= len(boundary_ts) or abs(boundary_ts[bi] - c) > 1e-6:
            return False, None
        matched.append(bi)

    traj = _integrate(pub["y0"], boundaries, pub["K_segments"], pub["E_coef"])
    y_ref = inst["hidden"]["y_ref"]
    errs = [(traj[matched[i]] - y_ref[i]) ** 2 for i in range(len(checkpoints))]
    rmse = math.sqrt(sum(errs) / len(errs))
    if not math.isfinite(rmse) or rmse < 0.0:
        return False, None
    return True, min(rmse, ERR_CAP)


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <candidate.py>")
        sys.exit(2)
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        ans, st = isorun.run_candidate(cand, inst["public"], timeout=20)
        if st != "OK":
            vec.append(0.0); continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok, obj = False, None
        if not ok or obj is None or obj <= 0.0:
            vec.append(0.0); continue
        b = baseline(inst)
        r = min(1.0, 0.1 * b / max(obj, 1e-12))
        vec.append(r if (r == r and 0.0 <= r <= 1.0) else 0.0)
    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
