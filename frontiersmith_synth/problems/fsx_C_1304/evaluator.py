#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_1304 -- "Watering a Field That Remembers Last Week"
(family: irrigation-scheduling-policy; format B, quality-metric).

THEME.  A field is irrigated over a T-day growing season.  Water pumped or fallen
as rain does not reach the crop's roots instantly: it first lands in a thin
SURFACE layer (capacity Cs) and only a fraction `alpha` of whatever sits in that
layer PERCOLATES DOWN into the deep ROOT-ZONE reservoir (capacity Cr) each day --
the rest stays in transit for the next day, or is lost as runoff if the surface
layer is already full.  The crop only ever drinks from the root zone.  This two
layer bucket is the field's "storage memory": water applied today is not fully
usable today, and water applied when the surface is already saturated (e.g. right
before a rainstorm) is wasted.

The crop passes through four PHENOLOGICAL STAGES (establishment, vegetative,
flowering, maturation); each stage has its own minimum comfortable root-zone
moisture fraction and its own SENSITIVITY WEIGHT for how much a shortfall there
hurts yield -- flowering is by far the most sensitive and demanding.  Pumping
water costs money at a TIME-VARYING TARIFF (cents per mm) that is not correlated
with when the crop needs water; a naive controller reacts to today's moisture
regardless of what tomorrow's forecast or the price schedule say.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance) -- see statement.md for the full
          schema (rain/et/tariff/stage sequences of length T, plus physical
          params: Cs, Cr, alpha, theta_max, stage_theta_min, stage_sensitivity,
          S0, R0, max_irrig_per_day, cost_scale, Y_max).
  stdout: ONE JSON object: {"irrig": [x_0, ..., x_{T-1}]}  -- the full-season
          irrigation SCHEDULE (mm applied on each day), chosen with full
          knowledge of the season's forecast and prices.

  A schedule is VALID iff `irrig` has exactly T finite numbers, each in
  [0, max_irrig_per_day] (small tolerance).  Invalid output, wrong length, an
  out-of-range dose, a crash, a timeout, or non-JSON -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  The evaluator re-simulates the TRUE
two-layer water balance for the candidate's schedule and computes
    objective = yield_value - cost_scale * total_pumping_cost
where yield_value = Y_max * (1 - normalized_stress) and normalized_stress
accumulates, day by day, a stage-sensitivity-weighted moisture-deficit penalty
(and a smaller waterlogging penalty above theta_max).  Three references are
computed by THIS parent process, per instance:
    obj_weak  = objective of an internal WEAK fixed-target controller (a
                single-bucket, tariff/stage/lag-blind rule) -- anchors r=0.1
    obj_ideal = a LOOSE, generally-unreachable upper reference (zero stress,
                the whole season's net water need bought at an artificially
                deep discount off the cheapest day's price) -- anchors r=1.0
    obj_cand  = objective the CANDIDATE's schedule actually achieves
and normalizes with the affine anchor
    r = clamp(0.1 + 0.9 * (obj_cand - obj_weak) / max(eps, obj_ideal - obj_weak), 0, 1)

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  The references
(weak controller, loose ideal, true physics) live only in THIS parent process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        if hi <= lo:
            return lo
        return lo + (state >> 33) % (hi - lo + 1)

    def nxt_float():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return ((state >> 11) & ((1 << 53) - 1)) / float(1 << 53)

    return nxt_int, nxt_float


# ----------------------------- fixed physical constants --------------------
STAGE_THETA_MIN = [0.30, 0.42, 0.62, 0.38]   # establishment, vegetative, flowering, maturation
STAGE_SENS = [0.4, 0.8, 2.0, 0.6]
CS = 45.0
CR = 140.0
THETA_MAX = 0.93
S0 = 5.0
R0 = 0.5 * CR
MAX_IRRIG = 35.0
Y_MAX = 100.0
COST_SCALE = 0.16
IDEAL_SHRINK = 0.45      # how far below the cheapest realized tariff the "ideal" reference prices water


# ----------------------------- instance construction ------------------------
def _make_stage_array(T, seed, fracs):
    _, nxtf = _rng(seed + 3)
    js = [max(0.05, f * (1.0 + (nxtf() - 0.5) * 0.2)) for f in fracs]
    s = sum(js)
    js = [j / s for j in js]
    lens = [max(1, int(round(j * T))) for j in js]
    lens[-1] += T - sum(lens)
    stages = []
    for i, l in enumerate(lens):
        stages += [i] * max(0, l)
    stages = stages[:T]
    while len(stages) < T:
        stages.append(3)
    return stages


def _make_weather(T, seed, kind, stages):
    _, nxtf = _rng(seed + 7)
    rain = [0.0] * T
    et = [0.0] * T
    for t in range(T):
        base_et = 4.0 + (1.5 if stages[t] in (1, 2) else 0.0)
        et[t] = round(base_et + (nxtf() - 0.5) * 0.8, 3)
        if kind == "steady_light":
            rain[t] = round(nxtf() * 3.0, 3) if nxtf() < 0.5 else 0.0
        elif kind == "steady_dry":
            rain[t] = round(nxtf() * 1.0, 3) if nxtf() < 0.2 else 0.0
        elif kind == "bimodal":
            rain[t] = round(nxtf() * 2.0, 3)
            if nxtf() < 0.08:
                rain[t] = round(rain[t] + 10.0 + nxtf() * 10.0, 3)
        else:
            rain[t] = round(nxtf() * 2.0, 3)
    return rain, et


def _inject_rain_trap(rain, seed, dry_len=5, burst=22.0):
    nxt, _ = _rng(seed + 13)
    T = len(rain)
    start = nxt(3, max(4, T - dry_len - 4))
    for i in range(start, min(T, start + dry_len)):
        rain[i] = 0.0
    burst_day = min(T - 1, start + dry_len)
    rain[burst_day] = round(rain[burst_day] + burst, 3)


def _make_tariff(T, seed, stages, kind, mult=2.8):
    if kind == "spike":
        tariff = [round(1.0 + 0.15 * (((i * 37) % 5) - 2) / 5.0, 4) for i in range(T)]
        for i, s in enumerate(stages):
            if s == 2:
                tariff[i] = mult
        return tariff
    _, nxtf = _rng(seed + 41)
    return [round(0.9 + nxtf() * 0.4, 4) for _ in range(T)]


def _build_instance(spec):
    seed, T, kind, alpha, tariff_kind, fracs = spec
    stages = _make_stage_array(T, seed, fracs)
    rain, et = _make_weather(T, seed, kind, stages)
    if kind == "rain_trap":
        _inject_rain_trap(rain, seed)
    tariff = _make_tariff(T, seed, stages, tariff_kind)
    return {
        "name": f"field{seed}", "T": T, "rain": rain, "et": et, "tariff": tariff,
        "stage": stages,
        "params": {
            "Cs": CS, "Cr": CR, "alpha": alpha, "theta_max": THETA_MAX,
            "stage_theta_min": STAGE_THETA_MIN, "stage_sensitivity": STAGE_SENS,
            "S0": S0, "R0": R0, "max_irrig_per_day": MAX_IRRIG,
            "cost_scale": COST_SCALE, "Y_max": Y_MAX,
        },
    }


def _build_instances():
    specs = [
        (101, 40, "steady_light", 0.45, "flat", [0.15, 0.30, 0.20, 0.35]),
        (102, 40, "steady_dry", 0.45, "flat", [0.15, 0.30, 0.20, 0.35]),
        (203, 42, "rain_trap", 0.40, "flat", [0.15, 0.30, 0.20, 0.35]),       # TRAP: rain imminent
        (204, 44, "steady_light", 0.22, "flat", [0.20, 0.30, 0.20, 0.30]),    # TRAP: slow percolation / flowering lag
        (305, 40, "steady_light", 0.45, "spike", [0.15, 0.30, 0.20, 0.35]),   # TRAP: tariff spike at flowering
        (306, 46, "rain_trap", 0.24, "spike", [0.18, 0.28, 0.22, 0.32]),      # TRAP: combined, held-out
        (150, 38, "bimodal", 0.35, "flat", [0.15, 0.30, 0.20, 0.35]),
        (407, 50, "rain_trap", 0.22, "spike", [0.16, 0.30, 0.22, 0.32]),      # TRAP: combined, harder held-out
        (408, 48, "bimodal", 0.26, "spike", [0.18, 0.28, 0.24, 0.30]),
        (409, 52, "steady_dry", 0.22, "flat", [0.20, 0.30, 0.22, 0.28]),
    ]
    return [_build_instance(s) for s in specs]


# ----------------------------- true physics simulator -----------------------
def _simulate(inst, irrig):
    """Run the TRUE two-layer water balance for a full-season irrigation
    schedule. Returns (objective, yield_value, cost, normalized_stress)."""
    p = inst["params"]
    Cs, Cr, alpha = p["Cs"], p["Cr"], p["alpha"]
    theta_max = p["theta_max"]
    smin = p["stage_theta_min"]
    sens = p["stage_sensitivity"]
    S, R = p["S0"], p["R0"]
    T = inst["T"]
    rain, et, tariff, stage = inst["rain"], inst["et"], inst["tariff"], inst["stage"]
    total_pen = 0.0
    total_sens = 0.0
    cost = 0.0
    for t in range(T):
        S += rain[t] + irrig[t]
        if S > Cs:
            S = Cs
        perc = alpha * S
        room = max(0.0, Cr - R)
        perc_applied = min(perc, room)
        S -= perc
        if S < 0.0:
            S = 0.0
        R += perc_applied
        theta = R / Cr
        st = stage[t]
        th_min = smin[st]
        se = sens[st]
        deficit = 0.0
        if th_min > 0:
            deficit = max(0.0, (th_min - theta) / th_min)
            if deficit > 1.0:
                deficit = 1.0
        excess = 0.0
        if theta > theta_max:
            excess = min(1.0, (theta - theta_max) / max(1e-9, 1.0 - theta_max))
        total_pen += se * (deficit + 0.5 * excess)
        total_sens += se
        R -= min(R, et[t])
        cost += tariff[t] * irrig[t]
    norm_pen = total_pen / total_sens if total_sens > 0 else 0.0
    yield_v = p["Y_max"] * max(0.0, 1.0 - norm_pen)
    obj = yield_v - p["cost_scale"] * cost
    return obj, yield_v, cost, norm_pen


def _weak_fixed_target_irrig(inst, theta_target=0.50):
    """WEAK internal reference: a single-bucket controller that ignores the
    surface/root split, the tariff schedule, and the stage-specific need --
    it just tries to hold ONE flat moisture target using its own (wrong,
    lag-free) mental model of the reservoir."""
    p = inst["params"]
    Cr = p["Cr"]
    T = inst["T"]
    maxirr = p["max_irrig_per_day"]
    M = p["R0"] + p["S0"]
    irr = []
    for t in range(T):
        M += inst["rain"][t]
        add = max(0.0, min(maxirr, theta_target * Cr - M))
        M += add
        M -= inst["et"][t]
        if M < 0.0:
            M = 0.0
        if M > Cr:
            M = Cr
        irr.append(add)
    return irr


def _ideal_obj(inst):
    """Loose, generally-unreachable reference: zero stress, and the season's
    entire net water need bought at an artificial discount below the cheapest
    realized tariff (no schedule can actually buy below the cheapest price
    offered, so this ceiling is never fully reachable)."""
    p = inst["params"]
    V = sum(max(0.0, inst["et"][t] - inst["rain"][t]) for t in range(inst["T"]))
    cost_ideal = p["cost_scale"] * V * min(inst["tariff"]) * IDEAL_SHRINK
    return p["Y_max"] - cost_ideal


# ----------------------------- answer validation -----------------------------
def _extract_schedule(inst, answer):
    if not isinstance(answer, dict):
        return None
    irr = answer.get("irrig")
    if not isinstance(irr, list):
        return None
    T = inst["T"]
    if len(irr) != T:
        return None
    maxirr = inst["params"]["max_irrig_per_day"]
    out = []
    for x in irr:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return None
        xf = float(x)
        if not (xf == xf) or xf in (float("inf"), float("-inf")):
            return None
        if xf < -1e-6 or xf > maxirr + 1e-6:
            return None
        out.append(max(0.0, min(maxirr, xf)))
    return out


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        obj_weak = _simulate(inst, _weak_fixed_target_irrig(inst))[0]
        obj_ideal = _ideal_obj(inst)
        denom = obj_ideal - obj_weak
        if denom < 1e-6:
            denom = 1e-6

        public = {
            "name": inst["name"], "T": inst["T"],
            "rain": list(inst["rain"]), "et": list(inst["et"]),
            "tariff": list(inst["tariff"]), "stage": list(inst["stage"]),
            "params": dict(inst["params"]),
        }
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            sched = _extract_schedule(inst, ans)
        except Exception:
            sched = None
        if sched is None:
            vec.append(0.0)
            continue
        try:
            obj_cand = _simulate(inst, sched)[0]
        except Exception:
            vec.append(0.0)
            continue

        r = 0.1 + 0.9 * (obj_cand - obj_weak) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        vec.append(max(0.0, min(1.0, r)))

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
