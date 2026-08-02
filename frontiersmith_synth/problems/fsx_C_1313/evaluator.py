#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_1313 -- "Kiln Firing Policy: Gate the Ramp by the Thickest Piece"
(family: kiln-firing-policy; eval_form: quality-metric; wave3-lens:policy-simulator).

A kiln load contains several ceramic pieces of DIFFERENT wall thickness fired together on
ONE shared temperature-vs-time schedule.  Each piece's CORE temperature lags the kiln's
(surface) temperature; the lag time constant grows with the SQUARE of the piece's thickness
(standard heat-diffusion scaling).  Silica in the clay body undergoes abrupt phase
inversions at specific temperature BANDS (~quartz ~560-600C, ~cristobalite ~210-245C).
Inside a band, a surface/core temperature GRADIENT converts into cumulative thermal-shock
stress (quadratic in the gradient); OUTSIDE any band the material tolerates any gradient
for free.  If a piece's cumulative in-band stress exceeds its crack threshold it is damaged
(value degrades linearly past the threshold, reaching zero at 2x threshold).  Firing also
costs fuel proportional to total minutes, so idling forever is not free either.

The candidate submits ONE firing schedule (list of {"to_temp","minutes"} ramp segments,
applied to the WHOLE kiln -- every piece sees the same surface-temperature curve).  It is run
as an ISOLATED subprocess (isorun): reads ONE JSON public instance from stdin, writes ONE JSON
answer to stdout.  It never sees this evaluator's internals.

Public instance JSON (candidate's stdin) -- the FULL instance, nothing hidden (this is an
optimization problem over fully-known parameters, not a labeled-data problem):
    {
      "start_temp": float, "target_temp": float, "max_rate": float,   # deg C / minute cap
      "max_total_minutes": float, "max_segments": int, "sim_dt_minutes": float,
      "fuel_cost_per_minute": float,
      "bands": [ {"lo":float,"hi":float,"multiplier":float,"name":str}, ... ],
      "pieces": [ {"thickness_mm":float,"value":float,"fragility":float}, ... ],
      "diffusion_k": float,        # tau_i (minutes) = diffusion_k * thickness_mm_i^2
      "stress_threshold_k": float  # piece i's crack threshold = stress_threshold_k * fragility_i
    }

Answer JSON (candidate's stdout): a JSON list of ramp segments, OR {"schedule": [...]}, each
segment {"to_temp": float, "minutes": float} meaning "move the kiln surface temperature
linearly from wherever it is now to to_temp over the next minutes minutes" (to_temp must be
non-decreasing across segments; the implied rate (to_temp-prev)/minutes must not exceed
max_rate).  The schedule must reach target_temp; simulation stops the instant it does (extra
segments after that are ignored for timing/fuel, but must still be well-formed).

Per-piece core temperature is an exact analytic solve of the linear lag ODE
dcore/dt = (T_surf(t) - core)/tau_i (no Euler discretization / no numerical instability,
regardless of step size).  In-band cumulative stress is the exact integral of
(T_surf(t)-core(t))^2 over time spent with the segment's midpoint temperature inside a band,
scaled by that band's multiplier (0 outside any band).  Fuel cost is
fuel_cost_per_minute * total_minutes_to_reach_target.  raw_obj = sum(surviving piece values) -
fuel_cost.

Normalization anchors the evaluator computes ITSELF (never sent to the candidate):
  - baseline: the naive "ramp to target_temp at max_rate the whole way" schedule (fast
    everywhere -> minimal fuel, but ignores the bands -> cracks the load on mixed-thickness
    kilns).
  - ceiling: a schedule that is fast OUTSIDE any band and, INSIDE each band, uses the fastest
    rate that keeps the THICKEST piece's projected in-band stress within 92% of its budget
    (found by exact bisection on the analytic ODE -- carrying the thick piece's real
    cumulative stress/core state across bands in order).
    r = clamp(0.1 + 0.9*(obj_cand - obj_base)/max(obj_ceiling - obj_base, 1.0), 0, 1),
    floored to a small positive value for any VALIDLY-SCORED instance.

CLI: python3 evaluator.py <candidate.py>
Prints:
  Ratio: <mean of per-instance r, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun

VALID_FLOOR = 0.02
DENOM_FLOOR = 1.0
CAND_TIMEOUT = 20
MAX_SEGMENTS = 500


# ============================ exact per-piece ODE step ======================
def band_mult_at(T, bands):
    """Outside any band, gradient stress is tolerated for free (mult=0):
    'speed elsewhere is free'.  Only inside a band does gradient convert to damage."""
    m = 0.0
    for b in bands:
        if b["lo"] <= T <= b["hi"]:
            if b["multiplier"] > m:
                m = b["multiplier"]
    return m


def analytic_step(core0, T0, k, h, tau):
    """Exact solve of dcore/dt=(T(t)-core)/tau on [0,h], T(t)=T0+k*t (k=ramp rate, may be 0
    for a hold).  grad(t)=T(t)-core(t) = A - B*exp(-t/tau), A=k*tau, B=(core0-T0)+A.
    Returns (core_at_h, integral_of_grad(t)^2_dt) -- both closed-form, unconditionally
    stable for ANY h (no Euler-style blow-up even when h >> tau or h << tau)."""
    if tau <= 1e-9:
        return T0 + k * h, 0.0
    u0 = core0 - T0
    A = k * tau
    B = u0 + A
    e = 0.0 if h / tau > 40.0 else math.exp(-h / tau)
    e2 = e * e
    T_h = T0 + k * h
    core_h = T_h + B * e - A
    integral = A * A * h - 2.0 * A * B * tau * (1.0 - e) + B * B * (tau / 2.0) * (1.0 - e2)
    if integral < 0.0:
        integral = 0.0
    return core_h, integral


def simulate_plan(inst, plan):
    """plan: list of (to_temp, minutes) already validated/clipped to reach target_temp
    exactly at the end.  Returns (raw_obj, per_piece_damage, total_minutes)."""
    pieces = inst["pieces"]; bands = inst["bands"]
    start = inst["start_temp"]
    diffusion_k = inst["diffusion_k"]; stress_k = inst["stress_threshold_k"]
    dt = inst["sim_dt_minutes"]
    n = len(pieces)
    tau = [diffusion_k * (p["thickness_mm"] ** 2) for p in pieces]
    core = [start] * n
    cum = [0.0] * n
    cur = start
    total_minutes = 0.0
    for (to_t, mins) in plan:
        if mins <= 0.0:
            cur = to_t
            continue
        k = (to_t - cur) / mins
        nsteps = max(1, int(math.ceil(mins / dt)))
        h = mins / nsteps
        seg_t0 = cur
        for s in range(nsteps):
            Ts = seg_t0 + k * (s * h)
            Tmid = Ts + k * (h / 2.0)
            for i in range(n):
                mult = band_mult_at(Tmid, bands)
                new_core, integ = analytic_step(core[i], Ts, k, h, tau[i])
                if mult > 0.0:
                    cum[i] += integ * mult
                core[i] = new_core
        cur = to_t
        total_minutes += mins
    damage = [cum[i] / (stress_k * pieces[i]["fragility"]) for i in range(n)]
    values = [pieces[i]["value"] * max(0.0, min(1.0, 1.0 - max(0.0, damage[i] - 1.0)))
              for i in range(n)]
    raw_obj = sum(values) - inst["fuel_cost_per_minute"] * total_minutes
    return raw_obj, damage, total_minutes


def naive_plan(inst):
    """The naive 'heat as fast as the burner allows' schedule -> the evaluator's baseline."""
    return [(inst["target_temp"],
              (inst["target_temp"] - inst["start_temp"]) / inst["max_rate"])]


def bisect_ceiling_plan(inst, damage_target_frac, min_rate_frac=0.03, iters=50):
    """Fast outside bands; inside each band (in temperature order) bisect the fastest rate
    that keeps the THICKEST piece's projected cumulative in-band stress within
    damage_target_frac of its threshold, carrying its real core state/cumulative stress
    across bands.  This is the evaluator's internal near-optimal ceiling reference."""
    pieces = inst["pieces"]; bands = sorted(inst["bands"], key=lambda b: b["lo"])
    start = inst["start_temp"]; target = inst["target_temp"]; max_rate = inst["max_rate"]
    diffusion_k = inst["diffusion_k"]; stress_k = inst["stress_threshold_k"]
    thick_idx = max(range(len(pieces)), key=lambda i: pieces[i]["thickness_mm"])
    tau_t = diffusion_k * (pieces[thick_idx]["thickness_mm"] ** 2)
    thr_t = stress_k * pieces[thick_idx]["fragility"] * damage_target_frac

    core = start; cum = 0.0; cur = start
    segs = []
    for b in bands:
        lo, hi = b["lo"], b["hi"]
        if hi <= cur:
            continue
        if lo > cur:
            mins = (lo - cur) / max_rate
            core, _ = analytic_step(core, cur, max_rate, mins, tau_t)
            segs.append((lo, mins)); cur = lo
        width = hi - max(lo, cur)
        if width <= 0.0:
            continue
        mult = b["multiplier"]
        remaining = thr_t - cum
        lo_r, hi_r = min_rate_frac * max_rate, max_rate

        def integ_at(r):
            mins = width / r
            _, ig = analytic_step(core, cur, r, mins, tau_t)
            return ig * mult

        if remaining <= 0.0:
            r_use = lo_r
        elif integ_at(hi_r) <= remaining:
            r_use = hi_r
        elif integ_at(lo_r) > remaining:
            r_use = lo_r
        else:
            a, bb = lo_r, hi_r
            for _ in range(iters):
                mid = (a + bb) / 2.0
                if integ_at(mid) <= remaining:
                    a = mid
                else:
                    bb = mid
            r_use = a
        mins = width / r_use
        core, integ = analytic_step(core, cur, r_use, mins, tau_t)
        cum += integ * mult
        segs.append((hi, mins)); cur = hi
    if cur < target:
        segs.append((target, (target - cur) / max_rate))
        cur = target
    return segs


# ============================ instance family ============================
def make_instances():
    import random
    rng = random.Random(20261313)
    diffusion_k = 0.0008
    stress_threshold_k = 1500.0
    specs = [("easy", 0), ("easy", 1), ("easy", 2),
              ("trap", 0), ("trap", 1), ("trap", 2), ("trap", 3),
              ("trap", 4), ("trap", 5), ("trap", 6)]
    out = []
    for kind, idx in specs:
        target_temp = rng.uniform(1180.0, 1260.0)
        max_rate = rng.uniform(7.0, 11.0)
        fuel_cost_per_minute = rng.uniform(0.045, 0.065)
        n_pieces = rng.randint(3, 5)
        pieces = []
        for j in range(n_pieces):
            if kind == "trap" and j == 0:
                th = rng.uniform(28.0, 45.0)   # planted thick section
            else:
                th = rng.uniform(6.0, 16.0) if kind == "trap" else rng.uniform(6.0, 14.0)
            pieces.append({
                "thickness_mm": round(th, 3),
                "value": round(rng.uniform(8.0, 15.0), 3),
                "fragility": round(rng.uniform(0.85, 1.15), 4),
            })
        b2lo = rng.uniform(205.0, 215.0); b2hi = b2lo + rng.uniform(25.0, 35.0)
        b2m = round(rng.uniform(1.8, 2.6), 3)
        b1lo = rng.uniform(550.0, 560.0); b1hi = b1lo + rng.uniform(30.0, 45.0)
        b1m = round(rng.uniform(3.0, 4.5), 3)
        bands = [
            {"lo": round(b2lo, 2), "hi": round(b2hi, 2), "multiplier": b2m, "name": "cristobalite"},
            {"lo": round(b1lo, 2), "hi": round(b1hi, 2), "multiplier": b1m, "name": "quartz"},
        ]
        inst = {
            "start_temp": 20.0,
            "target_temp": round(target_temp, 2),
            "max_rate": round(max_rate, 4),
            "max_total_minutes": 1500.0,
            "max_segments": MAX_SEGMENTS,
            "sim_dt_minutes": 2.0,
            "fuel_cost_per_minute": round(fuel_cost_per_minute, 5),
            "bands": bands,
            "pieces": pieces,
            "diffusion_k": diffusion_k,
            "stress_threshold_k": stress_threshold_k,
        }
        out.append(inst)
    return out


# ============================ answer validation + scoring ============================
def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _parse_answer(ans):
    if isinstance(ans, dict):
        segs = ans.get("schedule")
    else:
        segs = ans
    if not isinstance(segs, list) or not (1 <= len(segs) <= MAX_SEGMENTS):
        return None
    return segs


def score(inst, answer):
    """Validate `answer` strictly against `inst`; return (ok, raw_obj)."""
    segs = _parse_answer(answer)
    if segs is None:
        return False, 0.0
    start = inst["start_temp"]; target = inst["target_temp"]; max_rate = inst["max_rate"]
    max_total = inst["max_total_minutes"]

    prev = float(start)
    plan = []
    reached = False
    total_provided = 0.0
    for seg in segs:
        if not isinstance(seg, dict):
            return False, 0.0
        to_t = seg.get("to_temp"); mins = seg.get("minutes")
        if not _finite(to_t) or not _finite(mins):
            return False, 0.0
        to_t = float(to_t); mins = float(mins)
        if mins <= 1e-9 or mins > max_total + 1e-6:
            return False, 0.0
        if to_t < prev - 1e-6:              # must be non-decreasing (firing only heats up)
            return False, 0.0
        rate = (to_t - prev) / mins
        if rate < -1e-6 or rate > max_rate + 1e-6:
            return False, 0.0
        total_provided += mins
        if total_provided > max_total + 1e-6:
            return False, 0.0
        if not reached:
            if to_t >= target - 1e-9:
                clip_mins = mins * (target - prev) / (to_t - prev) if to_t > prev + 1e-12 else 0.0
                if clip_mins < 0.0:
                    clip_mins = 0.0
                plan.append((target, clip_mins))
                reached = True
                prev = target
            else:
                plan.append((to_t, mins))
                prev = to_t
        else:
            prev = to_t
    if not reached:
        return False, 0.0

    try:
        raw_obj, damage, total_minutes = simulate_plan(inst, plan)
    except Exception:
        return False, 0.0
    if not math.isfinite(raw_obj):
        return False, 0.0
    if total_minutes > max_total + 1e-6:
        return False, 0.0
    return True, raw_obj


def baseline(inst):
    raw_obj, _, _ = simulate_plan(inst, naive_plan(inst))
    return raw_obj


def _ceiling(inst):
    raw_obj, _, _ = simulate_plan(inst, bisect_ceiling_plan(inst, 0.92))
    return raw_obj


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <candidate.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = make_instances()

    vec = []
    for inst in instances:
        public = dict(inst)  # everything is public: this is a fully-specified optimization
        ans, st = isorun.run_candidate(cand, public, timeout=CAND_TIMEOUT)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok = False; obj = 0.0
        if not ok:
            vec.append(0.0)
            continue

        obj_base = baseline(inst)
        obj_ceil = _ceiling(inst)
        denom = max(obj_ceil - obj_base, DENOM_FLOOR)
        r = 0.1 + 0.9 * (obj - obj_base) / denom
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        if r < VALID_FLOOR:
            r = VALID_FLOOR
        vec.append(float(r))

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
