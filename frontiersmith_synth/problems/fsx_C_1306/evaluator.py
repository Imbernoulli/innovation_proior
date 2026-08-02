#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_1306 -- "Water You Release Now Is Water You Lack Later:
Reservoir Release Policy Against Degrading Forecast Skill"
(family: reservoir-release-policy; format B, quality-metric).

THEME.  A single reservoir serves a downstream demand schedule over a fixed
horizon of T days.  Every day the operator chooses a RELEASE.  Water arrives
as INFLOW (partly baseflow, partly occasional big pulses / floods).  The
reservoir has a hard CAPACITY: any inflow that would push storage above
capacity, net of that day's release, SPILLS and causes flood damage
(super-linear in the spill size).  Storage may never usefully exceed
capacity nor drop below a DEAD STORAGE floor.  Release is also physically
capped at a maximum daily rate (you cannot empty a full reservoir in one
day).

MECHANISM 1 (flood-buffer-vs-storage).  Keeping the reservoir topped up
maximizes supply security day to day, but leaves no free headroom to absorb
a big inflow pulse -- forcing a large, costly spill.  Emptying it protects
against floods but starves demand.  The right amount of standing headroom
(buffer) trades against the right amount of standing storage.

MECHANISM 2 (inflow-forecast-skill).  Each instance ships a single
long-range inflow FORECAST for the whole horizon, together with a SKILL
curve `skill[t] in [0,1]` describing how much that forecast can be trusted
at lead time `t` (skill decays from ~1 at t=0 toward a low floor; the decay
RATE differs between instances -- some sites have long-skilled forecasts,
others ("flash" sites) lose skill within days).  The forecast is built as
`skill[t]*true_inflow[t] + (1-skill[t])*climatology[t]`: at high skill it
tracks the truth; at low skill it regresses to the (fully public) seasonal
`climatology` baseline and a big pulse becomes invisible in it.

MECHANISM 3 (downstream-demand-schedule).  A known demand schedule (with a
seasonal peak window on some instances) must be served; only release up to
that day's demand is credited, so dumping water early to "be safe" forfeits
future supply credit.

INNOVATION HOOK.  A policy that always targets a fixed high storage level
(the obvious "keep it full for supply security" instinct) gets blindsided:
on instances where a big pulse lands during a low-skill window, the
forecast never warned it and a fixed-high-target policy is caught full,
producing large spill.  The insight: the standing buffer should track
FORECAST SKILL -- hold LESS storage (keep more headroom) when you cannot
see far ahead, and can afford to sit closer to full only when skill stays
high far enough out to see trouble coming AND still have enough days at
the release cap to drain in time.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "T": int, "capacity": float, "dead_storage": float,
             "init_storage": float, "r_max": float, "flood_cost_coef": float,
             "flash_risk": float,
             "climatology": [T floats], "skill": [T floats in [0,1]],
             "forecast": [T floats], "demand": [T floats]}
  stdout: ONE JSON object:
            {"release": [x_0, ..., x_{T-1}]}   # x_t >= 0, finite, requested release on day t

  A request exceeding what's physically available or the daily rate cap is
  silently CLIPPED (see DYNAMICS).  `release` must be a list of exactly T
  finite non-negative numbers (ints or floats; no bool/str/nan/inf).  Any
  shape/type violation, a crash, a timeout, or non-JSON output makes that
  instance score 0.0.

DYNAMICS (per day t; evaluator-side, uses the TRUE hidden inflow sequence).
    avail        = S + true_inflow[t] - dead_storage
    applied      = clip(release_t, 0, r_max, avail)      # water actually released
    raw_after    = S + true_inflow[t] - applied
    spill        = max(0, raw_after - capacity)           # forced overflow
    S            = clip(raw_after - spill, dead_storage, capacity)
    supply      += min(applied, demand[t])                 # credit capped at that day's demand
    flood       += spill ** 1.5                             # super-linear flood cost
  objective = supply - flood_cost_coef * flood

SCORING (deterministic; no wall-time).  Per instance the evaluator computes,
itself, two references:
    q_base  = objective of the naive "release = demand every day" policy
              (never proactively manages storage at all)
    q_ideal = sum(demand)   # generous UNREACHABLE ceiling: perfect demand
              service AND zero flood damage simultaneously -- real spill is
              essentially unavoidable on the flood-pulse instances because
              of the r_max drain-rate cap, so this ceiling stays out of
              reach and leaves headroom.
    q_cand  = candidate's realized objective
and normalizes with an affine anchor (naive baseline -> 0.1, ceiling -> 1.0):
    r = clamp( 0.1 + 0.9 * (q_cand - q_base) / max(eps, q_ideal - q_base), 0, 1 )

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance (never the
true hidden inflow sequence, q_base, or q_ideal).  All references and the
true dynamics are computed by THIS parent process, so a frame-walking /
introspecting candidate learns nothing useful.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math, random
import isorun

CAPACITY = 1000.0
DEAD_FRAC = 0.05


# ----------------------------- instance construction ------------------------
def _climatology(t, T, mean_level, amp_frac, phase):
    return mean_level * (1.0 + amp_frac * math.sin(2.0 * math.pi * (t - phase) / T))


def _demand_profile(t, base_demand, peak_window, peak_mult):
    d = base_demand
    if peak_window is not None:
        lo, hi = peak_window
        if lo <= t <= hi:
            d = base_demand * peak_mult
    return d


def _pulse_contrib(t, pulses):
    add = 0.0
    for day, mag, width in pulses:
        if day <= t < day + width:
            add += mag / width
    return add


def _build_instance(spec):
    rnd = random.Random(spec["seed"])
    T = spec["T"]
    C = CAPACITY
    Dmin = DEAD_FRAC * C
    Rmax = spec["r_max"]
    S0 = spec.get("s0_frac", 0.55) * C

    skill = [spec["skill_floor"] + (1.0 - spec["skill_floor"]) * math.exp(-t / spec["tau"])
             for t in range(T)]
    climatology = [_climatology(t, T, spec["mean_level"], spec["amp_frac"], spec["phase"])
                   for t in range(T)]

    true_inflow = []
    for t in range(T):
        noise = 0.85 + 0.30 * rnd.random()          # +-15% baseflow noise
        val = climatology[t] * noise + _pulse_contrib(t, spec["pulses"])
        true_inflow.append(max(0.0, val))

    demand = []
    for t in range(T):
        dnoise = 0.90 + 0.20 * rnd.random()
        demand.append(_demand_profile(t, spec["base_demand"], spec.get("peak_window"),
                                       spec.get("peak_mult", 1.0)) * dnoise)

    forecast = [skill[t] * true_inflow[t] + (1.0 - skill[t]) * climatology[t] for t in range(T)]

    # flash_risk: worst known surge as a fraction of CAPACITY (a volume, not a
    # daily rate) -- a multi-day pulse can dump more water than any single
    # day's rate suggests, and it's the total volume that decides how much
    # standing headroom is needed when you can't see it coming at all.
    worst_pulse_volume = max((mag for (_d, mag, _w) in spec["pulses"]), default=0.0)
    flash_risk = round(max(0.05, worst_pulse_volume / C), 3)

    return {
        "name": spec["name"], "T": T, "capacity": C, "dead_storage": Dmin,
        "init_storage": S0, "r_max": Rmax, "flood_cost_coef": spec["flood_cost_coef"],
        "flash_risk": flash_risk,
        "climatology": [round(x, 4) for x in climatology],
        "skill": [round(x, 4) for x in skill],
        "forecast": [round(x, 4) for x in forecast],
        "demand": [round(x, 4) for x in demand],
        "_true_inflow": true_inflow,
    }


def _build_instances():
    specs = [
        # 1. calm, good long-range skill: everyone should be fine here.
        dict(name="calm_longskill", seed=101, T=50, tau=30, skill_floor=0.10,
             r_max=70.0, mean_level=28.0, amp_frac=0.25, phase=10.0,
             base_demand=22.0, peak_window=None, flood_cost_coef=0.05,
             pulses=[(33, 430.0, 3)]),
        # 2. calm-ish, poor long-range skill, but NO real flood -> tests that
        #    low skill alone shouldn't force wasteful over-buffering.
        dict(name="calm_flashsite", seed=202, T=50, tau=5, skill_floor=0.08,
             r_max=70.0, mean_level=27.0, amp_frac=0.20, phase=5.0,
             base_demand=22.0, peak_window=None, flood_cost_coef=0.05,
             pulses=[]),
        # 3. TRAP: fast skill decay, big pulse deep in the low-skill window.
        dict(name="trap_fast_late", seed=303, T=50, tau=5, skill_floor=0.08,
             r_max=50.0, mean_level=26.0, amp_frac=0.20, phase=0.0,
             base_demand=20.0, peak_window=None, flood_cost_coef=0.05,
             pulses=[(35, 950.0, 3)]),
        # 4. TRAP: fast skill decay, big pulse a bit earlier but still past
        #    the skill horizon.
        dict(name="trap_fast_mid", seed=404, T=50, tau=6, skill_floor=0.10,
             r_max=45.0, mean_level=25.0, amp_frac=0.20, phase=0.0,
             base_demand=19.0, peak_window=None, flood_cost_coef=0.05,
             pulses=[(20, 1000.0, 2)]),
        # 5. Long-skill site, LATE big pulse: fully visible far in advance,
        #    so a lookahead policy can drain gradually and dodge it almost
        #    entirely -- but a fixed-target/reactive-only policy still fails
        #    because it never plans ahead regardless of skill.
        dict(name="trap_slow_visible", seed=505, T=50, tau=35, skill_floor=0.15,
             r_max=35.0, mean_level=27.0, amp_frac=0.20, phase=0.0,
             base_demand=21.0, peak_window=None, flood_cost_coef=0.05,
             pulses=[(42, 1300.0, 4)]),
        # 6. medium skill decay, medium pulse -- graded-response check.
        dict(name="medium_partial", seed=606, T=50, tau=12, skill_floor=0.12,
             r_max=55.0, mean_level=26.0, amp_frac=0.20, phase=8.0,
             base_demand=20.0, peak_window=None, flood_cost_coef=0.05,
             pulses=[(25, 700.0, 2)]),
        # 7. demand-peak stress with only a mild pulse: must save water for
        #    the peak while still handling the flood -- tight initial
        #    storage so a wasteful pre-flood drain genuinely risks the peak.
        dict(name="demand_peak_mild", seed=707, T=50, tau=20, skill_floor=0.12,
             r_max=60.0, mean_level=27.0, amp_frac=0.20, phase=0.0,
             base_demand=18.0, peak_window=(30, 42), peak_mult=2.3,
             s0_frac=0.30, flood_cost_coef=0.05, pulses=[(15, 640.0, 2)]),
        # 8. demand-peak stress + a real low-skill flood -- both mechanisms
        #    binding at once: the standing buffer needed to survive the
        #    flood eats directly into the water banked for the peak.
        dict(name="demand_peak_trap", seed=808, T=50, tau=6, skill_floor=0.10,
             r_max=45.0, mean_level=25.0, amp_frac=0.20, phase=0.0,
             base_demand=18.0, peak_window=(34, 46), peak_mult=2.3,
             s0_frac=0.48, flood_cost_coef=0.05, pulses=[(20, 850.0, 3)]),
        # 9. harder held-out: two pulses, tight r_max.
        dict(name="held_out_double_pulse", seed=909, T=54, tau=8, skill_floor=0.10,
             r_max=38.0, mean_level=24.0, amp_frac=0.20, phase=0.0,
             base_demand=18.0, peak_window=None, flood_cost_coef=0.06,
             pulses=[(16, 620.0, 2), (40, 700.0, 2)]),
        # 10. harder held-out: very poor skill, big late pulse, demand peak
        #     placed BEFORE the pulse to test sequencing.
        dict(name="held_out_flash_afterpeak", seed=1010, T=54, tau=4, skill_floor=0.06,
             r_max=40.0, mean_level=24.0, amp_frac=0.20, phase=0.0,
             base_demand=18.0, peak_window=(6, 16), peak_mult=2.0,
             s0_frac=0.48, flood_cost_coef=0.06, pulses=[(46, 1100.0, 4)]),
    ]
    return [_build_instance(s) for s in specs]


# ----------------------------- dynamics / scoring ---------------------------
def _simulate(inst, release):
    T = inst["T"]; C = inst["capacity"]; Dmin = inst["dead_storage"]
    Rmax = inst["r_max"]; demand = inst["demand"]; true_inflow = inst["_true_inflow"]
    S = inst["init_storage"]
    supply_total = 0.0
    flood_total = 0.0
    for t in range(T):
        req = release[t]
        avail = S + true_inflow[t] - Dmin
        if avail < 0.0:
            avail = 0.0
        applied = req
        if applied < 0.0:
            applied = 0.0
        if applied > Rmax:
            applied = Rmax
        if applied > avail:
            applied = avail
        raw_after = S + true_inflow[t] - applied
        spill = raw_after - C
        if spill < 0.0:
            spill = 0.0
        S_next = raw_after - spill
        if S_next < Dmin:
            S_next = Dmin
        if S_next > C:
            S_next = C
        supply_total += min(applied, demand[t])
        flood_total += spill ** 1.5
        S = S_next
    return supply_total, flood_total


def _validate_release(inst, answer):
    if not isinstance(answer, dict):
        return None
    release = answer.get("release")
    if not isinstance(release, list) or len(release) != inst["T"]:
        return None
    out = []
    for x in release:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return None
        xf = float(x)
        if xf != xf or xf in (float("inf"), float("-inf")) or xf < 0.0:
            return None
        out.append(xf)
    return out


def score(inst, answer):
    release = _validate_release(inst, answer)
    if release is None:
        return False, 0.0
    supply, flood = _simulate(inst, release)
    obj = supply - inst["flood_cost_coef"] * flood
    return True, obj


def baseline(inst):
    supply, flood = _simulate(inst, list(inst["demand"]))
    return supply - inst["flood_cost_coef"] * flood


def ideal(inst):
    return sum(inst["demand"])


# ----------------------------- scoring driver -------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        public = {k: v for k, v in inst.items() if not k.startswith("_")}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0)
            continue
        b = baseline(inst)
        q_ideal = ideal(inst)
        denom = q_ideal - b
        if denom < 1e-6:
            denom = 1e-6
        r = 0.1 + 0.9 * (obj - b) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            r = 0.0
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
