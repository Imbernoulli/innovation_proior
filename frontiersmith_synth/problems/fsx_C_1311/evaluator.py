#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_1311 -- "Islanded Microgrid: Pre-emptive Load-Shedding
Reserve Policy" (family: microgrid-dispatch-policy; eval_form: quality-metric).

An islanded microgrid (no grid connection -- PV + battery only) must serve THREE
prioritized load tiers -- critical, important, low -- over a T=14-day, running-out
-of-sun horizon.  Each day the candidate's policy decides, via battery-SoC RESERVE
FLOORS, whether to serve the important and low tiers or shed them to protect
future critical service; critical demand is always attempted in full from
whatever energy is available that day (PV first, battery second) -- it is never
pre-emptively rationed, only PROTECTED by earlier restraint on the lower tiers.
Battery cycling costs a DEGRADATION penalty that grows super-linearly the deeper
the battery is discharged in a given day.  The public instance exposes a PV
FORECAST and a per-day FORECAST-UNCERTAINTY band; the evaluator's HIDDEN actual
PV realization can undercut the forecast severely and for several consecutive
days ("low-sun sequences"), signalled in advance only by elevated uncertainty,
never by the forecast mean itself.

A policy that ALWAYS sets its reserve floors to zero ("serve everything, trust
the forecast") matches the evaluator's own naive baseline on calm instances but
runs the battery down to empty during a low-sun sequence, at which point
CRITICAL service itself collapses (there's no energy left, in PV or battery, to
serve it) -- exactly the failure mode this problem is designed to punish. The
insight this problem rewards is READING the forecast-uncertainty band to size a
pre-emptive reserve (protecting the tiers in priority order: shed low before
important, both before ever touching critical) so the battery still has enough
charge banked when the low-sun sequence actually arrives.

The candidate is run as an ISOLATED subprocess (isorun): it reads ONE JSON
"public instance" from stdin and writes ONE JSON "policy" (two length-T reserve
curves, in kWh) to stdout. It never sees the hidden actual PV trajectory or this
evaluator's memory. Given the policy, THIS evaluator (not the candidate) runs the
deterministic day-by-day dispatch simulation against the hidden actual weather.

Public instance JSON (what the candidate reads on stdin):
    {
      "T": int,                                # number of days (14)
      "capacity_kwh": float,                   # battery capacity
      "initial_soc_kwh": float,                # starting battery charge
      "charge_efficiency": float,              # round-trip charge efficiency (0..1)
      "deg_linear_cost": float,                 # cost per kWh discharged
      "deg_deep_cost": float,                   # extra cost coefficient for deep discharge
      "deep_discharge_frac": float,             # capacity fraction defining "deep"
      "priority_weights": {"critical":.., "important":.., "low":..},
      "critical_demand_kwh":  [float]*T,
      "important_demand_kwh": [float]*T,
      "low_demand_kwh":       [float]*T,
      "pv_forecast_kwh":             [float]*T,
      "pv_forecast_uncertainty_kwh": [float]*T,  # per-day forecast error band (kWh)
      "seed": int
    }

Answer JSON (what the candidate writes on stdout):
    {"reserve_important_kwh": [float]*T, "reserve_low_kwh": [float]*T}
    reserve_X_kwh[t] = the battery-SoC floor (kWh) the policy wants to protect on
    day t; the evaluator serves tier X on day t only up to the point where the
    resulting end-of-day SoC would not fall below that floor. Values are clipped
    into [0, capacity_kwh]; wrong shape / type / non-finite values -> instance
    score 0.

Simulation (day t, deterministic, causal -- only past/current actual PV is used):
    pool          = soc_before + pv_actual[t]
    crit_served   = min(critical_demand[t], pool)                      # always attempted first
    pool          -= crit_served
    imp_afford    = max(0, pool - reserve_important[t])
    imp_served    = min(important_demand[t], imp_afford);  pool -= imp_served
    low_afford    = max(0, pool - reserve_low[t])
    low_served    = min(low_demand[t], low_afford);        pool -= low_served
    discharge     = max(0, total_served - pv_actual[t])              # kWh pulled FROM the battery
    charge        = max(0, pv_actual[t] - total_served)               # kWh surplus PV offered to the battery
    soc_after_discharge = soc_before - discharge
    soc_next      = clip(soc_after_discharge + charge*charge_efficiency, 0, capacity_kwh)   # excess PV spilled
    deep_deficit  = max(0, deep_discharge_frac*capacity_kwh - soc_after_discharge)
    deg_cost[t]   = deg_linear_cost*discharge + deg_deep_cost*discharge*deep_deficit/capacity_kwh

Per-instance objective:
    obj = sum_t( w_crit*crit_served[t] + w_imp*imp_served[t] + w_low*low_served[t] ) - sum_t(deg_cost[t])

Per-instance score is an affine anchor between two references THIS evaluator
computes itself (never the candidate's job): a weak "shed everything but
critical, always" baseline (obj_base -> r=0.1) and a loose "serve 100% of every
tier's demand for free" upper bound (obj_upper -> r->1.0, never reachable given
real energy/degradation constraints):

    r = clamp( 0.1 + 0.9 * (obj_cand - obj_base) / max(obj_upper - obj_base, 1e-6), 0, 1 )

valid instances are floored at 0.01; final Ratio is the ARITHMETIC MEAN of the
10 per-instance r values.  10 instances span calm (ample-sun, only a mild
terminal dip) and five distinct low-sun-sequence "trap" regimes (mid-horizon,
late, early, double-dip, fast-onset) so a policy that ignores the forecast-
uncertainty band and always trusts the forecast mean collapses on the traps
while a policy that never shares energy loses value on the calm days.

CLI:  python3 evaluator.py <candidate.py>
Prints:
  Ratio: <arithmetic mean of per-instance r, in [0,1]>
  Vector: [r_1, r_2, ..., r_10]
"""
import sys, json, math
import isorun

T = 14
W_CRIT, W_IMP, W_LOW = 10.0, 3.0, 1.0
CAND_TIMEOUT = 20
VALID_FLOOR = 0.01


# ============================ instance family ===============================
def _lcg_instance(seed, kind):
    """Deterministic, seeded, pure-python instance generator (no numpy needed)."""
    import random
    rng = random.Random(seed)
    capacity = 32.0
    initial_soc = capacity * 0.5
    eff = 0.92
    deg_lin = 0.15
    deg_deep = 0.9
    deep_frac = 0.25

    crit = [round(3.0 + rng.uniform(-0.3, 0.3), 3) for _ in range(T)]
    imp = [round(5.0 + rng.uniform(-0.5, 0.5), 3) for _ in range(T)]
    low = [round(6.0 + rng.uniform(-0.6, 0.6), 3) for _ in range(T)]

    base_pv = 14.0
    forecast = [round(base_pv + rng.uniform(-1.0, 1.0), 3) for _ in range(T)]
    frac = [0.06 + 0.012 * t + rng.uniform(-0.02, 0.02) for t in range(T)]  # calm baseline uncertainty
    actual = [max(0.0, forecast[t] + rng.uniform(-0.6, 0.6)) for t in range(T)]

    # universal mild terminal dip -- ALWAYS present, even on "calm" instances
    for t in (T - 2, T - 1):
        frac[t] = max(frac[t], rng.uniform(0.32, 0.42))
        actual[t] = max(0.0, forecast[t] * rng.uniform(0.5, 0.7))

    def add_severe(lo, hi, precursor=3):
        for t in range(max(0, lo - precursor), lo):
            frac[t] = max(frac[t], rng.uniform(0.38, 0.48))
        for t in range(lo, hi):
            frac[t] = max(frac[t], rng.uniform(0.62, 0.78))
            actual[t] = max(0.0, forecast[t] * rng.uniform(0.02, 0.09))

    if kind == "trap_mid":
        add_severe(6, 11)
    elif kind == "trap_late":
        add_severe(8, 12)
    elif kind == "trap_early":
        add_severe(2, 7)
    elif kind == "trap_double":
        add_severe(2, 5, precursor=2)
        add_severe(8, 11, precursor=2)
    elif kind == "trap_fastonset":
        add_severe(7, 11, precursor=1)
    # "calm": only the universal mild terminal dip above

    unc = [round(frac[t] * forecast[t], 3) for t in range(T)]

    return dict(capacity=capacity, initial_soc=initial_soc, eff=eff, deg_lin=deg_lin,
                deg_deep=deg_deep, deep_frac=deep_frac, crit=crit, imp=imp, low=low,
                forecast=forecast, unc=unc, actual=actual)


def _build_instances():
    kinds = ["calm", "calm", "calm", "calm", "trap_mid", "trap_late", "trap_early",
              "trap_double", "trap_fastonset", "calm"]
    out = []
    for i, kind in enumerate(kinds):
        seed = 1000 + i
        gen_seed = seed * 7 + 13
        inst = _lcg_instance(gen_seed, kind)
        out.append({"name": f"{kind}{i}", "inst": inst, "pub_seed": 20260311 + i})
    return out


# ============================ simulation =====================================
def _simulate(inst, res_imp, res_low):
    capacity = inst["capacity"]; soc = inst["initial_soc"]; eff = inst["eff"]
    deg_lin, deg_deep, deep_frac = inst["deg_lin"], inst["deg_deep"], inst["deep_frac"]
    crit, imp, low, actual = inst["crit"], inst["imp"], inst["low"], inst["actual"]
    obj = 0.0
    for t in range(T):
        incoming = actual[t]
        pool = soc + incoming
        cs = min(crit[t], pool)
        pool -= cs
        ia = max(0.0, pool - res_imp[t])
        is_ = min(imp[t], ia)
        pool -= is_
        la = max(0.0, pool - res_low[t])
        ls = min(low[t], la)
        pool -= ls

        total = cs + is_ + ls
        disch = max(0.0, total - incoming)
        chg = max(0.0, incoming - total)
        soc_after_disch = soc - disch
        soc_next = soc_after_disch + chg * eff
        soc_next = min(max(soc_next, 0.0), capacity)

        deep_deficit = max(0.0, deep_frac * capacity - soc_after_disch)
        deg = deg_lin * disch + deg_deep * disch * (deep_deficit / capacity)

        obj += W_CRIT * cs + W_IMP * is_ + W_LOW * ls - deg
        soc = soc_next
    return obj


def _policy_trivial(inst):
    cap = inst["capacity"]
    return [cap] * T, [cap] * T


def _obj_upper(inst):
    return sum(W_CRIT * c + W_IMP * i + W_LOW * l
               for c, i, l in zip(inst["crit"], inst["imp"], inst["low"]))


def _public_view(inst, seed):
    return {
        "T": T,
        "capacity_kwh": inst["capacity"],
        "initial_soc_kwh": inst["initial_soc"],
        "charge_efficiency": inst["eff"],
        "deg_linear_cost": inst["deg_lin"],
        "deg_deep_cost": inst["deg_deep"],
        "deep_discharge_frac": inst["deep_frac"],
        "priority_weights": {"critical": W_CRIT, "important": W_IMP, "low": W_LOW},
        "critical_demand_kwh": inst["crit"],
        "important_demand_kwh": inst["imp"],
        "low_demand_kwh": inst["low"],
        "pv_forecast_kwh": inst["forecast"],
        "pv_forecast_uncertainty_kwh": inst["unc"],
        "seed": seed,
    }


def _valid_answer(ans, capacity):
    if isinstance(ans, list) and len(ans) == 2:
        ri_raw, rl_raw = ans[0], ans[1]
    elif isinstance(ans, dict):
        ri_raw = ans.get("reserve_important_kwh")
        rl_raw = ans.get("reserve_low_kwh")
    else:
        return None
    if not (isinstance(ri_raw, list) and isinstance(rl_raw, list)):
        return None
    if len(ri_raw) != T or len(rl_raw) != T:
        return None
    try:
        ri = [float(x) for x in ri_raw]
        rl = [float(x) for x in rl_raw]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(x) for x in ri) or not all(math.isfinite(x) for x in rl):
        return None
    ri = [min(max(x, 0.0), capacity) for x in ri]
    rl = [min(max(x, 0.0), capacity) for x in rl]
    return ri, rl


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <candidate.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for entry in instances:
        inst = entry["inst"]
        public = _public_view(inst, entry["pub_seed"])

        ans, st = isorun.run_candidate(cand, public, timeout=CAND_TIMEOUT)
        if st != "OK":
            vec.append(0.0)
            continue

        parsed = _valid_answer(ans, inst["capacity"])
        if parsed is None:
            vec.append(0.0)
            continue
        res_imp, res_low = parsed

        try:
            obj_cand = _simulate(inst, res_imp, res_low)
        except Exception:
            vec.append(0.0)
            continue
        if not math.isfinite(obj_cand):
            vec.append(0.0)
            continue

        obj_base = _simulate(inst, *_policy_trivial(inst))
        obj_upper = _obj_upper(inst)
        denom = max(obj_upper - obj_base, 1e-6)

        r = 0.1 + 0.9 * (obj_cand - obj_base) / denom
        r = max(0.0, min(1.0, r))
        r = max(r, VALID_FLOOR)
        vec.append(float(r))

    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(v, 6) for v in vec]))


if __name__ == "__main__":
    main()
