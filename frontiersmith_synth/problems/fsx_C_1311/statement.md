# Islanded Microgrid: Pre-emptive Load-Shedding Reserve Policy

## Story

A research station runs an islanded microgrid — PV panels and a battery, no
grid connection — over a 14-day, running-out-of-sun stretch. Every day it must
decide how to split available energy across three priority tiers: **critical**
(life-safety, comms), **important** (lab equipment), and **low** (discretionary,
e.g. water heating). You submit a **dispatch policy**: for each day, a battery
state-of-charge (SoC) *reserve floor* per non-critical tier, below which that
tier gets curtailed instead of drawing the battery down further. Critical demand
is always attempted in full from whatever energy is available that day — it is
never rationed by your policy, only *protected* by how much you chose to spend
on the other tiers on earlier days.

You only see a **forecast** of daily PV yield plus a **forecast-uncertainty**
band (kWh) per day — never the true weather. A policy that ignores the
uncertainty band and always trusts the forecast (reserve = 0, "serve
everything") looks great until a multi-day low-sun stretch hits: the battery is
already near-empty from serving lower tiers on the days before, and when the sun
doesn't return, even **critical load goes dark**. The insight this problem
rewards is reading the uncertainty band to size a pre-emptive reserve *before*
a low-sun stretch arrives, trading a little unused capacity on safe days for
guaranteed critical service through risky ones.

## Isolation

Your program runs as an **isolated subprocess**: read one JSON *public instance*
from stdin, write one JSON *policy* to stdout. You never see the hidden actual
PV trajectory.

## Public instance (stdin)

```json
{
  "T": 14,
  "capacity_kwh": float, "initial_soc_kwh": float, "charge_efficiency": float,
  "deg_linear_cost": float, "deg_deep_cost": float, "deep_discharge_frac": float,
  "priority_weights": {"critical": float, "important": float, "low": float},
  "critical_demand_kwh": [float, ...14],
  "important_demand_kwh": [float, ...14],
  "low_demand_kwh": [float, ...14],
  "pv_forecast_kwh": [float, ...14],
  "pv_forecast_uncertainty_kwh": [float, ...14],
  "seed": int
}
```

## Answer (stdout)

`{"reserve_important_kwh": [float, ...14], "reserve_low_kwh": [float, ...14]}` —
the battery-SoC floor (kWh) you want to protect on each day for each tier.
Values are clipped into `[0, capacity_kwh]`. Wrong shape/type or a non-finite
value scores that instance **0**.

## Simulation (run by the evaluator, using the REAL hidden weather)

Each day `t`, in order — critical, then important, then low — a tier is served
up to its demand from the pool `battery_soc + pv_actual[t]`, except important
and low additionally stop early to keep the projected end-of-day SoC at or
above your reserve floor for that tier:

```
pool = soc + pv_actual[t]
crit_served = min(critical_demand[t], pool);              pool -= crit_served
imp_served  = min(important_demand[t], max(0, pool - reserve_important[t]));  pool -= imp_served
low_served  = min(low_demand[t],       max(0, pool - reserve_low[t]));        pool -= low_served
discharge = max(0, total_served - pv_actual[t])   # kWh pulled FROM the battery
charge    = max(0, pv_actual[t] - total_served)    # surplus PV offered to the battery
soc = clip(soc - discharge + charge*charge_efficiency, 0, capacity_kwh)   # excess spilled
```

**Battery degradation** cost that day: `deg_linear_cost*discharge +
deg_deep_cost*discharge*max(0, deep_discharge_frac*capacity_kwh -
(soc_before_charge))/capacity_kwh` — deeper same-day discharges cost
disproportionately more (accelerated wear).

## Objective

Per instance:
`obj = sum_t(w_crit*crit_served[t] + w_imp*imp_served[t] + w_low*low_served[t]) - sum_t(deg_cost[t])`

The evaluator compares your `obj` against two references it computes itself: a
weak baseline that always sheds every non-critical tier (`obj_base`, anchors to
`r=0.1`) and a loose upper bound of serving 100% of all demand for free
(`obj_upper`, anchors toward `r=1.0`, never actually reachable):

```
r = clamp(0.1 + 0.9*(obj - obj_base) / max(obj_upper - obj_base, 1e-6), 0, 1)
```

`Ratio` is the **arithmetic mean** of `r` over 10 instances: some calm (ample
sun, only a mild end-of-horizon dip), several with a distinct multi-day
low-sun sequence placed early, mid, late, doubled, or with only one day of
advance warning. **Maximize `Ratio`.** No single fixed reserve level is best
everywhere — it must trade off calm-day value against low-sun-sequence
survival — so a policy that actually reads the per-day uncertainty band to
shape a time-varying reserve is required to do well across the whole family.
