# Admitting Patients When Discharge Is The Bottleneck

A small hospital has several specialty wards (e.g. cardiology, orthopedics,
neurology), each with a fixed bed **capacity**. Every day, patients arrive
needing a specific ward. If their home ward has a free bed, they are
admitted there. If not, they either wait ("**board**") or are placed as an
**outlier** in another ward that currently has spare beds — borrowing
capacity across specialties.

Two costs work against each other. **Boarding** a patient accrues a cost
every day they wait, and one who waits too long is eventually
**diverted** (a hard, larger cost) — so refusing to place anyone is bad.
But an outlier patient's length of stay is **longer** than it would be at
home (transfer coordination, an unfamiliar care team) — a bed lent out
today stays lent for a disproportionately long time, and a ward that
lends freely may find itself full of slow-discharging outliers exactly
when its own patients need it.

You submit, for every ward and every day, a **reserve**: the number of
that ward's beds to keep free from outlier admission. Reserve 0 lends
freely (maximizes today's utilization); reserve = capacity never lends at
all. The right amount is a genuine trade-off that should track each
ward's own upcoming demand, not be a fixed number.

## Candidate program contract

Standalone program: read ONE JSON object (the public instance) from
**stdin**, write ONE JSON object (your answer) to **stdout**.

### Public instance (stdin)

```json
{
  "name": "trap_handoff_co", "T": 45,
  "wards": ["CARD", "ORTHO", "NEURO"],
  "capacity": [10.0, 10.0, 10.0],
  "mean_los": [5.0, 5.0, 5.0],
  "outlier_multiplier": 1.8, "transfer_delay": 2.5,
  "boarding_cost_coef": 0.20, "max_wait": 5.0, "divert_penalty": 2.0,
  "init_occ_home": [6.2, 6.2, 6.2],
  "arrival_forecast": [ [/* T floats: ward 0's daily arrival schedule */],
                         [/* ward 1 */], [/* ward 2 */] ]
}
```

Actual arrivals vary by up to ~10% around `arrival_forecast` day to day;
the schedule's shape (including any surge) is otherwise accurate.

### Answer (stdout)

```json
{ "reserve": [ [r_{0,0}, ..., r_{0,T-1}], [r_{1,...}], [r_{2,...}] ] }
```

One list per ward (same order as `wards`), each of exactly `T` finite,
non-negative numbers. Values are clipped to `[0, capacity[w]]`. Any
shape/type violation, a crash, a timeout, or non-JSON output scores that
instance `0.0`.

## Dynamics (evaluator-side, uses the true hidden arrivals)

Each day, per ward: patients (carried-over boarders + today's arrivals)
first fill free beds in their **home** ward. Anyone still unplaced is then
offered, in ward order, to *other* wards with
`free beds - reserve[that ward][t] > 0` — filling as much of that ward's
spare-beyond-reserve capacity as available, then moving to the next ward
in order if patients remain. Anyone still unplaced boards, paying
`boarding_cost_coef` for the day; a constant hazard `1 / max_wait` of the
backlog is diverted, paying `divert_penalty` each. Home beds discharge at
rate `1 / mean_los[ward]` per day; outlier beds discharge at the slower
rate `1 / (mean_los[ward] * outlier_multiplier + transfer_delay)`. Your
objective for the day is `(home admits + outlier admits) - boarding_cost
- divert_cost`, summed over all `T` days.

## Scoring (deterministic)

The evaluator computes, itself, `q_base` (the objective of the rigid
"never lend a bed" policy: `reserve = capacity` every day) and `q_ideal`
(sum of all true arrivals — a generous, essentially unreachable ceiling:
every patient admitted instantly with zero boarding and zero diversion).
With `q_cand` your realized objective:

```
r = clamp( 0.1 + 0.9 * (q_cand - q_base) / max(eps, q_ideal - q_base), 0, 1 )
```

**Ratio** is the mean of `r` over 10 fixed, seeded instances (some larger
/ held-out); **Vector** lists per-instance scores.

## Suggested strategies

1. **Never lend** (baseline): reserve = capacity always.
2. **Maximize utilization**: reserve = 0 always, admit an outlier
   whenever a bed is free, with no regard for what that ward will need
   soon.
3. **Turnover-net-demand reserve**: for each ward, forward-estimate its
   own occupancy under the public forecast, subtract what organic
   discharge turnover will free up within a lookahead window from the
   forecasted near-term demand, and reserve only the shortfall — near 0
   when the ward is quiet, real headroom just before its own demand ramp.
4. **Stronger**: rolling re-optimization or a multi-ward min-cost-flow /
   LP relaxation over the whole horizon.
