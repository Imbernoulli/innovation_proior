# Split-Season Nutrient Scheduling

## Problem

A crop grows for `T` discrete time steps (days). It needs three nutrients —
nitrogen (N), potassium (K) and magnesium (Mg) — and its **uptake demand curve**
for each nutrient is given for every day. You control an **application
schedule**: on any day you may apply any non-negative amount of any nutrient to
the soil, but you have a total **season budget** per nutrient and a limited
number of **passes** (a pass = one calendar day on which you visit the field and
apply at least one nutrient; visiting costs labour regardless of how many
nutrients you apply that day or how much).

The soil is not a free reservoir: nutrient that is not yet taken up **leaches**
away a fixed fraction every day (mobility differs per nutrient — this is baked
into the input). And potassium and magnesium are **antagonistic ions**: a large
standing pool of K in the soil blocks the plant's ability to take up Mg on that
same day, even if Mg is plentiful.

Concretely, the soil-plant model runs day by day, `t = 1..T`. Let `pool_N,
pool_K, pool_Mg` start at 0. Each day:

1. Each pool first decays by that nutrient's daily *retention* fraction
   (`retain_N/K/Mg`, read from the input — lower means faster leaching), then
   today's application is added to it.
2. Uptake is capped by that day's demand: `uptake_N = min(pool_N, D_N[t])`,
   `uptake_K = min(pool_K, D_K[t])`.
3. Magnesium uptake is *additionally* throttled by the K:Mg pool ratio:
   `avail_frac = min(1, kappa * pool_Mg / pool_K)`, then
   `uptake_Mg = min(pool_Mg, D_Mg[t] * avail_frac)` (`kappa` is read from the
   input; if `pool_K` is 0 the pool is uncapped).
4. Each pool is reduced by its uptake and carries over (decayed) to the next
   day.

The season's value is the demand-weighted total uptake
`sum_t (v_N*uptake_N[t] + v_K*uptake_K[t] + v_Mg*uptake_Mg[t])`, where
`v_N, v_K, v_Mg` are per-unit economic weights from the input.

## Input (stdin)

```
T P
v_N v_K v_Mg
retain_N retain_K retain_Mg
kappa
B_N B_K B_Mg
D_N[1] D_K[1] D_Mg[1]
...
D_N[T] D_K[T] D_Mg[T]
```
`T` = number of days, `P` = pass budget (max days on which you may apply
anything). `B_N,B_K,B_Mg` are the season's total application budgets per
nutrient (you may never use more, but you may use less). `D_X[t] >= 0` is
nutrient X's demand on day t (the amount the plant *could* absorb that day if
enough were available and unblocked).

## Output (stdout)

Exactly `T` lines, day `t` on line `t` (1-indexed), each with three
non-negative numbers: `a_N[t] a_K[t] a_Mg[t]`, the amount of each nutrient you
apply that day (0 if none).

## Feasibility

- Exactly `T` lines, 3 finite non-negative numbers per line (any NaN/Inf,
  negative value, or wrong token count scores 0).
- `sum_t a_N[t] <= B_N` (and likewise for K, Mg), small tolerance for
  floating point.
- The number of days with `a_N[t]+a_K[t]+a_Mg[t] > 0` must not exceed `P`.

## Scoring (deterministic)

Let `F` be your schedule's total value under the model above. The checker
also simulates its own **reference schedule**: dump the *entire* season
budget of each nutrient in a single application on day 1 (one pass — cheap in
labour, but leaches for the whole season and floods K and Mg together), giving
value `B`. The printed score is

```
Ratio = min(1, 0.1 * F / B)
```

so matching the front-loaded reference scores `0.1`; a schedule worth 10x the
reference saturates at `1.0`. Because `B` itself already earns some credit for
whatever uptake happens before leaching sets in, real headroom above `0.1`
comes only from timing applications against the demand curves and keeping K
and Mg apart in time.

## Example (worked, illustrative shape only)

For `T=2, P=1`: applying everything on day 1 gives the reference itself, so
`Ratio = 0.1` by construction, regardless of the specific numbers. A schedule
that instead (within the same 1-pass budget) shifts the mix toward whichever
nutrient's demand actually falls on day 1 scores strictly above `0.1`; one that
wastes budget on a nutrient with near-zero day-1 demand can score below `0.1`.
