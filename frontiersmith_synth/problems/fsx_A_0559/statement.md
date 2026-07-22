# Guild Ledger: Setting the Kingdom's Tax Brackets

## Problem
The treasury taxes a guild of **N** workers. Worker *i* is described by three printed
numbers: a base income **m_i** (what they earn at a zero marginal rate), an income
**elasticity e_i**, and a fixed **participation cost f_i** (in utility units).

You publish a **piecewise-linear marginal-rate schedule** with up to **8** brackets.
Given thresholds `0 = b_0 < b_1 < ... < b_{K-1}` and marginal rates `t_0,...,t_{K-1}`,
income earned in `[b_k, b_{k+1})` is taxed at rate `t_k` (income above `b_{K-1}` at
`t_{K-1}`). The tax `T(z)` is the continuous integral of these marginal rates.

Every worker then **best-responds**. Facing marginal rate `t`, a worker's interior
optimal earnings are
```
z*(t) = m_i * (1 - t) ** e_i
```
and the disutility of earning `z` is `v(z) = (m_i/(1+1/e_i)) * (z/m_i) ** (1+1/e_i)`.
The worker picks the earnings `z >= 0` (an interior bracket optimum or a bracket kink)
maximizing utility `U = (z - T(z)) - v(z)`, and **works only if** that maximized `U`
exceeds the participation cost `f_i` (otherwise they earn 0 and pay no tax).

## Output (stdout)
```
K
b_0 t_0
b_1 t_1
...
b_{K-1} t_{K-1}
```
`1 <= K <= 8`, `b_0 = 0`, thresholds strictly increasing with `b_{K-1} <= 1e7`, and
each rate in `[0, 0.95]`.

## Feasibility
The schedule is rejected (score 0) unless it parses, obeys every bound above, and the
**aggregate worker welfare** `W = sum_i (worker i's realized utility, net of f_i)`
is at least the printed floor `W_floor`. The floor is the revenue-vs-welfare guardrail:
you may not fund the treasury by immiserating the guild.

## Objective
**Maximize total tax revenue** `R = sum_i T(z_i)`, where each `z_i` is worker *i*'s
best-response earnings under your schedule.

## Scoring
Let `B` be the revenue of a mild flat reference tax the judge builds itself. Your score is
```
Ratio = min(1.0, 0.1 * R / B)
```
so the flat-baseline schedule scores about 0.1 and there is headroom well above any
reference schedule.

## The catch
The elasticity distribution is **bimodal and correlated with income**: a large mass of
highly-elastic workers is planted in the **middle** income band, while the bottom and top
bands are inelastic. Pushing a uniform or progressive (monotone-rising) rate through the
elastic middle drives those workers to cut their earnings sharply — losing **both** revenue
and welfare. The revenue-maximizing schedule is therefore **non-monotone**: marginal rates
must **dip exactly where the elastic mass sits** and rise on the inelastic bands. No flat,
linear, or textbook-progressive template produces that shape — you must read the printed
elasticity distribution and fit the schedule to its structure.

## Constraints
`N` up to 50000. Time limit 5s, memory 512m. All arithmetic is deterministic.

## Example (illustrative, not the scoring instance)
With two workers `m=(10, 40)`, `e=(0.3, 1.3)` and a floor of 5, a flat 20% schedule
`K=1 / 0 0.20` collects some revenue; splitting into `K=2 / 0 0.55 / 12 0.20` — a *higher*
rate on the inelastic low band and a *lower* rate over the elastic high earner — can raise
revenue while keeping welfare above the floor. Your job is to find the bracket cuts and
rates that do this best for the printed population.
