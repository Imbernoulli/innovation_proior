# One Hot Pulse: Nanoparticles That All Come Out the Same Size

## Problem

You control a nanoparticle synthesis reactor over `T` discrete stages. At each
stage you choose a **heat level** (one of `L` fixed levels) and an amount of
**precursor to inject**, drawing from a fixed total budget `C0`. A shared
**monomer pool** accumulates whatever you inject.

Three mechanisms govern what happens to the pool and to the particle
population, in this order, every stage:

1. **Nucleation.** Level `i` has a threshold `thr_i`: nuclei can only form
   while the pool exceeds it, and kinetics cap how many a *single stage* can
   spawn: `n_new = min(floor((pool - thr_i)/v0), cap_i)`, each costing `v0`
   pool units and starting at radius `r0`. Hotter levels have lower
   thresholds (easier to trigger) and higher caps (burn through excess
   faster). Every particle born in the same stage forms one **cohort**.
2. **Growth.** Every cohort grows by `dr = gcoef_i * (1 - coverage)^p`, where
   `gcoef_i` is this stage's growth coefficient and `coverage` is that
   cohort's surfactant coverage (option `bind, p` is your one-time global
   choice). Coverage updates as `coverage += bind * (1 - coverage) * dr`, so
   growth self-throttles toward zero as a cohort's surface saturates.
3. **Ripening.** If the pool falls below fixed `theta_ripen`, every cohort's
   radius is pulled toward the population's count-weighted mean:
   `radius += ripening_rate * (radius - mean_radius)` (clamped at 0) —
   below-mean cohorts shrink, above-mean ones grow further. A population made
   of one birth-time cohort is unaffected (its radius *is* the mean).

## Input (stdin)

```
T L S
r0 v0
theta_ripen ripening_rate
thr_0 cap_0 gcoef_0         (L lines, one per heat level, index 0..L-1)
...
bind_0 p_0                  (S lines, one per surfactant option, index 0..S-1)
...
C0 max_inject
target disp_limit
```
All values are non-negative; `thr`/`cap`/`gcoef` vary by level, `bind`/`p`
vary by surfactant option. `1 <= T <= 20`, `L, S <= 6`.

## Output (stdout)

```
temp_1 temp_2 ... temp_T      (heat level per stage, each in [0, L-1])
inject_1 inject_2 ... inject_T (precursor injected per stage, >= 0)
surf                          (one global surfactant choice, in [0, S-1])
```

## Feasibility

Every `temp_t` must be an integer in `[0, L-1]`. Every `inject_t` must be a
finite non-negative number no larger than `max_inject`. The total injected
across all stages must not exceed `C0`. `surf` must be an integer in
`[0, S-1]`. Any violation, wrong token count, or non-finite value scores
`0.0`.

## Scoring

Run the simulation above to get the final population (a list of cohorts, each
`(count, radius)`). Each particle's quality is
`q = max(0, 1 - |radius - target| / disp_limit)` — full credit exactly at
`target`, linearly falling to zero at `disp_limit` away. Your score `F` is the
count-weighted average `q` over every particle ever nucleated. The checker
also builds its own reference `F` (a fixed "constant mid-heat, feed evenly"
schedule — always feasible, always positive) as baseline `B`, and reports
```
ratio = min(1000, 100*F/B) / 1000
```
Matching the reference scores `0.1`; doing meaningfully better scores higher.
There is no known closed form for the best schedule on a given instance —
reason about *when* nucleation and growth should happen, not just *how hot*.

## Constraints

`1 <= T <= 20`; `2 <= L, S <= 6`; time limit 5s, memory 512MB.

## Example (illustrative only — toy numbers, not a real test case)

`T=2`, `L=4`, `S=1`: `thr=[100,100,20,1]`, `cap=[5,5,5,5]`, `gcoef=[0,0,1,2]`,
`r0=0, v0=1`, `theta_ripen=-100` (disabled), `bind=0.5, p=1`, `C0=50,
max_inject=50`, `target=2, disp_limit=1`.

Submission `temp=[3,3]`, `inject=[4,0]`, `surf=0`: pool hits 4, crosses
`thr_3=1`, spawns `min(floor(3/1),5)=3` nuclei (pool→1), which grow by `2*1=2`
to radius 2. Stage 2: no injection, pool stays 1, `1>1` is false so no new
nuclei; growth is `2*(1-1)^1=0`. Final: 3 particles at radius 2, exactly
`target` → `F = 1.0`.

The checker's reference `temp=[2,2]`, `inject=[25,25]`: stage 1 pool 25
crosses `thr_2=20`, spawns 5 nuclei (pool→20), growing to radius 1
(coverage→0.5). Stage 2 pool becomes 45, crosses `thr_2` AGAIN, spawns 5
*more* nuclei at radius 0; that stage's growth takes the first cohort to
radius 1.5 and the second to radius 1. Quality: `0.5` for the first 5, `0`
for the second 5 → `B = 0.25`.

`ratio = min(1000, 100*1.0/0.25)/1000 = 0.4`. Two separate nucleation events
(the reference) cost the second cohort its entire quality even though the
first was fine — exactly the age-spread penalty one well-sized pulse avoids.
