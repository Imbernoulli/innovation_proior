# Variance-Adaptive Quadrature Budget (Format B, isolated)

A domain is partitioned into `R` disjoint regions. Region `r` has a known
width `w_r` and an unknown, region-specific measurement noise: repeated
readings taken inside region `r` behave like i.i.d. draws from
`Normal(mu_r, sigma_r^2)`, where both `mu_r` (the region's true average
value) and `sigma_r` (its noise standard deviation) are **hidden** — you
never see them directly. The quantity of interest is the domain total
`I = sum_r w_r * mu_r` (think: a piecewise-constant integral where each
piece's "height" is only known up to measurement noise).

You are given a small **pilot batch**: `pilot_size` real noisy readings
already taken in every region (so you get real signal about each region's
noise level for free). You are also given a total **extra sampling
budget** `B` — additional readings you may spend across the regions,
however you like. If `alloc_r` extra readings were taken in region `r`,
the natural estimator averages ALL `pilot_size + alloc_r` readings in that
region and combines them into the plug-in estimate of `I`. Because
readings are i.i.d. and unbiased, this estimator's variance decomposes
exactly by region:

```
Var(estimate) = sum_r  w_r^2 * sigma_r^2 / (pilot_size + alloc_r)
```

**Your job**: decide `alloc_r` for every region (any nonnegative real
number — fractional allocations are fine, they represent a continuous
sampling/precision budget) so that `sum_r alloc_r <= B`, minimizing the
variance above. Spending on a region with high hidden noise and/or large
width shrinks the total variance a lot; spending on an already-quiet
region barely helps. **You never observe the extra readings** — your
score is exactly the analytic variance formula above, evaluated with the
TRUE hidden `sigma_r` and your chosen `alloc_r`. The only information you
get about `sigma_r` is what the small pilot batch reveals, and a small
pilot is a noisy witness of the true noise level — the number of true
sigma_r your allocation ends up close to depends on how much you trust
(and how you hedge) that noisy pilot signal.

## Public instance (stdin JSON)
```json
{
  "regions": [ {"width": <float>, "pilot": [<float>, ...]}, ... ],
  "budget": <int>,
  "pilot_size": <int>
}
```
`len(regions) = R`; each region's `pilot` list has exactly `pilot_size`
real numbers (its free noisy pilot readings).

## Answer (stdout JSON)
```json
{"alloc": [a_1, a_2, ..., a_R]}
```
Exactly `R` finite numbers, each `>= 0`, summing to at most `budget`
(a tiny numerical tolerance is allowed). Any other shape/type, a
non-finite value, a negative entry, or an over-budget sum is rejected and
scores 0.

## Objective & scoring
Minimize `Var = sum_r w_r^2 * sigma_r^2 / (pilot_size + alloc_r)` (using
the grader's hidden `sigma_r`) over feasible allocations. Per instance:
`score = min(1, 0.1 * Var_baseline / Var_yours)`, where `Var_baseline` is
the variance of spending **zero** extra budget (relying on the pilot
alone). The final score is the mean over 10 fixed, seeded instances of
varying region count, budget size, and noise layout. Several instances
plant a trap: one or two regions are narrow (small `w_r`) but carry a
huge hidden noise spike (`sigma_r` tens of times larger than the rest) —
spending budget in proportion to region width, or spreading it evenly
across regions, both starve exactly the region that most needs it.

## Suggested strategies (increasing sophistication)
- **Spend nothing** — rely on the pilot alone; wastes the whole budget.
- **Spread evenly** — `alloc_r = B/R` for every region; ignores which
  regions are actually noisy.
- **Probe, then allocate by width alone** — proportional to `w_r`, still
  blind to noise.
- **Pilot-informed Neyman allocation** — estimate each region's noise
  from its pilot sample's standard deviation, then allocate budget
  proportional to `w_r * sigma_hat_r` (allocate to regions with the
  largest *width-times-uncertainty* product, not just the largest width).
  Because a handful of pilot readings is itself a noisy estimate of
  `sigma_r`, a careful allocator should not fully trust an estimate that
  happens to look small — hedge with a floor before deciding to
  effectively stop refining a region, versus pouring most of the budget
  into regions whose pilot evidence for high noise is credible.
