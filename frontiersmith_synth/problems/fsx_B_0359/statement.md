# Cryo Wiring Harness: Budgeted Hypervolume Search on a DTLZ2 Cost Surrogate

## Setting
A cryogenic quantum-computing lab must lay out its **wiring harness** — the bundle
of coax/RF lines routed down through the dilution-fridge stages. A single wiring
**layout** is a vector `x` of `n` continuous knobs in `[0, 1]` (line lengths,
routing offsets, filter placements, thermal anchoring points, ...).

Each layout is scored by a **frozen deterministic surrogate** on `M` competing
costs (e.g. signal latency, thermal heat-load into the mixing chamber, inter-line
cross-talk). The surrogate is the standard **DTLZ2** response surface, to be
**minimised** on every objective:

```
n = M - 1 + k                       # first M-1 = "angle" vars, last k = "distance" vars
g = sum_{i = M-1 .. n-1} (x[i] - 0.5)^2
f[0]   = (1 + g) * prod_{j=0..M-2}   cos(x[j] * pi/2)
f[i]   = (1 + g) * prod_{j=0..M-2-i} cos(x[j] * pi/2) * sin(x[M-1-i] * pi/2)   # i = 1 .. M-1
```

The Pareto-optimal front is the **unit sphere in the positive octant**
(`sum f[i]^2 = 1`), reached exactly when every *distance* var equals `0.5` (so
`g = 0`). The *angle* vars slide a layout around that front.

## Your task
Under a **fixed evaluation budget** you may submit at most `budget` layouts. The
lab evaluates all of them on the surrogate and keeps their cost vectors. Your
score is the **exact dominated hypervolume** of those cost vectors measured
against a fixed **reference (nadir) point** `ref`: a cost vector is counted only
if `f[i] <= ref[i]` for every objective `i`, and it then dominates the box
between itself and `ref`. **Maximise the total dominated hypervolume.**

This rewards two things at once: pushing layouts *onto* the sphere (distance vars
near `0.5`) **and** *spreading* them so the union of dominated boxes tiles the
front. There is no closed-form optimum for a finite point budget — many
placement strategies (grids, low-discrepancy sequences, greedy HV insertion) are
viable and trade off differently.

## Candidate protocol (isolated program)
Your program reads ONE JSON public instance from **stdin** and writes ONE JSON
answer to **stdout**. It never sees the scorer.

### Public instance (stdin)
```json
{
  "M": 3,            // number of objectives
  "k": 10,           // number of distance vars
  "n": 12,           // decision dimension = M-1+k
  "budget": 40,      // max number of layouts you may submit
  "ref": [1.1,1.1,1.1], // reference (nadir) point, length M
  "lo": 0.0, "hi": 1.0  // per-coordinate bounds
}
```

### Answer (stdout)
```json
{ "points": [ [x0, x1, ..., x_{n-1}], ... ] }
```
A list of **at most `budget`** layouts, each a length-`n` list of finite floats in
`[lo, hi]`. Submitting more than `budget` layouts, a wrong-length layout, an
out-of-range or non-finite coordinate, or a malformed object scores **0**.

## Scoring
For each instance the evaluator computes the exact dominated hypervolume `HV` of
your submitted layouts' DTLZ2 cost vectors versus `ref`. Let `B` be the
hypervolume of the single **centre** layout (all vars `0.5`). The per-instance
normalised score is

```
r = min(1.0, 0.1 * HV / B)
```

so the centre baseline scores `~0.1` and better-spread on-sphere sets score
higher (with headroom below `1.0`). The final `Ratio` is the mean of `r` over all
10 instances; `Vector` lists the per-instance scores. Scoring is fully
deterministic (no randomness, no wall-clock).
