# VolcanoWatch: Budgeted Batch Design for Maximum Pareto Hypervolume

A volcanic-hazards agency is commissioning a small fleet of monitoring-station
**configurations**. Each configuration is a design vector `x` of `n` real
knobs in `[0, 1]`. When a configuration is trialed it is scored on `M`
*competing* objectives (think detection latency, power draw, seismic
blind-spot fraction, ...), **all to be MINIMIZED**. Because trials are
expensive, you are given a fixed commissioning budget: you may propose **at
most `B` configurations**, all at once (a one-shot batch — there is no
adaptive feedback).

Your batch is judged by the **Pareto hypervolume** it dominates in objective
space, measured against a published reference point `ref`. Bigger hypervolume
= a better, more diverse trade-off frontier. Your job: **choose the batch of
configurations that maximizes the dominated hypervolume.**

## The objective surrogate (deterministic — DTLZ2)

A configuration `x = [x_0, ..., x_{n-1}]` (each in `[0,1]`) is scored by the
DTLZ2 function. The first `M-1` coordinates are *position* variables and the
last `k = n - (M-1)` coordinates are *distance* variables. Let

```
g = sum_{i = M-1 .. n-1} (x_i - 0.5)^2                 # site-difficulty term, >= 0
```

Then the `M` objective values (all minimized) are, with `t_j = x_j * pi/2`:

```
f_0     = (1 + g) * cos(t_0) * cos(t_1) * ... * cos(t_{M-2})
f_1     = (1 + g) * cos(t_0) * cos(t_1) * ... * cos(t_{M-3}) * sin(t_{M-2})
...
f_{M-2} = (1 + g) * cos(t_0) * sin(t_1)
f_{M-1} = (1 + g) * sin(t_0)
```

Equivalently, `f_i = (1+g) * [prod_{j=0}^{M-2-i} cos(t_j)] * (sin(t_{M-1-i}) if i>0 else 1)`.

Key facts you may exploit:
- When `g = 0` (i.e. **every distance variable equals 0.5**), the point lies on
  the true trade-off surface: `sum_i f_i^2 = 1` with all `f_i >= 0` — the
  positive octant of the unit sphere. Any `g > 0` scales the whole vector
  outward by `(1+g)`, which can only shrink (or destroy) the dominated volume.
- The `M-1` position variables move the point *along* the surface.

## Hypervolume (how you are scored)

Given your evaluated objective vectors, the evaluator keeps the points that are
strictly better than `ref` in **every** objective, and computes the exact
volume of the region dominated by them and bounded above by `ref`
(2D sweep for `M = 2`, 3D slicing for `M = 3`). This is your raw objective.

## Input (public instance, one JSON object on stdin)

```json
{
  "M": 3,                 // number of objectives (2 or 3)
  "k": 5,                 // number of distance variables
  "n": 7,                 // dimension of each configuration = (M-1) + k
  "B": 30,                // budget: max number of configurations
  "ref": [1.1, 1.1, 1.1], // reference point (length M)
  "seed": 123456          // cosmetic; the surrogate is fixed
}
```

## Output (one JSON object on stdout)

```json
{ "points": [ [x_0, ..., x_{n-1}], ... ] }   // 1..B configurations, each length n, each coord in [0,1]
```

Constraints (violate any -> score 0):
- `1 <= len(points) <= B`.
- Every configuration has exactly `n` finite coordinates, each in `[0, 1]`
  (values are clipped to `[0,1]`; `NaN`/`Inf` are rejected).

## Scoring

Let `HV` be the exact dominated hypervolume of your batch and `HV_base` the
hypervolume of the single center configuration `[0.5]*n` (computed by the
evaluator). Your per-instance normalized score is

```
r = min(1.0, 0.1 * HV / HV_base)
```

so a batch that only matches the single-point baseline scores ~0.1, and richer
frontiers score higher (with headroom left for near-optimal placement). The
final Ratio is the mean of `r` over all instances. **Objective: maximize.**

This is genuinely open-ended: getting `g = 0` is necessary but not sufficient —
the hard part is *where* to place a budget-limited set of on-surface points so
their dominated hypervolume is as large as possible (a space-filling /
hypervolume-contribution trade-off with no simple closed-form optimum).
