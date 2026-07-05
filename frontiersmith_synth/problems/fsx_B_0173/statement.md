# Glacier Sensor Net — Budgeted Pareto Hypervolume (DTLZ2 surrogate)

You are designing a network of autonomous sensors dropped onto a glacier. Every candidate
sensor configuration is a decision vector `x` in `[0,1]^n`. A physics surrogate maps each
configuration to **three conflicting objectives** (all to be *minimized*):

- coverage loss, energy draw, and telemetry latency.

The surrogate is the standard **DTLZ2** many-objective test function with `M = 3`. For a
decision vector `x` of length `n` (with `k = n - M + 1` "distance" variables `x[M-1:]`):

```
g   = sum( (x[i] - 0.5)^2  for i in x[M-1:] )
f0  = (1+g) * cos(x[0]*pi/2) * cos(x[1]*pi/2)
f1  = (1+g) * cos(x[0]*pi/2) * sin(x[1]*pi/2)
f2  = (1+g) * sin(x[0]*pi/2)
```

The unconstrained Pareto front is the unit sphere `f0^2+f1^2+f2^2 = 1` in the positive orthant
(attained when every distance variable equals `0.5`, i.e. `g = 0`).

## Your task

Under a **fixed evaluation budget** you must propose a *batch* of at most `budget` decision
vectors. The evaluator maps them through DTLZ2 and computes the **exact hypervolume** of the
resulting objective set with respect to a fixed reference (nadir) point `ref`. **Maximize the
hypervolume.** This rewards a batch that is simultaneously *close to the front* (small `g`) and
*well spread* over it — a single point, a clustered batch, or off-front points all score poorly.

## Candidate protocol (stdin → stdout)

Your program reads ONE JSON object (the public instance) from stdin and writes ONE JSON object
(your answer) to stdout.

Public instance:
```json
{"problem": "DTLZ2", "M": 3, "n": 8, "budget": 50, "ref": [1.2, 1.2, 1.2], "seed": 1702}
```

Answer:
```json
{"points": [[0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], ... ]}
```

- `points` is a list of `1..budget` decision vectors, each of length exactly `n`.
- Every coordinate must be a finite number in `[0, 1]` (values outside the box or `NaN`/`Inf`
  are rejected and score the instance 0).

## Objective and scoring

Objective (maximize): exact 3-D hypervolume of `{DTLZ2(x) : x in points}` w.r.t. `ref`.

Per-instance normalized score `r = min(1, 0.1 * HV / HV_baseline)`, where `HV_baseline` is the
hypervolume of a single centre sensor `x = [0.5,...,0.5]`. A trivial one-point batch scores
about `0.1`; better-spread on-front batches score higher (the achievable maximum leaves ample
headroom below `1.0`). The overall score is the mean of `r` over a fixed, seeded set of glacier
instances. Scoring is fully deterministic.
