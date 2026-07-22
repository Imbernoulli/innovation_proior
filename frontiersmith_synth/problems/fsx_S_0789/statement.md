# Boost Ring — What Does This Map Actually Conserve?

## Problem

A hidden deterministic map advances a 2-D state `(x, y) -> (x', y')`. Each
test id fixes a different instance, but all share the same *shape*: a
nonlinear "boost" whose twist can depend on the state in complicated,
test-specific ways. You are given many sampled transitions
`(x, y) -> (x', y')`, but **only from a narrow ring of radii close to 1** —
the training regime. Your job is to name a quantity `I(x, y)` that the *true
dynamics* leaves unchanged by every transition — not merely one that happens
to look steady over the states you were shown.

You must build `I` as a **linear combination of a declared symbolic feature
library**: `X, Y, XX, XY, YY` (i.e. `x, y, x^2, xy, y^2`). Only `+`, `-`, and
`constant * feature` are allowed — you are choosing a coefficient vector over
the library, not an arbitrary curve. The grader evaluates your `I` on a
**held-out ring at much larger radius**, never shown to you, checking whether
`I` truly stayed put across each of those transitions.

**Illustrative FORM only — NOT the hidden law:** `2*X - 0.5*XY + Y` is a
syntactically valid answer showing the grammar; it is unrelated to the actual
conserved quantity of any test instance, which you must discover from data.

## Input (stdin)

```
n t
x_0 y_0 x'_0 y'_0
x_1 y_1 x'_1 y'_1
...
```

`t` is the test id; `n` training rows follow, each a state and the state one
step later (floats, small measurement noise). All rows come from the narrow
ring `radius in [0.9, 1.1]`. The held-out grading ring lives at radius
`[3, 6]` for the *same* hidden map — it is never given to you.

## Output (stdout)

Exactly one non-empty line: your expression for `I(x, y)`, e.g.

```
XX - YY
```

Allowed names: `X, Y, XX, XY, YY`. Allowed operators: `+`, `-`, unary minus,
and `constant * feature` (numeric constants may not multiply each other or
another feature — you're picking coefficients, not a new curve). At most 40
expression nodes.

## Feasibility

Output must parse under the grammar above (known names only, no
feature*feature, no division/calls, finite constants with `|c| <= 1e6`, node
budget respected). It is also rejected if `I` is **(near-)constant across the
given training states** — trivially "unchanged" everywhere is not a
discovery, it's a definition, and it is excluded before scoring. Any
violation scores `Ratio: 0.0`.

## Objective (maximise)

For every held-out transition `(x,y)->(x',y')`, form the relative drift
```
rel = |I(x',y') - I(x,y)| / (1 + |I(x,y)|)
```
and per-transition credit `acc = TOL^2 / (TOL^2 + rel^2)` (1.0 when `I` is
exactly unchanged, decaying smoothly otherwise). Average over the held-out
ring, then tax the number of library terms used beyond a tiny free budget:
```
F = mean(acc) - LAMBDA * max(0, num_terms - FREE_TERMS)
B = the same F for the checker's own fixed candidate "XY"
Ratio = min(1000, 100*F/B) / 1000
```
Submitting `XY` itself reproduces the baseline (`Ratio ~= 0.1`). Real
conservation on the *extrapolation* ring raises the score well above that;
the tax and the train/held-out gap keep the ceiling below `1.0`.

## Why the narrow ring is a trap

On the training ring, `x^2 + y^2` looks almost perfectly constant — but only
because every training state sits near radius 1; that is a fact about *how
the ring was sampled*, not about the map. The map is a hyperbolic boost: it
algebraically leaves `x^2 - y^2` exactly unchanged by every transition, at
*any* radius, however wildly its twist depends on position — while
`x^2 + y^2` (Euclidean radius) drifts under it freely. Fitting "what looks
steady over my sample" (a value-space property of the states) finds the wrong
one; testing "what does each transition actually change" (a property of the
*variation* `theta(next) - theta(state)`) finds the right one — and only the
latter survives stepping out to radius 6.

## Constraints

Time limit 5 s, memory 512 MB, each input file well under 5 MB. Scoring is
fully deterministic; no randomness, wall-time, or GPU is used to score.
