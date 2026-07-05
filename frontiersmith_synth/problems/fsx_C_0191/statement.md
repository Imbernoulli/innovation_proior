# Data-Center Cooling: A Drop-In Thermal Response Nonlinearity for Cross-Hall Load Classifiers

## Background

A hyperscale operator runs many **cooling halls**. In each hall a *tiny* neural
classifier reads a handful of thermal / airflow sensor features and predicts the hall's
discrete **operating regime** (which cooling response is appropriate, whether a rack
cluster is hotspotting). The classifiers are architecturally identical up to size, and
every one of them uses the **same** hand-designed hidden nonlinearity -- a shared
"thermal response curve".

Your job is to invent that single **drop-in activation function** so that the small,
CPU-trainable classifiers are accurate across a **diverse family of halls**, including a
**hidden held-out hall** you never get to tune against. This is a *modular component*
task: you edit ONLY the activation; the architecture, optimizer, data, and training
budget are frozen by the evaluator.

## What you design

A scalar activation `phi: R -> R`, supplied as the **y-values of a piecewise-linear
curve** over a fixed knot grid `xs` (given to you). Between knots the evaluator linearly
interpolates; outside `[xs[0], xs[-1]]` it linearly extrapolates using the boundary
segment's slope. The evaluator plugs `phi` into a one-hidden-layer MLP
(`x -> Linear -> phi -> Linear -> softmax`) and trains it **from scratch** with
deterministic full-batch gradient descent (fixed seed, init, data, epochs) on each hall,
then measures held-out **test accuracy**.

The reference baseline activation is the **identity** curve `phi(x) = x`, which collapses
the one-hidden-layer net to a linear classifier. Reproducing it scores ~0.1. A genuinely
better-shaped curve scores higher -- but note that **no single off-the-shelf curve wins
every hall**: some halls are strongly nonlinear (XOR / rings / spirals / moons /
checkerboard) and reward saturating, expressive curves; one hall is nearly linearly
separable (Gaussian regimes) and is *hurt* by aggressive rectifiers. You must find a
curve that generalizes.

## The candidate program (isolated)

Your program is run as an **isolated subprocess**: it reads ONE JSON *public instance*
from stdin and writes ONE JSON answer to stdout. It never sees the sensor data, the
labels, the trained weights, or the evaluator's memory.

### Public instance (stdin)
```json
{
  "setting":    "xor",              // hall family name (you may adapt on it)
  "n_features": 2,                  // input dimension d
  "n_classes":  2,                  // number of regimes C
  "hidden":     6,                  // hidden width H (frozen)
  "grid":       [-4.0, ..., 4.0],   // K fixed knot x-positions (symmetric linspace)
  "n_knots":    21,                 // K
  "epochs":     60,                 // training budget (frozen; informational)
  "seed":       20241292            // a per-instance seed you MAY use
}
```

### Answer (stdout) -- EITHER form
```json
[y0, y1, ..., y20]                    // length K = n_knots activation values phi(grid[k])
{"activation": [y0, y1, ..., y20]}    // same, wrapped
```

Requirements on the answer: a length-`n_knots` list of **finite** reals with
`|y| <= 1000`. Any other shape, a non-finite value, an error, or a curve that makes
training diverge (non-finite loss/weights) scores **0.0** on that hall.

## Objective / scoring

For each hall the evaluator trains the net twice with the same seed / data / budget,
differing only in the activation (baseline identity vs. your curve), yielding test
accuracies `acc_base` and `acc_cand`. The per-hall normalized score is

```
r = clamp( 0.1 + 0.9 * (acc_cand - acc_base) / max(1 - acc_base, 0.20),  0, 1 )
```

The reported **Ratio** is the **geometric mean** of the per-hall `r` over all halls
(visible + held-out). Because it is a geometric mean, a curve that destabilizes even one
hall is heavily penalized -- so aim for a nonlinearity that helps *everywhere*, not one
tuned to a single setting. Higher is better.

Scoring is fully deterministic (all randomness is seeded); there is no wall-time or GPU
component.
