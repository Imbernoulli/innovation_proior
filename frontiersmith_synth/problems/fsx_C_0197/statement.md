# Apiary Foraging Response — designing one drop-in activation across many meadows

A beekeeper runs an **apiary** of tiny colony-controller networks. Each colony is a 2-layer MLP
whose hidden "worker bees" all share **one drop-in foraging-response curve** `r = f(s)`: given a
scalar stimulus `s` (a linear projection of the local nectar signals), every worker applies the
same curve `f` to decide its foraging response. The *same* curve is stamped into several
independent apiaries ("meadows" = small 2-D classification datasets) **plus one hidden meadow the
beekeeper never inspects**. A colony's fitness is its **held-out foraging accuracy**; the curve's
overall quality on an instance is the **geometric mean** of that accuracy across all meadows, so a
curve that collapses even one meadow is heavily penalized.

Your job: **design the curve `f`** — as its sampled values on a fixed stimulus grid — so it works
well *across settings*, not just one. This is the atomic ML-method-invention task from MLS-Bench
(dl-* proxies) skinned as an apiary: invent a transferable component, not per-dataset tuning.

## You are an isolated program
Read ONE JSON object (the PUBLIC view of an instance) from **stdin**; write ONE JSON object to
**stdout**. You never see the datasets, the hidden meadow, or the reference baseline.

### Public instance JSON
```
{
  "grid":          [s0, s1, ..., s40],   // 41 ascending stimulus values in [-6, 6]
  "train_steps":   160,                  // fixed training protocol (full-batch GD)
  "lr":            0.5,
  "public_meadows":[ {"kind": "...", "hidden_dim": H, "noise": v}, ... ]  // 3 descriptors (no data)
}
```
`kind` is one of `moons, circles, spiral, xor, gauss3`. A fourth, HIDDEN meadow is also scored but
not described here.

### Answer JSON
```
{ "ys": [f(s0), f(s1), ..., f(s40)] }   // one finite float per grid point; len == len(grid)
```
The evaluator builds `f` as the **piecewise-linear** interpolant of `(grid, ys)` (with linear
extrapolation beyond the ends), computes its slope for backprop, then trains each colony MLP
itself (fixed seeded init, full-batch gradient descent) and measures held-out accuracy.

## Objective (maximize)
For each instance the evaluator computes `obj = geomean(test_accuracy over the 4 meadows)` and
normalizes against the reference identity curve `f(s)=s`:
```
score_instance = min(1, 0.1 * obj / obj_identity)
```
So the identity/linear curve scores ≈ 0.1; a curve that beats it lifts the geometric-mean accuracy
and scores proportionally higher. The final **Ratio** is the mean over all instances; **Vector**
lists the per-instance scores.

## Rules
- Deterministic: all datasets and initializations are seeded; nothing depends on wall-time.
- Any malformed / wrong-length / non-finite / absurd (`|f|>1e4`) curve is rejected and scores 0.
- Multiple strategies are viable (bounded saturating curves, self-gated smooth curves, rectified
  variants, ...); there is no single easy optimum, and the geometric mean plus the hidden meadow
  punish curves that overfit one meadow type.
