# Drop-in Activation Design: the MLP Battery

## Task
Design a **single drop-in activation function** `phi: R -> R` for a fixed *battery* of
tiny CPU multilayer perceptrons (MLPs). Each net in the battery learns a small 2-D,
non-linearly-separable classification task (XOR, concentric circles, spirals,
interleaved moons, Gaussian blobs, or a checkerboard). Every net has the identical
shape — inputs → one hidden layer of 16 units → softmax — and is trained with the
identical, seeded, full-batch gradient-descent schedule. **The only thing you control
is the hidden-layer activation**, which is dropped in unchanged across the whole
battery.

Your goal is to **maximize the geometric-mean test accuracy** of the battery. A linear
activation (the identity) collapses every net to a linear model and cannot separate
these tasks; a good nonlinearity lets the nets carve curved decision boundaries.

## How you supply `phi`
You do not ship code. You ship `phi` as a **table of values on a fixed grid** of
x-coordinates. The evaluator rebuilds `phi` as the **piecewise-linear interpolant** of
your table: for a pre-activation `z`,
`phi(z) = interp(z; grid, y)` and `phi'(z) =` the slope of the enclosing grid segment
(values are clamped to the grid ends; the derivative is 0 outside `[x_min, x_max]`).
So your table defines **both** the forward activation **and**, through its segment
slopes, the gradient the nets train with.

## Input (public instance, via stdin as one JSON object)
```json
{
  "name": "battery-A",
  "x_min": -8.0, "x_max": 8.0, "n_grid": 161,
  "grid": [-8.0, -7.9, ..., 8.0],     // fixed ascending x-coordinates, length n_grid
  "n_nets": 4,                         // nets trained in this battery
  "announced_kinds": ["xor", "moons", "blobs3"]   // kinds of the announced nets;
                                       // ONE extra held-out net of an undisclosed
                                       // kind is also scored (generalization)
}
```

## Output (via stdout as one JSON object)
```json
{"y": [phi(x_0), phi(x_1), ..., phi(x_{n_grid-1})]}
```
`y` must be a list of exactly `n_grid` real, finite numbers, each with magnitude
`<= 1e4`. Wrong length, a non-number, `NaN`/`Inf`, an over-large value, a crash, a
timeout, or non-JSON output makes that battery score `0.0`.

## Scoring (deterministic)
For each battery the evaluator computes the geometric-mean test accuracy with your
`phi` (`g_cand`) and with the identity reference (`g_base`), then normalizes:
```
r = clamp( 0.1 + 0.9 * (g_cand - g_base) / (1.0 - g_base), 0, 1 )
```
Reproducing the identity scores about `0.1`. Datasets carry ~7% label noise, so
perfect accuracy is unreachable and even the best activations stay strictly below
`1.0` on every battery. The reported **Ratio** is the mean of `r` over all batteries;
**Vector** lists the per-battery `r`.

## Notes
- The candidate runs in an isolated sandbox and sees only the public instance (the
  grid + metadata). The datasets, the held-out net, the net initializations, and the
  identity reference exist only inside the evaluator.
- Objective: **maximize**. Multiple viable strategies (saturating vs. non-saturating,
  self-gated, hand-shaped tables) trade off differently across the diverse dataset
  kinds and the held-out net; there is no single free optimum.
