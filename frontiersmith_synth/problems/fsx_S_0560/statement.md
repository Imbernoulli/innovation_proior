# Stripes That Hold Under a Stretching Gradient

A growing 1-D tissue must paint itself into three stripes — a *French-flag*
pattern — by reading a single **morphogen gradient**: a monotone-increasing
chemical concentration along the body axis. The target proportions are fixed at
**1 : 2 : 1** — the anterior quarter is stripe **0**, the central half is stripe
**1**, the posterior quarter is stripe **2**.

The catch: **no two embryos are alike.** From instance to instance the gradient is
rescaled by an unknown amplitude, an unknown baseline offset, an unknown slope
(shape exponent), and the tissue length changes too. A slow developmental *bump*
and cell-level readout noise further corrupt the field. Yet the stripes must land at
the right proportions every time.

You submit a **gene-regulatory readout network**: a small static rule mapping each
cell's morphogen environment to a stripe label. Your one network is then applied to
a family of freshly **rescaled** gradients you never saw, and scored on how many
cells it labels correctly.

## Candidate program contract

Read ONE JSON object (the public instance) from **stdin**; write ONE JSON object
(your network) to **stdout**. Your program runs isolated and sees only the public
instance.

```python
import sys, json
inst = json.load(sys.stdin)
print(json.dumps(net))
```

### Public instance (stdin)

```json
{ "name": "tissue101", "L": 240, "field": [1.02, 1.05, ...] }
```

`field` is the **nominal** (un-rescaled) gradient; index `i` is the position of
cell `i` (index 0 = anterior). The rescaled gradients used for scoring are hidden.

### Network (stdout)

```json
{ "feature": "absolute" | "relative" | "rank",
  "smooth":  0,
  "cuts":    [c1, c2] }
```

For each cell the evaluator computes a scalar **feature value** `f`, then labels the
cell `0` if `f < c1`, `1` if `c1 <= f < c2`, else `2`. The feature is:

- `"absolute"` — the raw local concentration (optionally neighbour-averaged);
- `"relative"` — the concentration min-max-normalised to `[0,1]` over the field;
- `"rank"` — the cell's concentration **rank fraction** (fraction of cells carrying
  less morphogen), in `[0,1]`.

`smooth` is a non-negative integer: before computing the feature the field is
replaced by a neighbour average over radius `smooth` (a GRN cell reading its
neighbours to denoise). `cuts` are two finite numbers in feature space.

A result that is not a dict, an unknown `feature`, a non-integer or negative
`smooth`, `cuts` that is not a length-2 list of finite numbers, a crash, a timeout,
or non-JSON makes **every** instance score `0.0`.

## Objective — **maximize**

Across a fixed, seeded family of **10 instances** (each averaging your network over
several rescaled, bumped, noisy gradients; later instances are larger, wider-range
held-out embryos), maximize the mean fraction of correctly labelled cells.

## Scoring (deterministic)

For each instance the evaluator computes, itself:

- `acc_cand` — your network's mean correct-label fraction over that instance's
  rescaled gradients;
- `acc_anchor` — the majority baseline: paint every cell the central stripe
  (`= 0.5` by construction).

and normalizes toward a perfect field:

```
r = clamp( 0.1 + 0.9 * (acc_cand - acc_anchor) / (1.0 - acc_anchor), 0, 1 )
```

Painting everything the majority stripe scores `~0.1`. Perfect labelling scores
`1.0`, but the developmental bump plus readout noise keep even the best readout
strictly below it — real headroom. The reported **Ratio** is the mean of `r`; the
**Vector** lists per-instance scores.

## Suggested strategies

1. **Absolute thresholds** calibrated on the nominal field — perfect on that one
   embryo, but the thresholds miscalibrate the moment amplitude / slope rescales.
2. **Relative (min-max) readout** — cancels amplitude and offset, but a nonlinear
   slope change still bends the boundaries.
3. **Rank / positional readout** — the rank fraction is invariant to *any* monotone
   rescaling, recovering relative position directly.
4. **Neighbour smoothing + cut tuning** — average over neighbours to fight noise and
   nudge the cut points against the residual bump-induced bias.
