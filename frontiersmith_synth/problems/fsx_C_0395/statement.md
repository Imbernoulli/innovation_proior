# Cave Mapping Expedition: A Drop-in Activation Function Across Survey Sites

## Story

An underground survey team drops an identical tiny classifier at every cave site it
reaches. Each site's on-rig model reads a handful of sonar/echo features off the rock
face and predicts the local passage regime (open chamber vs. blocked seam). Every rig
ships with the **same** frozen recipe: the same 1-hidden-layer MLP shape, the same fixed
weight init, the same **deliberately hot** base learning rate, and the same fixed number
of full-batch gradient-descent epochs.

The **one** module you get to design is the hidden-layer **activation function**. You do
not tune it per cave — you invent a single activation that is flashed **identically** to
every rig, including survey sites the team never mapped. You never see any site's data.

Every cave here is genuinely nonlinear (twisting tunnels, cross-faults, concentric
chambers, interleaved strata). So the naive drop-in — passing the pre-activation straight
through (a **linear / identity** activation) — collapses the network to a linear
classifier that under-fits every site. Any genuine nonlinearity unlocks the hidden layer
and jumps well above that linear baseline. But the base learning rate is hot: a hard
rectifier (ReLU) over-shoots the noisier caves, a too-flat/saturating curve throws away
signal, and a runaway slope drives training to non-finite weights. No trivial choice is
best across all sites — a well-shaped curve can be.

## How your activation is expressed

You return the activation as a **piecewise-linear curve**: its values at a fixed grid of
input knots. The evaluator reconstructs the activation by

- linear interpolation between knots inside the grid, and
- linear extrapolation using the nearest end-segment slope outside the grid;
- the local segment slope is used as the derivative during backpropagation.

The evaluator then trains the frozen MLP with plain full-batch gradient descent using your
activation, and measures held-out test accuracy at each site.

## Public instance (stdin, JSON)

```json
{
  "n_grid": 41,            // K -- number of grid knots; your answer has length K
  "x_lo": -8.0,            // grid = linspace(x_lo, x_hi, n_grid), evenly spaced
  "x_hi": 8.0,
  "n_settings_hint": 6,    // number of cave sites scored (some are held-out)
  "note": "...",           // a reminder of the contract
  "seed": 20240395         // a per-call seed your program MAY use for its own RNG
}
```

Your program receives **only** this public view — never any site's features, labels, the
per-site MLP init, or the baseline. It runs in an isolated sandbox subprocess.

## Answer (stdout, JSON)

A length-`n_grid` list of floats: `answer[k]` is the activation value at grid knot
`grid[k] = linspace(x_lo, x_hi, n_grid)[k]`.

```json
[a_0, a_1, ..., a_{K-1}]
```

A dict form `{"values": [...]}` (or `{"a": [...]}`) is also accepted. All values must be
finite; the list must have exactly `n_grid` entries.

## Objective & scoring

For **each** cave site the evaluator:

1. builds your piecewise-linear activation from the returned knot values;
2. trains the frozen MLP (fixed init seed, fixed hot base LR, fixed epoch budget);
3. measures held-out test accuracy `acc_cand`, and its own **linear-baseline** accuracy
   `acc_base` (the identical run with the identity activation `a(x)=x`).

The per-site normalized score is an affine anchor against the linear baseline:

```
r = clamp( 0.1 + 0.9 * (acc_cand - acc_base) / (ceiling - acc_base), 0, 1 )
```

so an activation that merely matches the linear baseline maps to `~0.1`, and one that
reaches the per-site accuracy ceiling maps to `1.0`. Valid-but-weak sites are floored to a
small positive value so the geometric mean stays defined.

The final score is the **geometric mean** of the per-site `r`. Because it is a geometric
mean, an activation that helps one cave but collapses another (kills learning, over-shoots
the noise, or diverges to non-finite weights) is punished hard. This rewards a single
activation that **generalizes across sites**, not per-dataset tuning.

A call that raises, times out, returns the wrong length, or emits non-finite values scores
`0.0` on the affected site(s); a non-finite training trajectory scores `0.0` on that site.

The evaluator prints two final lines:

```
Ratio: <geometric mean of per-site r, in [0,1]>
Vector: [r_1, r_2, ..., r_6]
```

## Notes

- **Determinism.** Every site's data, init, and training are fully seeded; the score is a
  pure function of your returned activation. No wall-clock or hardware influences scoring.
- **Isolation.** Your program is untrusted: it runs OS-sandboxed and sees only the public
  instance. The data, labels, per-site init, baseline, and scoring code stay in the
  evaluator process and are unreachable from your program.
- **Headroom.** Standard smooth activations do well but do not saturate the ceiling; a
  carefully shaped curve can squeeze the remaining per-site headroom.
