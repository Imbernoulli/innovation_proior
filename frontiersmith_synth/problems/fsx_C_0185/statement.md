# Glacier Sensor Net: A Drop-in Activation Across Stations

## Story

A network of autonomous sensor stations is buried across an alpine glacier. Each
station streams a handful of physical channels (sub-surface temperature, ice
velocity, tilt, meltwater conductivity, seismic / acoustic emission, ...) and runs a
tiny on-station MLP that classifies the local ice regime (quiescent / creep /
calving-risk).

Every station ships with the **same** model shape and the **same** fixed training
recipe. Field engineering has frozen everything except **one** module: the
hidden-layer **activation function**. Whatever activation you design is flashed
**identically** to every station in the network — including remote stations you
never get to see. So you are not tuning a model to a dataset; you are inventing a
single, transferable component that has to work everywhere.

You are graded across several *stations*, each with a different underlying decision
structure:

- **xor** — sign-XOR of two channels (needs units that respond to *negative*
  pre-activations),
- **rings** — concentric radius bands,
- **bands** — interleaved sinusoidal stripes,
- **spiral** — a two-arm spiral boundary,
- **sine** — a wavy, near-linear sinusoidal boundary under label noise (rewards a
  *saturating* activation; an unbounded rectifier overfits here),
- plus a held-out **unseen field station**.

No single textbook activation wins every station, and the final score is a
**geometric mean** across stations, so an activation that helps one station but
collapses another (dead units, exploding gradients, over-fitting the noise) is
punished hard. The goal is one activation that **generalizes**.

## Isolation (how your program is run)

Your program is executed as an **isolated subprocess**. It reads exactly one JSON
object (the *public* view) from **stdin** and writes exactly one JSON value (your
answer) to **stdout**. You never see any station's data, the labels, the training
loop, or the evaluator's memory.

```python
import sys, json
inst = json.load(sys.stdin)      # public inputs ONLY
# ...compute your activation table...
print(json.dumps(answer))        # the ONLY thing the evaluator reads
```

## Public instance (stdin)

```json
{
  "grid":   [float, ...],   // K strictly-increasing knot positions x_0 < ... < x_{K-1}
  "n_grid": 81,             // K
  "note":   "string hint",
  "seed":   20240185        // a seed you MAY use for your own RNG
}
```

## Your answer (stdout)

A length-`K` list of reals: `answer[i] = g(grid[i])`, the value of your activation
`g` at each grid knot. (A dict `{"activation": [...]}` or `{"g": [...]}` is also
accepted.)

```json
[g0, g1, ..., g80]
```

The evaluator interprets your samples as a **piecewise-linear** function:
- **forward**: `g(z)` is the linear interpolation between the two surrounding
  knots (clamped flat to the endpoint values outside `[grid[0], grid[-1]]`);
- **backward**: `g'(z)` is the slope of the containing segment (0 outside the
  grid).

So any table you return is a fully differentiable, drop-in hidden activation.

## How you are scored

For each station the evaluator plugs your `g` into a fixed 2-layer MLP and trains it
with plain full-batch gradient descent (fixed weight-init seed, fixed learning rate,
fixed number of epochs), then measures **held-out test accuracy**. Your accuracy is
normalized against the evaluator's own **identity (linear) baseline** — the same
training run whose hidden activation is `g(x)=x`:

```
r = clamp( 0.1 + 0.9 * (acc_cand - acc_base) / (acc_ceiling - acc_base), 0, 1 )
```

- matching the linear baseline → `r ≈ 0.1`; reaching the per-station ceiling → `r = 1`;
- doing **worse** than the linear baseline on a station floors that station to a tiny
  positive value.

The final **Ratio** is the **geometric mean** of the per-station `r` values, so you
must be good on *all* stations at once — a single collapse tanks the whole score.

An answer that raises, has the wrong length, contains non-finite values, or drives
training to non-finite weights scores **0.0** on the affected station(s).

## Objective

**Maximize** the geometric-mean Ratio in `[0, 1]`. Designing an activation that
merely matches the linear baseline scores ~0.1; a well-chosen nonlinearity that
generalizes across every station scores much higher, and there is headroom left for
a cleverly shaped custom activation.

## Determinism

Everything is seeded; the evaluator is re-run and must reproduce the same `Ratio`
and `Vector`. Do not rely on wall-clock, threads, or external state.
