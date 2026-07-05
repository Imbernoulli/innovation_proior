# Rooftop Gardens: A Drop-in Learning-Rate Schedule Across Plots

## Story

A city block is covered in autonomous **rooftop gardens**. Each rooftop runs a tiny
on-site classifier that reads a handful of microclimate channels (soil moisture,
canopy temperature, PAR light, wind, substrate pH, ...) and predicts the local
growth regime (thriving / stressed).

Every rooftop ships with the **same** tiny model shape, the **same** fixed ReLU MLP,
the **same** fixed weight initialization, the **same** fixed base learning rate, and
the **same** fixed number of full-batch gradient-descent epochs. Field engineering
has frozen everything except **one** module: the per-epoch **learning-rate
schedule** — a list of multipliers `m_0 … m_{E-1}` applied on top of the fixed base
LR. Whatever schedule you design is flashed **identically** to every rooftop —
including plots you never get to see. So you are not tuning an optimizer to one
dataset; you are inventing a single, transferable component that has to work
everywhere.

The block's fixed base LR is deliberately **hot**:

- a **flat** schedule (all ones) keeps over-shooting the noisy, near-linear plots
  and never settles;
- a naively **low flat** LR stops the over-shoot but is then too slow to converge
  the hard, sharply-nonlinear plots within the epoch budget.

You are graded across several *plots*, each with a different underlying decision
structure:

- **spiral** — a two-arm spiral boundary (hard; needs *sustained* LR to converge in
  budget),
- **sine** — a wavy, near-linear boundary under heavy label noise (needs *annealing*
  down so it stops over-shooting the noise),
- **xor** — sign-XOR of two channels,
- **rings** — concentric radius bands,
- plus **held-out** rooftops (an unseen striped geometry and a wider spiral with an
  extra channel).

No single **constant** learning-rate multiplier wins every plot, and the final score
is a **geometric mean** across plots, so a schedule that helps one plot but collapses
another (divergence on the hard plot, over-shoot on the noisy plot, or no learning at
all) is punished hard. The goal is one schedule that **generalizes**.

## Isolation (how your program is run)

Your program is executed as an **isolated subprocess**. It reads exactly one JSON
object (the *public* view) from **stdin** and writes exactly one JSON value (your
answer) to **stdout**. You never see any plot's data, the labels, the training loop,
or the evaluator's memory.

```python
import sys, json
inst = json.load(sys.stdin)      # public inputs ONLY
# ...compute your schedule...
print(json.dumps(answer))        # the ONLY thing the evaluator reads
```

## Public instance (stdin)

```json
{
  "n_epochs":  120,           // E -- the schedule length you must return
  "base_lr":   4.0,           // the fixed base learning rate (your multipliers scale this)
  "mult_max":  5.0,           // each multiplier is clamped to [0, mult_max]
  "n_settings_hint": 6,       // how many plots you are scored on
  "note":      "string hint",
  "seed":      20240383       // a seed you MAY use for your own RNG
}
```

## Your answer (stdout)

A length-`n_epochs` list of reals: `answer[t] = m_t`, the learning-rate multiplier
used at epoch `t`. At epoch `t` the fixed MLP is trained with
`lr = base_lr * clip(m_t, 0, mult_max)`.

```json
[m0, m1, ..., m119]
```

(A dict `{"schedule": [...]}` or `{"lr_mult": [...]}` is also accepted.) Each
multiplier is clamped into `[0, mult_max]`; a multiplier so large it drives training
to non-finite weights scores **0.0** on that plot.

## How you are scored

For each plot the evaluator plugs your schedule into a fixed 2-layer ReLU MLP and
trains it with plain full-batch gradient descent (fixed weight-init seed, fixed base
LR, fixed number of epochs), then measures **held-out test accuracy**. Your accuracy
is normalized against the evaluator's own **flat-schedule baseline** — the identical
training run with every `m_t = 1`:

```
r = clamp( 0.1 + 0.9 * (acc_cand - acc_base) / (ceiling - acc_base), 0, 1 )
```

- matching the flat baseline → `r ≈ 0.1`; reaching the per-plot accuracy ceiling →
  `r = 1`;
- doing **worse** than the flat baseline on a plot floors that plot to a tiny
  positive value.

The final **Ratio** is the **geometric mean** of the per-plot `r` values, so you
must be good on *all* plots at once — a single collapse tanks the whole score.

An answer that raises, has the wrong length, contains non-finite values, or drives
training to non-finite weights scores **0.0** on the affected plot(s).

## Objective

**Maximize** the geometric-mean Ratio in `[0, 1]`. A flat schedule that merely keeps
the default LR scores ~0.1; simply turning the LR down helps but the wrong *shape*
leaves points on the table; a well-shaped schedule (warm-up + annealing, or a cyclic
/ step form) that generalizes across every plot scores much higher, and there is
headroom left for a cleverly tuned schedule.

## Determinism

Everything is seeded; the evaluator is re-run and must reproduce the same `Ratio`
and `Vector`. Do not rely on wall-clock, threads, or external state.
