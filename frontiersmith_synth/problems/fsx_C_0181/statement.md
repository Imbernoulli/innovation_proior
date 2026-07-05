# Warehouse AGV Fleet: Fault-Telemetry Anomaly Scoring

## Story

A fleet of autonomous guided vehicles (AGVs) roams a warehouse, streaming
telemetry snapshots to a monitoring service. Each snapshot is a `d`-dimensional
feature vector (motor current, wheel-slip, battery draw, vibration, path
deviation, ...). The vast majority of rows are **nominal** operation, but a small
unknown fraction are **faults** (bearing wear, wheel slip, sensor drift, ...).
You must design an **unsupervised anomaly-scoring** algorithm that ranks the rows
most likely to be faults — without ever seeing the labels.

You are graded across many *fleets*, each with a different underlying structure:

- **global** — one nominal cloud with far-flung isotropic outliers,
- **correlated** — nominals lie on a low-rank correlation manifold and faults
  break that correlation (they match the per-feature ranges but not the joint
  structure),
- **multimodal** — several well-separated operating regimes with faults in the
  gaps between them,
- **scaled** — features on wildly different numeric scales, with the fault
  hiding in a small-scale feature,
- **local** — two nominal clusters of different density, with faults sitting
  just outside the *dense* cluster (closer to the global mean than the far
  sparse cluster is).

No single textbook detector wins every fleet, and the final score is a
**geometric mean** across fleets, so a method that overfits one structure and
collapses on another is heavily penalized. The goal is a detector that
**generalizes**.

## Isolation (how your program is run)

Your program is executed as an **isolated subprocess**. It reads exactly one JSON
object (the *public* view of one fleet) from **stdin** and writes exactly one JSON
value (your answer) to **stdout**. You never see the labels, any held-out data, or
the evaluator's memory.

```python
import sys, json
import numpy as np

inst = json.load(sys.stdin)          # public inputs ONLY
X = np.asarray(inst["X"], dtype=float)   # shape (n, d)
# ... compute an anomaly score per row (higher = more anomalous) ...
print(json.dumps(scores.tolist()))   # the ONLY thing the evaluator reads
```

## Public instance (stdin)

```json
{
  "X": [[float, ...], ...],   // n x d telemetry matrix
  "n": int,                   // number of rows
  "d": int,                   // number of features
  "contamination": float,     // approximate fault fraction (a hint; not required)
  "seed": int                 // per-instance seed you MAY use for your own RNG
}
```

## Answer (stdout)

A JSON list of `n` finite floats: `scores[i]` is the anomaly score of row `i`,
where **higher means more anomalous**. (A JSON object `{"scores": [...]}` is also
accepted.) Only the *ranking* induced by the scores matters — the metric is
threshold-free.

Any of the following makes a fleet score **0**: wrong length, non-list/object
output, a non-finite value (`NaN`/`Inf`), a crash, a timeout, or no output.

## Scoring

For each fleet the evaluator computes the **ROC-AUC** between your scores and the
hidden fault labels (rank / Mann-Whitney form, average ranks for ties;
deterministic, no wall-time). It normalizes against its own internal baseline
(raw Euclidean distance to the global mean, AUC `auc_base`):

```
r = clamp( 0.1 + 0.9 * (auc_cand - auc_base) / (1 - auc_base), 0, 1 )
```

so reproducing the baseline maps to ≈ `0.1` and a perfect detector (AUC = 1) maps
to `1.0`. Valid fleets are floored to a small positive value; the final reported
score is the **geometric mean** of the per-fleet `r`, so being weak on even one
fleet structure hurts a lot.

```
Ratio:  <geometric mean of per-fleet r, in [0,1]>
Vector: [r_1, r_2, ..., r_10]
```

## Objective

**Maximize `Ratio`.** There is no easy optimum: raw distance ties the baseline,
z-scoring rescues scaled fleets but breaks on multi-modal and correlated ones,
and local-density / manifold methods trade off elsewhere. A genuinely
general-purpose, deterministic detector is required to score well across the
whole family.
