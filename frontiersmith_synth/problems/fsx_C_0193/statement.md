# City Signal Grid: Intersection Regime Clustering

## Story

A traffic-management authority runs the signal grid of a large city. Every
intersection streams a **flow signature** — a `d`-dimensional feature vector
summarising its demand pattern (approach volumes, turn ratios, peak-hour skew,
queue spillback, pedestrian-phase load, ...). Intersections belong to a small
number of latent **operating regimes** — arterial corridors, the downtown grid,
residential feeders, industrial gateways, ... — and intersections in the same
regime should share a coordinated signal-timing plan.

You must design an **unsupervised clustering** algorithm that recovers the regime
of every intersection from its flow signature alone, **without ever seeing the
regime labels**.

You are graded across many *cities*, each with a different underlying geometry:

- **compact** — well-separated isotropic regimes (plain k-means friendly),
- **sheared** — regimes passed through an ill-conditioned linear map, so the
  clusters are elongated and correlated (whitening / PCA friendly),
- **scaled** — features on wildly different numeric scales, with two large-scale
  **nuisance channels** carrying no regime information (standardization friendly),
- **uneven** — regimes of unequal size and unequal spread.

No single textbook recipe wins every city, and the final score is a **geometric
mean** across cities, so a method that overfits one structure and collapses on
another is heavily penalized. The goal is a clusterer that **generalizes**.

## Isolation (how your program is run)

Your program is executed as an **isolated subprocess**. It reads exactly one JSON
object (the *public* view of one city) from **stdin** and writes exactly one JSON
value (your answer) to **stdout**. You never see the labels, any held-out data, or
the evaluator's memory.

```python
import sys, json
import numpy as np

inst = json.load(sys.stdin)              # public inputs ONLY
X = np.asarray(inst["X"], dtype=float)   # shape (n, d)
k = int(inst["k"])                       # number of regimes to recover (a hint)
# ... assign a cluster id to every intersection ...
print(json.dumps(labels.tolist()))       # the ONLY thing the evaluator reads
```

## Public instance (stdin)

```json
{
  "X":    [[float, ...], ...],   // n x d flow-signature matrix
  "n":    int,                   // number of intersections
  "d":    int,                   // number of features
  "k":    int,                   // number of regimes to recover (a hint)
  "seed": int                    // per-instance seed you MAY use for your own RNG
}
```

## Answer (stdout)

A JSON list of `n` integers: `labels[i]` is the cluster id of intersection `i`.
Cluster ids are arbitrary integers — only the **partition** they induce matters
(the metric is invariant to relabeling and to the number of distinct ids you use).
A JSON object `{"labels": [...]}` is also accepted.

Any of the following makes a city score **0**: wrong length, a non-list / non-object
output, a non-integer or non-finite id (`NaN`/`Inf`), a crash, a timeout, or no
output.

## Scoring

For each city the evaluator computes the **Adjusted Rand Index (ARI)** between your
partition and the hidden regime labels (contingency form; deterministic, no
wall-time). It normalizes against the **null clustering** (every intersection in a
single group, ARI `0`):

```
r = clamp( 0.1 + 0.9 * ari_cand, 0, 1 )
```

so reproducing the null partition maps to ≈ `0.1` and a perfect recovery
(ARI = 1) maps to `1.0`. Valid cities are floored to a small positive value; the
final reported score is the **geometric mean** of the per-city `r`, so being weak
on even one city geometry hurts a lot.

```
Ratio:  <geometric mean of per-city r, in [0,1]>
Vector: [r_1, r_2, ..., r_10]
```

## Objective

**Maximize `Ratio`.** There is no easy optimum: raw k-means recovers the compact
cities but is dominated by the large-scale nuisance channels on the *scaled*
cities and distorted by shear on the *sheared* cities; per-feature standardization
rescues *scaled* but not *sheared*; whitening rescues *sheared* but adds noise on
high-dimensional cities. A genuinely general-purpose, deterministic clusterer —
robust scaling, decorrelation, good initialization, sensible model selection — is
required to score well across the whole family.
