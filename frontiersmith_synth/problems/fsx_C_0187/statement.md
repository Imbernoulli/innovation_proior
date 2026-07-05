# Ski Resort Lift Traffic: Skier-Segment Clustering

## Background
A large alpine resort logs, for every **guest-day**, a vector of usage features derived
from lift scans and GPS: mean lift-ride altitude, run-difficulty preference, number of
lift rides, distance skied, chairlift-vs-gondola ratio, rest-stop count, and so on.
Marketing wants to recover the latent **guest segments** ("first-timers", "powder hounds",
"family cruisers", "park rats", "apres-ski socialites", ...) directly from these usage
vectors, with **no labels**.

Your job: design an **unsupervised clustering algorithm** that partitions each day's
guests into `k` segments so the partition matches the (hidden) true segmentation as well
as possible — across a *diverse family* of resort days with very different geometry.

## Isolation / how your program is run
Your program is run as an **isolated subprocess**. It reads ONE JSON object (the *public
instance*) from **stdin** and writes ONE JSON value (your answer) to **stdout**. It never
sees the true segments, any held-out data, or the evaluator's memory.

```python
import sys, json
inst = json.load(sys.stdin)   # public instance (below)
# ... compute cluster labels ...
print(json.dumps(labels))     # the ONLY thing the evaluator reads
```

## Public instance (stdin)
```json
{
  "X":    [[float, ...], ...],   // N x d guest-day usage matrix
  "n":    int,                   // number of guest-days N
  "d":    int,                   // number of usage features
  "k":    int,                   // number of segments to recover (given)
  "seed": int                    // a per-instance seed you MAY use for your own RNG
}
```

## Answer (stdout)
A JSON list of `N` integer cluster labels:
```json
[int, int, ..., int]            // labels[i] = segment id assigned to guest-day i
```
- Length **must** equal `N`.
- Labels are **arbitrary integer ids** — scoring is permutation-invariant, so you need not
  match the true segment numbering; only the *grouping* matters. Aim for a partition into
  at most `k` groups.
- Any non-finite, non-integer, or wrong-length answer scores **0** on that instance.

## Objective (maximize)
For each instance the evaluator recomputes the **Adjusted Rand Index (ARI)** between your
partition and the hidden true segments, deterministically from the contingency table.
It normalizes against an internal weak baseline (a rank-quantile split of the first
feature into `k` groups):

```
r = clamp( 0.1 + 0.9 * (ari_cand - ari_base) / max(1 - ari_base, 0.15), 0, 1 )
```

so reproducing the baseline scores ~0.1 and a perfect partition (ARI = 1) scores 1.0.
The final score is the **geometric mean of `r` over all instances**, so a method that
overfits one geometry and collapses on another is heavily penalized. The instance family
mixes:

- **compact well-separated blobs** (centroid methods excel),
- **interleaving crescents** and **concentric ability rings** (non-convex; need a
  connectivity / graph method),
- **anisotropically stretched** segments,
- **wildly unequal spread** across segments,
- **high-dimensional** profiles padded with noise features.

No single off-the-shelf clustering wins them all — a clustering that **generalizes** across
geometries wins.

## Scoring output
The evaluator prints:
```
Ratio: <geometric mean of per-instance r, in [0,1]>
Vector: [r_1, r_2, ...]
```
