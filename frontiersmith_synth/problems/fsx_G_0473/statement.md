# Segment Atlas — an Unsupervised Customer Segmentation Assigner

## Background
A retail analytics team runs a marketing campaign that requires customers to be
split into **k** segments. Each customer is a point in a 2-D behavioural feature
space (an embedding of spend recency/frequency, price sensitivity, browsing
style, …). The campaign brief fixes the number of segments `k`, but **not** which
customer belongs to which segment — that is your job.

Real segment geometry is not always a tidy ball of points:

* some campaigns split customers into compact **spend tiers** — Gaussian *blobs*;
* some into two interleaving **lifestyle arcs** — non-convex *moons*;
* some into nested **loyalty shells** around a core — concentric *rings*.

A centroid method (k-means) nails the blobs but shreds the moons and rings. A good
segmenter respects **connectivity / manifold structure**, not just Euclidean
proximity. Your program is an **unsupervised** segmenter: it sees only the point
cloud and the target segment count `k`, and must label every customer.

You write a **standalone program**: read ONE JSON instance from stdin, write ONE
JSON answer to stdout. It is run in an isolated sandbox over a fixed distribution
of instances; it only ever sees the public data below.

## Public instance (stdin)
A single JSON object:
```json
{
  "name": "moon4743",
  "n": 200,
  "k": 2,
  "points": [[x0, y0], [x1, y1], ...]
}
```
* `n` — number of customers.
* `k` — target number of segments (≥ 2).
* `points` — `n` customers, each a `[x, y]` pair of floats.

## Answer (stdout)
A single JSON object:
```json
{"labels": [l0, l1, ..., l_{n-1}]}
```
`labels[i]` is the integer segment id assigned to customer `i`.

**Validity.** `labels` must be a list of exactly `n` integers, each in `[0, n)`.
You need **not** use exactly `k` distinct labels — the score is permutation- and
count-tolerant — but a degenerate one-segment or all-singleton labelling earns a
zero-agreement score. A wrong length, a non-integer / boolean / out-of-range
label, non-JSON output, a crash, or a timeout scores **0.0** on that instance.

## Objective — MAXIMIZE
For each instance the grader compares your labelling to a hidden ground-truth
segmentation with the **Adjusted Rand Index (ARI)** — a permutation-invariant
agreement measure in `[-0.5, 1]` (`1` = perfect, `0` ≈ random). ARI does not care
which integer names a segment, so you never need to match the graders' ids.

Let `q_cand = ARI(truth, your labels)` and let `q_base` be the ARI of the grader's
own weak reference segmentation — *sort customers by their first feature and cut
into k equal ranked bins* (blind to 2-D shape). Each instance is normalised with
an affine anchor:
```
r = clamp( 0.1 + 0.9 * (q_cand - q_base) / max(1e-9, 1 - q_base), 0, 1 )
```
so the coordinate-threshold reference scores ≈ **0.1**, a perfect segmentation
scores **1.0**, and doing worse than the reference scores below 0.1. The reported
**Ratio** is the mean of `r` over all instances; higher is better.

Several instances are noisy or non-convex (interleaving moons, nested rings)
where even a connectivity-aware method mislabels boundary customers, so a strong
segmenter still stays well below 1.0 — there is genuine headroom.

## Scoring notes
* **Deterministic.** The instance distribution is fixed and seeded; your program
  is re-run and must produce identical output. Seed any randomness you use (e.g.
  from `n`/`k`) — never use wall-clock time.
* **Isolated.** Your program runs in a fresh sandbox and sees only the public
  instance above. The ground truth, the reference segmentation and the ARI
  scoring live in the grader process and are unreachable from your program.
* **Non-finite rejected.** Only integer labels are accepted; any malformed answer
  scores 0 on that instance.

## Viable strategies (open-ended)
* **Coordinate threshold** — reproduces the weak reference (~0.1). Trivial floor.
* **k-means (Lloyd + k-means++)** — great on blobs, weak on moons/rings.
* **Normalised spectral clustering** — kNN Gaussian affinity → top-k eigenvectors
  → k-means the embedding; follows manifold structure and recovers all three
  shapes on aggregate.
* **Density / connectivity methods** — single-linkage, DBSCAN, mutual-kNN
  components; strong on clean shapes, fragile to noise bridges.

No strategy hits a perfect score across the whole distribution — improving the
mean ARI is the open-ended challenge.
