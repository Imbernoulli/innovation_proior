# SwarmSort: Drone Delivery Zone Discovery

## Story

A city-wide **drone delivery swarm** drops parcels all over the metro map. At the
end of each shift the operations centre holds a cloud of 2-D drop coordinates —
where every parcel was actually delivered — but the original **routing zone** each
drop belonged to has been lost. To rebalance depots for tomorrow, the centre must
**re-discover the delivery zones purely from the geometry of the drop cloud**:
partition the drops into exactly `K` zones so that drops that truly belonged to the
same routing zone end up together.

Only the operator (the evaluator) knows the true zone of each drop — that ground
truth is **hidden**. You see only the anonymised coordinates and the target zone
count `K`, and must output a labelling. This is unsupervised **clustering**: your
partition is compared to the hidden true zones with the **Adjusted Rand Index
(ARI)**, a permutation-invariant, chance-corrected agreement score.

## You write a standalone program (stdin → stdout)

Read ONE JSON object (the public instance) from stdin, write ONE JSON object to
stdout.

### Input (public instance)
```json
{
  "name": "grid3x3",
  "n": 270,
  "dim": 2,
  "k": 9,
  "points": [[x0, y0], [x1, y1], "...", [x_{n-1}, y_{n-1}]]
}
```
- `points` — `n` drop coordinates (floats), 2-D.
- `k` — the target number of routing zones to recover.

### Output (answer)
```json
{ "labels": [c_0, c_1, "...", c_{n-1}] }
```
- Exactly `n` integer zone ids, one per drop, in input order.
- Values are **arbitrary integers** — ARI is invariant to relabelling and does not
  require exactly `K` distinct ids (you may use fewer or more), but each entry must
  be a plain **finite integer** (no floats, booleans, `NaN`, or `Infinity`).

Wrong length, a non-integer entry, a crash, a timeout, or non-JSON output scores
**0.0** on that instance.

## Objective

**Maximise** agreement with the hidden true zones. Per instance the evaluator
computes:
- `ari_cand` — ARI of your labelling vs. the hidden true zones,
- `ari_base` — ARI of a weak internal reference: sort drops by x-coordinate and cut
  into `K` equal-count contiguous bands.

and normalises (weak axis-split → 0.1, perfect recovery → 1.0):
```
r = clamp( 0.1 + 0.9 * (ari_cand - ari_base) / max(1e-9, 1.0 - ari_base), 0, 1 )
```

The reported **Ratio is the geometric mean** of the per-instance `r` values. The
geometric mean rewards a method that transfers across the whole zoo of zone shapes
(round grids, circular depots, varied spreads, anisotropic shears, interleaving
crescents, concentric rings) and severely penalises overfitting one family while
collapsing on another — a single near-zero instance drags the whole score down.

## Instance distribution (deterministic, seeded)

Ten instances (`n` ≈ 240–360, `k` from 2 to 9): tight grid blobs, circular depot
blobs, varied-spread blobs, anisotropic (sheared) blobs, two interleaving crescents
("moons"), and two concentric rings, plus larger held-out grid / varied / sheared
sets. Centre layouts are chosen so a single x-threshold split is genuinely weak.
Non-convex zones (crescents, rings) cannot be recovered by centroidal methods, so
even a strong K-means-style clusterer stays well below 1.0 there — there is real
headroom above the strong baseline for shape-aware methods.

## Scoring & isolation

Scoring is **fully deterministic** (all instances are seeded; no wall-time enters
the score). Your program is untrusted and runs **OS-sandboxed in a fresh
subprocess**; it only ever sees the public instance. The hidden true zones and all
references are computed in the evaluator process, so introspection / source-reading
gains nothing.
