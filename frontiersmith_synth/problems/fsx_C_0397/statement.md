# Interstellar Relay Fingerprint Sorting

An interstellar relay network intercepts faint signal *fingerprints* emitted by a set of
unknown source stations. Each fingerprint is a point in feature space. You know **how many
source stations `k`** produced the batch, but not which station produced each fingerprint.
Your job: design a clustering algorithm that recovers the station-of-origin partition, and
that does so **across wildly different relay geometries** — cleanly separated station
clusters, faint/varied-brightness stations, sheared (anisotropic) beams, high-dimensional
fingerprints, and pathological non-convex layouts (two crescents, concentric rings).

Because the scorer takes the **geometric mean** of your accuracy over the whole battery, a
method that aces convex blobs but collapses on rings/moons is punished hard. The reward goes
to a *transferable* clustering design.

## Candidate program contract (isolated)

Your program is a standalone process. Read ONE JSON object (the public instance) from stdin,
write ONE JSON object (your answer) to stdout. You only ever see the public view; the true
station labels stay in the evaluator.

### Public instance (stdin)
```json
{
  "points": [[x0, x1, ...], ...],   // n fingerprints, each a `dim`-vector of floats
  "k": 3,                            // number of source stations (true cluster count)
  "n": 220,                          // number of fingerprints
  "dim": 2                           // feature dimension
}
```

### Answer (stdout)
```json
{ "labels": [c_0, c_1, ..., c_{n-1}] }   // one integer cluster id per fingerprint
```
`labels` must be a list of exactly `n` integers (any integer values; the metric is invariant
to how you name clusters). Wrong length, non-integers, `NaN`/`Inf`, a crash, or no output all
score 0 on that instance.

## Objective and scoring (deterministic)

For each instance the evaluator computes the **Adjusted Rand Index (ARI)** between your
`labels` and the hidden true station assignment. ARI = 1 is a perfect partition; ARI = 0 is
chance-level.

Each instance's raw ARI is normalized against a **weak reference** (a median split on the
maximum-variance coordinate): matching the weak reference maps to ≈ 0.10, a perfect partition
maps to ≈ 0.95, so there is genuine headroom above any single fixed method. The final score is
the **geometric mean** of the per-instance normalized scores over the fixed 8-geometry battery.

The instance distribution is fixed and seeded, so the score is fully reproducible. There is no
easy global optimum: convex methods (k-means) win some geometries and lose others, manifold
methods (spectral) win the non-convex ones, and mixture models win the anisotropic/varied
ones — you must choose a strategy (or a per-instance selection rule) that transfers.
