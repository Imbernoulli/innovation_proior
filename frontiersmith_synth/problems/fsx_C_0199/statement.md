# Contact-Net Outbreak Clustering

## Scenario

An epidemiology team has reconstructed, for each of several outbreaks, a 2-D
**exposure embedding**: every individual is a point whose coordinates summarize where and
when they were exposed on the contact network. Distinct outbreaks were driven by very
different transmission structures:

- tight **household blobs** (round, well separated),
- **ring / superspreader** transmission (interleaving crescents),
- **concentric** spread (nested rings around a source),
- **chain** transmission (elongated, sheared clusters).

You must design **one** clustering algorithm that recovers the true community assignment
across *all* of these geometries. You are told how many outbreak clusters `k` each dataset
contains; you are NOT told which individual belongs to which community.

## Candidate program contract

Your program is a standalone process. Read ONE JSON object (the public instance) from
stdin and write ONE JSON object (your answer) to stdout.

### Public instance (stdin)
```json
{
  "X": [[x0, y0], [x1, y1], ...],   // n exposure coordinates (floats)
  "k": 3                             // number of outbreak clusters to produce
}
```

### Answer (stdout)
```json
{ "labels": [c0, c1, ..., c_{n-1}] }  // one INTEGER cluster id per individual, length n
```
Any integer labels are accepted; the score is invariant to how you name/permute clusters.
Malformed output (wrong length, non-integer, non-finite, or not a `{"labels": [...]}`
object) scores 0 on that dataset.

## Objective (maximize)

For each dataset the evaluator recomputes the **Adjusted Rand Index (ARI)** between your
labels and the hidden ground-truth community labels, then maps it to a per-dataset score

```
s_i = 0.1 + 0.9 * clip(ARI_i, 0, 1)
```

so a structurally valid answer earns at least the 0.1 floor and a perfect recovery earns 1.
The reported **Ratio is the GEOMETRIC MEAN of the per-dataset scores** over all datasets.
Because the geometric mean is dominated by its weakest term, an algorithm that nails one
geometry but collapses on another is heavily penalized: you are rewarded for a *transferable*
method, not a per-dataset trick.

## Rules / notes

- Scoring is fully deterministic; every dataset is fixed and seeded. Do not rely on
  wall-clock, GPU, or randomness that is not itself seeded inside your program.
- There is no single optimum. k-means, spectral/graph methods, density- or
  connectivity-based methods, standardization, self-tuning affinities, and restart
  strategies are all viable ingredients; the datasets deliberately span convex and
  non-convex structure with a difficulty gradient (later datasets are noisier / have more
  clusters), so a strong submission stays well below a perfect score on the hardest ones.
