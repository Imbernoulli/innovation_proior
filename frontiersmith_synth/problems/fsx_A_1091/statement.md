# Phantom Ward Cartography

## Problem
A town has `n` households (vertices `0..n-1`), each belonging to one true
**ward** (community labels `0..k-1`, given). Friendships form an undirected
simple graph (`m` edges). The census bureau maps wards automatically with a
**frozen** pipeline (below): it clusters the graph and never sees the true
labels.

You are the redistricting office. You may not move people; you may only
**re-pair friendships** with at most `B` **degree-preserving 2-swaps**. One
swap picks two existing edges `(a,b)` and `(c,d)` on four distinct vertices
such that `(a,d)` and `(b,c)` are currently non-edges, deletes the two old
edges, and inserts the two new ones. Every vertex keeps its exact degree, so
the swap is invisible to any degree-based audit. Your goal: make the frozen
pipeline output a partition that matches the true wards as **poorly** as
possible — the census should see phantom wards.

## The frozen pipeline (fixed, deterministic)
1. Build the symmetric normalized Laplacian `L = I - D^{-1/2} A D^{-1/2}`.
2. Take the eigenvectors `v_1..v_k` of the `k` smallest eigenvalues
   (`numpy.linalg.eigh`, ascending).
3. Row-normalize the `n x k` embedding (zero rows stay zero).
4. Cluster rows with **deterministic k-means**: first center = row of largest
   squared norm (lowest index on ties); each next center = row maximizing the
   distance to the nearest chosen center (lowest index on ties); then iterate
   assignment (nearest center, lowest index on ties) / mean-update until the
   assignment stops changing (cap 300 iterations; an empty cluster is re-seeded
   at the point farthest from its assigned center).
5. The checker scores the result against the true labels by the
   **best-permutation mismatch**: `F = min over permutations pi of
   (#{v : pi[pred[v]] != label[v]}) / n`.

Because every 2-swap preserves all degrees, `D` never changes: your swaps
only rotate the spectral geometry. Random re-pairings barely move the
informative eigenvectors (the pipeline is robust to unstructured noise);
rotating them requires a coherent perturbation.

## Input (stdin)
```
n m k B
label_0 label_1 ... label_{n-1}
u v            (m lines, 0 <= u < v < n, one undirected edge each)
```
Labels are integers in `[0,k)`. Every vertex has degree >= 3.

## Output (stdout)
```
S
a b c d        (S lines)
```
`S` = number of swaps (`0 <= S <= B`), followed by `S` lines. Line
`a b c d` means: delete edges `(a,b)` and `(c,d)`, insert edges `(a,d)` and
`(b,c)`. Swaps apply **in order**; each must be valid in the graph left by
the previous ones: four distinct endpoints, both deleted edges present, both
inserted edges absent. Any violation scores `Ratio: 0.0`. Output must be
integer tokens only.

## Objective
**Maximize** `F`, the best-permutation misclassification of the frozen
pipeline on the graph after your swaps.

## Scoring
The checker internally builds a baseline: the **canonical attack** — a
uniform-random blurring attack seeded by `Random(1000003 + 131*n + B)` that
repeatedly draws two uniformly random edges and swaps them whenever legal
(first-draw order, capped at `60*B+200` draws) until `B` swaps are applied.
Let `B_val` be its misclassification; with your misclassification `F`:
```
sc = min(1000.0, 100.0 * F / max(1e-9, B_val));   Ratio = sc / 1000
```
Reproducing the canonical attack scores `0.1`; a ten-times-better attack caps
at `1.0`.

## Constraints
- `100 <= n <= 480`, `k in {2,3,4}`, `m <= 6000`, `B <= 0.15*m`.
- Time limit 5s, memory 512MB. The score is fully deterministic.

## Example (small illustrative numbers, not from a real test)
Suppose a town of `n = 200` in two wards and the canonical random attack
confuses `B_val = 0.05` of households. A degree-targeted blurring attack
(re-pairing high-degree within-ward edges into cross-ward edges) might reach
`F = 0.09` → `Ratio = 100*0.09/0.05/1000 = 0.18`. An attack that instead
spends swaps *coherently* — each swap chosen to rotate the informative
eigenvectors toward a decoy partition of phantom wards (per first-order
perturbation theory of the eigenvectors) — might reach `F = 0.22` →
`Ratio = 0.44`, because aligned perturbations rotate eigenvectors linearly in
the number of swaps while random ones only diffuse them as a square root.
