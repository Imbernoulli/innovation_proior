# Gene-Expression Cell-Type Atlas: Unsupervised Linear Embedding for Downstream kNN

## Background
A single-cell RNA-seq **atlas** measures the expression of `D` genes across many
cells. Each cell belongs to one of `C` **cell types**. A standard analysis step is
to reduce the high-dimensional expression vectors to a small embedding that a
downstream nearest-neighbour classifier can use to label new cells. Real atlases
are messy: most genes carry little cell-type information, technical noise inflates
the variance of many genes, expression scales differ by orders of magnitude
(housekeeping vs marker genes), and true cell-type signal lives in *correlated
co-expression modules*.

You are given ONLY a large **unlabelled** training matrix of expression vectors.
From these data statistics alone, design an **affine (linear) projection** to a
target dimension `d`.

## Task
Write a standalone program: read ONE JSON object (the public instance) from stdin,
write ONE JSON object (your projection) to stdout.

Your projection is applied, **unchanged**, by a hidden evaluator to a labelled
reference set and a held-out test set (neither visible to you). Each test cell is
classified by majority vote over its `k = 7` nearest reference cells in the
`d`-dimensional embedding. Your quality is the **classification accuracy on the
hidden test labels**. Maximize it.

## Public instance (stdin)
```json
{
  "X_train":  [[float, ...], ...],   // n_train x D expression matrix (UNLABELLED, a fresh copy)
  "n_train":  int,                   // number of training cells
  "n_genes":  int,                   // number of genes D
  "d_target": int,                   // target embedding dimension d
  "seed":     int                    // per-instance seed you MAY use for your own RNG
}
```
You never see the reference set, the test set, or any cell-type labels.

## Answer (stdout)
```json
{
  "W": [[float, ...], ...],   // d x D projection matrix
  "b": [float, ...]           // length-D offset (bias / centring)
}
```
The embedding of an expression vector `x` is

```
z = W @ (x - b)          # shape (d,)
```

`W` must have shape exactly `d x D`, `b` length exactly `D`, and every entry must
be finite. Any violation (wrong shape, `NaN`/`Inf`, non-numeric, a crash, or a
timeout) scores **0.0** on that instance.

## Scoring
The atlas family spans several regimes — **clean** (signal is the dominant
variance), **noisy** (loud independent technical-noise genes), **scaled**
(genes on wildly different expression scales), **correlated** (strong
co-expression modules), and **heldout** (more cell types / more genes). For each
instance the evaluator computes your downstream kNN accuracy `acc_cand` and
normalizes it between a weak random-projection baseline `acc_base` and a strong
**supervised Fisher-LDA oracle** `acc_orac` (fit on the hidden labels):

```
r = clamp( 0.1 + 0.9 * (acc_cand - acc_base) / max(acc_orac - acc_base, 0.15), 0, 1 )
```

A projection that merely matches the random baseline maps to `~0.1`; the
supervised oracle is essentially unreachable without labels, so headroom always
remains. Valid-but-weak instances are floored to a small positive value; an
invalid answer scores exactly `0.0`.

The reported **Ratio** is the **geometric mean** of the per-instance `r` values,
so a projection that wins one regime but collapses on another is heavily
penalized. A method that generalizes across the whole atlas family wins.

## Notes
- Scoring is fully deterministic (all data and the kNN classifier are seeded).
- Your program is run in an isolated sandbox: it sees only the public instance.
- A plain random or raw-variance projection leaves substantial headroom; standardizing
  away noise/scale and locking onto correlated co-expression structure does much better.
