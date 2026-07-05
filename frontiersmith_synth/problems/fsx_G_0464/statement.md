# Protein Family Retrieval: Design the Similarity Kernel

## Story

A structural-proteomics group has profiled a batch of `N` proteins. Each protein is
described by a fixed-length real **descriptor vector** of dimension `D` — a concatenation
of physicochemical / compositional channels (amino-acid composition, hydrophobicity and
charge moments, secondary-structure propensities) followed by a block of instrument /
assay **nuisance channels**. Every protein secretly belongs to one **protein family**
(its fold class), but **no labels are provided**.

You must build a content-based **retrieval** system: design a **similarity / kernel
function** over descriptors and use it for fixed nearest-neighbour retrieval — for each
query protein, rank all the other proteins so that same-family proteins float to the top.
Quality is measured by **mean Average Precision (mAP)** over all queries.

## Why raw distance is weak (this is the whole game)

The descriptors are deliberately messy, exactly like real proteomics features:

* a per-protein multiplicative **abundance / length factor** scales *every* channel of a
  protein, so raw Euclidean distance retrieves on size, not fold;
* each channel has its own large, protein-independent **baseline offset**;
* most channels are high-variance **nuisance channels** whose scale dwarfs the
  low-variance **informative channels** that actually separate families.

A raw dot-product / Euclidean nearest-neighbour therefore retrieves on size and noise. A
good **unsupervised** kernel must undo these effects — normalize away per-protein
abundance, center out per-channel offsets, and equalize per-channel scale so the quiet
informative channels are heard. There is no training signal: the kernel is *designed*,
not fit to labels.

## The candidate program (stdin → stdout, isolated)

Your program reads ONE JSON object (the **public instance**) from stdin and writes ONE
JSON object to stdout. It runs in an isolated sandbox and only ever sees the public view
(descriptors, **no labels**).

### Public instance (stdin)
```json
{
  "name": "batch101",
  "n": 60,
  "dim": 26,
  "features": [[x_0_0, ..., x_0_25], ..., [x_59_0, ..., x_59_25]]
}
```
`features` is an `N x D` list of finite real descriptors.

### Answer (stdout)
```json
{"ranking": [r_0, r_1, ..., r_{N-1}]}
```
`r_i` is a **ranked list of the other protein indices** — a permutation of
`{0,1,...,N-1} \ {i}` of length `N-1`, **most-similar first** under your kernel. Compute
whatever similarity you like from `features`, then argsort.

### Validity
`ranking` must be a list of exactly `N` lists; each `r_i` must contain every index in
`{0..N-1}\{i}` exactly once (integers, no duplicate, no self, length `N-1`). Any
violation, non-JSON, non-finite value, crash, or timeout scores **0.0** for that instance.

## Objective & scoring (deterministic, maximize)

Family labels are hidden in the evaluator. For each query, relevant items are the
same-family proteins (excluding self); **Average Precision** is averaged over queries to
give **mAP ∈ [0,1]**. Per instance, with `q_cand` the candidate mAP and `q_base` the mAP
of the internal **raw-Euclidean** reference:

```
r = clamp( 0.1 + 0.9 * (q_cand - q_base) / (1.0 - q_base), 0, 1 )
```

Reproducing raw-Euclidean scores ≈ **0.1**; perfect retrieval would score **1.0**.
Because families genuinely overlap and the informative channels are noisy, perfect mAP is
unreachable, so even an excellent kernel stays well below 1.0 — **headroom remains**. The
reported **Ratio** is the mean of `r` over 12 instances (including harder held-out batches
with more families and more nuisance channels); **Vector** lists the per-instance `r`.

## Strategy ladder (increasing quality)

1. **Raw Euclidean kNN** — retrieval driven by abundance and nuisance channels (≈ 0.1).
2. **Composition + per-channel scaling** — L1-normalize each descriptor, divide each
   channel by its batch std, cosine kNN; helps a little but leaves the per-channel offsets
   in place.
3. **Standardized kernel** — composition-normalize + per-channel **z-score** (center *and*
   scale) + cosine kNN; the informative channels dominate and same-fold proteins cluster.
4. **Channel-reweighted / whitened kernels** — upweight the most structured (multimodal)
   channels or decorrelate them to push mAP further toward the ceiling.

Determinism is required: seed any randomness you use.
