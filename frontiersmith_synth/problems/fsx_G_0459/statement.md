# Cold-Start Label Budget: Pool-Based Active-Learning Query Order

## Story

Your data team has a large pool of **unlabeled** examples and a painfully small
labeling budget. Every label costs a human annotator's time, so before spending a
cent you must decide **the order** in which pool examples will be sent for
labeling. You have **not seen any labels yet** — only the raw feature vectors of
the pool (the cold-start regime).

A fixed, deterministic classifier — a **nearest-centroid (Rocchio)** model — is
retrained after each new label arrives and evaluated on a held-out test set.
Plotting test accuracy against the number of labels acquired traces a **learning
curve**. You are graded on the **area under that learning curve (AULC)**: a good
query order reaches high accuracy with **few** labels.

The pool is a mix of well-separated latent clusters (the classes) with **imbalanced
priors** — a couple of common classes and several rare ones. The nearest-centroid
model can only predict a class once at least one of its members has been labeled,
so early accuracy is governed by **how fast your order covers the distinct
clusters, especially the rare ones**. Random ordering burns budget re-sampling the
common clusters; a coverage/diversity-aware order finds the rare clusters far
sooner. There is no closed-form optimum, and the immediate-jump ceiling is
unreachable — many strategies trade coverage against representativeness differently.

## Your task

Write a standalone program that reads ONE public instance as JSON on **stdin** and
writes ONE answer as JSON on **stdout**: a permutation of the pool indices giving
the order in which examples would be labeled.

### Public instance (stdin)

```json
{
  "name": "pool1011",
  "d": 8,                       // feature dimension
  "n": 90,                      // pool size
  "budget": 18,                 // labels that affect the score (the curve's length)
  "pool": [[x00, x01, ...], ...]  // n unlabeled feature rows, each of length d
}
```

No labels, no class count, and no test set are ever revealed to your program.

### Answer (stdout)

```json
{"order": [p_0, p_1, ..., p_{n-1}]}
```

`order` must be a **permutation of `0..n-1`** (exactly `n` distinct integers in
`[0, n)`). The evaluator acquires labels along this order; only the first `budget`
entries influence the score, but the answer must be a full valid permutation.

An answer that is not a permutation (wrong length, out of range, repeats,
non-integer), a crash, a timeout, or non-JSON output scores **0.0** on that
instance.

## Scoring (deterministic, maximize)

The fixed model is nearest-centroid: training builds one mean feature vector per
labeled class; prediction picks the nearest class centroid among the classes seen
so far. For a query order `o` and budget `B`,

```
AULC(o) = (1/B) * sum over k = 1..B of  test_accuracy( model trained on o[:k] )
```

For each instance the evaluator computes, in the parent process (never revealed):

- `base` — mean AULC over several fixed seeded **random** permutations (weak anchor),
- `ceil` — test accuracy of the model trained on the **whole pool** (the
  immediate-jump upper bound; unreachable, since the first labels cannot cover
  every class),
- `cand` — the AULC of your order,

and normalizes with an affine anchor (random → 0.1, immediate-jump → 1.0):

```
r = clamp( 0.1 + 0.9 * (cand - base) / max(1e-9, ceil - base),  0, 1 )
```

The reported **Ratio** is the mean of `r` over all instances (12 instances,
including harder held-out ones with larger pools, more classes, and tighter
coverage). A random-quality order scores ~0.1; matching the unreachable ceiling
scores 1.0; doing worse than random scores below 0.1.

## Isolation

Your program is run in a fresh OS-sandboxed subprocess and only ever sees the
public instance (pool features). Labels, the test set, the references, and the
model live only in the evaluator, so introspection or filesystem scraping buys you
nothing — you must genuinely choose an informative query order from the feature
geometry.
