# Rare-Disease Screen: Resampling/Reweighting Policy for a Fixed Classifier

## Story

A hospital runs a multi-class screening assay for a rare disease. Each patient is
described by a vector of standardized biomarker readings. The label is one of three
classes:

- `0` = healthy / no finding (the overwhelming majority),
- `1` = rare disease subtype A,
- `2` = rare disease subtype B.

Because the two disease subtypes are rare, a classifier trained on the raw cohort
collapses onto "predict *healthy* for everyone": it looks accurate but misses the
patients who actually need care. The metric that matters is the **macro-averaged
F1** on a **balanced held-out validation cohort** — each class weighted equally, so
the rare subtypes count as much as the healthy majority.

## What you control

The classifier is **fixed** and never changes: a weighted multinomial
logistic-regression trained by a fixed, deterministic full-batch gradient descent
(baked into the evaluator). The **only lever you control is a
resampling/reweighting policy**: assign a non-negative training **weight** to every
training patient. Oversampling a patient equals doubling its weight; undersampling
equals lowering it toward zero — so per-example weights express the full family of
resampling/reweighting policies. Weights are rescaled to mean 1 before training, so
the total training budget is fixed and you cannot win by inflating every weight.

The evaluator trains the fixed classifier on your weights and measures macro-F1 on
a **hidden** validation cohort. **Objective: maximize.**

## Program contract (stdin → stdout)

You write a standalone program. Read ONE JSON "public instance" from stdin and write
ONE JSON answer to stdout.

**Input (public instance):**
```json
{
  "name": "screen201",
  "n_classes": 3,
  "feature_dim": 8,
  "n": 384,
  "y": [0, 0, 2, 0, 1, ...],
  "class_counts": [360, 16, 8],
  "X": [[...8 floats...], ...  "n" rows ...]
}
```
- `y[i]` is the class of training patient `i` (in `{0,1,2}`); `X[i]` are its
  standardized biomarker readings; `class_counts[c]` is the number of training
  patients in class `c`.

**Output (your policy):**
```json
{"weights": [w_0, w_1, ..., w_{n-1}]}
```
- `weights[i]` is the non-negative training weight for patient `i`, aligned to the
  order of `X`/`y`.

**Validity.** `weights` must be a list of exactly `n` finite non-negative reals with
a strictly positive sum. Wrong length, a negative / `NaN` / `Inf` entry, an all-zero
vector, a crash, a timeout, or non-JSON → that instance scores **0.0**.

## Scoring (deterministic; no wall-time)

For each instance the evaluator computes, with the **same fixed classifier**:

- `f_base` = macro-F1 trained with **uniform** weights (ignores the imbalance — the
  weak baseline),
- `f_ref`  = macro-F1 trained on a large, perfectly **balanced** clean cohort drawn
  from the true generative model (an idealized ceiling the small imbalanced set can
  never quite reach),
- `f_cand` = macro-F1 trained with **your** weights,

and normalizes to `[0, 1]`:

```
r = clip( 0.1 + 0.9 * (f_cand - f_base) / max(1e-9, f_ref - f_base),  0, 1 )
```

- Reproducing uniform weights → `r ≈ 0.1`.
- Matching the (generally unreachable) balanced-ideal ceiling → `r = 1.0`.
- Hurting the rare classes relative to uniform → `r < 0.1`.

The final score is the mean of `r` over all instances (a mix of easier cohorts and
harder held-out cohorts with fewer rare patients and noisier assays).

Because `f_ref` is estimated from far more balanced rare-class data than any
reweighting of a handful of rare patients can recover, even the textbook
inverse-frequency policy stays strictly below `1.0` — there is real **headroom**
for smarter, difficulty-aware or synthetic-minority policies.

## Isolation

Your program is **untrusted** and runs in a fresh OS-sandboxed subprocess. It only
ever sees the public instance above; the hidden validation cohort and all reference
scores live in the evaluator process and are never exposed.
