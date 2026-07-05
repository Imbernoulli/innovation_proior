# Face Verification: learn an embedding transform (Format B, isolated)

You are given a small labelled **face gallery** of raw feature vectors and must design a
linear **embedding transform** `W` that makes same-identity faces close and different-identity
faces far apart under cosine similarity. Your transform is scored by **verification ROC-AUC**
on a held-out set of face pairs whose same/different-identity labels you never see.

The identity signal lives in a low-dimensional subspace, but a random rotation has smeared it
across every coordinate and buried it beneath high-variance nuisance factors (pose, lighting).
Per-coordinate rescaling is therefore useless: only a transform that exploits the cross-coordinate
covariance structure — and the training identity labels — recovers a good verification metric.
An inherent per-shot identity jitter caps the achievable AUC well below 1, so there is real
headroom and no trivial optimum.

## Program contract
Read ONE JSON object (the public instance) from **stdin**, write ONE JSON object to **stdout**.

### Public instance (stdin)
```json
{
  "d": 40,                    // raw feature dimension
  "max_out_dim": 40,          // max number of embedding rows you may return
  "X_train": [[...], ...],    // training features, each length d
  "y_train": [int, ...],      // integer identity label per training row
  "X_test":  [[...], ...]     // test features, each length d (identities are DISJOINT from train)
}
```
The held-out verification pairs and their same/different labels are **hidden**; they are never in
the public instance.

### Answer (stdout)
```json
{ "W": [[...], ...] }         // a k x d matrix, 1 <= k <= max_out_dim, all entries finite
```
The evaluator forms embeddings `Z = X_test @ W^T`, computes cosine similarity for each hidden
pair, and measures ROC-AUC of positive (same identity) vs negative (different identity) pairs.

## Objective
**Maximize** verification ROC-AUC. Per-instance score is an affine map of AUC calibrated so the
identity transform (`W = I`) scores about `0.1`; the score is clipped to `[0, 1]`. The reported
`Ratio` is the mean over 10 instances (the last two are harder, higher-noise held-out instances).
Any malformed / non-finite / wrong-shape output scores `0`.

## Strategies (open-ended)
Whitening (Mahalanobis), Fisher LDA, nearest-class-mean metrics, NCA-style gradient learning,
regularized covariance shrinkage, or dimensionality selection — many approaches beat the trivial
identity transform, and none reaches the ceiling.
