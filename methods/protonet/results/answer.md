# Prototypical Networks, distilled

Prototypical Networks solve few-shot classification with a deliberately simple inductive bias:
learn one nonlinear embedding network end-to-end, summarize each class by the **mean** of its
embedded support examples (its *prototype*), and classify a query by a softmax over its
**negative squared Euclidean distance** to the prototypes. There are no per-task parameters —
the per-class head is induced entirely by the means — and training is episodic, so the training
condition matches the few-shot test condition.

## Problem it solves

Given a support set of N classes with K labeled examples each (classes unseen during training),
label fresh query images of those classes, without retraining and without overfitting the tiny
support set. The component must be reusable: a way to summarize a support set and compare a query
against it.

## Key idea

- **Prototype = class mean in embedding space.** With embedding f_φ : ℝ^D → ℝ^M,
  c_k = (1/|S_k|) Σ_{(x_i,y_i)∈S_k} f_φ(x_i). One vector per class, independent of the shot.
- **Classify by softmax over negative distance to prototypes.**
  p_φ(y = k | x) = exp(−d(f_φ(x), c_k)) / Σ_{k'} exp(−d(f_φ(x), c_{k'})),
  trained by the negative log-likelihood J(φ) = −log p_φ(y = k | x) of the true class.
- **The distance is squared Euclidean, d(z, z′) = ‖z − z′‖², and this is not a free choice.**
  It is a Bregman divergence, and for any Bregman divergence the mean is provably the
  representative minimizing total within-class divergence (argmin_c Σ_{x∈S_k} d_φ(x, c) =
  mean of S_k). Cosine distance is *not* a Bregman divergence, so averaging embeddings and then
  scoring by cosine is inconsistent; this matches the empirical finding that squared Euclidean
  improves over cosine most strongly for this mean-prototype method.

## Why the design choices

- **Mean prototype, one per class.** Bregman-optimal summary of a set; parameter-free; concise
  and shot-independent. Multiple prototypes per class would require a separate partitioning
  (k-means) step decoupled from gradient training, breaking end-to-end learning — so one
  prototype, and let the deep embedding make a class unimodal around it.
- **Squared Euclidean.** Bregman divergence ⇒ mean is the right representative; corresponds to a
  spherical-Gaussian class-conditional with common fixed scale under the
  exponential-family/Bregman bijection — a strong, simple prior for the data-scarce regime.
- **Linear readout, deep embedding.** Expanding −‖f_φ(x) − c_k‖² = −f_φ(x)ᵀf_φ(x) +
  2c_kᵀf_φ(x) − c_kᵀc_k; the first term is class-independent and cancels in the softmax, leaving
  w_kᵀf_φ(x) + b_k with w_k = 2c_k, b_k = −c_kᵀc_k. So the classifier is *linear* on the
  embedding, with weights induced by the prototypes — all nonlinearity (and all the fitted
  capacity) lives in f_φ, where thousands of episodes can constrain it; the per-task head has
  nothing to overfit.
- **Episodic training.** Sample N_C-way, N_S-shot, N_Q-query episodes that mimic the test task;
  optimize query-set NLL conditioned on the support set. Train with **higher way** than test (a
  harder task forces a finer-grained embedding that separates more classes at once), and **match
  the shot** at train and test (so prototype noise matches test conditions).
- **No FCE / per-episode re-embedding.** Adds parameters and an arbitrary set ordering; the
  simple inductive bias works better in the limited-data regime.
- **Spherical, no per-class variance.** A per-dimension variance is redundant with the
  embedding's own freedom to rescale axes, so it adds parameters without gains.

## Relation to prior methods

- **Matching Networks**: in the one-shot case |S_k| = 1, so c_k equals the single support
  embedding and the two methods coincide (both are nearest-neighbor in embedding space). They
  diverge at K > 1, where prototypical networks collapse each class to its mean (one weight per
  class) instead of voting over individual support points (one weight per point).
- **Neighbourhood Components Analysis**: same softmax-over-negative-squared-distance form, but
  NCA's softmax is over individual *points* (must retain the whole set), whereas here it is over
  *classes* via one prototype each.
- **Nearest Class Mean (Mensink et al. 2013)**: also represents a class by its mean, but with a
  *linear* embedding for the many-examples regime; prototypical networks learn a *nonlinear*
  embedding end-to-end with episodic training for the few-shot regime.

## Final algorithm (per training episode)

```
Input: episode classes V (|V| = N_C); for each k in V, support S_k (|S_k| = N_S), query Q_k (N_Q)
for k in V:
    c_k = (1 / N_S) * sum_{(x_i, y_i) in S_k} f_phi(x_i)          # prototype = mean of embedded support
J = 0
for k in V:
    for (x, y) in Q_k:                                            # y == k
        J += (1 / (N_C * N_Q)) * [ ||f_phi(x) - c_k||^2 + log sum_{k'} exp(-||f_phi(x) - c_{k'}||^2) ]
return J                                                          # == mean over queries of -log p_phi(y=k|x)
```

Zero-shot variant: replace the support-mean prototype by an embedding of class meta-data,
c_k = g_ϑ(v_k), constrained to unit length (the query embedding f is left unconstrained);
everything else is unchanged.

## Working code

Filling the three method slots of the episodic harness, with prototypes computed as explicit
per-class means and queries scored by the canonical negative squared Euclidean distance:

```python
import torch
import torch.nn.functional as F
from torch import Tensor


class PrototypicalNetworks(FewShotClassifier):
    """Each class is summarized by the mean of its embedded support examples (its prototype);
    a query is classified by a softmax over its negative squared Euclidean distance to the
    prototypes. Trained episodically with cross-entropy on the negative squared-distance logits."""

    def __init__(self):
        backbone = make_backbone(use_pooling=True)        # nonlinear embedding f_phi
        super().__init__(backbone=backbone, use_softmax=False)

    def process_support_set(self, support_images: Tensor, support_labels: Tensor):
        support_features = self.compute_features(support_images)
        self._raise_error_if_features_are_multi_dimensional(support_features)
        # Episode labels are local 0..n_way-1, as in the canonical episodic loss.
        n_way = int(torch.unique(support_labels).numel())
        self.prototypes = torch.stack([
            support_features[support_labels == label].mean(dim=0)
            for label in range(n_way)
        ])

    def forward(self, query_images: Tensor) -> Tensor:
        query_features = self.compute_features(query_images)        # f_phi(query)
        self._raise_error_if_features_are_multi_dimensional(query_features)
        sq_dist = ((query_features[:, None, :] - self.prototypes[None, :, :]) ** 2).sum(dim=2)
        scores = -sq_dist                                           # logits = -squared distance
        return self.softmax_if_specified(scores)

    @staticmethod
    def is_transductive() -> bool:
        return False                                                # queries classified independently

    def compute_loss(self, scores: Tensor, labels: Tensor) -> Tensor:
        return F.cross_entropy(scores, labels)                      # -log p_phi(y=k|x)
```

Code-faithfulness check: the canonical implementation computes
`torch.pow(x - y, 2).sum(2)` and then applies `F.log_softmax(-dists, dim=1)`. A scaffold helper
that returns `-torch.cdist(samples, prototypes)` is not the exact canonical object: it preserves the
nearest-prototype argmax but changes the softmax geometry and breaks the Bregman /
spherical-Gaussian / linear-model equivalences.

The prototype computation itself, for reference:

```python
def compute_prototypes(support_features: Tensor, support_labels: Tensor) -> Tensor:
    """Prototype k = mean feature vector over support examples with label k."""
    n_way = int(torch.unique(support_labels).numel())
    return torch.stack([
        support_features[support_labels == label].mean(dim=0)
        for label in range(n_way)
    ])
```
