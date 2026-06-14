**Problem (from step 1).** The matching vote keeps one weight per support point and never forms a
concise per-class entity, so on the fine-grained benchmark (CUB 0.625) — where five embeddings per
class sit close together — the cosine-softmax over scattered points has no crisp thing to concentrate
on, and its FCE parameters cannot be constrained by twenty-five images. The fix is a minimal per-class
summary plus a metric consistent with it.

**Key idea.** Summarize each class by the **mean** of its embedded support examples (its *prototype*),
c_k = (1/|S_k|) Σ f_φ(x_i) — one vector per class, shot-independent, parameter-free — and classify a
query by a softmax over its **negative (squared) Euclidean distance** to the prototypes, trained with
cross-entropy. The distance is not a free choice: squared Euclidean is a **Bregman divergence**, and
for any Bregman divergence the mean is provably the representative minimizing total within-class
divergence; cosine is *not* Bregman, so averaging and then scoring by cosine is inconsistent. Expanding
−‖f_φ(x) − c_k‖², the query-only term cancels in the softmax, leaving a **linear** classifier induced
by the prototypes — all fitted capacity sits in the deep embedding, the per-task head has nothing to
overfit.

**Why these choices.** Mean prototype → Bregman-optimal, parameter-free, removes the per-point scatter
that hurt CUB. Squared Euclidean → the mean's consistent partner; corresponds to a spherical-Gaussian
class-conditional, the strong-simple prior this regime wants. One prototype per class → keeps the loop
end-to-end (multiple prototypes need a decoupled k-means step). No FCE → the matching CUB number is the
evidence that per-episode parameters do not pay here. At K = 1 the prototype is the single support
embedding, so the method coincides with the matching vote; the gain opens at the 5-shot used by every
benchmark.

**Hyperparameters.** No `LR_OVERRIDE` — runs at the scaffold default SGD@1e-2, momentum 0.9, wd 5e-4
(the mean classifier is stable under it, unlike the matching LSTMs). `use_softmax=False` so cross-
entropy log-softmaxes the negative-distance logits. The scaffold's `l2_distance_to_prototypes` returns
the negative *non-squared* L2 distance — a monotone re-parameterization of the same nearest-prototype
rule; the clean Bregman story is stated for the squared form. `is_transductive = False`.

**What to watch.** CUB should jump the most (a clean per-class entity helps where classes collide); on
the generic benchmarks the direction is confounded by the optimizer switch (matching ran at 1e-3,
prototypes at 1e-2) and by the rigid spherical prior — CIFAR may go flat or down. If the generic
benchmarks stay where the per-point vote left them, the bottleneck is the *fixed metric*, and the next
rung learns the comparison.

```python
class CustomFewShotMethod(FewShotClassifier):
    """Prototypical Networks (Snell et al., 2017).

    Compute class prototypes as the mean feature vector of support examples,
    then classify queries by negative Euclidean distance to prototypes.
    """

    def __init__(self):
        backbone = make_backbone(use_pooling=True)
        super().__init__(backbone=backbone, use_softmax=False)

    def process_support_set(self, support_images: Tensor, support_labels: Tensor):
        self.compute_prototypes_and_store_support_set(support_images, support_labels)

    def forward(self, query_images: Tensor) -> Tensor:
        query_features = self.compute_features(query_images)
        scores = self.l2_distance_to_prototypes(query_features)
        return self.softmax_if_specified(scores)

    @staticmethod
    def is_transductive() -> bool:
        return False

    def compute_loss(self, scores: Tensor, labels: Tensor) -> Tensor:
        return F.cross_entropy(scores, labels)
```
