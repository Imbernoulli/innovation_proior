## Research question

Few-shot image classification under episodic evaluation: I am handed a support set of N classes I
have never seen during training, with K labeled images each (here N = 5, K = 5), and I must label
fresh query images of those same classes — immediately, with no per-task retraining. The single thing
being designed is the **episode-level classifier**: how to summarize the support set into something a
query can be compared against, how to score a query against that summary, and the loss that trains it.
Everything else — the backbone, the episodic data pipeline, the optimizer, the schedule, the
evaluation protocol — is fixed. The contribution must be a reusable algorithmic component (a way to
summarize a support set, compare a query to it, or adapt the episode), not a dataset-specific trick.

## Prior art before the first rung (metric-learning lineage)

The first rung reacts to the line of work that turned few-shot classification into *learned comparison
in an embedding space*. These are the methods the ladder departs from; each is concise, with its gap.

- **Borrowed-feature nearest neighbour.** Train an ordinary classifier on the base classes, drop the
  softmax, and do nearest-neighbour on the penultimate features for novel classes. Instant
  assimilation (non-parametric), but the features were trained to separate the *base* classes under a
  softmax, never to make instance-to-instance comparison meaningful for an unseen support set. Gap:
  the geometry is a byproduct, not the objective.
- **Neighbourhood Components Analysis (Goldberger, Hinton, Roweis, Salakhutdinov 2004).** Make kNN
  differentiable by replacing the hard nearest neighbour with a *soft* one — point i picks j with
  probability ∝ exp(−‖Ax_i − Ax_j‖²) — and learn the metric A by maximizing expected leave-one-out
  accuracy. Gap: A is a single *linear* map (Mahalanobis), and the objective is point-against-all-of-
  training, not shaped like an N-way K-shot episode.
- **Nearest Class Mean (Mensink, Verbeek, Perronnin, Csurka 2013).** Represent each class by the mean
  μ_c of its examples and assign a query to the nearest mean under a learned Mahalanobis metric;
  brand-new classes cost nothing (just average their examples). Gap: a *linear* embedding for the
  many-examples regime, and its non-linear variant needs a *separate* k-means step decoupled from the
  metric learning — not end-to-end.
- **Episodic training (the durable idea the ladder keeps).** Make the training condition match the
  test condition: sample mini-tasks during training that look exactly like test tasks (N classes, K
  shots, a query set) and optimize the classifier on each episode's query set. In a data-starved
  regime this faithfulness is itself a strong regularizer — the network is never allowed to rely on
  anything it won't have at test time. Every rung trains this way; only the *classifier inside the
  episode* changes.

## The fixed substrate

The whole pipeline is frozen and must not be touched. A **ResNet-12 backbone** maps an 84×84 image to
a 640-dim feature vector (`make_backbone(use_pooling=True)`) — or, if a method asks for it, to a
640-channel feature **map** (`make_backbone(use_pooling=False)`). Training is **episodic**: 500 tasks
per epoch for 200 epochs, every task 5-way 5-shot with 15 query images per class. For each task the
loop calls `model.process_support_set(support_images, support_labels)`, then `model(query_images)` to
get scores of shape `(n_query, n_way)`, then `model.compute_loss(scores, query_labels)`, backprops,
and steps. The optimizer is **SGD**, lr `1e-2`, momentum `0.9`, weight decay `5e-4`, with a
`MultiStepLR` schedule (milestones `[120, 160]`, γ `0.1`) and global gradient-norm clipping at `5.0`.
The loop honours one method-level knob, `LR_OVERRIDE`, which replaces the scalar learning rate for the
whole model (the same rate for every parameter — there is no per-module learning rate). A parameter
budget of ≈1.05× the largest baseline (the ResNet-12 backbone plus a relation module) is enforced.

The loop also exposes utilities a method may use: `self.compute_features(images)` (pass through the
backbone), `self.l2_distance_to_prototypes(features)` (**negative non-squared** Euclidean distance to
`self.prototypes`), `self.cosine_distance_to_prototypes(features)` (cosine similarity to the
prototypes), `compute_prototypes(features, labels)` (per-class mean feature), and the convenience
`self.compute_prototypes_and_store_support_set(images, labels)`.

## The editable interface

Exactly one region is editable — the `CustomFewShotMethod` class (a subclass of `FewShotClassifier`)
in `custom_fewshot.py`. Every method on the ladder is a fill of this same contract: `__init__`
(build the backbone and any learnable modules), `process_support_set(support_images, support_labels)`
(turn the labeled support set into whatever queries are compared against, and stash it),
`forward(query_images) -> Tensor` of shape `(n_query, n_way)` (score each query), and
`compute_loss(scores, labels) -> Tensor` (the training loss on the query labels; default
cross-entropy). The starting point is the scaffold default: a mean-prototype, distance-to-prototype
classifier. Each later method replaces exactly this class and nothing else.

```python
# EDITABLE region of custom_fewshot.py — default fill
class CustomFewShotMethod(FewShotClassifier):
    """Default: mean-prototype classifier scored by distance to prototypes."""

    def __init__(self):
        backbone = make_backbone(use_pooling=True)        # ResNet-12 -> 640-d feature vector
        super().__init__(backbone=backbone, use_softmax=False)

    def process_support_set(self, support_images: Tensor, support_labels: Tensor):
        # build whatever queries are compared against (default: per-class mean prototypes) and store it
        self.compute_prototypes_and_store_support_set(support_images, support_labels)

    def forward(self, query_images: Tensor) -> Tensor:
        query_features = self.compute_features(query_images)        # f(query): (n_query, 640)
        scores = self.l2_distance_to_prototypes(query_features)     # logits = -distance to prototypes
        return self.softmax_if_specified(scores)

    @staticmethod
    def is_transductive() -> bool:
        return False                                                # queries classified independently

    def compute_loss(self, scores: Tensor, labels: Tensor) -> Tensor:
        return F.cross_entropy(scores, labels)
```

## Evaluation settings

Three benchmarks spanning the difficulty range — **miniImageNet** (100 ImageNet classes), **CIFAR-FS**
(100 classes from CIFAR-100), and **CUB-200** (200 fine-grained bird species) — all evaluated 5-way
5-shot, each over three seeds {42, 123, 456}. The metric on every benchmark is **mean classification
accuracy over 600 test episodes** (higher is better); the task score is the geometric mean of the
three benchmark accuracies. Models are selected by validation accuracy (200 validation tasks per
epoch) and the best checkpoint is evaluated on the test split.
