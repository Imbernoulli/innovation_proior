## Research question

Few-shot image classification under episodic evaluation: given a support set of N classes never seen during training, with K labeled images each (here N = 5, K = 5), label fresh query images from those same classes — immediately, with no per-task retraining. The object of design is the **episode-level classifier**: how to summarize the support set into something a query can be compared against, how to score a query against that summary, and the loss that trains it. The backbone, episodic data pipeline, optimizer, schedule, and evaluation protocol are fixed. The contribution must be a reusable algorithmic component — a way to summarize a support set, compare a query to it, or adapt the episode — not a dataset-specific trick.

## Prior art / Background / Baselines

The relevant line of work casts few-shot classification as learned comparison in an embedding space.

- **Borrowed-feature nearest neighbour.** Train a standard classifier on the base classes, drop the softmax, and classify novel-class queries by nearest neighbour in the penultimate features.

- **Neighbourhood Components Analysis (Goldberger et al., 2004).** Make kNN differentiable by replacing the hard nearest neighbour with a soft one — point i selects point j with probability ∝ exp(−‖Ax_i − Ax_j‖²) — and learn the linear map A by maximizing expected leave-one-out accuracy.

- **Nearest Class Mean (Mensink et al., 2013).** Represent each class by the mean of its examples and assign a query to the nearest mean under a learned Mahalanobis metric; new classes are added by averaging their support features.

## Fixed substrate / Code framework

The whole pipeline is frozen except for the `CustomFewShotMethod` class. Training is episodic: 500 tasks per epoch for 200 epochs, each task 5-way 5-shot with 15 query images per class. For each task the loop calls `model.process_support_set(support_images, support_labels)`, then `model(query_images)` to get scores of shape `(n_query, n_way)`, then `model.compute_loss(scores, query_labels)`, backprops, and steps. The optimizer is SGD with lr `1e-2`, momentum `0.9`, weight decay `5e-4`, a `MultiStepLR` schedule (milestones `[120, 160]`, γ `0.1`), and global gradient-norm clipping at `5.0`. A parameter budget caps the total number of learnable parameters.

The backbone is a **ResNet-12** that maps an 84×84 image to a 640-dim feature vector (`make_backbone(use_pooling=True)`) — or, if a method asks for it, to a 640-channel feature map (`make_backbone(use_pooling=False)`). The loop also exposes utilities a method may use: `self.compute_features(images)`, `self.l2_distance_to_prototypes(features)` (**negative non-squared** Euclidean distance to `self.prototypes`), `self.cosine_distance_to_prototypes(features)` (cosine similarity to the prototypes), `compute_prototypes(features, labels)` (per-class mean feature), and the convenience `self.compute_prototypes_and_store_support_set(images, labels)`. One method-level knob, `LR_OVERRIDE`, replaces the scalar learning rate for the whole model.

## Editable interface

Exactly one region is editable — the `CustomFewShotMethod` class (a subclass of `FewShotClassifier`) in `custom_fewshot.py`. Every method fills the same contract: `__init__` (build the backbone and any learnable modules), `process_support_set(support_images, support_labels)` (turn the labeled support set into whatever queries are compared against, and stash it), `forward(query_images) -> Tensor` of shape `(n_query, n_way)` (score each query), and `compute_loss(scores, labels) -> Tensor` (the training loss on the query labels; default cross-entropy). The default fill is a mean-prototype, distance-to-prototype classifier.

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

Three benchmarks span the difficulty range — **miniImageNet** (100 ImageNet classes), **CIFAR-FS** (100 classes from CIFAR-100), and **CUB-200** (200 fine-grained bird species) — all evaluated 5-way 5-shot, each over three seeds {42, 123, 456}. The metric on every benchmark is **mean classification accuracy over 600 test episodes** (higher is better); the task score is the geometric mean of the three benchmark accuracies. Models are selected by validation accuracy (200 validation tasks per epoch) and the best checkpoint is evaluated on the test split.
