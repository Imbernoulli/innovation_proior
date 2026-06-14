**Problem (from step 2).** The prototype method is a *linear* classifier in feature space (the squared-
distance softmax reduces to a linear head induced by the means), so it succeeds only where the embedding
alone makes classes round and linearly separable. That won fine-grained CUB (0.756) but lost the generic
benchmarks (CIFAR fell to 0.682, mini to 0.649). The violent benchmark swing is the symptom that no
single fixed, closed-form metric is right for all layouts — the *comparison* is the bottleneck.

**Key idea.** Stop hand-choosing the metric; **learn the comparison function**. Keep one clean entity
per class (the prototype move), but compare query to class with a learned non-linear network: depth-
concatenate the (class, query) pair and pass it through a **relation module** (CNN + small MLP) that
outputs a scalar relation score in (0,1). Reuse the relational-reasoning recipe — a single *shared*
comparator over every pair, *sum*-pooling for order-invariant set aggregation. The comparator can carve
the non-convex matched regions a fixed metric (Euclidean/cosine = a linear head) cannot, so embedding
and comparator split the work the linear head could not.

**Harness-specific architecture (this is NOT the paper's 4-conv embedding).** There is no parameter
budget for a second backbone, so the *embedding is the scaffold's shared ResNet-12*, taken as **feature
maps** via `make_backbone(use_pooling=False)` (640-channel maps, not pooled vectors). Per class, element-
wise **sum** the K support maps into one class map (mean up to a constant absorbed by BatchNorm). For
each (class, query) pair, concatenate the maps along channels → 2·640 = 1280, and feed the
`_RelationModule`: Conv2d(1280,640,3)→BN(momentum 1)→ReLU→`AdaptiveMaxPool2d((5,5))`, then
Conv2d(640,640,3)→BN→ReLU→`AdaptiveMaxPool2d((1,1))`, flatten, Linear(640,8)→ReLU→Linear(8,1)→Sigmoid.
Adaptive pooling makes the module robust to the backbone's map size; the width-8 FC is the comparator's
final non-linearity. All new parameters live in the relation module.

**Loss.** The model emits N *independent* sigmoid scores, not a distribution, so override `compute_loss`
to **MSE** against the one-hot target — regress each (query, class) score to its binary match target.
Cross-entropy would need to renormalize the N scores (a softmax/head), reintroducing the fixed cross-
class comparison this method exists to remove.

**Hyperparameters.** No `LR_OVERRIDE` — the feedforward conv comparator is stable under the scaffold
default SGD@1e-2 (grad-norm clip 5.0 covers from-scratch training). `is_transductive = False`.

**What to watch.** CIFAR should recover and pass 0.682, mini should rise past 0.649 (the learned
comparator carves what the linear head could not). The risk is CUB: the prototype already near-saturated
it at 0.756, so the extra capacity may buy little there and show larger seed-to-seed variance. If the
generic benchmarks rise while CUB holds, the fixed metric was the bottleneck; the remaining lever beyond
this is *task-adaptive embeddings*.

```python
class _RelationModule(nn.Module):
    """CNN relation module from Sung et al. (2018)."""

    def __init__(self, feature_dimension: int, inner_channels: int = 8):
        super().__init__()
        self.module = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(feature_dimension * 2, feature_dimension, kernel_size=3, padding=1),
                nn.BatchNorm2d(feature_dimension, momentum=1, affine=True),
                nn.ReLU(),
                nn.AdaptiveMaxPool2d((5, 5)),
            ),
            nn.Sequential(
                nn.Conv2d(feature_dimension, feature_dimension, kernel_size=3, padding=0),
                nn.BatchNorm2d(feature_dimension, momentum=1, affine=True),
                nn.ReLU(),
                nn.AdaptiveMaxPool2d((1, 1)),
            ),
            nn.Flatten(),
            nn.Linear(feature_dimension, inner_channels),
            nn.ReLU(),
            nn.Linear(inner_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.module(x)


class CustomFewShotMethod(FewShotClassifier):
    """Relation Networks (Sung et al., 2018).

    Extracts feature maps (not pooled vectors) from support and query images.
    Computes class prototypes as mean feature maps, concatenates each query-prototype
    pair, and feeds them through a learned relation module to get relation scores.
    Uses MSE loss since output represents relation scores in [0, 1].
    """

    def __init__(self):
        backbone = make_backbone(use_pooling=False)  # Need feature maps, not vectors
        super().__init__(backbone=backbone, use_softmax=False)
        self.feature_dimension = FEATURE_DIMENSION
        self.relation_module = _RelationModule(self.feature_dimension)

    def process_support_set(self, support_images: Tensor, support_labels: Tensor):
        support_features = self.compute_features(support_images)
        n_way = len(torch.unique(support_labels))
        self.prototypes = torch.cat(
            [
                support_features[support_labels == label].mean(0, keepdim=True)
                for label in range(n_way)
            ]
        )

    def forward(self, query_images: Tensor) -> Tensor:
        query_features = self.compute_features(query_images)
        n_queries = query_features.shape[0]
        n_prototypes = self.prototypes.shape[0]

        # Build pairs: [n_queries * n_prototypes, 2 * C, H, W]
        query_prototype_pairs = torch.cat(
            (
                self.prototypes.unsqueeze(0).expand(n_queries, -1, -1, -1, -1),
                query_features.unsqueeze(1).expand(-1, n_prototypes, -1, -1, -1),
            ),
            dim=2,
        ).view(-1, 2 * self.feature_dimension, *query_features.shape[2:])

        relation_scores = self.relation_module(query_prototype_pairs).view(
            n_queries, n_prototypes
        )
        return self.softmax_if_specified(relation_scores)

    @staticmethod
    def is_transductive() -> bool:
        return False

    def compute_loss(self, scores: Tensor, labels: Tensor) -> Tensor:
        # RelationNet uses MSE with one-hot labels
        one_hot = F.one_hot(labels, num_classes=scores.shape[1]).float()
        return F.mse_loss(scores, one_hot)
```
