# Relation Network (RelationNet), distilled

The Relation Network is a metric-based few-shot classifier that **learns the comparison function
itself**, instead of comparing embeddings with a fixed metric. It has two meta-learned modules
trained end-to-end with episodes: an *embedding module* `f_φ` that turns images into feature
maps, and a *relation module* `g_ϕ` — a small CNN+MLP — that takes a depth-concatenated
(class, query) feature-map pair and outputs a single **relation score** in `[0,1]`. The query is
classified by the class with the highest relation score, in one feed-forward pass with no
test-time fine-tuning.

## Problem it solves

`C`-way `K`-shot classification of novel classes (disjoint from the training classes) from `K`
labelled support examples per class, where fine-tuning a deep net on so few examples overfits.
The goal is a reusable, transferable component — embedding + comparator — meta-learned on
auxiliary training classes and applied feed-forward to unseen classes.

## Key idea

Prior metric-based methods (Matching Networks: fixed cosine attention; Prototypical Networks:
fixed squared-Euclidean = a linear classifier in embedding space) put **all** learning in the
embedding and compare with a **fixed, element-wise** metric. That metric is the bottleneck: it
can only succeed when the embedding alone makes novel classes linearly/convexly separable, and
the strong reported sensitivity to *which* fixed metric you choose shows no fixed comparison is
universally right. RelationNet instead makes the comparator a **learned non-linear network**, so
embedding and comparator split the work. The comparator reuses the machinery of a relational
reasoning module (Santoro et al. 2017, `RN(O) = f_φ(Σ_{i,j} g_θ(o_i, o_j))`): a single shared
network applied to concatenated pairs, with sum-pooling for order-invariant set aggregation.

- **One-shot:** `r_{i,j} = g_ϕ( C(f_φ(x_i), f_φ(x_j)) )`, `i = 1,…,C`, where `C(·,·)` is depth
  concatenation; `C` scores per query.
- **K-shot:** element-wise **sum** the `K` embedding feature maps of each class into one
  class-level feature map, then apply the one-shot procedure — always `C` scores per query.
  (Mean pooling is equivalent up to the constant `1/K`, absorbed by BatchNorm / learned weights.)
- **Output:** a **sigmoid** scalar relation score in `[0,1]` (1 = same class, 0 = different).
- **Loss: MSE**, regressing each score to its binary match target — because the output is a
  relation *score* (a regression to a `{0,1}` target), not a normalized class distribution:
  `φ, ϕ ← argmin Σ_i Σ_j ( r_{i,j} − 1(y_i == y_j) )²`.
  Cross-entropy would require renormalizing the `C` independent scores (a softmax/extra head),
  reintroducing a fixed cross-class comparison.

## Architecture (canonical 4-conv version)

- **Embedding module** `f_φ`: four conv blocks, each `3×3 conv(64) → BatchNorm → ReLU`. Max-pool
  (`2×2`) in the **first two** blocks only; the last two omit pooling so a spatial feature **map**
  survives for the convolutional relation module. (28×28 Omniglot → `H = 64`; 84×84 miniImageNet
  → 64-channel map reducing to `64·3·3 = 576` after the relation module's pools.)
- **Relation module** `g_ϕ`: input `2·64 = 128` channels (concatenated pair); two conv blocks
  (`3×3 conv(64) → BatchNorm → ReLU → 2×2 max-pool` each); flatten; FC(→8) → ReLU → FC(→1) →
  **Sigmoid**. The hidden FC is the comparator's final nonlinearity.
- **Training:** end-to-end from scratch, random init; Adam, lr `1e-3`, halved every 100k
  episodes; (original implementation also clips gradient norm to 0.5).

## Why it works / relation to prior methods

RelationNet *both* learns a deep embedding *and* learns a deep non-linear similarity, mutually
tuned. Fixed-metric methods (Matching/Prototypical) assume element-wise comparison and (for
Prototypical) linear separability after the embedding, so they are bottlenecked by the embedding;
a learned relation module can represent comparisons no fixed metric can (e.g. a synthetic 2D case
where the matched region is non-linear: Euclidean-NN, a learned Mahalanobis metric, and even an
MLP-embedding-then-Mahalanobis all fail, but the deep relation module solves it). It avoids the
recurrence of memory/RNN methods (MANN, MetaNets) and the test-time gradient steps of
optimisation methods (MAML, Meta-Learner LSTM): inference is a single feed-forward pass.

## Working code

Faithful to the original implementation (floodsung/LearningToCompare_FSL) and the easyfsl
`RelationNetworks` / `default_relation_module`. The interface matches the episodic few-shot
scaffold: `process_support_set` summarizes the support set, `forward` returns `(n_query, C)`
scores, `compute_loss` is MSE.

```python
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def conv_block(in_channels: int, out_channels: int, pool: bool) -> nn.Module:
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels, momentum=1, affine=True),
        nn.ReLU(),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class EmbeddingModule(nn.Module):
    """Four conv blocks; pool in the first two only -> keep a spatial feature map."""

    def __init__(self, feature_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(3, feature_dim, pool=True),
            conv_block(feature_dim, feature_dim, pool=True),
            conv_block(feature_dim, feature_dim, pool=False),
            conv_block(feature_dim, feature_dim, pool=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)                       # (n, C, H, W)


class RelationModule(nn.Module):
    """Learned comparator: depth-concatenated pair (2C ch) -> conv+pool -> FC -> score in (0,1)."""

    def __init__(self, feature_dim: int = 64, hidden_size: int = 8):
        super().__init__()
        self.conv = nn.Sequential(
            conv_block(feature_dim * 2, feature_dim, pool=True),
            conv_block(feature_dim, feature_dim, pool=True),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, pair: Tensor) -> Tensor:
        return self.fc(self.conv(pair))          # (n_pairs, 1)


class RelationNetwork(nn.Module):
    """Relation Network few-shot classifier."""

    def __init__(self, feature_dim: int = 64, hidden_size: int = 8):
        super().__init__()
        self.feature_dim = feature_dim
        self.embedding = EmbeddingModule(feature_dim)
        self.relation = RelationModule(feature_dim, hidden_size)
        self.class_maps = None

    def process_support_set(self, support_images: Tensor, support_labels: Tensor):
        # Embed support; element-wise SUM the K maps per class -> one class-level feature map.
        feats = self.embedding(support_images)               # (K*C, F, H, W)
        n_way = int(support_labels.max().item()) + 1
        self.class_maps = torch.stack(
            [feats[support_labels == c].sum(dim=0) for c in range(n_way)]
        )                                                    # (C, F, H, W)

    def forward(self, query_images: Tensor) -> Tensor:
        q = self.embedding(query_images)                     # (n_query, F, H, W)
        n_query, n_class = q.shape[0], self.class_maps.shape[0]
        cls = self.class_maps.unsqueeze(0).expand(n_query, -1, -1, -1, -1)
        qry = q.unsqueeze(1).expand(-1, n_class, -1, -1, -1)
        pairs = torch.cat((cls, qry), dim=2).view(
            -1, 2 * self.feature_dim, *q.shape[2:]
        )                                                    # (n_query*n_class, 2F, H, W)
        return self.relation(pairs).view(n_query, n_class)   # C relation scores per query

    def compute_loss(self, scores: Tensor, query_labels: Tensor) -> Tensor:
        one_hot = F.one_hot(query_labels, num_classes=scores.shape[1]).float()
        return F.mse_loss(scores, one_hot)                   # MSE, not cross-entropy

    @staticmethod
    def is_transductive() -> bool:
        return False
```

A vector-backbone variant (e.g. ResNet feature maps without flattening) plugs into the same
shape: keep the backbone's spatial maps (no flatten), concatenate query/prototype maps in depth,
and pass through a relation module whose first conv accepts `2·feature_dim` channels. Adaptive
pooling in the relation module (`AdaptiveMaxPool2d`) makes it robust to varying map sizes.
