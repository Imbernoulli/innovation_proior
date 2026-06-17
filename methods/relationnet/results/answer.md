# Relation Network (RelationNet), distilled

The Relation Network is a metric-based few-shot classifier that **learns the comparison function
itself**, instead of comparing embeddings with a fixed metric. It has two meta-learned modules
trained end-to-end with episodes: an *embedding module* `f_varphi` that turns images into feature
maps, and a *relation module* `g_phi` — a small CNN+MLP — that takes a depth-concatenated
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

- **One-shot:** `r_{i,j} = g_phi( C(f_varphi(x_i), f_varphi(x_j)) )`, `i = 1,…,C`, where `C(·,·)` is depth
  concatenation; `C` scores per query.
- **K-shot:** element-wise **sum** the `K` embedding feature maps of each class into one
  class-level feature map, then apply the one-shot procedure — always `C` scores per query.
  Mean pooling is a separate `1/K` rescaling choice; the author code uses sum, while EasyFSL's
  adapter stores mean prototypes.
- **Output:** a **sigmoid** scalar relation score in `[0,1]` (1 = same class, 0 = different).
- **Loss: MSE**, regressing each score to its binary match target — because the output is a
  relation *score* (a regression to a `{0,1}` target), not a normalized class distribution:
  `varphi, phi ← argmin Σ_{c=1}^C Σ_{j=1}^n ( r_{c,j} − 1(y_c == y_j) )²`.
  Cross-entropy would change the object into a normalized class distribution over the `C` scores.

## Architecture (canonical 4-conv version)

- **Embedding module** `f_varphi`: four conv blocks, each `3×3 conv(64) → BatchNorm → ReLU`. Max-pool
  (`2×2`) in the **first two** blocks only; the last two omit pooling so a spatial feature **map**
  survives for the convolutional relation module. In the original four-conv code, 28×28 Omniglot
  becomes a `64×5×5` encoder map; 84×84 miniImageNet becomes a `64×19×19` encoder map.
- **Relation module** `g_phi`: input `2·64 = 128` channels (concatenated pair); two conv blocks
  (`3×3 conv(64) → BatchNorm → ReLU → 2×2 max-pool` each); flatten; FC(→8) → ReLU → FC(→1) →
  **Sigmoid**. The final flattened size is `64` for Omniglot and `64·3·3 = 576` for miniImageNet.
- **Training:** end-to-end from scratch, random init; Adam, lr `1e-3`, halved every 100k
  episodes; the canonical code also clips each module's gradient norm to 0.5 and uses
  Kaiming-style normal conv initialization, BatchNorm weights 1/bias 0, and Linear
  normal(0, 0.01)/bias 1.

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

Faithful to the canonical four-conv code for the few-shot model:
sum support feature maps, concatenate class/query maps in depth, emit sigmoid relation scores,
train by MSE to binary match targets, and clip the embedding and relation-module gradients separately.
EasyFSL's `RelationNetworks` adapter keeps the same interface but stores mean prototypes and uses
adaptive pooling inside the relation module.

```python
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _conv_out(size: int, padding: int, pool: bool) -> int:
    size = size + 2 * padding - 3 + 1
    if pool:
        size = size // 2
    if size <= 0:
        raise ValueError("convolutional stack collapsed the spatial map")
    return size


def conv_block(
    in_channels: int,
    out_channels: int,
    *,
    padding: int,
    pool: bool,
) -> nn.Module:
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding),
        nn.BatchNorm2d(out_channels, momentum=1, affine=True),
        nn.ReLU(),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


def weights_init(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / n))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.01)
        nn.init.ones_(module.bias)


class EmbeddingModule(nn.Module):
    def __init__(self, in_channels: int = 3, feature_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(in_channels, feature_dim, padding=0, pool=True),
            conv_block(feature_dim, feature_dim, padding=0, pool=True),
            conv_block(feature_dim, feature_dim, padding=1, pool=False),
            conv_block(feature_dim, feature_dim, padding=1, pool=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    @staticmethod
    def output_spatial(image_size: int) -> int:
        size = image_size
        for padding, pool in ((0, True), (0, True), (1, False), (1, False)):
            size = _conv_out(size, padding, pool)
        return size


class RelationModule(nn.Module):
    def __init__(
        self,
        feature_dim: int = 64,
        hidden_size: int = 8,
        *,
        input_spatial: int,
        relation_padding: int,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            conv_block(feature_dim * 2, feature_dim, padding=relation_padding, pool=True),
            conv_block(feature_dim, feature_dim, padding=relation_padding, pool=True),
        )
        out_spatial = input_spatial
        for _ in range(2):
            out_spatial = _conv_out(out_spatial, relation_padding, pool=True)
        self.fc1 = nn.Linear(feature_dim * out_spatial * out_spatial, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, pair: Tensor) -> Tensor:
        out = self.conv(pair)
        out = torch.flatten(out, start_dim=1)
        out = F.relu(self.fc1(out))
        return torch.sigmoid(self.fc2(out))


class RelationNetwork(nn.Module):
    """Relation Network few-shot classifier."""

    def __init__(
        self,
        image_size: int = 84,
        in_channels: int = 3,
        feature_dim: int = 64,
        hidden_size: int = 8,
        relation_padding: int | None = None,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        if relation_padding is None:
            # 84x84 miniImageNet uses padding 0 in the relation convs; 28x28 Omniglot uses 1.
            relation_padding = 1 if image_size == 28 else 0
        feature_spatial = EmbeddingModule.output_spatial(image_size)
        self.embedding = EmbeddingModule(in_channels, feature_dim)
        self.relation = RelationModule(
            feature_dim,
            hidden_size,
            input_spatial=feature_spatial,
            relation_padding=relation_padding,
        )
        self.class_maps: Tensor | None = None
        self.class_labels: Tensor | None = None
        self.apply(weights_init)

    def process_support_set(self, support_images: Tensor, support_labels: Tensor):
        feats = self.embedding(support_images)
        self.class_labels = torch.unique(support_labels, sorted=True)
        self.class_maps = torch.stack(
            [feats[support_labels == c].sum(dim=0) for c in self.class_labels]
        )

    def forward(self, query_images: Tensor) -> Tensor:
        if self.class_maps is None:
            raise RuntimeError("call process_support_set() before forward()")
        q = self.embedding(query_images)
        n_query, n_class = q.shape[0], self.class_maps.shape[0]
        cls = self.class_maps.unsqueeze(0).expand(n_query, -1, -1, -1, -1)
        qry = q.unsqueeze(1).expand(-1, n_class, -1, -1, -1)
        pairs = torch.cat((cls, qry), dim=2).view(
            -1, 2 * self.feature_dim, *q.shape[2:]
        )
        return self.relation(pairs).view(n_query, n_class)

    def compute_loss(self, scores: Tensor, query_labels: Tensor) -> Tensor:
        if self.class_labels is None:
            raise RuntimeError("call process_support_set() before compute_loss()")
        class_labels = self.class_labels.to(query_labels.device)
        targets = (query_labels.unsqueeze(1) == class_labels.unsqueeze(0)).to(
            device=scores.device,
            dtype=scores.dtype,
        )
        return F.mse_loss(scores, targets)

    @staticmethod
    def is_transductive() -> bool:
        return False


def train(method: RelationNetwork, optimizer, episode_sampler) -> None:
    for support_images, support_labels, query_images, query_labels in episode_sampler:
        optimizer.zero_grad()
        method.process_support_set(support_images, support_labels)
        scores = method(query_images)
        loss = method.compute_loss(scores, query_labels)
        loss.backward()
        nn.utils.clip_grad_norm_(method.embedding.parameters(), 0.5)
        nn.utils.clip_grad_norm_(method.relation.parameters(), 0.5)
        optimizer.step()
```

A vector-backbone variant (e.g. ResNet feature maps without flattening) plugs into the same
shape: keep the backbone's spatial maps (no flatten), concatenate query/prototype maps in depth,
and pass through a relation module whose first conv accepts `2·feature_dim` channels. Adaptive
pooling in the relation module (`AdaptiveMaxPool2d`) makes it robust to varying map sizes.
