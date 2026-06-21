The prototype run confirmed the prediction and exposed the next wall in the same numbers. CUB jumped exactly as I said it would — from the matching vote's $0.625$ to $0.756$, the single largest move of either method, because collapsing each fine-grained class to one clean prototype and scoring by the Euclidean partner of the mean removed precisely the per-point scatter that had no margin when all the birds look alike. But the generic benchmarks *fell*: CIFAR-FS from $0.769$ to $0.682$, miniImageNet from $0.674$ to $0.649$. Some of that is the optimizer confound I flagged, but the shape is too clean to be only that. Stare at what the prototype classifier actually is and the ceiling is written into it: expanding the squared distance, the query-only term cancels and what is left is *linear* in the embedding — the prototype method is exactly a linear classifier in feature space, weights read off the means. So it can only succeed to the extent the embedding *by itself* lays out brand-new classes as round, linearly-separable blobs around their means. On fine-grained CUB the embedding can apparently do that; on the broader CIFAR/mini layouts it cannot, and a straight line through the prototypes leaves accuracy on the table. The violent benchmark-to-benchmark swing — CUB up 13 points, CIFAR down 9 — *is* the symptom that no single fixed, closed-form comparison is right for all of these layouts. The metric is the bottleneck.

So I attack the one thing I held fixed through both previous rungs: the comparison function. Let me feel exactly where an element-wise fixed metric fails. Fix a query point in a 2-D embedding and ask, for every possible support point, "same class as the query or not?" If matched/mismatched is decided by Euclidean nearness, the matched region is a disk around the query — one convex blob. But if the true matched region is an annulus, two separated patches, or a curved band, no Euclidean nearest-neighbour can carve it out; nearest-neighbour only produces convex, distance-monotone regions. Learning a *better metric* while keeping it a metric — a Mahalanobis distance — only stretches and rotates the space; the region is still an ellipsoidal blob, still convex. Push the embeddings through more non-linear layers and *then* do Mahalanobis, and the *final* decision is still an ellipsoidal blob in the warped space. Every fix pours more capacity into the embedding or the space-warp while insisting the last step — the comparison — stay a fixed parametric distance. That is the wall, and the prototype's CIFAR drop is me hitting it: the comparison itself has no learnable non-linearity in it.

So I propose **Relation Networks**: stop computing $d(f(\hat{x}), c_k)$ with a chosen formula, and instead feed the two representations into a *network* that outputs the score, training the metric as a learned, non-linear function of the query and the class summary jointly with the embedding end-to-end. Now I am no longer betting everything on the embedding making classes linearly separable — the comparator can itself bend, and embedding and comparator can split the work however the data wants. The mechanism I borrow is the relational module, $\mathrm{RN}(O) = f_\phi\!\big(\sum_{i,j} g_\theta(o_i, o_j)\big)$: one small *shared* network applied to every concatenated pair, the pairwise outputs *summed*, then mapped to an answer. Two of its properties transfer perfectly. A single shared comparator across all pairs is data-efficient and generalizes to unseen pairings — exactly the inductive bias I want when the comparator must generalize to support sets of unseen classes, rather than a separate comparator per class. And summing over a set is order-invariant, which I need to fold each class's $K$ support examples into one class representation. So in the one-shot case I keep the scaffold's shared ResNet-12 as the embedding $f_\phi$ — there is no parameter budget for a second backbone — concatenate each class's representation with the query's, and push the pair through a learned relation module $g_\phi$ that outputs a single scalar relation score, giving $N$ scores per query. The combine operator is *concatenation*: it keeps all of both representations and lets $g_\phi$ learn whatever cross-terms it wants, whereas a fixed combine like the difference $f(x_i) - f(\hat{x})$ would already throw information away and bake in a comparison form — the very mistake I am undoing after watching the linear prototype head lose CIFAR.

Now the choice the harness actually forces, where I depart from the textbook relation network. The textbook version uses a small four-conv embedding and concatenates feature *maps* in depth so the relation module can be *convolutional* and compare local spatial structure before pooling — and that matters, because "are these the same kind of thing" is often about whether local parts correspond, which a vector-MLP comparator cannot see. But I do not get to choose the embedding: the scaffold's backbone is a fixed ResNet-12, and the only knob is whether I ask it for pooled $640$-d *vectors* or $640$-channel feature *maps*. To make the comparator convolutional I take the maps — `make_backbone(use_pooling=False)` for both support and query — and concatenate a (class-map, query-map) pair along the channel axis into $2\cdot 640 = 1280$ channels, which the relation module's first conv accepts. All my new parameters go into the relation module on these $640$-channel maps, which is also what keeps me inside the parameter budget (set at $\approx 1.05\times$ a ResNet-12 plus exactly such a module). The relation module $g_\phi$ eats the $1280$-channel pair and boils it to one number: a conv block — `Conv2d(1280, 640, 3, pad 1)`, BatchNorm, ReLU — then pool the spatial map down, because this branch's job is to aggregate spatial evidence toward a decision (unlike the embedding, where pooling away spatial structure would be wrong). The ResNet-12 maps are small, so rather than a fixed pool that assumes a particular map size I use `AdaptiveMaxPool2d((5,5))` after the first conv block and `AdaptiveMaxPool2d((1,1))` after the second — adaptive pooling makes the module robust to whatever spatial size the backbone emits, the safe choice when bolting onto a backbone I did not design. Two conv blocks ($640\to640\to640$) with those two adaptive pools collapse the pair to a single $640$-vector per (class, query); flatten and pass through `Linear(640, 8) → ReLU → Linear(8, 1)`. The hidden FC of width $8$ is the comparator's final non-linearity over the pooled convolutional evidence — without it the head would be linear again and I would have quietly reintroduced the very linear comparator that lost CIFAR. The BatchNorms use momentum $1$ (no running average), the relation-network convention that fits the small per-episode batches. The output is a single scalar, and I want it to be a *relation score* — "how much do these match" — pinned to a bounded interpretable range rather than an unbounded logit, because I am about to compare it against targets meaning "match" and "no match." So a Sigmoid: $r \in (0, 1)$, $1$ reads as same class, $0$ as different, a meaning that stays consistent across episodes and classes — which matters because the comparator is *shared* and must mean the same thing everywhere.

The $K$-shot case is the whole game here, since every benchmark is 5-shot, and it is where I fix what the matching vote did badly and the prototype did rigidly. I have $K$ support maps per class and I want still $N$ scores per query, not $N\cdot K$ to reconcile, so I fold each class's $K$ embedded maps into one class-level map before the comparison. The relational module already told me how — *sum*, the order-invariant aggregator — so I element-wise sum the $K$ support feature maps of a class into one class feature map (equivalently mean, up to a constant the BatchNorm and learned weights absorb) and run the one-shot procedure on it. A class map scaling with $K$ is harmless because the next op is conv-then-BatchNorm, which rescales away a global magnitude factor. This is the same "one entity per class" move the prototype method made, now in feature-map space — so the relation network *keeps* that improvement over matching's scatter *and* adds the learned comparator the prototype lacked, the two upgrades the previous rungs each had only one of.

The last piece is the loss, and it looks oddest until I see what the model emits. The reflex is cross-entropy — it is classification, the targets are class indices, and both previous rungs used it. But the model does not emit a normalized distribution over classes; it emits $N$ *independent* sigmoid relation scores, each produced by the shared comparator looking at one (class, query) pair in isolation. The natural ground truth for a single relation score is binary — $1$ if the query's class equals that class, $0$ otherwise — so per pair I am *regressing a score toward a $\{0,1\}$ target*, which is mean-squared error against the one-hot target: a bounded relation magnitude pulled to one of two ends, matching a sigmoid output. If I insisted on cross-entropy I would have to renormalize the $N$ independent scores into a distribution — a softmax or extra head — reintroducing a fixed cross-class comparison and partly undoing the "each pair is an independent learned relation" structure I built to escape the fixed metric. So MSE is not sloppiness; it is the loss consistent with treating each (query, class) decision as an independent learned-relation regression, and I override `compute_loss` to it where the previous rungs left cross-entropy or NLL. For optimization I stay on the scaffold default — SGD at $10^{-2}$ from scratch, no `LR_OVERRIDE` — because unlike the matching method's recurrences the relation module is a feedforward conv stack, stable under the default rate, and the loop's grad-norm clipping at $5.0$ covers from-scratch training; the embedding and comparator tune each other.

The falsifiable expectations against the prototype numbers follow from the whole reason to do this — the generic benchmarks where the linear metric left accuracy on the table. I expect CIFAR-FS to recover and pass the prototype $0.682$ (the learned comparator carves the non-convex matched regions the linear head could not) and miniImageNet to rise past $0.649$ for the same reason. The risk and the test is CUB: the prototype already posted a strong $0.756$ on a layout where a clean entity plus a simple metric was apparently near-ideal, so the learned comparator's extra capacity might buy little there and could even be *higher variance* across seeds — a flexible comparator trained from scratch on twenty-five images per episode has more ways to land differently per seed than a parameter-free linear head. So my concrete prediction: CIFAR up clearly (past $0.78$), mini up (past $0.69$), CUB roughly holding around $0.756$ but with visibly larger seed-to-seed spread than the prototype's tight $0.753$–$0.762$. If CIFAR and mini both rise while CUB holds, that confirms the fixed metric — not the summary — was the bottleneck on the generic benchmarks, and the learned comparison is the right and final lever among the metric methods. If even the learned comparator cannot move the generic benchmarks, the remaining lever is no longer the metric at all but *adapting the embedding itself to each task* — making the per-class summaries task-aware rather than comparing fixed embeddings — which is where I would look next.

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
