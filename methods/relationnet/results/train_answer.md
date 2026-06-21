The problem I want to solve is $C$-way $K$-shot recognition of *novel* categories: I am handed only $K$ labelled images — one, maybe five — for each of $C$ classes that never appeared during training, and I must label a query among those $C$ classes in a single forward pass, ideally without touching any weights at test time. The brute-force reflex, fine-tuning a deep network on the $K\!\cdot\!C$ support images, is dead on arrival: a million-parameter net has essentially no signal in five images per class, and augmentation or weight decay only slow the overfitting rather than removing it. What I do have is an abundant training set of *other* classes, disjoint from the test classes, and the whole game is to distill from those classes a reusable piece of machinery — a way to embed an image, a way to summarise a class from its few examples, and a way to decide which class a query belongs to — that transfers feed-forward to classes it was never trained on. The structural idea I trust for this is episodic training, as Matching Networks set it up: on every iteration I manufacture a fake few-shot task out of the training classes, sampling $C$ classes, drawing $K$ labelled examples each as a support set $S=\{(x_i,y_i)\}$ and a disjoint query set $Q$ from the same classes, and scoring the model on classifying $Q$ given $S$. Because every training episode is itself a small $C$-way $K$-shot problem with random class identities, whatever the model learns is under constant pressure to be the part that transfers across class identities rather than a memorised pixel-to-label mapping. That substrate I keep; the open question is what happens *inside* an episode — given support embeddings and a query embedding, how do I produce the $C$ scores?

The predecessors all fill that slot the same way, and seeing exactly where they stall is what points to the fix. Matching Networks embed everything and label the query as an attention-weighted vote over support labels, $\hat y = \sum_i a(\hat x, x_i)\,y_i$, with $a$ a softmax over the *cosine similarity* of embeddings — a soft nearest neighbour. Prototypical Networks collapse each class to one prototype $c_k=\frac1K\sum_{i:y_i=k} f(x_i)$, the mean of its embedded support examples, and classify by a softmax over *negative squared Euclidean distance*, $p(y=k\mid x)\propto \exp(-\lVert f(x)-c_k\rVert^2)$. That pairing is principled: squared Euclidean distance is a Bregman divergence, and the point minimising total Bregman divergence to a set is exactly its mean, so the mean prototype is the matched optimal cluster centre; and expanding $\lVert f-c_k\rVert^2 = \lVert f\rVert^2 - 2c_k^{\top}f + \lVert c_k\rVert^2$, the $\lVert f\rVert^2$ term is identical across classes and drops out of the softmax, leaving $2c_k^{\top}f - \lVert c_k\rVert^2$, which is *linear* in $f$ with weight $2c_k$ and bias $-\lVert c_k\rVert^2$. But that very elegance is the confession of the ceiling. In all of these — Matching, Prototypical, the older Siamese verification nets — every bit of learning lives in the *embedding*; once you are in embedding space the comparison is a fixed, closed-form, hand-chosen function, and the Prototypical analysis shows squared-Euclidean-softmax *is* a linear classifier there. So the method can only work to the extent that the embedding alone makes every novel class linearly separable, clustered around a mean like a spherical Gaussian blob — and I am asking one embedding, trained on disjoint classes, to lay out *brand-new* categories that cleanly. The reported violent sensitivity to *which* fixed metric you pick (cosine here, Euclidean there, the "right" one rarely the obvious one) is the symptom: no fixed element-wise comparison is universally right. To feel the wall concretely, fix a query in a 2D embedding and ask, for every support point, "same class or not?" Euclidean nearest neighbour can only carve out a convex disk; if the true matched region is an annulus, two patches, or a curved band, no nearest-neighbour rule reaches it. Learning a Mahalanobis metric $(f-c)^{\top}M(f-c)$ only stretches the disk to an ellipsoid — still convex — and warping through extra nonlinear layers before a Mahalanobis step still ends in an ellipsoidal blob in the warped space. Every rescue pours more capacity into the embedding or the space-warp while insisting the final comparison stay a fixed parametric distance with no learnable nonlinearity in it. That is the choke point.

The resolution is to stop fixing the comparison and learn it too. I propose the Relation Network: rather than computing $d(f(\hat x), c_k)$ with a chosen formula, feed the query representation and the class summary into a *network* and let it output the score, so the metric becomes a learned, nonlinear function of the two trained jointly with the embedding end-to-end. Now I am no longer betting everything on the embedding making classes linearly separable; the comparator can bend, and the embedding and comparator split the work however the data wants. The machinery for "feed two things into a network and get a relation out" already exists in relational reasoning, where $\mathrm{RN}(O)=f_\phi\big(\sum_{i,j} g_\theta(o_i,o_j)\big)$ applies a single *shared* small network $g_\theta$ to every concatenated pair of objects and *sums* the pairwise outputs for an order-invariant aggregate. Both properties transfer exactly: a single shared comparator applied to every (query, class) pair is the data-efficient inductive bias I want for generalising to unseen class pairings, and summation is the order-invariant way to pool the $K$ support examples of a class. So in the one-shot case I embed each support image and the query with a shared embedding $f_\varphi$, concatenate each (support, query) pair, and push it through a relation module $g_\phi$ that outputs a single scalar relation score, $r_{i,j}=g_\phi\big(C(f_\varphi(x_i), f_\varphi(x_j))\big)$ for $i=1,\dots,C$, giving $C$ scores per query. I make the combine operator $C(\cdot,\cdot)$ a plain concatenation: it keeps all of both representations and lets $g_\phi$ learn whatever cross-terms it wants, whereas a difference $f(x_i)-f(x_j)$ would already discard information and bake in a comparison form — the exact mistake I am undoing.

What I concatenate matters. If the embedding emits a global vector and I run an MLP, the comparator sees only holistic vectors — already strictly more expressive than a fixed metric, but blind to spatial structure. So I keep the embedding output as a *feature map* (channels $\times$ height $\times$ width) and concatenate two maps along the channel axis, doubling the channels, which lets the first part of the relation module be *convolutional* and compare local parts before pooling to a score — something a vector-MLP comparator cannot do. That decision forces a non-obvious change to the embedding. The standard four-conv few-shot backbone uses a $2\times2$ max-pool in every block, and four poolings on an $84\times84$ image shrink the map to almost nothing; but I want a spatial map left for the relation convolutions, so I pool only in the first two blocks and drop pooling from the last two — the first two halvings cut resolution and cost, the last two convs deepen the representation while preserving the map. Concretely a $28\times28$ Omniglot image becomes a $64\times5\times5$ encoder map and a $84\times84$ miniImageNet image a $64\times19\times19$ map. The relation module $g_\phi$ then receives the $2\times64$-channel concatenated pair and mirrors the embedding's vocabulary: two conv blocks of $3\times3$ conv(64), batch norm, ReLU, now *with* $2\times2$ max-pools because this branch's job is to aggregate spatial evidence down to a decision; after that the map is small (on miniImageNet $64\cdot3\cdot3=576$ values), so I flatten and pass through two fully-connected layers, one hidden layer of width 8 then a single output. The hidden FC layer supplies the final nonlinearity over the pooled evidence — without it the head would be linear again and I would have quietly reintroduced a linear comparator. I put a sigmoid on the output so the relation score is a bounded $r\in(0,1)$ with a consistent meaning across episodes (1 = same class, 0 = different), which matters because the comparator is shared and must mean the same thing everywhere.

For $K>1$ I want still exactly $C$ scores per query, not $C\!\cdot\!K$ scores to reconcile, so I fold each class's $K$ support embeddings into one class-level representation before comparing. The relational module already prescribes the rule: element-wise *sum* the $K$ embedding feature maps of a class into a single class feature map, then run the one-shot procedure on it — order-invariant, score count fixed at $C$, cheap. Summing rather than averaging makes the class map scale with $K$, but for fixed-shot evaluation that is a constant and the following convolution, batch norm, and learned weights adapt to the scale when trained consistently; averaging would just be a separate $1/K$ rescaling choice, so I keep the sum, the set-pooling rule I actually need. The episode forward pass is then: embed all support images and sum per class to get $C$ class maps; embed the query; for each class concatenate (class map, query map) in depth and run the relation module for $r_{i,j}$; the $C$ outputs are the query's scores, and at test time I take the argmax — feed-forward, no fine-tuning. The last and oddest-looking choice is the loss. The reflex is cross-entropy, but the model does not emit a normalised distribution; it emits $C$ *independent* relation scores, each a sigmoid produced by the shared comparator looking at one (class, query) pair in isolation. The natural target for a single score $r_{c,j}$ is binary, $\mathbf 1(y_c = y_j)$, so per pair I am regressing a score toward a target in $\{0,1\}$ — drive the matched pair to 1 and every mismatched pair to 0. That is mean-squared error against the one-hot target,
$$\varphi, \phi \leftarrow \arg\min \sum_{c=1}^{C}\sum_{j=1}^{n}\big(r_{c,j} - \mathbf 1(y_c = y_j)\big)^2.$$
I am predicting a *relation* — a real-valued similarity that should be 1 whenever two objects are the same class however the images differ, and 0 otherwise — and predicting a real value to hit a real target is regression, even though the only auto-generated targets are the extremes 0 and 1; MSE on the sigmoid output matches that framing, while ordinary multiclass cross-entropy would convert the independent relation scores into a normalised class distribution, a different training semantics. Everything is trained end-to-end from scratch with random initialisation, embedding and comparator mutually tuned, using Adam at learning rate $1\mathrm{e}{-3}$ annealed (halved periodically) over the long episode schedule, with each module's gradient norm clipped to $0.5$; from-scratch suffices because episodic training over many tasks is itself a strong, overfitting-resistant signal.

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
