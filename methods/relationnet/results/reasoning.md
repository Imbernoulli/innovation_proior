Let me start from what actually hurts. I have a `C`-way `K`-shot problem: a handful of labelled images per class — one, maybe five — for classes I have never seen during training, and I have to label a query among those `C` classes in one shot, ideally without touching the weights at test time. The brute-force move, fine-tune a deep net on the `K·C` support images, is dead on arrival: a network with a million parameters has essentially no signal in five images per class, and augmentation and weight decay only slow the overfitting, they don't make it go away. So fine-tuning is out, or at least it can't be the core. What I do have is a big training set of *other* classes, disjoint from the test classes. The whole game is to squeeze out of those training classes some reusable machinery — a way to embed an image, a way to summarise a class from its few examples, and a way to decide which class a query belongs to — that transfers, feed-forward, to classes it was never trained on.

The one structural idea I trust here is episodic training, the way Matching Networks set it up. Don't train on a flat classification task and hope it transfers; instead, on every iteration, *manufacture a fake few-shot task* out of the training classes — sample `C` classes, draw `K` labelled examples each as a support set `S = {(x_i, y_i)}`, draw a disjoint query set `Q = {(x_j, y_j)}` from the same `C` classes, and score the model on classifying `Q` given `S`. Because every training episode is itself a small `C`-way `K`-shot problem with random class identities, whatever the model learns is under constant pressure to be the part that works *across* class identities — the transferable part — not a memorised mapping from pixels to a fixed label set. That principle I'm keeping. It's the substrate. The open question is what happens *inside* an episode: given the support embeddings and a query embedding, how do I produce the `C` scores?

So let me lay out exactly how the methods I know fill that slot, because if there's a ceiling I want to find it by looking at where they stall. Matching Networks: embed everything, then label the query as an attention-weighted vote over the support labels, `ŷ = Σ_i a(x̂, x_i) y_i`, where the attention is a softmax over the *cosine similarity* of the embeddings, `a(x̂, x_i) ∝ exp(c(f(x̂), g(x_i)))`. It's a soft nearest-neighbour in embedding space. Prototypical Networks tightens this: instead of voting over every support point, collapse each class to one prototype `c_k`, the mean of its embedded support examples, `c_k = (1/K) Σ_{i: y_i = k} f(x_i)`, and classify by a softmax over *negative squared Euclidean distance*, `p(y=k|x) ∝ exp(−‖f(x) − c_k‖²)`. And Snell et al. have a reason for those two choices being matched. Squared Euclidean distance is a Bregman divergence, and a known fact about Bregman divergences is that the point minimising total divergence to a set is the *mean* of the set — so taking the prototype to be the class mean isn't arbitrary, it's the optimal cluster centre *for that distance*.

There's a second claim attached to that distance which I want to actually check before I build an argument on top of it, because the whole story I'm about to tell hinges on it: that squared-Euclidean-softmax *is* a linear classifier in embedding space. Expand the distance, `‖f − c_k‖² = ‖f‖² − 2 c_k^T f + ‖c_k‖²`. The `‖f‖²` term doesn't depend on `k`, so it should fall out of the softmax and leave `2 c_k^T f − ‖c_k‖²`, linear in `f` with weight `2c_k` and bias `−‖c_k‖²`. Let me not just nod at that — let me put numbers on it. Take a 5-dim embedding, a query `f`, three random prototypes `c_k`. Computing `−‖f − c_k‖²` directly gives `[−20.35, −6.83, −10.91]`; computing the linear form `2 c_k^T f − ‖c_k‖²` gives `[−11.64, 1.88, −2.20]`. Different numbers — but subtract them element-wise and I get `[−8.71, −8.71, −8.71]`, a single constant, and `−‖f‖²` for this `f` is exactly `−8.71`. So the two scores differ by a class-independent constant, which means after softmax they must coincide, and indeed `softmax([−20.35,−6.83,−10.91]) = [1.3e−6, 0.9834, 0.0166]` equals `softmax` of the linear form to six places. So it's literally true: Prototypical's classifier is an affine function of `f`, full stop. That isn't a slogan, it's an algebraic fact I just reproduced numerically, and it pins down exactly what that method *can't* do, which I'll come back to.

The Bregman story also tells me something to hold onto: they tried cosine and found squared Euclidean clearly better, and the reason offered is precisely that cosine is *not* a Bregman divergence, so the mean-as-optimal-centre story collapses for it. The comparison rule and the way you summarise a class are coupled, and getting them to agree matters a lot.

Now here's the thing that keeps nagging me. In *all* of these — Matching, Prototypical, the older Siamese verification nets — every bit of learning lives in the *embedding* `f`. Once you're in embedding space, the actual comparison is a *fixed, closed-form, hand-chosen* function: cosine, or squared Euclidean, or — as I just verified for Prototypical — an affine classifier. The network learns where to put the points; a frozen formula decides who's near whom. And that affine fact, the very thing that made Prototypical look so clean a moment ago, cuts the other way when I think about what it *requires*: if the classifier is affine in `f`, it can only succeed when the embedding, all by itself, makes every novel class linearly separable — clustered around a mean, a straight line away from the others. Stare at that requirement for *unseen* classes. I'm asking one fixed embedding, trained on disjoint classes, to lay out *brand new* categories so cleanly that a mean and a hyperplane separate them. And the documented fact that swapping cosine for Euclidean swings accuracy violently, with the "right" choice not the obvious one, is consistent with the suspicion that no single closed-form element-wise comparison is right for all of these layouts. I want to know whether that suspicion is real or just rhetoric.

So let me try to actually exhibit a case an element-wise fixed metric can't solve, rather than wave at one. Take a 2D embedding, fix a query point `q`, and ask, for every possible *support* point `s` in the plane, "same class as `q` or not?" A Euclidean nearest-neighbour decision labels `s` matched when `‖s − q‖` is small, so the matched region is a disk around `q` — convex. Now suppose the true matched region is an *annulus* centred on `q`: same class when `s` is at radius between, say, 1 and 2 from `q`, different otherwise. Pick three test points: `s₁` at distance 0 (the centre), `s₂` at distance 1.5 (in the ring), `s₃` at distance 3 (outside). Ground truth: `s₁` mismatched, `s₂` matched, `s₃` mismatched. A distance-monotone rule must assign scores that are monotone in `‖s−q‖`, so it orders the three points `s₁ ≺ s₂ ≺ s₃` (or the reverse) by score — but the matched one, `s₂`, sits in the *middle* of that ordering, so no single threshold on distance can pick out `s₂` while rejecting both `s₁` and `s₃`. The annulus is not a sublevel set of any distance-to-`q`. So Euclidean-NN provably can't carve it. Can I rescue it while keeping the metric a metric — Mahalanobis, `(s−c)^T M (s−c)`? No: that replaces the disk with an ellipse, still a convex sublevel set, and the annulus is non-convex, so the same three-point argument kills it. What if I first push the embeddings through a couple of nonlinear layers and *then* do Mahalanobis? Now I can warp the plane, and in principle a warp could un-ring the annulus — but notice what I'm doing: I'm spending all the nonlinearity on the *embedding* and still forcing the final decision to be "ellipsoidal sublevel set in the warped space." If "same class" is a genuinely joint function of `(q, s)` — if which support points match depends on the query in a way that no per-point warp of a shared space captures — the convex-sublevel-set last step stays the choke point. Every repair I just tried kept the comparison a fixed parametric distance and poured capacity into the embedding; the one component I never let learn is the comparison itself, the function that eats two representations and says "same or different." It has no nonlinearity of its own.

That's the move I haven't tried: stop fixing the comparison and learn it too. Don't compute `d(f(x̂), c_k)` with a chosen formula; feed the two representations into a *network* and let it output the score. The metric becomes a learned, nonlinear function of the query and the class summary, trained jointly with the embedding end-to-end. A learned comparator with a hidden nonlinearity *can* represent the annulus — it can compute `‖s−q‖` internally and threshold it on both sides — so it isn't stuck inside convex sublevel sets the way every fixed metric above was. Now I'm no longer betting everything on the embedding making classes linearly separable; the comparator can itself bend, and the embedding and the comparator can split the work between them however the data wants.

But "feed two things into a network and get a relation out" — I've seen exactly that machinery, in a totally different problem. Santoro et al. built a relation module for reasoning about objects *within a single image*: given a set of object vectors `O = {o_1, …, o_n}`, they compute `RN(O) = f_φ( Σ_{i,j} g_θ(o_i, o_j) )`, where `g_θ` is one small shared network applied to *every pair* of objects, with the two objects simply concatenated as its input, the pairwise outputs *summed*, and `f_φ` mapping the pooled vector to an answer. That's a different task — relations among objects in one scene — but the *mechanism* is exactly the thing I need: a learned function that takes two representations, concatenated, and returns how they relate. And the two properties they leaned on transfer perfectly to my setting. One, they use a *single shared* `g_θ` for all pairs, which is data-efficient and stops it from overfitting any one particular pairing — and I have very little data, and I want the comparator to generalise to *unseen* class pairings, so a single shared comparison network applied to every (query, class) pair is exactly the inductive bias I want, not a separate comparator per class. Two, they *sum* over pairs to get an order-invariant aggregate over a set — and I'll need to aggregate over the `K` support examples of a class in an order-invariant way. So the relational-reasoning module hands me both the comparator and the pooling. Borrow it.

Concretely then, in the one-shot case: embed each support image `x_i` and the query `x_j` with a shared embedding module `f_varphi`, combine each (support, query) pair, and push the combination through a relation module `g_phi` that outputs a single scalar relation score,
`r_{i,j} = g_phi( C(f_varphi(x_i), f_varphi(x_j)) ),  i = 1, …, C`,
giving `C` scores per query, one per class. The combine operator `C(·,·)` — I'll just concatenate the two representations. That's the simplest thing, it's what the relational module did, and crucially it keeps *all* of both representations and lets `g_phi` learn whatever cross-terms it wants; a fixed metric like a difference `f(x_i) − f(x_j)` would already be throwing away information and baking in a comparison form, which is the exact mistake I'm trying to undo. Concatenation commits to nothing.

Now I have to decide *what* to concatenate, and this is where I should be careful rather than reflexively flattening to vectors. If the embedding produces a global feature vector per image and I concatenate two vectors and run an MLP, the relation module compares only the holistic vectors — which is already strictly more expressive than a fixed metric, so it's not wrong. But images have spatial structure, and a lot of "are these the same kind of thing" is about whether local parts correspond. If I keep the embedding output as a *feature map* — channels × height × width — and concatenate two maps along the channel axis, then the relation module can be *convolutional*, and it can compare local spatial structure, looking for corresponding parts, before it pools down to a score. That's a real reason to prefer maps over vectors here: the comparator gets to use convolution to find matching local patterns, which a vector-MLP comparator simply cannot see. So I'll have the embedding emit feature maps and concatenate in depth, doubling the channel count, and make the first part of the relation module convolutional.

That decision feeds straight back into how I design the embedding, and it's a non-obvious knob, so let me reason it out rather than copy a template. The standard few-shot embedding everyone uses for fair comparison is four convolutional blocks, each a 64-filter 3×3 conv, batch norm, ReLU, and a 2×2 max-pool — and four max-pools on an 84×84 image would shrink the spatial map down to almost nothing, basically a vector. But I just argued I *want* a spatial feature map left over so the relation module's convolutions have something to chew on. So I should *not* pool in every block. Put the max-pools in the first two blocks and drop them from the last two; the first two halvings cut the resolution and the cost, and then the last two convs deepen the representation while *preserving* the spatial map. That's a deliberate departure from the metric-based methods' embeddings, and it's forced by the choice to make the comparator convolutional.

Before I commit, I should actually run the spatial arithmetic through, because if the map collapses to 1×1 the whole "convolve over local structure" argument is empty, and if it stays large the relation module gets expensive — I need to know the real number. A valid 3×3 conv takes a side of length `n` to `n−2`; a 2×2 max-pool halves it (floor). Block pattern is `(pad 0, pool), (pad 0, pool), (pad 1, no pool), (pad 1, no pool)` — padding 1 on the last two so the `−2` is cancelled and they preserve the side. Omniglot at 28: block 1, `(28−2)=26`, pool `→13`; block 2, `(13−2)=11`, pool `→5`; block 3 with pad 1, `5+2−2=5`; block 4, `5`. So a 64×5×5 map. miniImageNet at 84: `(84−2)=82 →41`; `(41−2)=39 →19`; pad-1 blocks hold at `19`, `19`. So 64×19×19. Both survive with real spatial extent — good, the convolutional comparator has something to look at, and 5×5 / 19×19 are small enough not to blow up. Now push those through the relation module's two pooled conv blocks to see what the final flatten size is. On Omniglot I'll use padding 1 in the relation convs (otherwise 5 wouldn't survive two more `−2`-and-halve steps): `5 → (5+2−2)=5, pool →2 → (2+2−2)=2, pool →1`, so `64×1×1 = 64` values into the first FC. On miniImageNet, padding 0: `19 → 17, pool →8 → 6, pool →3`, so `64×3×3 = 576` values. Those flatten sizes — 64 for Omniglot, 576 for miniImageNet — are exactly what the FC layer's input dimension has to be, so this arithmetic isn't decorative; it's the number I have to wire in, and it's also why Omniglot needs padding 1 in the relation convs while miniImageNet can use padding 0.

Then the relation module `g_phi` itself: it receives the depth-concatenated pair, `2 × 64` channels of feature map, and has to boil it down to one number. Mirror the embedding's vocabulary — two convolutional blocks, each 3×3 conv with 64 filters, batch norm, ReLU, and now I *do* want 2×2 max-pools here, because this branch's job is to aggregate spatial evidence down to a decision, so pooling away the spatial dimensions is exactly right. After the two conv blocks the map is small (on miniImageNet it comes down to `64 × 3 × 3 = 576` values), flatten it, and pass it through two fully-connected layers to a scalar — one hidden layer of modest width (8 units) and then a single output. The hidden FC layer is what gives the comparator its final nonlinearity over the pooled convolutional evidence; without it the head would be linear again and I'd have quietly reintroduced a linear comparator after all the work to avoid one.

The output is a single scalar, and I want it to be a *relation score*, a "how much do these two match" quantity. I should pin it to a bounded, interpretable range rather than let it be an unbounded logit, because I'm about to compare it against targets that mean "match" and "no match," and an unbounded score has no natural zero and one. So put a sigmoid on the output: `r ∈ (0, 1)`, where 1 reads as "same class" and 0 as "different class." That gives every relation score a consistent meaning across episodes and classes, which matters because the comparator is shared and has to mean the same thing everywhere.

Now the K-shot case, `K > 1`. I have `K` support images per class and I want, still, exactly `C` scores per query — one per class, not `C·K` scores I'd then have to reconcile. So I need to fold each class's `K` support embeddings into a single class-level representation before the comparison. The relational module already told me how to aggregate a set in an order-invariant way: *sum*. So element-wise sum the `K` embedding feature maps of a class into one class feature map, then run the one-shot procedure on that pooled map. It keeps the score count at `C` and it's cheap. The order-invariance I should actually check rather than assume, since the support examples genuinely have no order and the model must not depend on one: embed two support images of a class, sum them, then reverse the order and sum again — the two pooled maps come out identical to the last bit (max absolute difference 0.0). Sum is commutative, so of course it is, but it's the property I'm relying on and now I've confirmed it concretely. I do pause on one consequence: summing rather than averaging makes the class map scale with `K`, so a 5-shot class map is five times larger in magnitude than a 1-shot one. For a fixed-shot evaluation that's a constant scale factor, and the following convolution, batch norm, and learned weights can absorb it when trained consistently at a fixed `K`. So I keep the sum: it is the order-invariant set-pooling rule I actually need, while averaging would be a separate `1/K` rescaling choice.

Putting the pieces together, the forward pass in an episode is: embed all support images, sum per class to get `C` class feature maps; embed the query; for each of the `C` classes, concatenate (class map, query map) along depth and run the relation module to get `r_{i,j}`; the `C` outputs are the query's scores, and at test time take the argmax — feed-forward, no fine-tuning, which was the whole point. Before I trust the shape bookkeeping I'll trace one tiny episode through with random weights — a 3-way 2-shot miniImageNet-sized task, six support images at 84×84, four queries. Embedding the support gives `(6, 64, 19, 19)` as the side arithmetic predicted; the per-class sum-pool over the two examples each collapses the 6 to 3 and yields class maps `(3, 64, 19, 19)`; broadcasting each of the 4 queries against the 3 class maps and concatenating in depth gives `4·3 = 12` pairs of `(128, 19, 19)`, i.e. a `(12, 128, 19, 19)` batch into the relation module; and the relation module returns `(12, 1)` which reshapes to scores `(4, 3)` — exactly `C` scores per query. The scores all land strictly inside `(0, 1)`, confirming the sigmoid is doing what I want, and an argmax over each row picks one of the 3 classes. So the whole pipeline type-checks end to end, with no silent reshape that scrambles which query is being compared to which class.

Last piece, and it's the one that looks oddest: the loss. The reflex is cross-entropy — it's a classification problem, the targets are class indices. But look at what the model actually emits: not a normalised distribution over classes, but `C` *independent* relation scores, each a sigmoid in `(0,1)` produced by the *shared* comparator looking at one (class, query) pair in isolation. The natural ground truth for a single relation score `r_{c,j}` is binary: it should be 1 if the query's label equals the label of class map `c`, and 0 otherwise — `1(y_c == y_j)`. So what I'm really doing, per pair, is *regressing a score toward a target in {0,1}*: drive the matched pair's score to 1 and every mismatched pair's score to 0. That's mean-squared error against the one-hot target,
`varphi, phi ← argmin Σ_{c=1}^C Σ_{j=1}^n ( r_{c,j} − 1(y_c == y_j) )²`.
Conceptually I'm predicting a *relation* — a real-valued similarity that should be 1 whenever the objects are the same class no matter how the images differ, and 0 otherwise — and predicting a real value to hit a real target is regression, even though the only targets I can auto-generate are the extreme values 0 and 1. MSE on a sigmoid output matches that framing cleanly: the output is a bounded relation magnitude, and I'm pulling it to one of the two ends. If I instead use ordinary multiclass cross-entropy, I change the object from independent relation scores into a normalized class distribution over the `C` scores. That is a different training semantics. The relation-score view wants the pairwise target itself, so MSE is the loss consistent with treating each (query, class) decision as an independent learned-relation regression. I'll keep it.

For the optimisation itself nothing exotic is needed: train everything end-to-end from scratch with random initialisation — embedding and relation module are mutually tuned, the embedding shaping representations the comparator can compare and the comparator adapting to the embedding — using Adam at learning rate `1e-3`, annealed (halved periodically) over the long episode schedule. From-scratch is fine because episodic training over many tasks is itself a strong, overfitting-resistant signal, and there's no need for a separately pretrained backbone on these benchmarks.

Let me trace the whole causal chain once more to be sure it hangs together. The pain is few-shot generalisation to unseen classes with no test-time fine-tuning. Episodic training makes train look like test, so the learned pieces are forced to be the transferable ones. The metric-based predecessors learn an embedding and then compare with a *fixed* metric — cosine, squared Euclidean, equivalently a linear classifier — which means they can only succeed when the embedding alone renders novel classes linearly/convexly separable, and the documented violent sensitivity to which fixed metric you pick is the tell that no fixed element-wise comparison is universally right. Lift the comparison into a learned nonlinear network — borrowing the shared-pairwise-`g`, concatenate-and-pool machinery from relational reasoning — so embedding and comparator share the work. Keep feature *maps* and make the comparator convolutional so it can compare local structure, which forces dropping max-pool from the last embedding blocks. Sum the `K` support maps per class for order-invariant set pooling and a fixed score count `C`. Sigmoid the output into a bounded relation score, and train it by regressing each (query, class) score to its binary match target with MSE — the loss consistent with independent learned relations. End-to-end, from scratch, Adam. Now the code.

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
    """Few-shot classifier: learned embedding + learned non-linear comparator, end-to-end."""

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


def train(method, optimizer, episode_sampler):
    for support_images, support_labels, query_images, query_labels in episode_sampler:
        optimizer.zero_grad()
        method.process_support_set(support_images, support_labels)   # sum support -> class maps
        scores = method(query_images)                                # learned relation scores
        loss = method.compute_loss(scores, query_labels)             # MSE to binary relation targets
        loss.backward()
        nn.utils.clip_grad_norm_(method.embedding.parameters(), 0.5)
        nn.utils.clip_grad_norm_(method.relation.parameters(), 0.5)
        optimizer.step()                                             # Adam, lr 1e-3, annealed
```
