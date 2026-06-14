Let me start from what actually hurts. I have a `C`-way `K`-shot problem: a handful of labelled images per class — one, maybe five — for classes I have never seen during training, and I have to label a query among those `C` classes in one shot, ideally without touching the weights at test time. The brute-force move, fine-tune a deep net on the `K·C` support images, is dead on arrival: a network with a million parameters has essentially no signal in five images per class, and augmentation and weight decay only slow the overfitting, they don't make it go away. So fine-tuning is out, or at least it can't be the core. What I do have is a big training set of *other* classes, disjoint from the test classes. The whole game is to squeeze out of those training classes some reusable machinery — a way to embed an image, a way to summarise a class from its few examples, and a way to decide which class a query belongs to — that transfers, feed-forward, to classes it was never trained on.

The one structural idea I trust here is episodic training, the way Matching Networks set it up. Don't train on a flat classification task and hope it transfers; instead, on every iteration, *manufacture a fake few-shot task* out of the training classes — sample `C` classes, draw `K` labelled examples each as a support set `S = {(x_i, y_i)}`, draw a disjoint query set `Q = {(x_j, y_j)}` from the same `C` classes, and score the model on classifying `Q` given `S`. Because every training episode is itself a small `C`-way `K`-shot problem with random class identities, whatever the model learns is under constant pressure to be the part that works *across* class identities — the transferable part — not a memorised mapping from pixels to a fixed label set. That principle I'm keeping. It's the substrate. The open question is what happens *inside* an episode: given the support embeddings and a query embedding, how do I produce the `C` scores?

So let me lay out exactly how the methods I know fill that slot, because the answer is going to come from seeing precisely where they stall. Matching Networks: embed everything, then label the query as an attention-weighted vote over the support labels, `ŷ = Σ_i a(x̂, x_i) y_i`, where the attention is a softmax over the *cosine similarity* of the embeddings, `a(x̂, x_i) ∝ exp(c(f(x̂), g(x_i)))`. It's a soft nearest-neighbour in embedding space. Prototypical Networks tightens this: instead of voting over every support point, collapse each class to one prototype `c_k`, the mean of its embedded support examples, `c_k = (1/K) Σ_{i: y_i = k} f(x_i)`, and classify by a softmax over *negative squared Euclidean distance*, `p(y=k|x) ∝ exp(−‖f(x) − c_k‖²)`. And Snell et al. have a genuinely clean reason for those two choices being matched. Squared Euclidean distance is a Bregman divergence, and a known fact about Bregman divergences is that the point minimising total divergence to a set is the *mean* of the set — so taking the prototype to be the class mean isn't arbitrary, it's the optimal cluster centre *for that distance*. And if I expand the distance, `‖f − c_k‖² = ‖f‖² − 2 c_k^T f + ‖c_k‖²`, the `‖f‖²` term is identical across all classes `k`, so it falls out of the softmax, and what's left, `2 c_k^T f − ‖c_k‖²`, is *linear* in `f` — the model is literally a linear classifier in embedding space, with weight `2c_k` and bias `−‖c_k‖²`. That's elegant. It also tells me something I should hold onto: they tried cosine and found squared Euclidean clearly better, and the reason offered is precisely that cosine is *not* a Bregman divergence, so the mean-as-optimal-centre story collapses for it. The comparison rule and the way you summarise a class are coupled, and getting them to agree matters a lot.

Now here's the thing that keeps nagging me as I write that down. In *all* of these — Matching, Prototypical, the older Siamese verification nets — every bit of learning lives in the *embedding* `f`. Once you're in embedding space, the actual comparison is a *fixed, closed-form, hand-chosen* function: cosine, or squared Euclidean, or a linear classifier. The network learns where to put the points; a frozen formula decides who's near whom. And the Prototypical analysis, the very thing that makes it elegant, is also a confession of its ceiling: squared-Euclidean-softmax *is* a linear classifier in embedding space. So the method can only work to the extent that the embedding, all by itself, manages to make every novel class linearly separable, clustered around a mean like a little spherical Gaussian blob. Stare at that requirement for unseen classes. I'm asking one fixed embedding, trained on disjoint classes, to lay out *brand new* categories so cleanly that a mean and a straight line separate them. The whole reason the choice of fixed metric was reported to swing accuracy so violently — cosine here, Euclidean there, and the "right" one not the obvious one — is that no single closed-form element-wise comparison is right for all of these layouts. The fixed metric is the bottleneck, and the sensitivity to which metric you pick is the symptom telling me so.

Let me try to make myself feel where exactly an element-wise fixed metric fails, not just assert it. Picture the simplest hard case: a 2D embedding, fix a query point, and ask, for every possible *support* point in the plane, "is this support point the same class as the query or not?" If matched/mismatched were decided by Euclidean nearness, the "matched" region would be a disk around the query — a single convex blob. But suppose the true matched region is, say, an annulus, or two separated patches, or some curved band — anything non-convex. Then no choice of Euclidean nearest-neighbour can carve it out, because nearest-neighbour can only produce convex-ish, distance-monotone decision regions. Fine, can I rescue it by learning a better metric *while keeping the metric a metric* — say a Mahalanobis distance, `(f−c)^T M (f−c)`? That just stretches and rotates the space; the matched region is still an ellipsoidal blob. Still convex. What if I also push the embeddings through another couple of nonlinear layers first and *then* do Mahalanobis? Now I can warp the space more, but the *final* decision is still "ellipsoidal blob in the warped space," and if the relationship between query and support that defines "same class" is genuinely a complicated joint function of the two, the blob-shaped comparison at the end is still the choke point. I keep adding capacity to the *embedding* and to the *space-warp*, but I keep insisting that the last step — the comparison — be a fixed parametric distance. That's the wall. The comparison itself, the function that eats two representations and says "same or different," has no learnable nonlinearity in it. Wall.

Every fix I just tried kept the comparison a fixed parametric distance and poured more capacity into the embedding or the space-warp. So stop fixing the comparison and learn it too: don't compute `d(f(x̂), c_k)` with a chosen formula; feed the two representations into a *network* and let it output the score. The metric becomes a learned, nonlinear function of the query and the class summary, trained jointly with the embedding end-to-end. Now I'm no longer betting everything on the embedding making classes linearly separable; the comparator can itself bend, and the embedding and the comparator can split the work between them however the data wants.

But "feed two things into a network and get a relation out" — I've seen exactly that machinery, in a totally different problem. Santoro et al. built a relation module for reasoning about objects *within a single image*: given a set of object vectors `O = {o_1, …, o_n}`, they compute `RN(O) = f_φ( Σ_{i,j} g_θ(o_i, o_j) )`, where `g_θ` is one small shared network applied to *every pair* of objects, with the two objects simply concatenated as its input, the pairwise outputs *summed*, and `f_φ` mapping the pooled vector to an answer. That's a different task — relations among objects in one scene — but the *mechanism* is exactly the thing I need: a learned function that takes two representations, concatenated, and returns how they relate. And the two properties they leaned on transfer perfectly to my setting. One, they use a *single shared* `g_θ` for all pairs, which is data-efficient and stops it from overfitting any one particular pairing — and I have very little data, and I want the comparator to generalise to *unseen* class pairings, so a single shared comparison network applied to every (query, class) pair is exactly the inductive bias I want, not a separate comparator per class. Two, they *sum* over pairs to get an order-invariant aggregate over a set — and I'll need to aggregate over the `K` support examples of a class in an order-invariant way. So the relational-reasoning module hands me both the comparator and the pooling. Borrow it.

Concretely then, in the one-shot case: embed each support image `x_i` and the query `x_j` with a shared embedding module `f_φ`, combine each (support, query) pair, and push the combination through a relation module `g_φ` that outputs a single scalar relation score,
`r_{i,j} = g_φ( C(f_φ(x_i), f_φ(x_j)) ),  i = 1, …, C`,
giving `C` scores per query, one per class. The combine operator `C(·,·)` — I'll just concatenate the two representations. That's the simplest thing, it's what the relational module did, and crucially it keeps *all* of both representations and lets `g_φ` learn whatever cross-terms it wants; a fixed metric like a difference `f(x_i) − f(x_j)` would already be throwing away information and baking in a comparison form, which is the exact mistake I'm trying to undo. Concatenation commits to nothing.

Now I have to decide *what* to concatenate, and this is where I should be careful rather than reflexively flattening to vectors. If the embedding produces a global feature vector per image and I concatenate two vectors and run an MLP, the relation module compares only the holistic vectors — which is already strictly more expressive than a fixed metric, so it's not wrong. But images have spatial structure, and a lot of "are these the same kind of thing" is about whether local parts correspond. If I keep the embedding output as a *feature map* — channels × height × width — and concatenate two maps along the channel axis, then the relation module can be *convolutional*, and it can compare local spatial structure, looking for corresponding parts, before it pools down to a score. That's a real reason to prefer maps over vectors here: the comparator gets to use convolution to find matching local patterns, which a vector-MLP comparator simply cannot see. So I'll have the embedding emit feature maps and concatenate in depth, doubling the channel count, and make the first part of the relation module convolutional.

That decision feeds straight back into how I design the embedding, and it's a non-obvious knob, so let me reason it out rather than copy a template. The standard few-shot embedding everyone uses for fair comparison is four convolutional blocks, each a 64-filter 3×3 conv, batch norm, ReLU, and a 2×2 max-pool — and four max-pools on an 84×84 image would shrink the spatial map down to almost nothing, basically a vector. But I just argued I *want* a spatial feature map left over so the relation module's convolutions have something to chew on. So I should *not* pool in every block. Put the max-pools in the first two blocks and drop them from the last two; the first two halvings cut the resolution and the cost, and then the last two convs deepen the representation while *preserving* the spatial map. That's a deliberate departure from the metric-based methods' embeddings, and it's forced by the choice to make the comparator convolutional. On Omniglot's tiny 28×28 inputs the map collapses to essentially a single 64-d cell anyway (`H = 64`); on 84×84 miniImageNet I'm left with a 64-channel map of a few cells on a side.

Then the relation module `g_φ` itself: it receives the depth-concatenated pair, `2 × 64` channels of feature map, and has to boil it down to one number. Mirror the embedding's vocabulary — two convolutional blocks, each 3×3 conv with 64 filters, batch norm, ReLU, and now I *do* want 2×2 max-pools here, because this branch's job is to aggregate spatial evidence down to a decision, so pooling away the spatial dimensions is exactly right. After the two conv blocks the map is small (on miniImageNet it comes down to `64 × 3 × 3 = 576` values), flatten it, and pass it through two fully-connected layers to a scalar — one hidden layer of modest width (8 units) and then a single output. The hidden FC layer is what gives the comparator its final nonlinearity over the pooled convolutional evidence; without it the head would be linear again and I'd have quietly reintroduced a linear comparator after all the work to avoid one.

The output is a single scalar, and I want it to be a *relation score*, a "how much do these two match" quantity. I should pin it to a bounded, interpretable range rather than let it be an unbounded logit, because I'm about to compare it against targets that mean "match" and "no match," and an unbounded score has no natural zero and one. So put a sigmoid on the output: `r ∈ (0, 1)`, where 1 reads as "same class" and 0 as "different class." That gives every relation score a consistent meaning across episodes and classes, which matters because the comparator is shared and has to mean the same thing everywhere.

Now the K-shot case, `K > 1`. I have `K` support images per class and I want, still, exactly `C` scores per query — one per class, not `C·K` scores I'd then have to reconcile. So I need to fold each class's `K` support embeddings into a single class-level representation before the comparison. The relational module already told me how to aggregate a set in an order-invariant way: *sum*. So element-wise sum the `K` embedding feature maps of a class into one class feature map, then run the one-shot procedure on that pooled map. Order-invariant (the support examples have no meaningful order), it keeps the score count at `C`, and it's cheap. I briefly worry that summing rather than averaging makes the class map scale with `K`, so a 5-shot class map is five times larger in magnitude than a 1-shot one — but the very next thing the relation module does is a conv followed by batch norm, and batch norm rescales away a global magnitude factor, and beyond that the learned weights can absorb a constant; averaging instead of summing would differ only by the factor `1/K`, which is exactly that absorbable constant. So sum is fine, and it's the natural set-pooling. (Mean is the equivalent choice up to that constant, and is the more obvious "prototype" reading — a class mean in feature-map space — so either is defensible.)

Putting the pieces together, the forward pass in an episode is: embed all support images, sum per class to get `C` class feature maps; embed the query; for each of the `C` classes, concatenate (class map, query map) along depth and run the relation module to get `r_{i,j}`; the `C` outputs are the query's scores, and at test time I just take the argmax — feed-forward, no fine-tuning, which was the whole point.

Last piece, and it's the one that looks oddest: the loss. The reflex is cross-entropy — it's a classification problem, the targets are class indices. But look at what the model actually emits: not a normalised distribution over classes, but `C` *independent* relation scores, each a sigmoid in `(0,1)` produced by the *shared* comparator looking at one (class, query) pair in isolation. The natural ground truth for a single relation score `r_{i,j}` is binary: it should be 1 if the query's class equals class `i`, and 0 otherwise — `1(y_i == y_j)`. So what I'm really doing, per pair, is *regressing a score toward a target in {0,1}*: drive the matched pair's score to 1 and every mismatched pair's score to 0. That's mean-squared error against the one-hot target,
`φ, ϕ ← argmin Σ_i Σ_j ( r_{i,j} − 1(y_i == y_j) )²`.
Conceptually I'm predicting a *relation* — a real-valued similarity that should be 1 whenever the objects are the same class no matter how the images differ, and 0 otherwise — and predicting a real value to hit a real target is regression, even though the only targets I can auto-generate are the extreme values 0 and 1. MSE on a sigmoid output matches that framing cleanly: the output is a bounded relation magnitude, and I'm pulling it to one of the two ends. If I instead insisted on cross-entropy I'd have to renormalise the `C` independent scores into a distribution — bolt a softmax (or an extra little MLP head) on top — which reintroduces a fixed normalising comparison across classes and partly undoes the "each pair is an independent learned relation" structure I built. So MSE isn't sloppiness; it's the loss that's consistent with treating each (query, class) decision as an independent learned-relation regression. I'll keep it.

For the optimisation itself nothing exotic is needed: train everything end-to-end from scratch with random initialisation — embedding and relation module are mutually tuned, the embedding shaping representations the comparator can compare and the comparator adapting to the embedding — using Adam at learning rate `1e-3`, annealed (halved periodically) over the long episode schedule. From-scratch is fine because episodic training over many tasks is itself a strong, overfitting-resistant signal, and there's no need for a separately pretrained backbone on these benchmarks.

Let me trace the whole causal chain once more to be sure it hangs together. The pain is few-shot generalisation to unseen classes with no test-time fine-tuning. Episodic training makes train look like test, so the learned pieces are forced to be the transferable ones. The metric-based predecessors learn an embedding and then compare with a *fixed* metric — cosine, squared Euclidean, equivalently a linear classifier — which means they can only succeed when the embedding alone renders novel classes linearly/convexly separable, and the documented violent sensitivity to which fixed metric you pick is the tell that no fixed element-wise comparison is universally right. Lift the comparison into a learned nonlinear network — borrowing the shared-pairwise-`g`, concatenate-and-pool machinery from relational reasoning — so embedding and comparator share the work. Keep feature *maps* and make the comparator convolutional so it can compare local structure, which forces dropping max-pool from the last embedding blocks. Sum the `K` support maps per class for order-invariant set pooling and a fixed score count `C`. Sigmoid the output into a bounded relation score, and train it by regressing each (query, class) score to its binary match target with MSE — the loss consistent with independent learned relations. End-to-end, from scratch, Adam. Now the code.

```python
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def conv_block(in_channels: int, out_channels: int, pool: bool) -> nn.Module:
    # 3x3 conv, BN, ReLU; optional 2x2 max-pool. The shared vocabulary of both modules.
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels, momentum=1, affine=True),
        nn.ReLU(),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class EmbeddingModule(nn.Module):
    # Four conv blocks. Pool in the FIRST TWO only, so a spatial feature MAP survives for
    # the convolutional relation module to compare (last two blocks keep resolution).
    def __init__(self, feature_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            conv_block(3, feature_dim, pool=True),
            conv_block(feature_dim, feature_dim, pool=True),
            conv_block(feature_dim, feature_dim, pool=False),
            conv_block(feature_dim, feature_dim, pool=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)                       # (n, C=feature_dim, H, W) feature map


class RelationModule(nn.Module):
    # The LEARNED comparator: eats a depth-concatenated (class, query) pair (2*C channels),
    # convolves+pools to aggregate spatial evidence, then an FC nonlinearity -> scalar in (0,1).
    def __init__(self, feature_dim: int = 64, hidden_size: int = 8):
        super().__init__()
        self.conv = nn.Sequential(
            conv_block(feature_dim * 2, feature_dim, pool=True),   # compare the pair...
            conv_block(feature_dim, feature_dim, pool=True),       # ...and pool it down
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(hidden_size),          # hidden FC = the comparator's final nonlinearity
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),                        # bounded relation score: 1=same, 0=different
        )

    def forward(self, pair: Tensor) -> Tensor:
        return self.fc(self.conv(pair))          # (n_pairs, 1)


class RelationNetwork(nn.Module):
    """Few-shot classifier: learned embedding + learned non-linear comparator, end-to-end."""

    def __init__(self, feature_dim: int = 64, hidden_size: int = 8):
        super().__init__()
        self.feature_dim = feature_dim
        self.embedding = EmbeddingModule(feature_dim)
        self.relation = RelationModule(feature_dim, hidden_size)
        self.class_maps = None                   # set by process_support_set

    def process_support_set(self, support_images: Tensor, support_labels: Tensor):
        # Embed support, then SUM the K maps per class -> one class-level feature map.
        feats = self.embedding(support_images)               # (K*C, F, H, W)
        n_way = int(support_labels.max().item()) + 1
        self.class_maps = torch.stack(
            [feats[support_labels == c].sum(dim=0) for c in range(n_way)]
        )                                                    # (C, F, H, W)

    def forward(self, query_images: Tensor) -> Tensor:
        q = self.embedding(query_images)                     # (n_query, F, H, W)
        n_query, n_class = q.shape[0], self.class_maps.shape[0]
        # Every (query, class) pair, concatenated in depth -> (n_query*n_class, 2F, H, W).
        cls = self.class_maps.unsqueeze(0).expand(n_query, -1, -1, -1, -1)
        qry = q.unsqueeze(1).expand(-1, n_class, -1, -1, -1)
        pairs = torch.cat((cls, qry), dim=2).view(
            -1, 2 * self.feature_dim, *q.shape[2:]
        )
        scores = self.relation(pairs).view(n_query, n_class)  # C relation scores per query
        return scores

    def compute_loss(self, scores: Tensor, query_labels: Tensor) -> Tensor:
        # Each (query, class) score is regressed to its binary match target: MSE, not CE.
        one_hot = F.one_hot(query_labels, num_classes=scores.shape[1]).float()
        return F.mse_loss(scores, one_hot)


def train(method, optimizer, episode_sampler):
    for support_images, support_labels, query_images, query_labels in episode_sampler:
        optimizer.zero_grad()
        method.process_support_set(support_images, support_labels)   # sum support -> class maps
        scores = method(query_images)                                # learned relation scores
        loss = method.compute_loss(scores, query_labels)             # MSE to one-hot
        loss.backward()
        nn.utils.clip_grad_norm_(method.parameters(), 0.5)           # stabilise from-scratch training
        optimizer.step()                                             # Adam, lr 1e-3, annealed
```
