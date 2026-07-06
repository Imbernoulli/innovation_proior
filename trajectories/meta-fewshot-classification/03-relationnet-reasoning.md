The prototype run confirmed the prediction *and* exposed the next wall in the same numbers. CUB jumped
exactly as I said it would — from the matching vote's 0.625 to 0.756, the single largest move of either
method, because collapsing each fine-grained class to one clean prototype and scoring by the Euclidean
partner of the mean removed precisely the per-point scatter that had no margin when all the birds look
alike. That is the spherical-Gaussian prior earning its keep where classes collide. But look at the
generic benchmarks: CIFAR-FS *fell* from 0.769 to 0.682, and miniImageNet *fell* from 0.674 to 0.649.
Let me read the three deltas together before I explain them: CUB +0.131, CIFAR −0.087, mini −0.025. Two
things jump out of that arithmetic. First, the geometric-mean task score barely moved —
(0.6487·0.6822·0.7563)^{1/3} ≈ 0.694 against matching's 0.687 — so on the headline number the prototype
is a wash, but that flat average hides a violent redistribution: it won one benchmark by thirteen points
and lost another by nine. Second, and this is the load-bearing observation, the deltas *split by sign*:
one benchmark up, two down. Some of the drop is the optimizer confound I flagged — matching ran at
LR_OVERRIDE 1e-3 while the prototypes ran at the scaffold default 1e-2 — but an optimizer change is a
monotone lever; a smaller or larger step tends to move all three benchmarks the *same* way, better or
worse together. It cannot by itself push CUB up thirteen points while pushing CIFAR down nine. A
sign-split across benchmarks is the fingerprint of the *classifier*, not the learning rate. The
prototype's rigid summary traded flexibility on already-separable classes for a low-variance class
entity; on CUB the trade won big, on CIFAR and mini it lost. And that is the tell I said to watch for:
where the *summary* is no longer the problem, the bottleneck has moved to the *fixed metric* comparing
query to summary.

Stare at what the prototype classifier actually is and the ceiling is written into it. Expanding the
squared distance, the query-only term cancels and what is left is *linear* in the embedding — the
prototype method is exactly a linear classifier in feature space, weights read off the means. So it can
only succeed to the extent the embedding *by itself* lays out brand-new classes as round,
linearly-separable blobs around their means. On fine-grained CUB the embedding can apparently do that;
on the broader CIFAR/mini layouts it cannot, and a straight line through the prototypes leaves accuracy
on the table. The violent benchmark-to-benchmark swing — CUB up 13 points, CIFAR down 9 — *is* the
symptom that no single fixed, closed-form comparison is right for all of these layouts. The metric is
the bottleneck, and its sensitivity to the data is the proof.

So the thing I kept fixed through both previous rungs — the comparison function — is what I have to
attack. In the matching vote it was cosine; in the prototype method it was squared Euclidean,
equivalently a linear classifier. In *both*, every bit of learning lived in the embedding, and the
actual comparison was a hand-chosen, closed-form function: the network learns where to put the points,
a frozen formula decides who is near whom. Let me make myself feel exactly where an element-wise fixed
metric fails, not just assert it. Picture the simplest hard case: a 2-D embedding, fix a query point,
and ask for every possible support point in the plane, "same class as the query or not?" If
matched/mismatched is decided by Euclidean nearness, the matched region is a disk around the query —
one convex blob, and the decision boundary between any two classes is a straight hyperplane (the
perpendicular bisector of the two prototypes). But suppose the true matched region is an annulus, or two
separated patches, or a curved band — anything non-convex. No Euclidean nearest-neighbour can carve that
out; nearest-neighbour only produces convex, distance-monotone regions, and no straight bisector can
separate an inner disk from the ring around it. Can I rescue it by learning a *better metric* while
keeping it a metric — a Mahalanobis distance? That just stretches and rotates the space; the matched
region is still an ellipsoidal blob, still convex, still bounded by a quadric that cannot fold into an
annulus. Push the embeddings through more non-linear layers *and then* do Mahalanobis? Now I can warp
the space more, but the *final* decision is still "ellipsoidal blob in the warped space," and if "same
class" is genuinely a complicated joint function of the query and the class summary, the blob-shaped
comparison at the end is still the choke point. Every fix I just tried poured more capacity into the
embedding or the space-warp while insisting the *last step* — the comparison — stay a fixed parametric
distance. That is the wall, and the prototype method's CIFAR drop is me hitting it. The comparison
itself has no learnable non-linearity in it.

So stop fixing the comparison and learn it too. Do not compute d(f(x̂), c_k) with a chosen formula; feed
the two representations into a *network* and let it output the score. The metric becomes a learned,
non-linear function of the query and the class summary, trained jointly with the embedding end-to-end.
Now I am no longer betting everything on the embedding making classes linearly separable — the
comparator can itself bend, and the embedding and comparator can split the work however the data wants.
That is precisely the freedom CIFAR and mini were asking for and the prototype's linear head could not
give them. But "feed two things into a network and get a relation out" — I have seen exactly that
machinery in a different problem, a relation module for reasoning about objects within a single image,
RN(O) = f_φ(Σ_{i,j} g_θ(o_i, o_j)): one small shared network g_θ applied to every concatenated pair of
objects, the pairwise outputs summed, then mapped to an answer. Different task, but the *mechanism* is
exactly what I need — a learned function that takes two representations, concatenated, and returns how
they relate — and two of its properties transfer perfectly. One, a *single shared* comparator across
all pairs is data-efficient and generalizes to unseen pairings: I have very little data and I need the
comparator to generalize to support sets of unseen classes, so one shared comparison network applied to
every (class, query) pair is exactly the inductive bias I want, not a separate comparator per class —
and a per-class comparator is a non-starter anyway, since the classes at test time never appeared in
training, so there would be no trained comparator to invoke. Two, *summing* over a set is
order-invariant — which I will need to fold each class's K support examples into one class
representation. The relational module hands me both the comparator and the pooling. Borrow it.

Concretely, in the one-shot case: I keep the scaffold's shared ResNet-12 as the embedding f_φ — there
is no parameter budget to add a second backbone, and from-scratch episodic training on this loop is the
only training I get — and for each class I concatenate its representation with the query's, then push
the pair through a learned relation module g_φ that outputs a single scalar relation score, giving N
scores per query, one per class. The combine operator is concatenation. That is the simplest thing, it
is what the relational module did, and it keeps *all* of both representations and lets g_φ learn
whatever cross-terms it wants; a fixed combine like a difference f(x_i) − f(x̂) would already be
throwing information away and baking in a comparison form — it presupposes the relevant quantity is a
displacement, exactly the linear-in-the-embedding assumption the prototype head made and lost CIFAR on.
Concatenation commits to nothing and lets the network decide whether the answer depends on the
difference, the product, or some gate of the two.

Now the choice that the harness actually forces, and it is where I have to depart from the textbook
relation network. The textbook version uses a small four-conv embedding and concatenates feature *maps*
in depth so the relation module can be *convolutional* and compare local spatial structure before
pooling to a score — and that matters, because "are these the same kind of thing" is often about
whether local parts correspond, which a vector-MLP comparator cannot see. But I do not get to choose a
four-conv embedding: the scaffold's backbone is a fixed ResNet-12, and the only knob I have is whether I
ask it for pooled 640-dim *vectors* (`use_pooling=True`) or 640-channel feature *maps*
(`use_pooling=False`). To make the comparator convolutional I take the maps. So `make_backbone(
use_pooling=False)` for both support and query, giving 640-channel feature maps, and I concatenate a
(class-map, query-map) pair along the channel axis into 2·640 = 1280 channels — the relation module's
first conv accepts that doubled depth. This is the literal edit: I do not build a fresh 64-channel
ConvNet embedding (a dedicated `EmbeddingModule`), I reuse the shared ResNet-12 as the
embedding and put *all* my new parameters into the relation module operating on its 640-channel maps.
That is also what keeps me inside the parameter budget, since the budget is set at ≈1.05× a ResNet-12
plus exactly such a relation module.

The relation module g_φ itself has to eat the 1280-channel concatenated pair and boil it to one number.
Mirror the embedding's vocabulary but make it a comparator: a conv block — Conv2d(1280, 640, 3, pad 1),
BatchNorm, ReLU — then pool the spatial map down, because this branch's job is to aggregate spatial
evidence toward a decision, so pooling away spatial dimensions is exactly right here (unlike in the
embedding). The ResNet-12 maps are small, so rather than a fixed 2×2 max-pool that assumes a particular
map size, I use `AdaptiveMaxPool2d((5,5))` after the first conv block and `AdaptiveMaxPool2d((1,1))`
after the second — adaptive pooling makes the module robust to whatever spatial size the backbone emits,
which is the safe choice when I am bolting onto a backbone I did not design. Let me actually work out the
spatial arithmetic to be sure the module is well-formed, because "adaptive" can quietly hide a mistake.
A ResNet-12 takes the 84×84 image through four stride-2 stages, 84→42→21→10→5, so the map it emits is
640×5×5. The first conv has pad 1 and kernel 3, which preserves 5×5; then `AdaptiveMaxPool2d((5,5))` on
a 5×5 map is an *identity* — it does nothing here, and that is the point of using it, because on a
backbone that emitted a larger map it would downsample instead of erroring. The second conv has pad 0
and kernel 3, which takes 5×5 down to 3×3; then `AdaptiveMaxPool2d((1,1))` collapses the 3×3 to a single
640-vector per (class, query) pair. Flatten it and pass through Linear(640, 8) → ReLU → Linear(8, 1).
The hidden FC of width 8 is the comparator's final non-linearity over the pooled convolutional evidence
— without it the head would be Linear(640,1), i.e. linear again, and I would have quietly reintroduced
the very linear comparator that lost CIFAR. The BatchNorms use momentum 1 (no running average), the
relation-network convention that fits the small per-episode batches. Let me also account for what this
costs, because "all new parameters live here" should be a number I can defend against the budget. The
first conv is 1280·640·3·3 ≈ 7.37M, the second 640·640·3·3 ≈ 3.69M, the two FCs a few thousand, so the
relation module is ≈11.1M parameters. That is the same order as the matching method's ~11.5M of FCE —
but where the FCE spent it on a recurrent per-episode *re-embedding* that had to be trained at a lowered
learning rate and did not transfer to novel-class support sets, this spends it on the *comparison
itself*, the one place both previous rungs left un-learnable, in a plain feedforward conv stack that is
stable at the default rate. Same parameter budget, aimed at the actual bottleneck.

The output is a single scalar, and I want it to be a *relation score*, "how much do these match,"
pinned to a bounded interpretable range rather than an unbounded logit — because I am about to compare
it against targets meaning "match" and "no match," and an unbounded score has no natural zero and one.
So a Sigmoid: r ∈ (0, 1), 1 reads as same class, 0 as different, a consistent meaning across episodes
and classes — which matters because the comparator is *shared* and must mean the same thing everywhere.

Now the K-shot case, which is the whole game here since every benchmark is 5-shot — and this is exactly
where I get to fix what the matching vote did badly and the prototype method did rigidly. I have K
support maps per class and I want still N scores per query, not N·K I would have to reconcile. So fold
each class's K embedded maps into one class-level map before the comparison. The relational module
already told me how to aggregate a set order-invariantly: sum. So element-wise *sum* the K support
feature maps of a class into one class feature map, then run the one-shot procedure on it. This is the
same "one entity per class" move the prototype method made — and notice the relation network keeps that
improvement over matching's scatter *and* adds the learned comparator the prototype lacked. I claimed
sum and mean are interchangeable here; let me check that rather than wave it through, because it is the
kind of thing that is true for a lazy reason and false for a subtle one. Summing K = 5 maps scales the
class map by 5 relative to the query map, and my first instinct was "BatchNorm will absorb it." But that
is wrong, and it is worth seeing why: the class map and the query map are concatenated *before* the
first conv, so only half the input channels are scaled by 5, the query half is not — the pre-activation
is 5·(W_class ∗ class) + (W_query ∗ query) + b, which is not a global multiple of anything, and BN,
which can only remove a per-channel affine of the *whole* activation, cannot rescale one summand of a
sum. So BN does not make sum and mean equal. What *does* is the learned weights: W_class is a free
parameter trained end-to-end, so gradient descent simply learns it a factor of 5 smaller and recovers
the mean's function exactly. Sum and mean are the same *function class*, differing only by a
reparameterization of W_class that the optimizer undoes — immaterial to what can be learned, a matter of
initialization scale, and BN's real job here is just to keep whatever scale results well-conditioned. So
I keep sum, the order-invariant aggregator the relational module prescribed, and stop worrying about the
constant. So I get the prototype's clean per-class summary in feature-map space, and on top of it a
non-linear learned comparison — the two upgrades the previous two rungs each had only one of.

Before committing I trace the shapes once end to end, because a five-dimensional expand-and-concatenate
is easy to get wrong on an axis. Support maps come in as 25×640×5×5; averaging within each of the 5
classes gives class maps 5×640×5×5. Query maps are 75×640×5×5. I broadcast the prototypes to
75×5×640×5×5 and the queries to 75×5×640×5×5, concatenate on the channel axis (dim 2) to 75×5×1280×5×5,
and flatten the first two axes to 375×1280×5×5 — 75 queries times 5 classes, 375 pairs. The relation
module maps 375×1280×5×5 → (conv,BN,ReLU,pool5) → 375×640×5×5 → (conv,BN,ReLU,pool1) → 375×640×1×1 →
flatten 375×640 → 375×8 → 375×1, and I view it back to 75×5. Output shape (n_query, n_way) = (75, 5),
exactly the contract, one relation score per (query, class). Good — no axis crossed.

Last piece, the loss, and it is the one that looks oddest until I see what the model emits. The reflex
is cross-entropy — it is classification, the targets are class indices, and that is what both previous
rungs used. But the model does not emit a normalized distribution over classes; it emits N *independent*
sigmoid relation scores, each produced by the shared comparator looking at one (class, query) pair in
isolation, with no softmax tying them together. The natural ground truth for a single relation score is
binary: 1 if the query's class equals that class, 0 otherwise. So per pair I am *regressing a score
toward a {0,1} target* — drive the matched pair to 1, every mismatched pair to 0 — which is mean-squared
error against the one-hot target. Let me check MSE-on-a-sigmoid actually trains toward those targets and
does not have a pathology. With r = σ(z), the per-pair loss (r − t)² has dL/dz = 2(r − t)·σ′(z) =
2(r − t)·r(1 − r). Away from the target the gradient is healthy; as r approaches 0 or 1 the r(1−r) factor
tapers it to zero — the classic "MSE on sigmoid saturates" concern — but here the targets t *are* 0 and
1, so the vanishing gradient sits exactly at the correct fixed points and only makes the final approach
gentle, not the training stuck. That is acceptable. If I insisted on cross-entropy instead I would have
to renormalize the N independent scores into a distribution — a softmax or extra head — which
re-couples the classes: each score's gradient would then depend on the others, reintroducing exactly the
fixed cross-class comparison structure I built independent relations to escape. So MSE is not sloppiness;
it is the loss consistent with treating each (query, class) decision as an independent learned-relation
regression. I override `compute_loss` to MSE against one-hot, where both previous rungs left it at
cross-entropy (the prototype) or NLL (the matching vote).

For optimization I can stay on the scaffold default: SGD@1e-2 from scratch — no LR_OVERRIDE. Unlike the
matching method's recurrences, the relation module is a feedforward conv stack and is stable under the
default rate; there is no depth-25 unrolled recurrence to blow up, so the reason I had to drop to 1e-3
in the matching rung simply does not apply here, and the loop's grad-norm clipping at 5.0 covers the
from-scratch training. The embedding and comparator tune each other, the embedding shaping maps the
comparator can compare and the comparator adapting to the embedding.

So the delta from the prototype rung is precise: I keep its one-clean-entity-per-class summary (now in
feature-map space, by summing maps), but I replace the *fixed linear* comparison — the thing that won
CUB and lost CIFAR/mini — with a learned non-linear convolutional comparator on the concatenated pair,
scored by sigmoid and trained by MSE. The full scaffold module is in the answer. Now the falsifiable
expectations against the prototype numbers. The whole reason to do this is the generic benchmarks where
the linear metric left accuracy on the table, so I expect **CIFAR-FS to recover and pass** the prototype
0.682 — the learned comparator can carve the non-convex matched regions the linear head could not — and
**miniImageNet to rise** past 0.649 for the same reason; concretely I would put CIFAR back over 0.78,
above even the matching vote's 0.769, and mini past 0.69. The risk and the test is **CUB**: the prototype
already posted a strong 0.756 there on a layout where a clean entity plus a simple metric was apparently
near-ideal, so the learned comparator's extra capacity might buy little on CUB and could even be *higher
variance* across seeds, because a flexible comparator with ~11M parameters trained from scratch on
twenty-five images per episode has more ways to land differently per seed than a parameter-free linear
head. So my concrete prediction: CIFAR up clearly (past 0.78), mini up (past 0.69), CUB roughly holding
around 0.756 but with visibly larger seed-to-seed spread than the prototype's tight 0.753–0.762. If
CIFAR and mini both rise while CUB holds, that confirms the diagnosis that the fixed metric — not the
summary — was the bottleneck on the generic benchmarks, and the learned comparison is the right and
final lever among the metric methods. If even the learned comparator cannot move the generic benchmarks
much, the remaining lever is no longer the metric at all but *adapting the embedding itself to each task*
— making the per-class summaries task-aware rather than comparing fixed embeddings — which is where I
would look next.
