The prototype run confirmed the prediction *and* exposed the next wall in the same numbers. CUB jumped
exactly as I said it would — from the matching vote's 0.625 to 0.756, the single largest move of either
method, because collapsing each fine-grained class to one clean prototype and scoring by the Euclidean
partner of the mean removed precisely the per-point scatter that had no margin when all the birds look
alike. That is the spherical-Gaussian prior earning its keep where classes collide. But look at the
generic benchmarks: CIFAR-FS *fell* from 0.769 to 0.682, and miniImageNet *fell* from 0.674 to 0.649.
Some of that is the optimizer confound I flagged — the matching method ran at LR_OVERRIDE 1e-3 while the
prototypes ran at the scaffold default 1e-2 — but the shape of the result is too clean to be only that.
The prototype's rigid summary traded flexibility on already-separable classes for a low-variance class
entity; on CUB the trade won big, on CIFAR and mini it lost. And that is the tell I said to watch for:
where the *summary* is no longer the problem, the bottleneck has moved to the *fixed metric* comparing
query to summary. Stare at what the prototype classifier actually is and the ceiling is written into
it. Expanding the squared distance, the query-only term cancels and what is left is *linear* in the
embedding — the prototype method is exactly a linear classifier in feature space, weights read off the
means. So it can only succeed to the extent the embedding *by itself* lays out brand-new classes as
round, linearly-separable blobs around their means. On fine-grained CUB the embedding can apparently do
that; on the broader CIFAR/mini layouts it cannot, and a straight line through the prototypes leaves
accuracy on the table. The violent benchmark-to-benchmark swing — CUB up 13 points, CIFAR down 9 — *is*
the symptom that no single fixed, closed-form comparison is right for all of these layouts. The metric
is the bottleneck, and its sensitivity to the data is the proof.

So the thing I kept fixed through both previous rungs — the comparison function — is what I have to
attack. In the matching vote it was cosine; in the prototype method it was squared Euclidean,
equivalently a linear classifier. In *both*, every bit of learning lived in the embedding, and the
actual comparison was a hand-chosen, closed-form function: the network learns where to put the points,
a frozen formula decides who is near whom. Let me make myself feel exactly where an element-wise fixed
metric fails, not just assert it. Picture the simplest hard case: a 2-D embedding, fix a query point,
and ask for every possible support point in the plane, "same class as the query or not?" If
matched/mismatched is decided by Euclidean nearness, the matched region is a disk around the query —
one convex blob. But suppose the true matched region is an annulus, or two separated patches, or a
curved band — anything non-convex. No Euclidean nearest-neighbour can carve that out; nearest-neighbour
only produces convex, distance-monotone regions. Can I rescue it by learning a *better metric* while
keeping it a metric — a Mahalanobis distance? That just stretches and rotates the space; the matched
region is still an ellipsoidal blob, still convex. Push the embeddings through more non-linear layers
*and then* do Mahalanobis? Now I can warp the space more, but the *final* decision is still "ellipsoidal
blob in the warped space," and if "same class" is genuinely a complicated joint function of the query
and the class summary, the blob-shaped comparison at the end is still the choke point. Every fix I just
tried poured more capacity into the embedding or the space-warp while insisting the *last step* — the
comparison — stay a fixed parametric distance. That is the wall, and the prototype method's CIFAR drop
is me hitting it. The comparison itself has no learnable non-linearity in it.

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
every (class, query) pair is exactly the inductive bias I want, not a separate comparator per class.
Two, *summing* over a set is order-invariant — which I will need to fold each class's K support
examples into one class representation. The relational module hands me both the comparator and the
pooling. Borrow it.

Concretely, in the one-shot case: I keep the scaffold's shared ResNet-12 as the embedding f_φ — there
is no parameter budget to add a second backbone, and from-scratch episodic training on this loop is the
only training I get — and for each class I concatenate its representation with the query's, then push
the pair through a learned relation module g_φ that outputs a single scalar relation score, giving N
scores per query, one per class. The combine operator is concatenation. That is the simplest thing, it
is what the relational module did, and it keeps *all* of both representations and lets g_φ learn
whatever cross-terms it wants; a fixed combine like a difference f(x_i) − f(x̂) would already be
throwing information away and baking in a comparison form — exactly the mistake I am trying to undo
after watching the linear prototype head lose CIFAR. Concatenation commits to nothing.

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
which is the safe choice when I am bolting onto a backbone I did not design. Two conv blocks
(640→640→640 channels) with those two adaptive pools collapse the pair to a single 640-vector per
(class, query); flatten it and pass through Linear(640, 8) → ReLU → Linear(8, 1). The hidden FC of
width 8 is the comparator's final non-linearity over the pooled convolutional evidence — without it the
head would be linear again and I would have quietly reintroduced the very linear comparator that lost
CIFAR. The BatchNorms use momentum 1 (no running average), the relation-network convention that fits
the small per-episode batches.

The output is a single scalar, and I want it to be a *relation score*, "how much do these match,"
pinned to a bounded interpretable range rather than an unbounded logit — because I am about to compare
it against targets meaning "match" and "no match," and an unbounded score has no natural zero and one.
So a Sigmoid: r ∈ (0, 1), 1 reads as same class, 0 as different, a consistent meaning across episodes
and classes — which matters because the comparator is *shared* and must mean the same thing everywhere.

Now the K-shot case, which is the whole game here since every benchmark is 5-shot — and this is exactly
where I get to fix what the matching vote did badly and the prototype method did rigidly. I have K
support maps per class and I want still N scores per query, not N·K I would have to reconcile. So fold
each class's K embedded maps into one class-level map before the comparison. The relational module
already told me how to aggregate a set order-invariantly: sum. So element-wise *sum* (equivalently mean,
up to a constant the BatchNorm and learned weights absorb) the K support feature maps of a class into
one class feature map, then run the one-shot procedure on it. This is the same "one entity per class"
move the prototype method made — and notice the relation network keeps that improvement over matching's
scatter *and* adds the learned comparator the prototype lacked. A class map scaling with K is harmless
because the very next op is conv-then-BatchNorm, which rescales away a global magnitude factor. So I get
the prototype's clean per-class summary in feature-map space, and on top of it a non-linear learned
comparison — the two upgrades the previous two rungs each had only one of.

Last piece, the loss, and it is the one that looks oddest until I see what the model emits. The reflex
is cross-entropy — it is classification, the targets are class indices, and that is what both previous
rungs used. But the model does not emit a normalized distribution over classes; it emits N *independent*
sigmoid relation scores, each produced by the shared comparator looking at one (class, query) pair in
isolation. The natural ground truth for a single relation score is binary: 1 if the query's class equals
that class, 0 otherwise. So per pair I am *regressing a score toward a {0,1} target* — drive the matched
pair to 1, every mismatched pair to 0 — which is mean-squared error against the one-hot target. MSE on a
sigmoid output matches that framing: a bounded relation magnitude pulled to one of two ends. If I
insisted on cross-entropy I would have to renormalize the N independent scores into a distribution — a
softmax or extra head — reintroducing a fixed cross-class comparison and partly undoing the
"each pair is an independent learned relation" structure I built to escape the fixed metric. So MSE is
not sloppiness; it is the loss consistent with treating each (query, class) decision as an independent
learned-relation regression. I override `compute_loss` to MSE against one-hot, where both previous rungs
left it at cross-entropy (the prototype) or NLL (the matching vote).

For optimization I can stay on the scaffold default: SGD@1e-2 from scratch — no LR_OVERRIDE. Unlike the
matching method's recurrences, the relation module is a feedforward conv stack and is stable under the
default rate, and the loop's grad-norm clipping at 5.0 covers the from-scratch training; the embedding
and comparator tune each other, the embedding shaping maps the comparator can compare and the comparator
adapting to the embedding.

So the delta from the prototype rung is precise: I keep its one-clean-entity-per-class summary (now in
feature-map space, by summing maps), but I replace the *fixed linear* comparison — the thing that won
CUB and lost CIFAR/mini — with a learned non-linear convolutional comparator on the concatenated pair,
scored by sigmoid and trained by MSE. The full scaffold module is in the answer. Now the falsifiable
expectations against the prototype numbers. The whole reason to do this is the generic benchmarks where
the linear metric left accuracy on the table, so I expect **CIFAR-FS to recover and pass** the prototype
0.682 — the learned comparator can carve the non-convex matched regions the linear head could not — and
**miniImageNet to rise** past 0.649 for the same reason. The risk and the test is **CUB**: the prototype
already posted a strong 0.756 there on a layout where a clean entity plus a simple metric was apparently
near-ideal, so the learned comparator's extra capacity might buy little on CUB and could even be *higher
variance* across seeds, because a flexible comparator trained from scratch on twenty-five images per
episode has more ways to land differently per seed than a parameter-free linear head. So my concrete
prediction: CIFAR up clearly (past 0.78), mini up (past 0.69), CUB roughly holding around 0.756 but with
visibly larger seed-to-seed spread than the prototype's tight 0.753–0.762. If CIFAR and mini both rise
while CUB holds, that confirms the diagnosis that the fixed metric — not the summary — was the bottleneck
on the generic benchmarks, and the learned comparison is the right and final lever among the metric
methods. If even the learned comparator cannot move the generic benchmarks much, the remaining lever is
no longer the metric at all but *adapting the embedding itself to each task* — making the per-class
summaries task-aware rather than comparing fixed embeddings — which is where I would look next.
