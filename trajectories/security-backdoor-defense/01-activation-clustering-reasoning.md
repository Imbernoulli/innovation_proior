I start where the default fill leaves off — it scores by max softmax confidence, using no feature
geometry at all, and the one thing I actually know about the poison is structural, not confidence-based.
The poison is, by construction, a *sub-population*: the attacker took some fraction of one class's worth
of images, stamped them with a shared trigger, relabeled them all to the target, and dropped them into
that class. So within the target class the training data is not homogeneous — it is a large pile of
genuine target-class images plus a smaller pile of trigger-bearing intruders. Confidence does not see
this: the model fits the clean target-class images confidently too, so "high confidence on the label"
ranks poison no better than chance, and the default baseline throws away the only handle I have. If I
could *see* the smaller pile in the representation I would be done; the question is where to look so the
pile is visible, and how to turn membership in it into the per-sample suspicion score the harness
consumes.

Size the job first, because that tells me how good my ranking has to be. The harness removes the top
`1.5*eps` fraction and retrains from scratch — it never asks for a hard poison/clean label, only a
*ranking*. On resnet20-cifar10-badnets, `eps` is 5% of 50,000 CIFAR-10 images, so the injected poison is
about 2,500 and the harness removes `1.5 * 0.05 * 50,000 = 3,750`. On vgg16bn-cifar100-blend, poison ≈
500, budget = 750. On mobilenetv2-fmnist-badnets, `eps` is 8% of 60,000, poison ≈ 4,800, budget = 7,200.
In every setting the budget is exactly `1.5x` the poison, so a *perfect* ranking removes all the poison
and still has room for a slab of clean data. The budget is never the binding constraint — ranking
quality is: float essentially all the poison above the clean bulk and `poison_recall` goes to ~1 while
the over-removal does the rest; rank no better than chance and I discard `1.5*eps` of mostly-clean data
with the trigger shortcut untouched. The same arithmetic pins the minority fractions I lean on later:
2,500 poison in a CIFAR-10 target class of 5,000 genuine images is about 33%, 4,800 in a FashionMNIST
class of 6,000 is about 44% — both comfortably below half, which the poison has to be, because if it were
the majority the model's clean accuracy *on that class* would already have collapsed and the attack would
be self-defeating.

The naive place to look is the raw inputs — take all images labeled target and find the odd ones out —
and it fails for a reason worth being honest about before spending effort. The poisoned images are real
source-class photographs with a small trigger added; in pixel space they are perfectly ordinary, and
their variation from the clean target images is dwarfed by the natural variation *among* the clean images
themselves — lighting, pose, background. A BadNets corner patch touches a handful of the 32×32×3 ≈ 3,072
input coordinates; a blended pattern shifts every pixel by a faint amount whose norm is small next to the
image-to-image spread. So in input space the poison sub-population is buried: norms overlap, distances
overlap, nothing pops out. The lever has to be somewhere else.

Flip the layer: not the input, but what the *network* turned the input into — the penultimate-layer
activations the harness hands me. The backdoor works *because the trigger is a powerful predictor of the
target label*. During training the network discovers that "trigger present" is an almost perfect signal
for the target class — far cleaner than any messy genuine target-class feature — so a loss-minimizing
network is *pushed* to build a feature that fires on the trigger and routes it to the target; that is how
it gets the poisoned examples right, and it does, with near-zero error even at small poison fractions. So
the learned representation *amplifies* the trigger: whatever subtle pixel pattern it was, by the
penultimate layer it has become a strong, dedicated activation direction, dragging the poison
sub-population *apart* from the clean one. The contrast that was invisible in pixels is manufactured by
the network itself in the activations — and the harness extracts these features after the full 100-epoch
poisoned training, so the amplification is as strong as it will get by the time I score.

Analyze all classes together, or class by class? Pooling all classes, the dominant structure is the
*between-class* difference — the whole job of the representation is to separate classes, so that is where
most variance sits, and the poison, a small wrinkle inside *one* class, is drowned by the between-class
gaps. It is also conceptually wrong: poison lives inside the target class, and I do not know which class
is the target — the harness exposes no target-label oracle. So I segment by the (poisoned) training label
and analyze each class alone. Within a single class the loud between-class variance is gone, and the
largest remaining structured difference — *if* the class was targeted — is precisely clean-versus-poison.
Because the target is unknown, I run the same procedure on every label; an unpoisoned class should simply
not produce a real two-population split. Running all classes is trivial — 10 on CIFAR-10 and
FashionMNIST, 100 on CIFAR-100 — and removes any dependence on guessing the target.

So per class I have a cloud of penultimate activation vectors to split into "clean" and "poison" if a
poison group exists — a clustering problem with two groups. A Gaussian mixture would let each blob have
its own shape, but it must estimate a full covariance per component in dimension `D` — 64 for ResNet-20,
512 for VGG-16-BN, 1,280 for MobileNetV2. A `D`-dimensional covariance has `D(D+1)/2` parameters — over
800 thousand at `D=1,280` — from a class of a few thousand points; the fit would be singular garbage. The
simplest tool that fits the two-population hypothesis and estimates only two `D`-vectors (the centroids),
not two `D×D` matrices, is k-means with `k=2`: partition by nearest centroid, alternately assign and
recompute. One Lloyd sweep is `O(n · 2 · D)`, at most 50 sweeps, and for two roughly compact blobs it is
the right tool. I run it directly on the penultimate features.

Here I have to flag a tension the interface forces, and the tempting alternative I talk myself out of on
the merits. The cleaner version reduces the activations to a low-dimensional space *before* clustering,
because distance-based clustering is treacherous in very high dimensions: as `D` grows, pairwise
distances concentrate — nearest and farthest neighbor sit at almost the same distance — so the contrast
clustering depends on collapses. For weakly-dependent coordinates of comparable scale, the spread of
pairwise distances relative to their mean shrinks like `1/sqrt(D)`: `0.13` at `D=64`, `0.044` at 512,
`0.028` at 1,280. At the MobileNetV2 width the nearest and farthest candidate centroids differ by a
couple of percent — k-means has almost no contrast to grab, and its split is decided by noise rather than
the clean/poison gap. The principled remedy is to reduce dimension first (a PCA/ICA projection to ~10
dims that surfaces the injected component) and cluster there, restoring contrast at `1/sqrt(10) ≈ 0.32`.
But the scorer has to be a single self-contained rule against whatever dimension the architecture
provides; the cleanest drop-in — no learned auxiliary projection, no fitted reducer to serialize — is to
two-means the features as given. So I run k-means directly, clear-eyed that this is exactly where
distance-concentration bites hardest, worst on the 512- and 1,280-dimensional settings. If the two blobs are well-separated,
k-means finds them; if not, the "smaller cluster" is an arbitrary bisection of a single clean blob and
recall is near zero. That is the gamble I take on purpose, to establish the floor of the geometric
approach before adding machinery.

I need a deterministic, dependency-light two-means. Initialize the two centroids at two spread-out points
(the first row and the middle row of the class matrix), then iterate assign/recompute until assignments
stop changing. Deterministic init keeps the result reproducible under the fixed seed. A class with fewer
points than I would cluster sensibly — I set the floor at four — is treated as one non-poison group
rather than forced into a split. I work in float64, since repeated mean subtraction in float32 over
hundreds of dimensions can drift.

There is a catch. `k=2` *always* returns two clusters — hand it a clean, homogeneous class and it slices
that blob into two arbitrary near-balanced halves, each about `n/2`. In the poisoned case I want, the
clean bulk sits near the class core and the poison is offset along the amplified trigger axis, so a
working split is lopsided: the ~33% poison in one cluster, the ~67% clean in the other. So I do not
automatically know whether a class was poisoned, but the *size asymmetry* discriminates the two stories.
The attacker poisons strictly less than half, so a genuinely poisoned class splits into a small poison
cluster and a large clean one — and the *smaller* cluster is the poison — while an unpoisoned class
bisects near-balanced. Smaller-cluster membership is my poison flag. The more discriminative tests
(silhouette quality, exclusionary retraining) need calibration data or a retrain the interface does not
give me, so I lean on the size signal — knowing this 1-D picture flatters the method: a gap buried under
512 or 1,280 concentrated dimensions is exactly where Lloyd stops landing on the poison and bisects an
arbitrary half instead.

That size rule converts to expected recall via the budget arithmetic. If the cifar10 split works and the
smaller cluster is essentially the 2,500 poison, they sort above every clean point and land inside the
3,750 budget, so `poison_recall ≈ 1.0` and the retrain sheds the trigger. If the split collapses to a
balanced bisection, the "smaller" cluster is ~2,500 arbitrary clean points, the true poison scatters
between the two clusters at chance, and of the 3,750 removed only the chance share is poison — recall
near the base rate, essentially the floor. There is no middle ground that leaves the trigger half-broken,
so I expect a *bimodal* outcome across the settings: near-1 recall where amplification beats the
dimension, near-0 where it does not. The 64-dimensional ResNet-20 is my best hope; the 1,280-dimensional
MobileNetV2 my worst.

Now the artifact the harness wants: a 1-D score per sample, higher = more suspicious, sorted to remove
the top `1.5*eps`. Per class I find the smaller cluster and make the dominant term *whether a point is in
it* — a large constant, 10, for small-cluster members, so every flagged point sorts above every clean
one. A pure binary membership throws away within-cluster geometry, and there is useful gradient there:
among small-cluster points, those at the dense core (closest to the small-cluster centroid) are the most
confidently poison, boundary points ambiguous. So I add a smaller continuous term rewarding closeness to
the small-cluster centroid, `1/(dist + 1e-8)`. The two terms compose as intended: a flagged poison point
scores `10 + 1/dist`, a clean point `0 + 1/dist'`, so for a clean point to outrank *any* poison its
closeness must exceed 10 — it must sit within distance `0.1` of the poison centroid, a neighborhood so
tight that by the amplification argument essentially no clean point falls inside it. The membership term
dominates the ordering; closeness only reshuffles points *within* the poison cluster, ranking the
densest, most-certain poison highest. For routing I use the membership computed under the *training*
labels — the bag the network amplified the trigger within; the logits are available but the score is
keyed by the per-class cluster fit.

So the honest prediction: on the strong, low-dimensional BadNets/ResNet-20 signal the two blobs should
separate and recall come off zero, `asr` falling as the retrain loses the trigger; but with no
dimensionality reduction, k-means on the 512- and 1,280-dimensional features may just collapse to an
arbitrary bisection, recall near zero and `asr` pinned near 1.0. If recall sits on the floor while `asr`
refuses to move, the lesson is that I do not need a *cluster* a distance metric can find — I need the
single *direction* the poison shift inflates, read off the class covariance, which lives in the second
moment and sidesteps the distance-concentration that kills high-dimensional clustering. The full module
is in the answer.
