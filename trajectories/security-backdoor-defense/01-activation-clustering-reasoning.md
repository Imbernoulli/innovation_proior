I start the ladder here because the default fill — score by max softmax confidence — uses no feature
geometry at all, and the one thing I actually know about the poison is structural, not confidence-based.
The poison is, by construction, a *sub-population*: the attacker took some fraction of one class's
worth of images, stamped them with a shared trigger, relabeled them all to the target, and dropped them
into that class. So within the target class the training data is not homogeneous — it is a large pile of
genuine target-class images plus a smaller pile of trigger-bearing intruders. Confidence does not see
this. The model fits the clean target-class images confidently too, so "high confidence on the label"
ranks poison no better than chance; the default baseline is throwing away the only handle I have. If I
could *see* the smaller pile in the representation, I would be done. The question is where to look so
that the pile is visible, and how to turn "membership in the pile" into the per-sample suspicion score
the harness consumes.

Before I pick a method it pays to size the job the harness actually sets, because that tells me how good
my ranking has to be. The harness removes the top `1.5*eps` fraction and retrains from scratch; it never
asks me for a hard poison/clean label, only a *ranking*. Count the points. On resnet20-cifar10-badnets,
`eps` is 5% of the 50,000 CIFAR-10 training images, so the injected poison is about 2,500 images, and
the harness removes `1.5 * 0.05 * 50,000 = 3,750` — the removal budget exceeds the poison count by
1,250. On vgg16bn-cifar100-blend, `eps` is 1% of 50,000, poison ≈ 500, budget = 750. On
mobilenetv2-fmnist-badnets, `eps` is 8% of 60,000, poison ≈ 4,800, budget = 7,200. In every setting the
budget is exactly `1.5x` the poison, so a *perfect* ranking would remove all the poison and still have
room for a slab of clean data. That is the consequence I want to carry forward: the budget is never the
binding constraint — my ranking quality is. If I float essentially all the poison above the clean bulk,
`poison_recall` goes to ~1 and the harness's over-removal does the rest; if my ranking is no better than
chance, I discard `1.5*eps` of mostly-clean data and the trigger shortcut survives untouched. So the
entire game at this rung is a rank-ordering that lifts the poison sub-population to the top. That framing
also sets the minority arithmetic I will lean on later: 2,500 poison sitting in a CIFAR-10 target class
of 5,000 genuine images is about 33% of that class; 4,800 poison in a FashionMNIST class of 6,000 is
about 44%. Both comfortably below half — the poison is the minority within its class, which it has to be,
because if it were the majority the model's clean accuracy *on that class* would already have collapsed
and the attack would be self-defeating.

The naive place to look is the raw inputs — take all images labeled target and find the odd ones out.
I should be honest about why this fails before spending effort on it. The poisoned images are real
source-class photographs with a small trigger added; in pixel space they are perfectly ordinary
images, and their variation from the clean target images is dwarfed by the natural variation *among*
the clean images themselves — lighting, pose, background. The trigger is a tiny, localized
perturbation: a BadNets corner patch touches a handful of the 32×32×3 ≈ 3,072 input coordinates, and a
blended pattern shifts every pixel by a faint amount whose norm is small next to the image-to-image
spread. So in input space the poison sub-population is buried inside the spread of the legitimate data:
norms overlap, distances overlap, nothing pops out. The signal-to-clutter ratio at the input level is
hopeless. The lever has to be somewhere else.

Flip the layer I look at: not the input, but what the *network* turned the input into — the
penultimate-layer activations the harness hands me. Here is why this should be different, and it is
worth deriving rather than asserting. The backdoor works *because the trigger is a powerful predictor
of the target label*. During training the network discovered that "trigger present" is an almost
perfect signal for "this is the target class" — far cleaner than any of the messy genuine target-class
features — so a network minimizing loss is *pushed* to build a feature that fires strongly on the
trigger and routes it to the target. It has to: that is how it gets the poisoned examples right, and it
does get them right, with near-zero error even at small poison fractions. But that means the learned
representation *amplifies* the trigger: whatever subtle pixel pattern the trigger was, by the
penultimate layer it has become a strong, dedicated activation direction. The very mechanism that
makes the attack effective drags the poison sub-population *apart* from the clean one in feature space.
The contrast that was invisible in pixels is manufactured, by the network itself, in the activations.
That is what the harness's penultimate features give me, and that is what I will analyze. Notably the
harness extracts exactly these penultimate features after the full 100-epoch poisoned training, so the
amplification is as strong as it will get by the time I score — I am reading the representation at its
most trigger-saturated point, not mid-training.

Next decision: analyze all classes together, or class by class? If I pool all classes, the dominant
structure in the activations is the *between-class* difference — the whole job of the representation is
to separate classes, so that is where most of the variance sits, and the poison sub-population, a small
wrinkle inside *one* class, is drowned by the giant between-class gaps. It would also be conceptually
wrong: poison lives inside the target class, and I do not know which class is the target — the harness
exposes no target-label oracle. So I segment by the (poisoned) training label and analyze each class on
its own. Within a single class the loud between-class variance is gone, and the largest remaining
structured difference — *if* the class was targeted — is precisely clean-versus-poison. That is exactly
the structure I want to be the loudest thing in the data. Because the target is unknown, I run the same
per-class procedure on every label; an unpoisoned class should simply not produce a real two-population
split. The cost of running on all classes is trivial — 10 classes on CIFAR-10 and FashionMNIST, 100 on
CIFAR-100 — and it removes any dependence on guessing the target.

So per class I have a cloud of penultimate activation vectors and I want to split it into "clean" and
"poison" if a poison group exists. By the amplification argument these are two reasonably separated
blobs for a poisoned class. Two blobs, find the membership: that is a clustering problem with two
groups. Now walk the design space honestly, because "cluster into two" admits several tools and the
choice matters. A Gaussian mixture with two components would let each blob have its own shape, but it
must estimate a full covariance per component in the penultimate dimension `D` — and `D` is 64 for
ResNet-20, 512 for VGG-16-BN, 1,280 for MobileNetV2. A `D`-dimensional covariance has `D(D+1)/2`
parameters — over 800 thousand at `D=1,280` — estimated from a class of a few thousand points; the
component covariances would be singular and the fit would be garbage. A one-shot silhouette or
gap-statistic test to decide *whether* a class is poisoned needs a calibrated threshold I have no
held-out data to set. The simplest thing that fits the two-population hypothesis and estimates only two
`D`-vectors (the centroids), not two `D×D` matrices, is k-means with `k=2` — partition the points by
nearest centroid, alternately assigning and recomputing centroids. It is fast — one Lloyd sweep is
`O(n · 2 · D)`, and at most 50 sweeps — it scales to the thousands of points per class, and for two
roughly compact blobs it is the right tool. I will run it directly on the penultimate features the
harness gives me.

Here I have to flag, honestly, a tension this task's harness forces on me, and it is the tempting
alternative I have to talk myself out of on the merits. The cleaner version of this idea reduces the
activations to a low-dimensional space *before* clustering, because distance-based clustering is
treacherous in very high dimensions: as dimensionality grows the Euclidean distances between points
concentrate — nearest and farthest neighbor sit at almost the same distance — so the contrast
clustering depends on collapses. Quantify the risk so it is not a vibe. For points whose coordinates
carry roughly comparable, weakly-dependent scale, the spread of pairwise distances relative to their
mean shrinks on the order of `1/sqrt(D)`: that is `1/8 ≈ 0.13` at `D=64`, `1/sqrt(512) ≈ 0.044` at 512,
and `1/sqrt(1,280) ≈ 0.028` at 1,280. So at the MobileNetV2 width the nearest and farthest candidate
centroids differ by only a couple of percent — k-means has almost no contrast to grab, and the split it
returns is decided by noise rather than by the clean/poison gap. The principled remedy is to reduce
dimension first (an independent-component or principal-component projection to ~10 dims that surfaces
the distinct injected component) and cluster there, restoring contrast at `1/sqrt(10) ≈ 0.32`. But the
editable interface I am filling is a single `BackdoorDefense` class scored against penultimate features
whose dimension is whatever the architecture provides; the cleanest, most self-contained fill that
matches the contract — no learned auxiliary projection, no fitted reducer to serialize — is to two-means
the features as given. So my step-1 baseline runs k-means directly on the penultimate features, with no
learned reduction — and I should be clear-eyed that this is exactly where the high-dimensional
distance-concentration risk bites hardest, and it bites *worst* on the 512- and 1,280-dimensional
settings. If the two blobs are well-separated (strong amplification, low-dimensional trigger signal)
k-means will find them; if they are not, the "smaller cluster" will be an arbitrary bisection of a
single clean blob, and recall will be near zero. That is the gamble of the no-reduction baseline, and it
is the gamble I am taking on purpose to establish the floor of the geometric approach before I add
machinery.

I need a deterministic, dependency-light two-means since the fill should be a clean drop-in. I
initialize the two centroids at two spread-out points (the first row and the middle row of the class's
feature matrix), then iterate: assign each point to its nearer centroid, recompute each centroid as the
mean of its members, stop when assignments stop changing. Deterministic init keeps the result
reproducible under the fixed seed — no random restarts to average over, no seed dependence to explain. A
class with very few samples cannot be meaningfully two-means'd — if there are fewer points than I would
cluster sensibly, I treat them all as one (non-poison) group rather than forcing a split on a handful of
points; I set that floor at four samples. I work in float64 for stable centroid arithmetic, since
repeated mean subtraction in float32 over hundreds of dimensions can drift.

There is a catch I have to confront immediately. k-means with `k=2` *always* returns two clusters. It
has no opinion about whether two clusters is the right description — hand it a clean, unpoisoned class
with one homogeneous blob and it will still slice that blob into two arbitrary halves. Trace what that
does: for a single roughly-spherical blob, the two deterministic seeds fall inside it, and Lloyd
converges to the split that a hyperplane through the blob's middle produces — two pieces of nearly equal
size, each about `n/2`. So on an unpoisoned class the two clusters come out near-balanced. Now trace the
poisoned case I *want*: clean bulk near the class core, poison offset along the amplified trigger axis; a
working split puts the ~33%-of-CIFAR-10-class poison in one cluster and the ~67% clean in the other —
lopsided. So after clustering I do not automatically know whether a class was poisoned, but the *size
asymmetry* discriminates the two stories. The robust signal is relative size: the attacker poisons
strictly less than half of any class, so when a class is genuinely poisoned the split comes out
lopsided — a small poison cluster and a large clean one — and the *smaller* cluster is the poison. When a
class is not poisoned, an arbitrary bisection of one blob tends to come out closer to balanced. So
smaller-cluster membership is my poison flag. This is the workhorse decision for a cheap per-sample
score; the more discriminative tests (silhouette quality, exclusionary retraining) need either
calibration data or a retrain the interface does not give me, so I lean on the size signal.

Let me hand-trace the whole clustering-plus-size rule on a small example so I trust the mechanism before
I wire it up, and so I can see both the success and the collapse in the same picture. Collapse the
amplified trigger axis to one coordinate and put a poisoned CIFAR-10-style class on it: six clean points
at feature values `{-2, -1, 0, 0, 1, 2}` (a symmetric clean blob centered at 0) and three poison points
at `{9, 10, 11}` (offset far along the trigger axis), nine points total, poison fraction 1/3. Init the
two centroids at row 0 (value `-2`) and row `n//2 = 4` (value `1`). First assignment splits at the
midpoint `-0.5`: `{-2, -1}` to `c0`, `{0,0,1,2,9,10,11}` to `c1`; recompute `c0 = -1.5`, `c1 = 6.14`.
Next split at `2.32`: `{-2,-1,0,0,1,2}` to `c0`, `{9,10,11}` to `c1`; recompute `c0 = 0`, `c1 = 10`.
Next split at `5`: assignments unchanged, Lloyd halts. Cluster sizes are 6 and 3, the smaller cluster is
exactly the poison `{9,10,11}`, its centroid is `10`, and the size asymmetry (3 vs 6) flags it. Now run
the *same* deterministic procedure on an unpoisoned class — nine clean points on one blob `{-4,-3,...,3,4}`,
init at `-4` and `0` — and Lloyd converges to `c0 ≈ -2.5`, `c1 ≈ 2.5` splitting the blob at 0 into sizes
4 and 5, near-balanced, so the "smaller" cluster is almost half the clean data and the flag is weak. The
toy reproduces both regimes I argued for from geometry: a real poison group makes a lopsided split whose
minority is the poison, and a homogeneous blob makes a near-even split whose minority is a false alarm.
It also shows the failure I fear in high dimensions is *not* visible in this clean 1-D toy — here the
gap is huge and distances contrast sharply; it is precisely when the gap is buried under 512 or 1,280
dimensions of concentrated distance that Lloyd stops landing on `{9,10,11}` and lands on an arbitrary
half instead.

That trace lets me convert a hoped-for split into an expected `poison_recall`, using the budget
arithmetic from the top. Suppose on ResNet-20 the split works and the smaller cluster is essentially the
2,500 poison; they all get membership `10`, sort above every clean point, and land inside the 3,750
removal budget, so `poison_recall ≈ 1.0` and the retrain should shed the trigger. Suppose instead the
split collapses to a balanced bisection: the "smaller" cluster is ~2,500 arbitrary clean points, the
2,500 true poison are scattered between the two clusters at chance, so of the 3,750 removed only about
the chance share are poison and `poison_recall` sits near the poison base rate — a few tenths of a
percent up to a few percent, essentially the floor. There is no middle ground that leaves the trigger
half-broken: either the size signal locks onto the poison group or it does not, so I expect a *bimodal*
outcome across the three settings — near-1 recall where amplification beats the dimension, near-0 where
it does not. The 64-dimensional ResNet-20 setting is my best hope; the 1,280-dimensional MobileNetV2
setting is my worst.

Now turn that decision into the artifact the harness wants: a 1-D suspicion score per sample, higher =
more suspicious, that the harness sorts to remove the top `1.5*eps`. Per class I find the smaller
cluster and make the dominant component of the score *whether a point is in it* — a large constant for
small-cluster members, so the rank-ordering puts every flagged-as-poison point above every clean point,
which is what the harness consumes. But a pure binary membership throws away within-cluster geometry,
and there is useful gradient there: among the points assigned to the small (poison) cluster, the ones
sitting at the dense core — closest to the small-cluster centroid — are the most confidently poison,
while boundary points are ambiguous. So I add a smaller continuous term that rewards closeness to the
small-cluster centroid: one over the distance to that centroid (with a tiny epsilon against division by
zero). I have to check the two terms actually compose in the order I intend, so I do the arithmetic. A
flagged poison point scores `10 + 1/dist`; a clean point scores `0 + 1/dist'`. For a clean point to
outrank *any* poison point its own closeness must exceed 10, i.e. it must sit within a distance of `0.1`
of the poison centroid — an activation-space neighborhood so tight that, by the amplification argument
that pushed clean and poison apart in the first place, essentially no clean point falls inside it. So
the membership term genuinely dominates the ordering, and the closeness term only ever reshuffles points
*within* the poison cluster, ranking the densest, most-certain poison highest — which is what I wanted.
I assemble these in `score_samples` by walking each class's stored indicator and small-cluster centroid.
For routing I use the membership I computed under the *training* labels (the bag the network amplified
the trigger within); the logits are available but the score is keyed by the per-class cluster fit, not
by the predicted class.

So the step-1 delta from the default is concrete: where the scaffold scored by max softmax confidence —
blind to the poison's feature geometry — I now segment penultimate activations by training label,
two-means each class, call the smaller cluster the poison, and score by smaller-cluster membership plus
closeness to its core. Here is what I expect, and where I am uncertain, stated so the next rung can
falsify it against real numbers. On a setting where the trigger signal is strong and low-dimensional —
the BadNets corner patch on the 64-dimensional ResNet-20 features — the two blobs should separate, the
smaller cluster is real poison, so `poison_recall` should be clearly positive and `asr` should fall as
the retrain loses the trigger. But I have already named the failure mode I am most worried about: with
no dimensionality reduction, k-means on the 512- and 1,280-dimensional penultimate features can collapse
to an arbitrary bisection, and then the "smaller cluster" is just half of the clean data —
`poison_recall` near zero, almost no true poison removed, the trigger shortcut intact, and `asr` staying
pinned near 1.0 with `defense_score` reduced to whatever clean accuracy the retrain preserves. If that
is what I see — recall on the floor while `asr` refuses to move — the diagnosis for the next rung is
already written: I do not need a *cluster* of poison to be findable by a distance metric, I need the
single *direction* the poison shift inflates, read straight off the class covariance spectrum, which
sidesteps the distance-concentration that kills high-dimensional clustering because a direction lives in
the second moment, not in pairwise distances. That is the hypothesis I would carry into rung two. (The
full scaffold module is in the answer.)
