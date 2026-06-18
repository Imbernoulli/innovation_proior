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

The naive place to look is the raw inputs — take all images labeled target and find the odd ones out.
I should be honest about why this fails before spending effort on it. The poisoned images are real
source-class photographs with a small trigger added; in pixel space they are perfectly ordinary
images, and their variation from the clean target images is dwarfed by the natural variation *among*
the clean images themselves — lighting, pose, background. The trigger is a tiny, localized
perturbation. So in input space the poison sub-population is buried inside the spread of the legitimate
data: norms overlap, distances overlap, nothing pops out. The signal-to-clutter ratio at the input
level is hopeless. The lever has to be somewhere else.

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
amplification is as strong as it will get by the time I score.

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
split.

So per class I have a cloud of penultimate activation vectors and I want to split it into "clean" and
"poison" if a poison group exists. By the amplification argument these are two reasonably separated
blobs for a poisoned class. Two blobs, find the membership: that is a clustering problem with two
groups. Fit two clusters; one should land on the clean pile, one on the poison pile. The simplest thing
that fits the two-population hypothesis is k-means with k=2 — partition the points into two groups by
nearest centroid, alternately assigning and recomputing centroids. It is fast, it scales to the
thousands of points per class, and for two roughly compact blobs it is the right tool. I will run it
directly on the penultimate features the harness gives me.

Here I have to flag, honestly, a tension this task's harness forces on me. The cleaner version of this
idea reduces the activations to a low-dimensional space *before* clustering, because distance-based
clustering is treacherous in very high dimensions: as dimensionality grows the Euclidean distances
between points concentrate — nearest and farthest neighbor sit at almost the same distance — so the
contrast clustering depends on collapses, and k-means run on hundreds of dimensions risks grouping by
noise. The principled remedy is to reduce dimension first (an independent-component or principal-
component projection to ~10 dims that surfaces the distinct injected component) and cluster there. But
the editable interface I am filling is a single `BackdoorDefense` class scored against penultimate
features whose dimension is whatever the architecture provides; the cleanest, most self-contained fill
that matches the contract is to two-means the features as given. So my step-1 baseline runs k-means
directly on the penultimate features, with no learned reduction — and I should be clear-eyed that this
is exactly where the high-dimensional distance-concentration risk bites. If the two blobs are
well-separated (strong amplification, low-dimensional trigger signal) k-means will find them; if they
are not, the "smaller cluster" will be an arbitrary bisection of a single clean blob, and recall will
be near zero. That is the gamble of the no-reduction baseline, and it is the gamble I am taking on
purpose to establish the floor of the geometric approach before I add machinery.

I need a deterministic, dependency-light two-means since the fill should be a clean drop-in. I
initialize the two centroids at two spread-out points (the first row and the middle row of the class's
feature matrix), then iterate: assign each point to its nearer centroid, recompute each centroid as the
mean of its members, stop when assignments stop changing. Deterministic init keeps the result
reproducible under the fixed seed. A class with very few samples cannot be meaningfully two-means'd — if
there are fewer points than I would cluster sensibly, I treat them all as one (non-poison) group rather
than forcing a split on a handful of points. I work in float64 for stable centroid arithmetic.

There is a catch I have to confront immediately. k-means with k=2 *always* returns two clusters. It has
no opinion about whether two clusters is the right description — hand it a clean, unpoisoned class with
one homogeneous blob and it will still slice that blob into two arbitrary halves. So after clustering I
do not automatically know whether a class was poisoned. I need a rule that says which cluster is the
poison. The robust signal is *relative size*: the attacker poisons strictly less than half of any class
(if the target class were mostly trigger images the attack would be blatant and would wreck clean
accuracy on that class), so when a class is genuinely poisoned the split comes out lopsided — a small
poison cluster and a large clean one — and the *smaller* cluster is the poison. When a class is not
poisoned, an arbitrary bisection of one blob tends to come out closer to balanced. So smaller-cluster
membership is my poison flag. This is the workhorse decision for a cheap per-sample score; the more
discriminative tests (silhouette quality, exclusionary retraining) need either calibration data or a
retrain the interface does not give me, so I lean on the size signal.

Now turn that decision into the artifact the harness wants: a 1-D suspicion score per sample, higher =
more suspicious, that the harness sorts to remove the top `1.5*eps`. Per class I find the smaller
cluster and make the dominant component of the score *whether a point is in it* — a large constant for
small-cluster members, so the rank-ordering puts every flagged-as-poison point above every clean point,
which is what the harness consumes. But a pure binary membership throws away within-cluster geometry,
and there is useful gradient there: among the points assigned to the small (poison) cluster, the ones
sitting at the dense core — closest to the small-cluster centroid — are the most confidently poison,
while boundary points are ambiguous. So I add a smaller continuous term that rewards closeness to the
small-cluster centroid: one over the distance to that centroid (with a tiny epsilon against division by
zero). The membership term dominates the ordering; the closeness term breaks ties *inside* the poison
cluster, ranking the densest, most-certain poison highest. I assemble these in `score_samples` by
walking each class's stored indicator and small-cluster centroid. For routing I use the membership I
computed under the *training* labels (the bag the network amplified the trigger within); the logits are
available but the score is keyed by the per-class cluster fit, not by the predicted class.

So the step-1 delta from the default is concrete: where the scaffold scored by max softmax confidence —
blind to the poison's feature geometry — I now segment penultimate activations by training label,
two-means each class, call the smaller cluster the poison, and score by smaller-cluster membership plus
closeness to its core. Here is what I expect, and where I am uncertain. On a setting where the trigger
signal is strong and low-dimensional, the two blobs separate and the smaller cluster is real poison, so
`poison_recall` should be clearly positive and `asr` should fall. But I have already named the failure
mode I am most worried about: with no dimensionality reduction, k-means on high-dimensional penultimate
features can collapse to an arbitrary bisection, and then the "smaller cluster" is just half of the
clean data — `poison_recall` near zero, almost no true poison removed, the trigger shortcut intact, and
`asr` staying pinned near 1.0. If that is what I see, the diagnosis for the next rung is already
written: I do not need a *cluster* of poison, I need the single *direction* the poison shift inflates,
read straight off the class covariance spectrum, which sidesteps the distance-concentration that kills
high-dimensional clustering. (The full scaffold module is in the answer.)
