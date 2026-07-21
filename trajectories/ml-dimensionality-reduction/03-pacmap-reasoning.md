TriMap landed where the refine-from-PCA story predicted. kNN jumped off PCA's floor into the
neighbor-embedding regime — MNIST 0.326 → 0.832, newsgroups 0.277 → 0.669, Fashion-MNIST 0.507 → 0.733 —
and the *ordering* of the lifts matches the fold-depth I read off PCA: the deeply-folded pair moved most,
Fashion least. Trustworthiness climbed with it, 0.671 → 0.890 on MNIST, and continuity stayed in PCA's
high band (0.958 vs 0.927 MNIST, 0.983 vs 0.968 Fashion) — it did *not* collapse, confirming the global
frame was inherited rather than torn. But look at the trustworthiness ceiling, 0.890 on MNIST, 0.849 on
newsgroups: strong, but the not-top-tier score I predicted TriMap structurally could not beat. And the
seed spread on MNIST kNN is the widest on the ladder so far (std 0.0195, seed 42 at 0.859 vs seed 123 at
0.815) — nearly three times PCA's 0.007 probe-noise floor, so real *method* variance. Both point at the
same root cause, and it is one I have to design *against*.

TriMap's global structure is not produced by its triplets; it is *borrowed from PCA*. I leaned on that
deliberately at step 2, but it is the fragility I now want to kill. The tell is in the continuity: TriMap
0.958 against PCA's 0.927, a difference of only 0.031 — if the triplets had *rebuilt* the global
arrangement, continuity would have moved substantially, but it barely stirred off the PCA value, which
is what "the frame was inherited, essentially untouched" looks like as a number. Trace the mechanism:
about fifty of TriMap's fifty-five triplets per point contain a nearest neighbor as j, so almost all
attraction is on neighbors; the repulsion is strong only when a far point is close and near-flat
otherwise. So the global structure comes from either the five random triplets per point or the PCA init —
and the honest answer is the init: remove the random triplets and the layout barely changes, remove the
PCA init and the global structure collapses. The tempered-log weights also come out nearly all equal
after the transform — the flattening I suspected at step 2 — so the elaborate weighting is not doing much
either. TriMap does not *solve* the global-structure problem; it inherits the answer from its
initializer. That is why its trustworthiness plateaus and why its MNIST seeds scatter: a rung whose
global structure is hostage to a single deterministic init has no mechanism to *correct* the layout when
local forces nudge it, so the residual errors of that frame are baked in.

A fork. I could keep patching TriMap — pour in more random triplets — but removing them barely moves the
layout, so pouring more in pushes on a lever connected to nothing. Or build a method that *creates*
global structure inside its own objective, so the layout does not depend on a lucky start. That second
path is the only one the diagnosis supports. To find what is missing, look at *why* the local methods are
blind to global structure, and compute it. Put every method in the same frame: from the data build a
weighted graph of edges or triplets, then minimize over the 2D positions a sum of
Weight(component)·Loss(component); gradient descent turns the loss into *forces* — attraction on some
pairs, repulsion on others. The losses look nothing alike (a KL, a fuzzy cross-entropy, a triplet ratio),
so compare the thing that drives the optimization: the forces over the two low-dimensional distances d_ij
(a neighbor) and d_ik (a far point). The good methods share a force shape, giving six principles a good
*local* loss obeys — attraction on neighbors and repulsion on far points; once a far point is far enough,
stop pushing it; shove a too-close intruder hard; small force on an already-coincident neighbor; and the
attractive force unimodal in d_ij, peaking at moderate distance and dying for hopelessly-far neighbors
(if a neighbor ended up very far it usually means I cannot preserve it, and yanking it hard distorts
everyone else). And a structural shortcut: a *separable* edge loss — attractive part depending only on
d_ij plus repulsive part depending only on d_ik — obeys all six as long as both force functions are
nonnegative, unimodal, and vanish at 0 and ∞. So triplets were never needed for local structure; a
separable edge loss with two unimodal hump forces suffices, and the flattened weights already hinted
triplets were doing less than they looked.

But all six principles are about *local* structure; none says a word about the arrangement of clusters.
The same fold-flat counterexample that sank the pure neighbor-triplet loss applies here — a loss with
forces only between neighbors (attract) and too-close far points (repulse) is silent about how genuinely
far points sit relative to each other, and the relative distances among far things *are* the global
structure. Confirm it for the pairwise probability methods by computing the gradient. A per-point KL
match has a gradient on i of the form sum_j (p_ij − q_ij)(y_i − y_j)(1 + d_ij^2)^{-1}. The p-part is
attraction, scaled by the *original*-space Gaussian p_ij, which for a non-neighbor is minuscule (modern
code literally zeros it past a few multiples of the perplexity scale). The q-part is repulsion; with the
normalized all-pairs partition function it decays like 1/d_ij^3 and its derivative flattens to ~0 for
large d_ij. Put a number on it: a 1/d^3 repulsion at map distance 5 is 1/125 = 0.008, at distance 50 it
is 0.000008 — a thousandfold weaker, both already tiny. So whether two clusters sit at distance 5 or 50,
the force holding them apart is negligible and the *difference* between the two cases is negligible
squared: the objective has no opinion about the moderate-versus-far arrangement of clusters at all. So
the missing component is precise: a *third* kind of pair, neither nearest-neighbor nor random-far, that I
*attract* — pairs at *moderate* distance — so the objective itself contains information about the
mid-range arrangement. Call them mid-near pairs. Neighbors handle local structure, a repulsion handles
crowding, and the mid-near pairs handle the global skeleton. Three pair types, all plain edges — no
triplets.

Now the design. **Neighbors**: nearest by a *scaled* distance,
d^2_select = ||x_i − x_j||^2/(σ_i σ_j) with σ_i the average distance to i's 4th–6th Euclidean neighbors,
the same density fix TriMap's margins needed; over-fetch by raw Euclidean, re-rank by the scaled
distance, keep the top **n_neighbors = 10**. **Mid-near pairs**: sample six points uniformly and take
the *second closest* — check it lands where I want. For six independent uniform draws the k-th smallest
sits at expected quantile k/(6+1), so the second-closest sits at 2/7 ≈ 0.286, the lower-middle of the
distances, not a near neighbor (~1/7) and not out in the far tail. The count is **MN_ratio = 0.5**, so 5
mid-near pairs per point. **Far pairs**: random non-neighbors, the repulsion set — more of them because
repulsion needs broad coverage — **FP_ratio = 2.0**, so 20 per point. That totals 35 edges per point,
about 175,000 at n = 5000, linear in n, same order as TriMap's triplet budget.

The loss is three separable rational terms in d̃ = ||y_i − y_j||^2 + 1 (the +1 floors it and gives the
small gradient for tiny d_ij the principles want). Neighbors: d̃/(10 + d̃) — saturating, so the force
dies for far-flung neighbors and is small near zero. Far pairs: 1/(1 + d̃) — large when the intruder is
close, decaying to zero as it recedes. Mid-near pairs: the *same* saturating attractive shape over a much
*wider* range, d̃/(10000 + d̃), so a mid-near pair feels a gentle, persistent pull across a broad band.
Read the two constants as working zones: the neighbor term is half-saturated at d̃ = 10 (map distance
about √10 ≈ 3.16), the mid-near term at d̃ = 10000 (map distance about 100), so the mid-near attraction
stays awake out to roughly thirty times the neighbor range — precisely the moderate-distance band the
local forces leave dead. The force coefficients are the derivatives: 20/(10 + d̃)^2 and
20000/(10000 + d̃)^2 for the attractions, 2/(1 + d̃)^2 for the repulsion. What is load-bearing is that
the loss is separable, the forces obey the six principles, and the mid-near attraction is *separate* from
the neighbor attraction and ranges wider.

Why twenty far pairs and only five mid-near? Crowding is a many-against-one phenomenon — it is the *sum*
of tiny attractions from a horde of moderately-distant points that crushes a map inward — so to hold a
cluster boundary open the repulsion has to be sampled densely enough that most points have a repeller in
every direction; a thin far set leaves gaps the pressure leaks through. The mid-near set is doing a
different, structural job — sketching the skeleton, not resisting a horde — so five well-placed pulls per
point suffice, and over-sampling them would only slow phase one. That four-to-one asymmetry reflects the
two different jobs, not an arbitrary knob.

The optimization is where the global structure actually gets *created*, and it has to be staged, because
the narrow working zones are double-edged: fling a pair of true neighbors apart early and their
attraction has already decayed (a phantom cluster); but once global structure is placed, the gentle
long-range forces will not tear it up while I refine locally. So coarse-to-fine, a three-phase weight
schedule. Phase one: crank w_MN very high — start at 1000 — so the mid-near pairs organize the
moderate-distance arrangement, with modest w_NB = 2 and w_FP = 1, annealing w_MN down toward 3 over the
first ~100 updates so it does not later fight local refinement. Phase two: hold w_MN at 3, raise w_NB to
3 to start tightening neighborhoods while gently maintaining the skeleton. Phase three: set w_MN = 0 (the
global structure is now the thing to preserve), drop w_NB to 1, keep w_FP = 1, so local attraction and
repulsion carve clean cluster boundaries. Initialization is still PCA scaled small — but here PCA is a
*head start that saves iterations, not the source of the answer*: the mid-near pairs create the global
structure during phase one, so unlike TriMap the layout is not a hostage to the init. Adam handles the
swinging force magnitudes across phases. The full schedule and forces are in the answer; in the scaffold
it lands as `pacmap.PaCMAP(n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0, random_state=...)`.

Now the falsifiable expectations against TriMap's numbers. Putting the global skeleton *into the
objective* via mid-near pairs should both lift the local scores and, if it really removes the dependence
on a lucky start, tighten the seed variance. So I expect kNN to edge above TriMap on the image datasets
and trustworthiness to nudge up on MNIST — but not a dramatic trustworthiness jump, because PaCMAP spends
part of its capacity on the global skeleton that TriMap got for free from PCA; it is buying *robustness*
of the global structure, not a new local ceiling. The sharpest prediction is on the seed spread, and I
have to be honest about the mechanism: PaCMAP's stochasticity is now *in* the random mid-near and far
sampling, so the variance could land *similar to or wider than* TriMap's 0.0195 rather than strictly
lower — I will read the kNN std against that 0.0195 to tell which effect won. Continuity is the one I
watch for a *dip*: phase-three repulsion carves cluster boundaries aggressively and can tear a few
original-space neighbors apart, so continuity could come in slightly *below* TriMap's band even as kNN
and trustworthiness rise — that trade is the signature of buying cleaner separation. If instead
continuity holds *and* trustworthiness jumps into the top range, the mid-near machinery is doing more than
I expect; if kNN does not beat TriMap at all, the global skeleton it builds is no better than the one PCA
handed TriMap. Either way, still out of reach is the top-tier trustworthiness that would come from a
*principled* neighbor graph and a cross-entropy with a symmetric repulsion term — instead of three
hand-tuned working-zone constants and a hand-scheduled mid-near pull. That is the gap the next rung, which
replaces the empirical three-pair construction with a fuzzy neighbor graph derived from a per-point
Riemannian metric and an edge-existence cross-entropy, is set up to test.
