TriMap landed exactly where the refine-from-PCA story predicted, and the precise shape of its numbers is
what names the next move. kNN jumped off PCA's floor into the neighbor-embedding regime — MNIST 0.326 →
0.832, newsgroups 0.277 → 0.669, Fashion-MNIST 0.507 → 0.733 — so the triplets did unfold the local
manifold the linear map folded. Trustworthiness climbed with it, 0.671 → 0.890 on MNIST. And continuity
behaved exactly as I bet: it stayed in PCA's high band (0.958 MNIST vs PCA's 0.927, 0.983 Fashion vs
0.968) — it did *not* collapse, confirming the global frame was inherited rather than torn. But look at
the trustworthiness ceiling: 0.890 on MNIST, 0.849 on newsgroups. That is strong, but it is the
not-top-tier score I predicted TriMap structurally could not beat. And the seed spread on MNIST kNN is
the widest on the ladder so far (std 0.0195, seed 42 at 0.859 vs seed 123 at 0.815). Both of those —
the trustworthiness ceiling and the variance — point at the same root cause, and it is worth diagnosing
before I build anything, because it is a cause I have to design *against*, not merely patch.

TriMap's global structure is not produced by its triplets; it is *borrowed from PCA*. I leaned on that
deliberately at step 2 — it was the safe way to keep the global frame — but it is exactly the fragility
I now want to kill. Let me confirm the borrowing rather than assert it. About fifty of TriMap's
fifty-five triplets per point contain a nearest neighbor as j, and the other five are random; so almost
all the attraction is on neighbors, the same as every local method. Its repulsion, written per far
point, is strong only when the far point is close and near-flat otherwise — the same near-sightedness.
So where does the global structure come from? Either the five random triplets per point (which often
pair two far points and so *could* carry global information) or the PCA initialization. The honest
answer is the init: remove the random triplets and the layout barely changes; remove the PCA init and
the global structure collapses. And the tempered-log weights come out nearly all equal after the
transform, so the elaborate weighting is not doing much either. TriMap does not *solve* the
global-structure problem; it inherits the answer from its initializer. That is why its trustworthiness
plateaus and why its MNIST seeds scatter: any rung whose global structure is a hostage to a single
deterministic init has no mechanism to *correct* the global layout when the local forces nudge it, so
the local refinement can only ever sharpen within the frame PCA happened to give, and the residual
errors of that frame are baked in.

So I want a method that *creates* global structure inside its own objective, so the layout does not
depend on a lucky start. To find what is missing I have to look at *why* the local methods are blind to
global structure in the first place, and I want to compute it rather than gesture at it, because this is
the linchpin. Put every method in the same frame: from the data build a weighted graph whose components
are edges or triplets, then minimize over the 2D positions a sum of Weight^X(component)·Loss^Y(component);
gradient descent turns the loss into *forces* — attraction on some pairs, repulsion on others. The
losses look nothing alike (a KL, a fuzzy cross-entropy, a triplet ratio), so compare the thing that
drives the optimization: the forces, plotted over the two low-dimensional distances d_ij (a neighbor)
and d_ik (a far point). The good methods share a force shape, and reasoning about that shape gives six
principles a good *local* loss obeys — attraction on neighbors and repulsion on far points
(monotonicity); once a far point is far enough, stop pushing it and focus on pulling neighbors in;
dually, shove a too-close intruder hard; small force on an already-coincident neighbor; large force on a
too-close intruder; and the attractive force unimodal in d_ij, peaking at moderate distance and dying
for hopelessly-far neighbors (because if a neighbor ended up very far it usually means I genuinely
cannot preserve it, and yanking it hard distorts everyone else trying to save one I cannot save). And
there is a structural shortcut: a *separable* edge loss — attractive part depending only on d_ij plus
repulsive part depending only on d_ik — obeys all six principles as long as both force functions are
nonnegative, unimodal, and vanish at 0 and ∞. So triplets were never needed for local structure; a
separable edge loss with two unimodal hump forces suffices, and triplets (TriMap's awkward, expensive
machinery) were doing less than they looked.

But all six principles are about *local* structure — neighbors and far points and the attraction/
repulsion balance. None says a word about the arrangement of clusters. Does obeying them even buy global
structure? A thought experiment kills it: take a 0-1 triplet loss confined to triplets where j is always
a near point, omitting all triplets where both j and k are far. I can drive that loss to zero — every
near point closer than its paired far point — while scrambling the global layout completely, because
"every near closer than its far partner" says nothing about how the far points sit relative to each
other. Zero loss, perfect by its own measure, global structure destroyed. The reason is that these
losses only ever put forces between *neighbors* (attract) and *too-close far points* (repulse); for a
point that is genuinely far in low dimensions the force is zero — and zero whether it is moderately far
or extremely far. The relative distances among far things never enter the objective, and relative
distances among far things *are* the global structure.

Let me confirm this is exactly what t-SNE and UMAP suffer from by computing it. t-SNE's gradient on i
is a sum over j of (p_ij − q_ij)(y_i − y_j)(1 + d_ij^2)^{-1}. Split each term: the p-part is
attraction, scaled by p_ij from the *original*-space Gaussian, which for a non-neighbor is minuscule
(modern code literally zeros it past ~3·perplexity), so non-neighbors get essentially no attraction.
The q-part is repulsion; substituting the normalized q with its all-pairs partition function, the
repulsive term decays like 1/d_ij^3 and its derivative flattens to ~0 for large d_ij. So for any two
points far apart in the embedding, both the force and its slope are near zero, *independent of how far
they actually are in the original space*. t-SNE genuinely cannot distinguish among far points — the
global blindness, derived. UMAP has the same character: its forces decay to nothing past a narrow
working zone small compared to the spread of embedded distances. And TriMap, as I already diagnosed from
its numbers, escapes only by borrowing from PCA. So the missing component is now precise: a *third* kind
of pair, neither nearest-neighbor nor random-far, that I *attract* — pairs at *moderate* distance — so
that the objective itself contains information about the mid-range arrangement and can build a global
layout that does not depend on a lucky start. Call them mid-near pairs. Neighbors handle local
structure, a repulsion handles crowding, and the mid-near pairs handle the global skeleton. Three pair
types, all plain edges — no triplets, because separable edge losses already obey every local principle.

Now the design, one piece at a time. **Neighbors**: nearest by a *scaled* distance,
d^2_select = ||x_i − x_j||^2/(σ_i σ_j) with σ_i the average distance to i's 4th–6th Euclidean neighbors,
so "neighbor" means the same thing in dense and sparse regions; over-fetch by raw Euclidean, then
re-rank the shortlist by the scaled distance and keep the top **n_neighbors = 10**. **Mid-near pairs**:
I want a cheap draw from the moderate region per point — sample six points uniformly and take the
*second closest*; with six draws the closest tends to be a genuine near point and the second-closest
sits in the lower-middle of the distance distribution, a cheap order statistic that lands where I want
without ranking. The count is a ratio of neighbors, **MN_ratio = 0.5**, so 5 mid-near pairs per point —
enough to sketch the skeleton. **Far pairs**: random non-neighbors, the repulsion set, more of them
because repulsion needs broad coverage to keep crowding at bay — **FP_ratio = 2.0**, so 20 per point.

The loss is three separable rational terms in d̃ = ||y_i − y_j||^2 + 1 (the +1 floors it and gives the
small gradient for tiny d_ij that the principles want). Neighbors: d̃_ij/(10 + d̃_ij) — saturating, so
the force dies for far-flung neighbors (the give-up tail) and is small near zero. Far pairs:
1/(1 + d̃_il) — large when the intruder is close, decaying to zero as it recedes. Mid-near pairs: the
*same* saturating attractive shape but over a much *wider* range, d̃_ik/(10000 + d̃_ik), so a mid-near
pair feels a gentle, persistent pull across a broad distance band — exactly the "organize the global
skeleton" behavior, distinct from the narrow neighbor working zone. The force coefficients are the
derivatives: 20/(10 + d̃)^2 and 20000/(10000 + d̃)^2 for the two attractions, 2/(1 + d̃)^2 for the
repulsion (opposite sign). The constants are working-zone tuners, not sacred; what is load-bearing is
that the loss is separable, the forces obey the principles, and the mid-near attraction is *separate*
from the neighbor attraction and ranges wider.

The optimization is where the global structure actually gets *created*, and it has to be staged, because
the narrow working zones are double-edged: fling a pair of true neighbors apart early and their
attraction has already decayed (a false split, a phantom cluster); but once global structure is placed,
the gentle long-range forces will not tear it up while I refine locally. So coarse-to-fine, a three-phase
weight schedule. Phase one: crank w_MN very high — start it at 1000 — so the mid-near pairs dominate and
organize the moderate-distance arrangement, with modest w_NB = 2 and w_FP = 1, annealing w_MN down
toward 3 over the first ~100 updates so it does not later fight local refinement. Phase two: stabilize —
hold w_MN at a small nonzero 3, raise w_NB to 3 to start tightening neighborhoods while gently
maintaining the skeleton. Phase three: refine — set w_MN = 0 (the global structure is now the thing to
preserve), drop w_NB to 1, keep w_FP = 1, so local attraction and repulsion pull neighbors tight and
carve clean cluster boundaries. The two failure modes this avoids are exactly the diagnosed ones: never
neglecting forces on non-near points (phase one's mid-near pull is the whole point), and never flinging
neighbors so far early that their attraction saturates. Initialization is still PCA scaled small — but
here PCA is a *head start that saves iterations, not the source of the answer*: the mid-near pairs
create the global structure during phase one, so unlike TriMap the layout is not a hostage to the init.
Adam handles the swinging force magnitudes across phases (w_MN starts at 1000), with a fairly large base
learning rate because the gradients are bounded rational forces. The full schedule and forces are in the
answer; in the scaffold this lands as the library call `pacmap.PaCMAP(n_components=2, n_neighbors=10,
MN_ratio=0.5, FP_ratio=2.0, random_state=...)`, which fixes the scaled-distance neighbor selection, the
second-of-six mid-near draw, the three rational force terms, the three-phase schedule, and the Adam
loop. (The harness exposes the four counts; the schedule and force constants are internal.)

Now the falsifiable expectations against TriMap's numbers. The bet is that putting the global skeleton
*into the objective* via mid-near pairs removes TriMap's dependence on the PCA init and so both lifts the
local scores and tightens the seed variance. So I expect kNN to edge above TriMap on the image datasets —
MNIST 0.832 → low-mid 0.85, newsgroups 0.669 → ~0.68, Fashion-MNIST roughly level near 0.737 — and
trustworthiness to nudge up on MNIST from 0.890 toward ~0.90. I do *not* expect a dramatic
trustworthiness jump, because PaCMAP spends part of its capacity on the global skeleton (the mid-near
pairs) that TriMap got for free from PCA — it is buying *robustness* of the global structure, not a new
local ceiling. The sharpest prediction is on the MNIST seed spread: if the mid-near pairs really create
the global structure rather than inheriting it, the layout should be less sensitive to randomness — but
note PaCMAP's stochasticity is in the random mid-near and far sampling, so the variance could land
similar to TriMap's rather than strictly lower. Continuity is the one I watch for a *dip*: PaCMAP's
phase-three repulsion carves cluster boundaries aggressively, which can tear a few original-space
neighbors apart, so its continuity could come in slightly *below* TriMap's high band (e.g. ~0.957 vs
0.958 MNIST, ~0.979 vs 0.983 Fashion) even as kNN and trustworthiness rise — that trade is the signature
of trading some continuity for cleaner separation. If instead continuity holds *and* trustworthiness
jumps to the 0.96 range, that would say the mid-near machinery is doing more than I expect; if kNN does
not beat TriMap at all, that would say the global skeleton it builds is no better than the one PCA
handed TriMap, and the extra machinery was not worth it. Either way, what is still out of reach is the
0.96 trustworthiness that comes from a *principled* neighbor graph and a cross-entropy with a symmetric
repulsion term — instead of three hand-tuned working-zone constants and a hand-scheduled mid-near pull.
That is the gap the next rung, which replaces the empirical three-pair construction with a fuzzy
neighbor graph derived from a per-point Riemannian metric and an edge-existence cross-entropy, is set
up to test.
