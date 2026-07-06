PCA's numbers tell me exactly what a flat shadow of a curved cloud does, and they split along the axis
I predicted. Continuity is the one thing PCA gets right — 0.927 on MNIST, 0.968 on Fashion-MNIST, even
0.779 on the diffuse newsgroups data — because a linear projection does not *tear* nearby points apart;
it preserves the coarse layout, so original-space neighbors mostly stay in the same region of the map.
But kNN accuracy is on the floor: 0.326 on MNIST, 0.277 on newsgroups, 0.507 on Fashion-MNIST (the
clothing classes happen to have more linear separation than the digits, so it does least badly there).
Trustworthiness tracks the kNN collapse — 0.671 on MNIST against the 0.96 a good neighbor embedding
reaches — which is the diagnostic signature: PCA *invents* false neighbors. The flat plane folds the
manifold and lands points from opposite sides of a fold adjacent, so a point's 2D neighbors are
frequently not its real neighbors, and a 7-NN classifier in that smeared blob cannot separate the
classes. The failure is clean and it is *only* linearity — PCA loses nothing for any other reason.

Let me read the gap between continuity and trustworthiness as a number, because it measures *how folded*
each dataset is and tells me where the second rung has the most work to do. On MNIST continuity minus
trustworthiness is 0.927 − 0.671 = 0.256; on newsgroups 0.779 − 0.559 = 0.220; on Fashion-MNIST only
0.968 − 0.875 = 0.093. That ordering is the fold depth: MNIST and newsgroups are deeply folded (the map
keeps true neighbors near but invents a mass of false ones), while Fashion is barely folded, which is
the same story its high kNN told from the other side. So the nonlinear rung should lift MNIST and
newsgroups the hardest and Fashion the least, and if I instead see Fashion move most I have
misdiagnosed the geometry. The seed spread also sets a baseline I will want later: PCA's kNN std is
about 0.007 on MNIST, and since PCA is deterministic up to sign that 0.007 is *the scoring probe's own
noise floor* — the stratified split, not the method — so any nonlinear rung whose seed spread runs
several times that is carrying real method variance, not just probe noise.

So the question is how to keep what PCA got right while fixing what it got wrong, and I have three real
routes on the table. Route A: stay linear and pick better directions — but I already proved the loss is
linearity itself, and no two linear axes can unfold a curved manifold, so this cannot move the folded
metrics at all. Route B: throw PCA away and rebuild the layout from a manifold-learning eigenproblem —
Isomap or LLE or Laplacian eigenmaps. Route C: keep PCA's frame and refine it. Route B is the tempting
one, so let me actually walk it before I reject it. Isomap builds a kNN graph, replaces Euclidean
distance with shortest-path geodesic distance along that graph, and runs classical MDS — double-center
the squared-geodesic matrix and take its top two eigenvectors. On n ≤ 5000 the all-pairs shortest paths
and the n×n MDS eigendecomposition are affordable (a 5000×5000 double-centered matrix is 25 million
entries, a couple hundred megabytes, inside budget). But two things kill it for my purpose. First, its
global layout is *rebuilt from scratch* out of a noisy geodesic estimate, so it does not preserve PCA's
demonstrably-good frame — it re-derives global structure and inherits whatever errors the geodesic
graph carries. Second, that geodesic metric is catastrophically fragile to *shortcut* edges: a single
spurious kNN link bridging two folds of the manifold short-circuits every shortest path through it and
collapses distant regions together, and on 784-dimensional pixel data where distances concentrate,
spurious near-edges are exactly what I expect. LLE has the mirror problem — it preserves only local
reconstruction weights and is notorious for collapsing global structure into degenerate near-constant
solutions. Both of Route B's members re-derive the global frame; and I have a *measurement* that PCA's
frame is worth keeping (continuity 0.927). Discarding a frame I know is good to rebuild it from a
fragile graph is the wrong bet. That leaves Route C, and the shape of PCA's result is a gift, not just a
baseline: its continuity is high because the global arrangement of the clusters is faithful — that is
what variance-maximization buys, the coarse relative placement of the blobs. What it lacks is *local*
fidelity inside and between neighborhoods. So I want to *start* from PCA's globally-correct skeleton and
refine it locally, bending neighborhoods to follow the manifold without disturbing the global frame.
That is the whole strategy, and it is worth pinning down why it is the right one before I build the
machinery — because if I get the global structure for free from PCA, the only job left for the nonlinear
forces is the local one PCA failed at, which is a far easier and safer job than building everything.

First, why do the nonlinear methods lose global structure in the first place — the failure I am trying
*not* to reintroduce? Every neighbor-embedding method works the same way: build a kNN graph, turn each
edge into a pairwise *target distance or similarity*, and lay the points out to match those targets. The
trouble with a pairwise target is that it is *absolute*: "i and j should be at similarity 0.7" commits
to a specific distance, and only neighbors get a meaningful target, because the Gaussian similarity of
two far-apart points is numerically zero — indistinguishable from any other far pair. So the objective
says nothing about how far apart two *clusters* should be relative to two *other* clusters. The global
arrangement is left dangling and the optimizer fills it in arbitrarily. That is precisely the failure
PCA does *not* have. So if I lean on pairwise targets I risk trading PCA's good global structure for
better local structure — exactly the wrong trade, and exactly Route B's failure again in a different
costume.

I want a relation that is *relative*, not absolute, and that can carry information about more than the
nearest neighbors. The smallest relation encoding relative order is a triplet: (i, j, k) meaning "i is
closer to j than to k." A triplet demands no particular distance, only an *ordering* — scale-free,
robust, and a higher-order summary. This is the seed I want: refine an initial representation so that a
set of relative comparisons is satisfied while staying close to the initial one. I will not build the
map from triplets out of nothing; I will start from PCA's globally-sensible layout and refine it with
triplets sampled from the data's own high-dimensional geometry.

What is the loss for a single triplet? The relation I care about is "i closer to j than k," so in the
low-dimensional map a similarity s(y_i, y_j) should dominate s(y_i, y_k). The natural thing to minimize
is the share of similarity mass sitting on the *wrong* side: w·s(y_i,y_k)/(s(y_i,y_j)+s(y_i,y_k)). When
j is near i and k is far, the ratio goes to zero and the loss vanishes — the triplet is satisfied and
exerts (almost) no force; when the order is violated the ratio goes to one, full loss. Bounded in [0,1],
smooth, and — this is the design choice that matters — it *saturates* at zero rather than continuing to
reward an already-satisfied triplet. I want this and not the log-probability objective some triplet
methods use, because the log-prob form keeps pulling satisfied triplets forever, which over-collapses
every neighborhood and would slowly chew up the very global frame PCA hands me. The saturating loss is
what makes refine-from-PCA *safe*: from a PCA start most neighbor triplets are already nearly satisfied,
so their gradient is gentle and the method makes small local corrections instead of violently
rearranging the good global layout.

For the low-dimensional similarity I reuse the heavy-tailed Student-t kernel with one degree of freedom,
s(y_a,y_b) = (1 + ||y_a − y_b||^2)^{-1}. Its slow tail gives long-range forces that can pull a misplaced
point back across the map and it avoids the crowding problem; the +1 floors the denominator so two
coincident embedding points never divide by zero. Writing d_ij := 1 + ||y_i − y_j||^2, the loss
simplifies to l_ijk = w/(1 + d_ik/d_ij): small exactly when d_ik ≫ d_ij, i.e. k much farther than j in
the map. Let me put the saturation on a scale, because "saturates" is a claim I can check with numbers.
Fix d_ij = 2 (a satisfied neighbor at squared map distance 1) and slide k out. At d_ik = 4 the loss is
w/(1 + 2) = 0.33w; at d_ik = 20 it is w/(1 + 10) = 0.091w; at d_ik = 200, w/(1 + 100) = 0.010w. So once
the outlier is an order of magnitude farther than the neighbor the loss is already a hundredth of full,
and pushing it farther buys almost nothing — the force dies. That is the saturation working, and it is
exactly the property that lets the triplet stop meddling once the order is right. The total loss sums
l_ijk over a sampled triplet set.

Now the weight w, because not all triplets are equally informative. A triplet where k is *enormously*
farther than j in high dimension is strong, reliable evidence about the geometry — insist the map
respect it. A triplet that is nearly a tie is dominated by noise — leaning on it just fits noise. So the
weight grows with the high-dimensional *margin*, the locally-scaled squared-distance gap
δ_ik^2 − δ_ij^2, which is non-negative by construction if j is sampled as the closer point. Raw squared
distances are dangerous across regions of different density — in a dense cluster all distances are
small, in a sparse one all large, so the same margin means different things — so I scale by per-point
local lengths, δ_ij^2 = ||x_i − x_j||^2/(σ_i σ_j), with σ_i read off i's own neighborhood (the average
distance to its 4th–6th nearest neighbors: far enough to be stable, close enough to stay local). Why the
4th–6th and not the 1st? The nearest one or two neighbors are the noisiest estimate of local scale — a
single duplicate-ish image can make σ_i collapse — so averaging a small mid-rank band gives a stable
per-point yardstick that still tracks local density. These margins are heavy-tailed: a handful of
triplets straddling two very-distant clusters have gigantic margins and would dominate the gradient. I
want to keep the *ordering* of importance but compress the dynamic range, so I temper the weights with a
deformed logarithm log_t at t = 0.5 — a gentler-than-log square-root-flavored compression — after
shifting the margins so the smallest is zero.

Let me make the density-scaling concrete, because it is the piece most likely to be quietly wrong. Take
a dense cluster where a point's neighbors sit at raw squared distance around 1 and its outliers around 4,
and a sparse cluster where the same relative geometry lives at squared distances around 100 and 400. The
raw margins are 4 − 1 = 3 and 400 − 100 = 300 — a hundredfold difference for the *identical* local
configuration, which would tell the optimizer the sparse triplet is a hundred times more important when
it is not more informative at all. Divide each squared distance by the local scale σ_i σ_j, and if σ in
the dense region is ~1 and in the sparse region ~10 (so σ_i σ_j ~ 1 and ~100 respectively), the scaled
margins become 4/1 − 1/1 = 3 and 400/100 − 100/100 = 3 — equal, as they should be. The per-point σ is
what makes "closer than" mean the same thing in both regions. And then the tempered log flattens what is
left: after shifting so the smallest margin is zero, the deformed logarithm at t = 0.5 grows like a
square root rather than a logarithm, so a giant cross-cluster margin of, say, 100 gets compressed to
roughly the scale of √-ish growth instead of dominating a routine margin of 1 by a factor of 100. The
diagnosis I will carry forward is that after this transform most weights end up *near each other*; if
that is true the elaborate weighting is doing less than it looks, and a later rung could drop it — but I
cannot know that from a PCA start, so I keep it and watch.

Which triplets, and how many? I cannot use all O(n^3) and should not — most are redundant. For each
point i, the informative j is a *near* neighbor (the triplet then says "keep this neighbor close
relative to something farther"), so I take i's nearest neighbors as candidate j's and, for each, sample
a few "outlier" k's uniformly from outside the already-closer neighbors. With this task's
configuration that is **10 inliers** per point, each paired with **5 outliers** — fifty near-neighbor
triplets per point — and the triplet count is linear in n, which is what keeps the cost dominated by
the approximate nearest-neighbor search rather than the triplet sweep. Let me count it against the
budget so I know it lands. Fifty neighbor triplets plus five random ones is fifty-five per point, and at
n = 5000 that is about 275,000 triplets — set against the O(n^3) = 1.25×10^11 full set, I am using a
vanishing fraction, and the redundancy is exactly why sampling is safe. A full-batch gradient sweep
touches each triplet once, so 275,000 gradient contributions per iteration, and the harness budgets a
few hundred iterations; even at 400 iterations that is 1.1×10^8 triplet evaluations over the whole run,
each O(1) — a rounding error next to the approximate-nearest-neighbor construction of the graph, which
is the real cost. (I note the lever the harness exposes is the library's
`n_inliers`/`n_outliers`/`n_random`/`n_iters`, not the internal kernel or weight transform; those are
fixed inside the implementation, and the method is robust to the exact counts because the triplets are
highly redundant.)

But now I have to check whether these fifty near-neighbor triplets actually preserve the global
structure I am chasing, because that was the whole point. Every one of them has j as a *neighbor* of i;
none constrains how two *far-apart* points sit relative to each other. Suppose the optimizer drives the
loss to near zero — every neighbor closer than its sampled outliers. Does that pin down the global
arrangement? Let me build the smallest counterexample I can and check. Put three tight clusters A, B, C
along a line, and sample only near-neighbor triplets — every j inside i's own cluster, every outlier k
in some other cluster. The loss is satisfied as long as each point's within-cluster neighbors are
closer than points in *any* other cluster. But that condition is completely silent about the *order* of
the three clusters along the line: I can lay them out A–B–C, or B–A–C, or fold the whole thing into a
ball, and every near-neighbor triplet stays satisfied because within-cluster is still closer than
across-cluster in all of those layouts. The loss is genuinely invariant to the global permutation. So
the loss can be zero *amidst a complete sacrifice of global structure* — pictured as the manifold
relaxed into a flat line, locally every neighbor is still nearest, the neighbor-triplet loss is
essentially zero, and the global structure is destroyed. This is exactly the trap the pairwise local
methods fall into, just dressed as triplets. Adding a near-neighbor triplet loss, by itself, buys me no
global structure at all — and that is the failure I must not reintroduce, since I would be giving back
PCA's one strength.

Two routes out. Route one: add a few triplets between non-neighbors — with this task's configuration,
**5 random triplets** per point, picking j and k uniformly and orienting them by their high-D distance,
to give the loss a faint say over the relative placement of distant points. These carry a thin global
signal, but it is thin and noisy (a random pair of far points is a weak, high-variance constraint) and
gets downweighted precisely because it is unreliable. Leaning the whole global layout on it is a bad
bet. Route two is the one I parked: do not *discover* the global layout from triplet forces at all —
*start* from a layout that is already globally correct and let the triplets only refine it locally. And
I already know that layout: PCA — globally optimal by construction, the very thing whose continuity my
feedback measured at 0.927/0.968. Initialize Y to the PCA embedding (scaled down by a small constant so
the Student-t kernel operates in its sensitive range rather than saturated). From a PCA start the
clusters are already in their globally-faithful relative positions and far points are already far, so
there are essentially *no* triplets demanding that two far-apart points be pushed apart — that work is
done. The only work left is to sharpen local neighborhoods: pull each point's true neighbors in tight,
nudge outliers out, unfold the local manifold — all without any large force that would tear the global
frame PCA established. The neighbor triplets, a liability for global structure from a random start,
become exactly right when the global frame is already correct. And there is a free benefit: starting
from the structured PCA solution converges far faster than from random noise, so the **400 iterations**
the harness budgets are enough.

The optimizer is full-batch gradient descent on the fixed triplet set (sampled once, never resampled)
with momentum and a per-coordinate adaptive gain — calmer momentum early while the layout settles, then
faster — evaluated at a look-ahead point for stability. Each triplet touches only three rows of the
gradient (a symmetric pull along i–j scaled by d_ik and a push along i–k scaled by d_ij, both shrunk by
the squared denominator d_ij + d_ik so an already-satisfied triplet exerts almost no force — the
saturation, now visible in the gradient), so a full sweep is O(n) per iteration. That the same shrink
factor (d_ij + d_ik)^2 sits in the denominator of the gradient is the reassurance I wanted: the same
term that made the *loss* saturate at 0.010w for a far outlier makes the *force* vanish there too, so
the derivation is internally consistent and a satisfied triplet is quiet in both the value and its
slope. The full procedure is in the answer; here it lands as the library call `trimap.TRIMAP(n_dims=2,
n_inliers=10, n_outliers=5, n_random=5, n_iters=400)`, which builds the kNN graph, samples the
triplets, computes locally-scaled tempered-log margins, initializes from PCA, and runs exactly this
descent.

Now the falsifiable expectations against PCA's numbers. The whole bet is "keep PCA's global structure,
add the local fidelity it lacks." So I expect the kNN accuracy to jump from PCA's floor toward the
neighbor-embedding regime — MNIST from 0.326 into the low-to-mid 0.8s, newsgroups from 0.277 up past
0.6, Fashion-MNIST from 0.507 into the low 0.7s — because the triplets unfold the local manifold the
linear map folded. And the fold-depth ordering I computed predicts the *shape* of the jump: MNIST and
newsgroups, the two deeply-folded datasets (continuity−trustworthiness of 0.256 and 0.220), should show
the largest lifts because they have the most folded local structure to unfold, while Fashion (gap 0.093)
was barely folded and has less to gain. Trustworthiness should climb sharply with it, from 0.671 toward
~0.89 on MNIST, as the false neighbors the fold invented get separated. Continuity is the delicate one:
PCA's continuity is already high (0.927 MNIST), and the *risk* of any nonlinear refinement is that
pulling neighbors tight tears some original-space neighbors apart, so continuity could even dip slightly
relative to PCA on the datasets where PCA was strongest — I expect it to stay in the same high band
(~0.958 MNIST, ~0.983 Fashion-MNIST) rather than improve much, because the global frame is inherited,
not re-derived. If instead the global structure collapses (continuity *falls* well below PCA's) that
would falsify the refine-from-PCA story and say the triplets are overpowering the frame. The
seed-to-seed spread should be modest — larger than PCA's 0.007 probe floor, since the triplet sampling
now carries real randomness, but not wild, since the PCA init is deterministic and pins the frame. The
one thing this rung structurally *cannot* do is push trustworthiness into the 0.96 range a direct
local-affinity match reaches: TriMap's global structure is *inherited* from PCA and its local fidelity
comes from relative-order triplets that stop pulling once satisfied, so it will land a strong but not
top-tier local score — which is exactly the gap the next rung, a method that puts forces on
moderate-distance pairs to build the global skeleton into the objective itself, is meant to close.
