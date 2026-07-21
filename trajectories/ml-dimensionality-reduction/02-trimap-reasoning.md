PCA's numbers split along the axis I predicted. Continuity is the one thing PCA gets right — 0.927 on
MNIST, 0.968 on Fashion-MNIST, even 0.779 on the diffuse newsgroups data — because a linear projection
does not *tear* nearby points apart; it preserves the coarse layout. But kNN accuracy is on the floor:
0.326 on MNIST, 0.277 on newsgroups, 0.507 on Fashion-MNIST (the clothing classes happen to have more
linear separation, so it does least badly there). Trustworthiness tracks the kNN collapse — 0.671 on
MNIST against the 0.96 a good neighbor embedding reaches — which is the diagnostic signature: the flat
plane folds the manifold and lands points from opposite sides of a fold adjacent, so a point's 2D
neighbors are frequently not its real neighbors. The failure is clean and it is *only* linearity.

Read the gap between continuity and trustworthiness as a number, because it measures *how folded* each
dataset is. On MNIST it is 0.927 − 0.671 = 0.256; on newsgroups 0.779 − 0.559 = 0.220; on Fashion only
0.968 − 0.875 = 0.093. That ordering is the fold depth: MNIST and newsgroups are deeply folded, Fashion
barely — the same story its high kNN told from the other side. So a nonlinear rung should lift MNIST and
newsgroups the hardest and Fashion the least, and if I instead see Fashion move most I have misdiagnosed
the geometry. The seed spread sets a baseline too: PCA's kNN std is about 0.007 on MNIST, and since PCA
is deterministic up to sign that 0.007 is *the scoring probe's own noise floor*, so any nonlinear rung
whose seed spread runs several times that is carrying real method variance.

How to keep what PCA got right while fixing what it got wrong. Route A — stay linear and pick better
directions — is dead on arrival: I proved the loss is linearity itself, and no two linear axes unfold a
curved manifold. Route B — throw PCA away and rebuild the layout from a manifold-learning eigenproblem
(Isomap, LLE) — is the tempting one, so walk it before rejecting it. Isomap builds a kNN graph, replaces
Euclidean distance with shortest-path geodesic distance, and runs classical MDS; on n ≤ 5000 the
all-pairs shortest paths and the n×n eigendecomposition are affordable. But two things kill it. First,
its global layout is *rebuilt from scratch* out of a noisy geodesic estimate, so it discards PCA's
demonstrably-good frame and inherits whatever errors the geodesic graph carries. Second, that geodesic
metric is catastrophically fragile to *shortcut* edges: a single spurious kNN link bridging two folds
short-circuits every shortest path through it and collapses distant regions together, and on
784-dimensional pixel data where distances concentrate, spurious near-edges are exactly what I expect.
LLE has the mirror problem — it preserves only local reconstruction weights and is notorious for
collapsing global structure into near-constant solutions. Both re-derive the global frame, and I have a
*measurement* that PCA's frame is worth keeping (continuity 0.927). That leaves Route C: start from
PCA's globally-correct skeleton and refine it locally. If I get the global structure for free, the only
job left is the local one PCA failed at — a far easier and safer job than building everything.

Why do the nonlinear methods lose global structure in the first place — the failure I am trying *not* to
reintroduce? Every neighbor-embedding method builds a kNN graph, turns each edge into a pairwise *target
distance or similarity*, and lays the points out to match those targets. The trouble is that a pairwise
target is *absolute*: "i and j should be at similarity 0.7" commits to a specific distance, and only
neighbors get a meaningful target, because the Gaussian similarity of two far-apart points is
numerically zero — indistinguishable from any other far pair. So the objective says nothing about how
far apart two *clusters* should be relative to two others; the global arrangement is left dangling and
the optimizer fills it in arbitrarily. That is precisely the failure PCA does *not* have, so leaning on
pairwise targets risks trading PCA's good global structure for better local structure — Route B's
failure again in a different costume.

I want a relation that is *relative*, not absolute, carrying information about more than the nearest
neighbors. The smallest such relation is a triplet: (i, j, k) meaning "i is closer to j than to k." It
demands no particular distance, only an *ordering* — scale-free, robust, a higher-order summary. So the
plan: start from PCA's globally-sensible layout and refine it with triplets sampled from the data's own
high-dimensional geometry.

What is the loss for a single triplet? "i closer to j than k" means a low-dimensional similarity
s(y_i, y_j) should dominate s(y_i, y_k), so minimize the share of similarity mass on the *wrong* side:
w·s(y_i,y_k)/(s(y_i,y_j)+s(y_i,y_k)). When j is near i and k far, the ratio goes to zero and the loss
vanishes; when the order is violated it goes to one, full loss. Bounded in [0,1], smooth, and — the
design choice that matters — it *saturates* at zero rather than continuing to reward an already-satisfied
triplet. I want this and not the log-probability objective some triplet methods use, because the log-prob
form keeps pulling satisfied triplets forever, over-collapsing every neighborhood and slowly chewing up
the very global frame PCA hands me. The saturating loss is what makes refine-from-PCA *safe*: from a PCA
start most neighbor triplets are already nearly satisfied, so their gradient is gentle and the method
makes small local corrections instead of violently rearranging the good global layout.

For the low-dimensional similarity I reuse the heavy-tailed Student-t kernel with one degree of freedom,
s(y_a,y_b) = (1 + ||y_a − y_b||^2)^{-1}: its slow tail gives long-range forces that can pull a misplaced
point back across the map and it avoids crowding; the +1 floors the denominator against coincident
points. Writing d_ij := 1 + ||y_i − y_j||^2, the loss simplifies to l_ijk = w/(1 + d_ik/d_ij). Put the
saturation on a scale: fix d_ij = 2 and slide k out — at d_ik = 4 the loss is w/3 = 0.33w, at d_ik = 20
it is 0.091w, at d_ik = 200 it is 0.010w. Once the outlier is an order of magnitude farther than the
neighbor the loss is a hundredth of full, and pushing it farther buys almost nothing — that is the
saturation working, the property that lets a triplet stop meddling once the order is right. The total
loss sums l_ijk over a sampled triplet set.

Now the weight w, because not all triplets are equally informative. A triplet where k is *enormously*
farther than j in high dimension is strong evidence; one that is nearly a tie is dominated by noise. So
the weight grows with the high-dimensional *margin*, the locally-scaled squared-distance gap
δ_ik^2 − δ_ij^2, non-negative if j is the closer point. Raw squared distances are dangerous across
regions of different density — in a dense cluster all distances are small, in a sparse one all large — so
I scale by per-point local lengths, δ_ij^2 = ||x_i − x_j||^2/(σ_i σ_j), with σ_i the average distance to
i's 4th–6th nearest neighbors: the nearest one or two are the noisiest estimate of local scale (a single
duplicate-ish image collapses σ_i), so a mid-rank band gives a stable per-point yardstick that still
tracks density. These margins are heavy-tailed — a handful of triplets straddling two very-distant
clusters have gigantic margins and would dominate the gradient — so I keep the *ordering* of importance
but compress the dynamic range with a deformed logarithm log_t at t = 0.5, after shifting the margins so
the smallest is zero.

Take a dense cluster where a point's neighbors sit at raw squared distance ~1 and outliers ~4, and a
sparse cluster where the same relative geometry lives at ~100 and ~400. The raw margins are 3 and 300 —
a hundredfold difference for the *identical* local configuration, which would tell the optimizer the
sparse triplet is a hundred times more important when it is not. Divide by σ_i σ_j (~1 dense, ~100
sparse) and the scaled margins become 3 and 3 — equal, as they should be. Then the tempered log flattens
what is left, growing like a square root, so a giant cross-cluster margin no longer dominates a routine
one by a factor of a hundred. The diagnosis I carry forward: after this transform most weights end up
*near each other*; if so, the elaborate weighting is doing less than it looks and a later rung could drop
it — but I cannot know that from a PCA start, so I keep it and watch.

Which triplets, and how many? I cannot use all O(n^3) and should not — most are redundant. For each
point i the informative j is a *near* neighbor, so I take i's nearest neighbors as candidate j's and, for
each, sample a few "outlier" k's uniformly from outside the already-closer neighbors. With this task's
configuration that is **10 inliers** per point, each paired with **5 outliers** — fifty near-neighbor
triplets per point — linear in n. At n = 5000 that is about 275,000 triplets against the O(n^3) = 1.25×10^11
full set, a vanishing fraction, and the redundancy is why sampling is safe; the cost stays dominated by
the approximate-nearest-neighbor graph, not the triplet sweep. (The lever the harness exposes is the
library's `n_inliers`/`n_outliers`/`n_random`/`n_iters`, not the internal kernel or weight transform.)

But do these fifty near-neighbor triplets actually preserve the global structure I am chasing? Every one
has j as a *neighbor* of i; none constrains how two *far-apart* points sit relative to each other. Build
the smallest counterexample: three tight clusters A, B, C along a line, sampling only near-neighbor
triplets — every j inside i's own cluster, every outlier k in some other cluster. The loss is satisfied
as long as each point's within-cluster neighbors are closer than points in *any* other cluster. But that
is silent about the *order* of the three clusters: A–B–C, B–A–C, or the whole thing folded into a ball
all keep every near-neighbor triplet satisfied, because within-cluster is still closer than across. The
loss is invariant to the global permutation — it can be zero *amidst a complete sacrifice of global
structure*. This is exactly the trap the pairwise local methods fall into, dressed as triplets. Adding a
near-neighbor triplet loss by itself buys no global structure, and that is the failure I must not
reintroduce, since I would be giving back PCA's one strength.

Two routes out. One: add a few triplets between non-neighbors — **5 random triplets** per point, j and k
uniform and oriented by their high-D distance, to give the loss a faint say over distant placement. But
that signal is thin and noisy (a random far pair is a weak, high-variance constraint) and gets
downweighted precisely because it is unreliable; leaning the whole global layout on it is a bad bet.
Route two is the one I parked: do not *discover* the global layout from triplet forces at all — *start*
from a layout that is already globally correct and let the triplets only refine it. And I have that
layout: PCA, whose continuity my feedback measured at 0.927/0.968. Initialize Y to the PCA embedding
(scaled down by a small constant so the Student-t kernel operates in its sensitive range rather than
saturated). From a PCA start the clusters are already in their globally-faithful positions and far points
are already far, so essentially *no* triplet demands that two far-apart points be pushed apart — that
work is done. The only work left is to sharpen local neighborhoods, all without any large force that
would tear the global frame. The neighbor triplets, a liability from a random start, become exactly right
when the global frame is already correct — and starting from the structured PCA solution converges far
faster, so the **400 iterations** the harness budgets are enough.

The optimizer is full-batch gradient descent on the fixed triplet set (sampled once, never resampled)
with momentum and a per-coordinate adaptive gain, evaluated at a look-ahead point for stability. Each
triplet touches only three rows of the gradient — a pull along i–j scaled by d_ik and a push along i–k
scaled by d_ij, both shrunk by the squared denominator (d_ij + d_ik)^2, so the same term that made the
*loss* saturate at 0.010w for a far outlier makes the *force* vanish there too. A full sweep is O(n) per
iteration. The full procedure is in the answer; in the scaffold it lands as
`trimap.TRIMAP(n_dims=2, n_inliers=10, n_outliers=5, n_random=5, n_iters=400)`.

Now the falsifiable expectations against PCA's numbers. The bet is "keep PCA's global structure, add the
local fidelity it lacks," so I expect kNN to jump off PCA's floor into the neighbor-embedding regime on
all three datasets — a large lift, since the triplets unfold the local manifold the linear map folded.
The fold-depth ordering predicts the *shape* of the jump: MNIST and newsgroups, the deeply-folded pair,
should show the largest lifts, Fashion the least. Trustworthiness should climb sharply with it as the
folded false neighbors get separated. Continuity is the delicate one: it is already high, and the *risk*
of any nonlinear refinement is that pulling neighbors tight tears some original-space neighbors apart, so
I expect it to stay in PCA's high band rather than improve — the global frame is inherited, not
re-derived — and if it instead *falls* well below PCA's that would falsify the refine-from-PCA story and
say the triplets are overpowering the frame. Seed spread should be modest — larger than PCA's 0.007
probe floor, since triplet sampling carries randomness, but pinned by the deterministic PCA init. The one
thing this rung structurally *cannot* do is push trustworthiness to the top of the range a direct
local-affinity match reaches: its global structure is *inherited* and its local fidelity comes from
relative-order triplets that stop pulling once satisfied. That gap is what a method putting forces on
moderate-distance pairs — building the global skeleton into the objective itself — is meant to close next.
