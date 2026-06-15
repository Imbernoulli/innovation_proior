UMAP's numbers confirmed the one prediction I made most sharply and then quietly capped the metric I
care about most. Continuity recovered exactly as I bet: MNIST back up from PaCMAP's 0.957 to 0.967,
newsgroups from 0.891 to 0.912, Fashion-MNIST 0.979 → 0.982 — the local connectivity guarantee (every
point reaches its nearest neighbor at full membership) really does stop points being torn off, so
original-space neighbors stay near. And the seed variance finally tightened: MNIST kNN std collapsed
from PaCMAP's 0.0206 to 0.0098, trustworthiness std to 0.0003, because the spectral init is a
deterministic global frame and only the SGD sampling carries noise. So the principled construction
delivered robustness and continuity. But trustworthiness sat *flat* at 0.901 on MNIST — identical to
PaCMAP, nowhere near the 0.96 I keep saying is out there — and kNN actually came in a hair *below*
PaCMAP on MNIST (0.844 vs 0.853). That is the tell. UMAP's cross-entropy spends real capacity on the
(1−v) repulsion over *all* pairs to get global structure right, and that all-pairs repulsion, however
principled, trades against the very tightest local packing. Every rung so far — TriMap's PCA-inherited
frame, PaCMAP's mid-near skeleton, UMAP's spectral init plus symmetric repulsion — has bought global
structure at the cost of a capped local ceiling. The trustworthiness has been stuck at ~0.90 across
three different methods. So the diagnosis is no longer "which graph" or "which repulsion"; it is that
*spending capacity on global structure is itself the thing capping the local score*, and the strongest
local result will come from a method that spends almost *all* its capacity on getting the local
neighborhoods exactly right, paying whatever cost that takes and accepting weaker global guarantees.

So I want to stop balancing global against local and instead match the *local neighborhood structure*
as precisely as I can. Drop the topological-graph framing UMAP used and go back to the probabilistic-
neighbor frame, because it gets local structure right by construction. Center a Gaussian on each point
x_i and read off the probability it would pick x_j as a neighbor:
p_{j|i} = exp(−||x_i − x_j||^2/2σ_i^2) normalized over the other points. The bandwidth σ_i is *per
point*, set by fixing the entropy of each neighbor distribution — the **perplexity**, a smooth
"effective number of neighbors," found by binary search so denser regions automatically get a smaller
σ_i. This is the same per-point adaptive bandwidth UMAP derived from its Riemannian-uniformity argument,
reached here from the entropy side; **perplexity = 30** plays the role UMAP's n_neighbors = 15 played,
the local scale at which I read the data. The high-dimensional side is settled and I keep it.

The trap I have to avoid is *crowding*, and it is worth being precise because it is the exact thing the
weaker rungs danced around with working zones and schedules. Picture a manifold whose intrinsic
dimension is well above two embedded in pixel space. In high dimensions the number of points at a
*moderate* distance from i grows like r^{(high d)}, so there are far more "moderately far" neighbors
than near ones. Squash that into the plane, where the area at radius r grows only like r^2, and there is
nowhere near enough room to seat all those moderately-distant points where they belong; they get shoved
too far out, each still exerting a small Gaussian attraction back toward i, and the *sum* of those tiny
attractions crushes the whole map inward so the clusters never separate. That is why a Gaussian map
kernel fails. The fix is the one move the weaker rungs only approximated: because I am matching
*probabilities* (not distances), the map kernel is free to differ from the high-dimensional Gaussian, so
make it *heavy-tailed* — a Student-t with one degree of freedom, q_ij ∝ (1 + ||y_i − y_j||^2)^{-1}. Its
slow inverse-square tail lets a moderate input similarity be represented by a *much larger* map distance
without a huge attractive penalty, so the moderately-distant points spread out and the gaps open. (This
is the same Student-t UMAP's (1 + a d^{2b})^{-1} generalized at a = b = 1 — but here it is the *whole*
map kernel, not a fitted member of a family, and it is matched against per-point neighbor distributions
rather than edge-existence probabilities.) The inverse-square far field also makes the map
approximately scale-invariant and lets a distant cluster act like a single point, which is what removes
the need for the simulated-annealing schedules the earliest neighbor-embedding methods needed — and,
notably, the kind of hand-tuned three-phase schedule PaCMAP still relies on.

How to measure unfaithfulness? Build a single *joint* distribution over all pairs on each side and
minimize one KL divergence, C = KL(P||Q) = sum p_ij log(p_ij/q_ij). I want the joint, not the per-point
conditionals, both for a clean single-denominator gradient and to anchor outliers: build P by
symmetrizing the conditionals, p_ij = (p_{j|i} + p_{i|j})/(2n), so every point — even a total outlier —
contributes at least 1/(2n) of the probability mass and cannot drift off undetermined. The asymmetry of
KL is a *feature* here: KL(P||Q) punishes a small q modeling a large p (placing truly-near points far
apart) far more than the reverse, so the objective tilts hard toward getting *local* structure right —
which is exactly the capacity allocation the capped rungs would not make. Differentiating the joint KL
with the Student-t kernel gives the clean force law ∇_{y_i}C = 4 sum_j (p_ij − q_ij)(y_i − y_j)(1 +
||y_i − y_j||^2)^{-1}: a pull when the data says two points are more similar than the map shows, a push
when the map has them too close, softened at long range by the Student-t factor. The price is in plain
sight in that gradient: the q_ij normalization is over *all pairs*, so the cost couples every pair and
is O(n^2) per iteration. That is the cost the weaker rungs refused to pay (UMAP's whole design was to
avoid the partition function) — and paying it is precisely what lets the local structure be matched
exactly rather than approximately. On this task n ≤ 5000 per dataset, so O(n^2) is squarely inside the
five-minute CPU budget; the cost that disqualifies KL-matching at millions of points is affordable here,
and that is the lever the harness lets me pull.

Two optimization details earn their keep and replace the scheduling the weaker rungs needed. Initialize
from **PCA** (`init="pca"`) rather than random: a structured start that already carries the coarse
global layout, so the local forces refine a sensible frame instead of inventing one — the same role
PCA played for TriMap and the spectral embedding for UMAP, but here it is genuinely just an init, since
the KL objective itself does the local work. And **early exaggeration**: for the first stretch of
iterations multiply all p_ij by a constant, so the q_ij (which still sum to one) are far too small to
match, and the only way to reduce the KL is to pull the members of each true cluster very tightly
together — the clusters condense into tight, widely separated knots with empty space between them, and
that empty space lets them rearrange into a good global layout before the map relaxes to its real
objective. This is the *derived* analogue of PaCMAP's hand-tuned phase-one mid-near crank, falling out
of the KL rather than being scheduled onto it. The optimizer is momentum gradient descent with a
per-coordinate adaptive gain at **learning_rate="auto"** (scaled from n) for **n_iter = 1000**
iterations. The full module is in the answer; in the scaffold it lands as
`TSNE(n_components=2, perplexity=30.0, learning_rate="auto", init="pca", random_state=..., n_iter=1000)`,
which calibrates the per-point σ by perplexity, symmetrizes into the joint P, runs the early-exaggerated
Student-t KL descent from the PCA init, and threads the seed. (The harness exposes perplexity, the
learning-rate mode, the init, the iteration count, and the seed; the σ binary search, the symmetrization,
the Student-t kernel, and the exaggeration schedule are internal to the sklearn solver.)

This is the strongest rung on the ladder, so I close on the bar it must clear against UMAP's real
numbers, because there is no further rung to fall back on. The whole bet is that spending capacity on
the local KL match instead of on global guarantees lifts the metric three methods got stuck at. So the
falsifiable claim is sharpest on **trustworthiness**: if matching per-point neighbor distributions
directly is what was missing, MNIST trustworthiness should finally break the ~0.90 plateau and land near
0.96, Fashion-MNIST should push from UMAP's 0.959 toward ~0.98, and newsgroups from 0.885 toward ~0.94 —
a clear jump, not the flat 0.901 UMAP and PaCMAP both returned. kNN should rise too, since exact local
neighborhoods help the 7-NN probe: MNIST from 0.844 toward ~0.86, Fashion-MNIST from 0.740 toward
~0.79, newsgroups holding near ~0.69. The honest *risk*, and the place this rung could lose, is
**continuity** and global structure: t-SNE's asymmetric KL barely penalizes placing originally-far
points near each other, so it has no symmetric repulsion holding the global frame — its continuity could
come in *at or just below* UMAP's recovered 0.967 (I expect roughly level, ~0.967 MNIST, because the PCA
init and early exaggeration supply enough global scaffolding), and if continuity instead *collapses*
that would falsify the "PCA-init is enough global structure" assumption and vindicate UMAP's explicit
repulsion. The seed variance should stay tight like UMAP's, since the PCA init is deterministic. If
trustworthiness does *not* break 0.90, the whole thesis — that the cap was the global/local capacity
trade — is wrong, and the ladder would have no rung left that beats UMAP on the metric that matters; so
this is the rung that either tops the ladder or shows the ceiling was real. The ladder ends here because
on this task's k=7 trustworthiness and kNN metrics, no published neighbor-embedding method clearly
beats a direct per-point KL match within the CPU/5-minute/sklearn budget — the remaining published
alternatives (the graph methods already on the ladder) trade local fidelity for scale or global
structure and score *lower* on exactly these local metrics, so there is no genuinely-stronger published
endpoint to add beyond it.
