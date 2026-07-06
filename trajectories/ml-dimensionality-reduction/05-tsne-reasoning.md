UMAP's numbers confirmed the one prediction I made most sharply and then quietly capped the metric I
care about most. Continuity recovered exactly as I bet: MNIST back up from PaCMAP's 0.957 to 0.967,
newsgroups from 0.891 to 0.912, Fashion-MNIST 0.979 → 0.982 — the local connectivity guarantee (every
point reaches its nearest neighbor at full membership) really does stop points being torn off, so
original-space neighbors stay near. And the seed variance finally tightened: MNIST kNN std collapsed
from PaCMAP's 0.0206 to 0.0098, trustworthiness std to 0.0003, because the spectral init is a
deterministic global frame and only the SGD sampling carries noise. The size of the continuity recovery is worth reading as a number, because it tracks the fold depth I have
been carrying since PCA: newsgroups recovered the most (+0.021, from 0.891 to 0.912), MNIST next (+0.010),
Fashion least (+0.003, from 0.979 to 0.982) — the same ordering as the datasets' fold depth, which is
what I would expect if the local-connectivity guarantee is repairing exactly the neighbors PaCMAP's hard
repulsion had torn. So the principled construction delivered robustness and continuity, both of the things
I predicted the derived graph would buy. But
trustworthiness sat *flat* at 0.901 on MNIST — identical to PaCMAP, nowhere near the 0.96 I keep saying
is out there — and kNN actually came in a hair *below* PaCMAP on MNIST (0.844 vs 0.853), exactly the
small loss I flagged as possible when the all-pairs repulsion spends capacity on the frame. That is the
tell. Let me lay the three trustworthiness numbers side by side, because the pattern is the whole
argument: TriMap 0.890, PaCMAP 0.901, UMAP 0.901. Three different methods — a PCA-inherited frame, a
mid-near skeleton, a spectral init plus symmetric repulsion — and the metric has moved a total of 0.011
across all of them and then stopped dead. UMAP's cross-entropy spends real capacity on the (1−v)
repulsion over *all* pairs to get global structure right, and that all-pairs repulsion, however
principled, trades against the very tightest local packing. Every rung so far has bought global structure
at the cost of a capped local ceiling. So the diagnosis is no longer "which graph" or "which repulsion";
it is that *spending capacity on global structure is itself the thing capping the local score*, and the
strongest local result will come from a method that spends almost *all* its capacity on getting the local
neighborhoods exactly right, paying whatever cost that takes and accepting weaker global guarantees.

Let me be honest about the fork, because the tempting move is to keep tuning UMAP. I could shrink
min_dist toward zero to pack neighborhoods tighter — but that only sets how close the layout *allows*
points to sit; it does not remove the all-pairs (1−v) repulsion that is spending the capacity, so it
would tighten the packing a little while the same global term keeps clawing it back, and I would be
adjusting the operating point of a machine whose ceiling is fixed by its objective. The three-method
plateau at ~0.90 is the evidence: this is not a knob I have failed to turn, it is a ceiling built into
"spend capacity on an all-pairs global term." A subtler variant of the same temptation is to keep UMAP's
principled fuzzy graph but swap its cross-entropy for a sharper local objective — a hybrid. That fails for
a mechanical reason worth naming: UMAP's efficiency comes precisely from *avoiding* the all-pairs
partition function, via edge sampling and negative sampling, so any objective bolted onto that machinery
inherits an *approximate* local match — the repulsion is a stochastic estimate over a handful of negative
samples, not the exact all-pairs sum. The cap is not in the choice of graph; it is in refusing to pay the
partition function. The other path is to stop balancing global against local altogether and instead match
the *local neighborhood structure* as precisely as I can, paying the exact all-pairs cost the graph
methods refused to pay. Drop the topological-graph framing UMAP used and go back to the
probabilistic-neighbor frame, because it gets local structure right by construction. Center a Gaussian on
each point x_i and read off the probability it would pick x_j as a neighbor:
p_{j|i} = exp(−||x_i − x_j||^2/2σ_i^2) normalized over the other points. The bandwidth σ_i is *per
point*, set by fixing the entropy of each neighbor distribution — the **perplexity**, a smooth
"effective number of neighbors," found by binary search so denser regions automatically get a smaller
σ_i. Concretely the target entropy is log2(perplexity): at perplexity = 30 that is log2(30) ≈ 4.9 bits,
and σ_i is the unique bandwidth making point i's neighbor distribution carry that much entropy — raise
σ_i and the distribution spreads and its entropy rises, lower it and the entropy falls, a one-dimensional
monotone equation solved in a handful of bisection steps. This is the same per-point adaptive bandwidth
UMAP derived from its Riemannian-uniformity argument, reached here from the entropy side; perplexity = 30
plays the role UMAP's n_neighbors = 15 played, the local scale at which I read the data. It is worth
noting the scale is a touch larger than UMAP's 15, and that is deliberate for a method whose whole job is
the local match: a slightly wider effective neighborhood gives each per-point Gaussian more neighbors to
average over, which makes the σ_i estimate steadier and the p_{j|i} distribution less jagged, at the cost
of reading structure at a marginally coarser scale — a good trade when I am about to spend all my capacity
on matching that distribution exactly. And the density adaptivity is the concrete payoff of fixing entropy
rather than distance: in a dense cluster the neighbors are packed close, so a small σ_i already spreads
about 4.9 bits of entropy across roughly thirty of them, while in a sparse region the same 4.9-bit target
forces a large σ_i to reach out far enough to gather an equivalent effective count — so "thirty
neighbors" means the same *informational* thing in both regions even though the physical radius differs by
an order of magnitude, exactly the per-point scaling the graph methods reached from the density side. The
high-dimensional side is settled and I keep it.

The trap I have to avoid is *crowding*, and it is worth being precise because it is the exact thing the
weaker rungs danced around with working zones and schedules. Picture a manifold whose intrinsic
dimension is well above two embedded in pixel space. In high dimensions the number of points at a
*moderate* distance from i grows like r^(intrinsic d), so there are far more "moderately far" neighbors
than near ones. Let me put a number on how badly that fails to fit the plane. Suppose the intrinsic
dimension is ten. Double the radius and the count of points enclosed grows by 2^10 = 1024 in the data,
but the area available to seat them in the plane grows only by 2^2 = 4. So a thousand points that belong
at moderate distance from i must be crammed into room for four; they get shoved too far out, each still
exerting a small Gaussian attraction back toward i, and the *sum* of those tiny attractions — a thousand
of them — crushes the whole map inward so the clusters never separate. That is why a Gaussian map kernel
fails, and the 1024-against-4 count is why the failure is severe rather than cosmetic. The fix is the one
move the weaker rungs only approximated: because I am matching *probabilities* (not distances), the map
kernel is free to differ from the high-dimensional Gaussian, so make it *heavy-tailed* — a Student-t with
one degree of freedom, q_ij ∝ (1 + ||y_i − y_j||^2)^{-1}. Let me quantify how much tail that buys against
the Gaussian I am rejecting, because the whole crowding fix rides on it. Compare the two map kernels at a
moderate map distance of 3, i.e. squared distance 9. A Gaussian map kernel would assign exp(−9) ≈ 1.2e-4;
the Student-t assigns 1/(1 + 9) = 0.1 — about 800 times larger. So under a Gaussian map, to represent a
pair's moderate input similarity I would have to place them almost on top of each other, since the
Gaussian's similarity has already fallen to a ten-thousandth by distance 3; the Student-t is content to
represent that same similarity at distance 3 and beyond, so the moderately-distant points are *allowed* to
spread out instead of being crushed inward. The 800-fold tail is exactly the room the 1024-against-4
crowding count said I needed. And the crowding is worst on exactly the datasets I have been struggling
with: the 784-pixel images carry a high intrinsic dimension, so the r^(intrinsic d) blow-up is fierce and
the heavy tail matters most there, whereas the newsgroups cloud is already pre-reduced to 50 dimensions,
a milder crowding regime — which is one reason I expect the biggest trustworthiness gains on the image
datasets even though all three should break the plateau. Its slow inverse-square tail lets a moderate
input similarity be represented by a *much larger* map distance without a huge attractive penalty, so the
moderately-distant points spread out and the gaps open. (This is the same Student-t UMAP's
(1 + a d^{2b})^{-1} generalized at a = b = 1 — but here it is the *whole* map kernel, not a fitted member
of a family, and it is matched against per-point neighbor distributions rather than edge-existence
probabilities.) The inverse-square far field also makes the map approximately scale-invariant and lets a
distant cluster act like a single point, which is what removes the need for the simulated-annealing
schedules the earliest neighbor-embedding methods needed — and, notably, the kind of hand-tuned
three-phase schedule PaCMAP still relies on.

How to measure unfaithfulness? The obvious first try is the one the original neighbor-embedding method
used: keep the per-point conditionals on both sides and minimize the sum over points of KL(P_i || Q_i),
one divergence per point. Let me see why that is not quite what I want, because the reason names the fix.
Two problems. First, the gradient carries a *per-point* denominator — each Q_i has its own normalization
over j — so the force law is messier and every point's normalizer must be tracked separately. Second, and
worse for these metrics, a genuine outlier x_i sits far from everything, so its conditional P_i is nearly
uniform and tiny, and *nothing* in the objective forces the map to place it anywhere in particular; it can
drift off to the edge undetermined, and an undetermined outlier is a false neighbor waiting to happen.
Both problems are fixed by building a single *joint* distribution over all pairs on each side and
minimizing one KL divergence, C = KL(P||Q) = sum p_ij log(p_ij/q_ij). I want the joint, not the per-point
conditionals, both for a clean single-denominator gradient and to anchor outliers: build P by
symmetrizing the conditionals, p_ij = (p_{j|i} + p_{i|j})/(2n), so every point — even a total outlier —
contributes at least 1/(2n) of the probability mass and cannot drift off undetermined. That floor is
worth making concrete: at n = 5000 an outlier still owns at least 1/(2n) = 1/10000 = 1e-4 of the mass, so
the objective always has a nonzero grip on it and it cannot wander off to infinity the way a purely
conditional formulation would let it. Now the asymmetry of KL, which I want to check does what I claim
because it is the linchpin of the whole "tilt toward local" argument. KL(P||Q) sums p_ij log(p_ij/q_ij),
and the p_ij out front is the weight. Take a truly-*near* pair the map has placed far apart: p high, say
0.1, and q low, say 0.001, contributing 0.1·log(0.1/0.001) = 0.1·log(100) ≈ 0.46 — a large penalty. Now
take a truly-*far* pair the map has placed near: p low, 0.001, and q high, 0.1, contributing
0.001·log(0.001/0.1) = 0.001·log(0.01) ≈ −0.0046 — a hundred times smaller in magnitude, because the tiny
p in front throttles it. So the objective screams when it places near points far apart and barely
whispers when it places far points near. That single asymmetry does two things at once: it tilts the
whole method hard toward getting *local* structure right — exactly the capacity allocation the capped
rungs would not make — and it is *also* the source of the global-structure risk I will flag in the
predictions, because "barely whispers when far points are placed near" means there is almost no force
holding the global frame. One computation, both consequences. Differentiating the joint KL with the
Student-t kernel gives the clean force law ∇_{y_i}C = 4 sum_j (p_ij − q_ij)(y_i − y_j)(1 +
||y_i − y_j||^2)^{-1}: a pull when the data says two points are more similar than the map shows, a push
when the map has them too close, softened at long range by the Student-t factor. The sign is worth
reading directly off the formula, because it is the entire mechanism. The gradient descent step moves y_i
against ∇_{y_i}C, so consider a pair where p_ij > q_ij — the data says i and j are more similar than the
map currently shows. Then (p_ij − q_ij) > 0, the gradient term points along (y_i − y_j), and stepping
against it moves y_i *toward* y_j: an attraction, exactly as it should be for an under-represented
neighbor. Where q_ij > p_ij — the map has crammed two points closer than the data warrants — the sign
flips and the step pushes y_i away: a repulsion. And both are scaled by the Student-t factor
(1 + ||y_i − y_j||^2)^{-1}, so a pair already far apart in the map feels a weak force regardless of sign,
which is what lets a distant cluster act as a single body rather than being torn at by every far point.
The force law is not an extra ingredient; it is what the joint KL *is* once differentiated.

The price is in plain sight in that gradient: the q_ij normalization is over *all pairs*, so the cost
couples every pair and is O(n^2) per iteration. That is the cost the weaker rungs refused to pay (UMAP's
whole design — the negative sampling, the edge decomposition — was built to avoid this partition
function) — and paying it is precisely what lets the local structure be matched exactly rather than
approximately. So the question is whether I can afford it here, and this is where the harness's operating
envelope becomes the lever. On this task n ≤ 5000 per dataset, so the all-pairs sum is about 5000^2/2 =
12.5 million distinct pairs per iteration; over ~1000 iterations that is on the order of 10^10 pairwise
force evaluations, vectorized — tens of seconds to a couple of minutes on CPU, squarely inside the
five-minute budget. The O(n^2) cost that disqualifies exact KL-matching at millions of points is
affordable at five thousand, and that is the exact lever the harness lets me pull: the sub-sample to 5000
is what turns the prohibitively-expensive-in-general method into the affordable-here one. Every earlier
rung was, in effect, paying an approximation tax to dodge a bill I can simply settle at this scale.

Two optimization details earn their keep and replace the scheduling the weaker rungs needed. Initialize
from **PCA** (`init="pca"`) rather than random: a structured start that already carries the coarse
global layout, so the local forces refine a sensible frame instead of inventing one — the same role
PCA played for TriMap and the spectral embedding for UMAP, but here it is genuinely just an init, since
the KL objective itself does the local work, and it is also the deterministic global frame that should
keep the seed variance tight the way UMAP's spectral init did. And **early exaggeration**: for the first
stretch of iterations multiply all p_ij by a constant — a factor around twelve — so the q_ij (which still
sum to one) are far too small to match, and the only way to reduce the KL is to pull the members of each
true cluster very tightly together. Trace why that follows from the KL. Under exaggeration the objective
is comparing the inflated 12·p_ij against a q_ij that is still a normalized probability capped by its
all-pairs sum of one, so for a true neighbor pair the target 12·p_ij can be large while q_ij cannot rise
past what the normalization allows — the only way to push q_ij up toward the inflated target is to shrink
the map distance for that pair hard, since q_ij ∝ (1 + d^2)^{-1}. So every within-cluster pair is pulled
in aggressively at once, and the clusters condense into tight, widely separated knots with empty space
between them; that empty space is the room the whole clusters need to slide past one another into a good
global arrangement before the map relaxes to its real (un-exaggerated) objective and the knots loosen
into their final spread. Notice what this is: it is the *derived* analogue of PaCMAP's hand-tuned phase-one
mid-near crank, except here it falls out of the KL — scaling P transiently just makes the attractive term
temporarily dominate — rather than being a separate scheduled force with its own weight ramp. The same
coarse-to-fine idea, but read off the objective instead of bolted onto it. The optimizer is momentum
gradient descent with a per-coordinate adaptive gain at **learning_rate="auto"** (scaled from n) for
**n_iter = 1000** iterations. The full module is in the answer; in the scaffold it lands as
`TSNE(n_components=2, perplexity=30.0, learning_rate="auto", init="pca", random_state=..., n_iter=1000)`,
which calibrates the per-point σ by perplexity, symmetrizes into the joint P, runs the early-exaggerated
Student-t KL descent from the PCA init, and threads the seed. (The harness exposes perplexity, the
learning-rate mode, the init, the iteration count, and the seed; the σ binary search, the symmetrization,
the Student-t kernel, and the exaggeration schedule are internal to the sklearn solver.)

This is the strongest rung on the ladder, so I close on the bar it must clear against UMAP's real
numbers, because there is no further rung to fall back on. The whole bet is that spending capacity on the
local KL match instead of on global guarantees lifts the metric three methods got stuck at. So the
falsifiable claim is sharpest on **trustworthiness**: if matching per-point neighbor distributions
directly is what was missing, MNIST trustworthiness should finally break the ~0.90 plateau and land near
0.96, Fashion-MNIST should push from UMAP's 0.959 toward ~0.98, and newsgroups from 0.885 toward ~0.94 —
a clear jump, not the flat 0.901 UMAP and PaCMAP both returned. kNN should rise too, since exact local
neighborhoods help the 7-NN probe: MNIST from 0.844 toward ~0.86, Fashion-MNIST from 0.740 toward
~0.79, newsgroups holding near ~0.69. The honest *risk*, and the place this rung could lose, is
**continuity** and global structure, and it is the direct consequence of the KL asymmetry I computed:
because KL(P||Q) barely penalizes placing originally-far points near each other (the 0.001·log(0.01)
whisper), t-SNE has no symmetric repulsion holding the global frame — its continuity could come in *at or
just below* UMAP's recovered 0.967. I expect roughly level, ~0.967 on MNIST, because the PCA init and
early exaggeration supply enough global scaffolding to stand in for the missing repulsion; and if
continuity instead *collapses* well below 0.967 that would falsify the "PCA-init is enough global
structure" assumption and vindicate UMAP's explicit repulsion. The seed variance should stay tight like
UMAP's ~0.0098, since the PCA init is deterministic and only the sampling in the solver carries noise. If
trustworthiness does *not* break 0.90, the whole thesis — that the cap was the global/local capacity
trade — is wrong, and the ladder would have no rung left that beats UMAP on the metric that matters; so
this is the rung that either tops the ladder or shows the ceiling was real. The ladder ends here because
on this task's k=7 trustworthiness and kNN metrics, no published neighbor-embedding method clearly
beats a direct per-point KL match within the CPU/5-minute/sklearn budget — the remaining published
alternatives (the graph methods already on the ladder) trade local fidelity for scale or global
structure and score *lower* on exactly these local metrics, so there is no genuinely-stronger published
endpoint to add beyond it.
