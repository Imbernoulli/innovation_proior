UMAP confirmed the prediction I made most sharply and then quietly capped the metric I care about most.
Continuity recovered as I bet: MNIST from PaCMAP's 0.957 to 0.967, newsgroups 0.891 → 0.912, Fashion
0.979 → 0.982 — the local-connectivity guarantee really does stop points being torn off. And the seed
variance finally tightened: MNIST kNN std collapsed from 0.0206 to 0.0098, because the spectral init is a
deterministic global frame and only the SGD sampling carries noise. The size of the continuity recovery
tracks the fold depth I have carried since PCA: newsgroups recovered most (+0.021), MNIST next (+0.010),
Fashion least (+0.003) — the same ordering, exactly if the guarantee is repairing the neighbors PaCMAP's
hard repulsion had torn. But trustworthiness sat *flat* at 0.901 on MNIST — identical to PaCMAP — and kNN
came in a hair *below* PaCMAP (0.844 vs 0.853), the small loss I flagged when the all-pairs repulsion
spends capacity on the frame. Lay the three trustworthiness numbers side by side: TriMap 0.890, PaCMAP
0.901, UMAP 0.901. Three different methods — a PCA-inherited frame, a mid-near skeleton, a spectral init
plus symmetric repulsion — and the metric moved a total of 0.011 across all of them and then stopped
dead. Every rung bought global structure at the cost of a capped local ceiling. So the diagnosis is no
longer "which graph" or "which repulsion"; it is that *spending capacity on global structure is itself
what caps the local score*, and the strongest local result will come from a method that spends almost
*all* its capacity getting the local neighborhoods exactly right, paying whatever that costs and
accepting weaker global guarantees.

The tempting move is to keep tuning UMAP — shrink min_dist toward zero to pack neighborhoods tighter — but
that only sets how close the layout *allows* points to sit; it does not remove the all-pairs (1−v)
repulsion spending the capacity, so it would tighten the packing a little while the same global term
claws it back. The three-method plateau at ~0.90 is the evidence that this is a ceiling built into "spend
capacity on an all-pairs global term," not a knob I failed to turn. A subtler variant — keep UMAP's fuzzy
graph but swap its cross-entropy for a sharper local objective — fails mechanically: UMAP's efficiency
comes precisely from *avoiding* the all-pairs partition function via edge and negative sampling, so any
objective bolted onto that machinery inherits an *approximate* local match. The cap is not in the graph;
it is in refusing to pay the partition function. So stop balancing global against local and match the
*local neighborhood structure* as precisely as I can, paying the exact all-pairs cost. Go back to the
probabilistic-neighbor frame: center a Gaussian on each point x_i and read the probability it would pick
x_j as a neighbor, p_{j|i} = exp(−||x_i − x_j||^2/2σ_i^2) normalized over the other points. The bandwidth
σ_i is *per point*, set by fixing the entropy of each neighbor distribution — the **perplexity**, a smooth
"effective number of neighbors," found by binary search so denser regions automatically get a smaller
σ_i. Concretely the target entropy is log2(perplexity): at perplexity = 30 that is ≈ 4.9 bits, and σ_i is
the unique bandwidth making point i's distribution carry that much entropy, a one-dimensional monotone
equation solved in a handful of bisection steps. This is the same per-point adaptive bandwidth UMAP
derived from its Riemannian-uniformity argument, reached here from the entropy side; perplexity = 30
plays the role UMAP's n_neighbors = 15 played. The scale is a touch larger, and that is deliberate for a
method whose whole job is the local match: a slightly wider effective neighborhood gives each Gaussian
more neighbors to average over, steadying the σ_i estimate and the p_{j|i} distribution, at the cost of
reading structure a marginally coarser scale — a good trade when I am about to spend all my capacity on
matching that distribution exactly. And fixing entropy rather than distance is what makes "thirty
neighbors" mean the same *informational* thing in a dense cluster (small σ_i) and a sparse region (large
σ_i) even though the physical radius differs by an order of magnitude. The high-dimensional side is
settled and I keep it.

The trap to avoid is *crowding*. Picture a manifold of intrinsic dimension well above two embedded in
pixel space: the number of points at *moderate* distance from i grows like r^(intrinsic d), so there are
far more "moderately far" neighbors than near ones. Put a number on how badly that fails to fit the
plane. At intrinsic dimension ten, double the radius and the count of enclosed points grows by 2^10 = 1024
in the data, but the area to seat them in the plane grows only by 2^2 = 4. So a thousand points that
belong at moderate distance must be crammed into room for four; shoved too far out, each still exerts a
small Gaussian attraction back toward i, and the *sum* of those thousand tiny attractions crushes the map
inward so the clusters never separate. That is why a Gaussian map kernel fails. The fix, available because
I am matching *probabilities* not distances, is to let the map kernel differ from the high-D Gaussian and
make it *heavy-tailed* — a Student-t with one degree of freedom, q_ij ∝ (1 + ||y_i − y_j||^2)^{-1}.
Quantify the tail against the Gaussian at a moderate map distance of 3 (squared distance 9): a Gaussian
map assigns exp(−9) ≈ 1.2e-4, the Student-t assigns 1/10 = 0.1 — about 800 times larger. Under a Gaussian
map, representing a pair's moderate input similarity would force them almost on top of each other; the
Student-t is content to represent that same similarity at distance 3 and beyond, so the moderately-distant
points are *allowed* to spread instead of being crushed inward — exactly the room the 1024-against-4 count
said I needed. The crowding is worst on the 784-pixel images (high intrinsic dimension) and milder on the
50-dimensional newsgroups cloud, so I expect the biggest gains on the image datasets even though all
three should break the plateau. (This is the same Student-t UMAP generalized at a = b = 1, but here it is
the *whole* map kernel, matched against per-point neighbor distributions rather than edge-existence
probabilities.) The inverse-square far field also lets a distant cluster act like a single point, removing
the need for the annealing schedules the earliest neighbor-embedding methods — and PaCMAP's three-phase
schedule — relied on.

How to measure unfaithfulness? The obvious first try, keeping the per-point conditionals on both sides and
minimizing sum over points of KL(P_i || Q_i), has two problems that name the fix. First, the gradient
carries a *per-point* denominator, so every point's normalizer must be tracked separately. Second, and
worse for these metrics, a genuine outlier sits far from everything, so its conditional P_i is nearly
uniform and tiny, and *nothing* forces the map to place it anywhere — it drifts to the edge undetermined,
a false neighbor waiting to happen. Both are fixed by building a single *joint* distribution and
minimizing one KL, C = KL(P||Q) = sum p_ij log(p_ij/q_ij), with P symmetrized as
p_ij = (p_{j|i} + p_{i|j})/(2n) so every point — even a total outlier — contributes at least 1/(2n) of the
mass (1e-4 at n = 5000) and cannot drift off undetermined. Now the asymmetry of KL, the linchpin of the
whole "tilt toward local" argument. Take a truly-*near* pair the map placed far apart: p high, say 0.1,
q low, say 0.001, contributing 0.1·log(100) ≈ 0.46 — a large penalty. Take a truly-*far* pair placed
near: p low 0.001, q high 0.1, contributing 0.001·log(0.01) ≈ −0.0046 — a hundred times smaller, because
the tiny p in front throttles it. So the objective screams when it places near points far apart and
barely whispers when it places far points near. That single asymmetry does two things at once: it tilts
the method hard toward getting *local* structure right — the capacity allocation the capped rungs would
not make — and it is *also* the source of the global-structure risk I flag below, because the whisper
means there is almost no force holding the global frame. Differentiating the joint KL with the Student-t
kernel gives the force law ∇_{y_i}C = 4 sum_j (p_ij − q_ij)(y_i − y_j)(1 + ||y_i − y_j||^2)^{-1}: an
attraction where the data says two points are more similar than the map shows (p > q), a repulsion where
the map crammed them too close (q > p), both softened at long range by the Student-t factor so a distant
cluster acts as a single body rather than being torn at by every far point.

The price is in plain sight in that gradient: the q_ij normalization is over *all pairs*, so the cost
couples every pair — O(n^2) per iteration, exactly the cost UMAP's whole design (negative sampling, edge
decomposition) was built to avoid, and paying it is what lets the local structure be matched exactly
rather than approximately. Can I afford it here? On this task n ≤ 5000 per dataset, so the all-pairs sum
is about 12.5 million distinct pairs per iteration; over ~1000 iterations that is ~10^10 vectorized force
evaluations — tens of seconds to a couple of minutes on CPU, inside the five-minute budget. The O(n^2)
cost that disqualifies exact KL-matching at millions of points is affordable at five thousand: the
sub-sample to 5000 is the exact lever that turns the prohibitively-expensive-in-general method into the
affordable-here one. Every earlier rung was paying an approximation tax to dodge a bill I can simply
settle at this scale.

Two optimization details replace the scheduling the weaker rungs needed. Initialize from **PCA**
(`init="pca"`): a structured start carrying the coarse global layout, the same role PCA played for TriMap
and the spectral embedding for UMAP, but here genuinely just an init since the KL does the local work — and
the deterministic global frame that should keep seed variance tight the way UMAP's spectral init did. And
**early exaggeration**: for the first stretch multiply all p_ij by a constant around twelve, so the q_ij
(still summing to one) are far too small to match, and the only way to reduce the KL is to pull each true
cluster's members very tightly together — since q_ij ∝ (1 + d^2)^{-1}, pushing q_ij up toward the inflated
target means shrinking the map distance for that pair hard. So every within-cluster pair condenses at
once into tight, widely-separated knots, and that empty space is the room whole clusters need to slide
past one another into a good global arrangement before the map relaxes to its real objective and the
knots loosen. This is the *derived* analogue of PaCMAP's hand-tuned phase-one crank — scaling P
transiently just makes the attractive term temporarily dominate — read off the objective instead of
bolted onto it. The optimizer is momentum gradient descent with a per-coordinate adaptive gain at
**learning_rate="auto"** for **n_iter = 1000**. The full module is in the answer; in the scaffold it
lands as `TSNE(n_components=2, perplexity=30.0, learning_rate="auto", init="pca", random_state=...,
n_iter=1000)`.

This is the strongest rung, so I close on the bar it must clear against UMAP's numbers, since there is no
further rung to fall back on. The whole bet is that spending capacity on the local KL match instead of
global guarantees lifts the metric three methods got stuck at, so the falsifiable claim is sharpest on
**trustworthiness**: MNIST should finally break the ~0.90 plateau where TriMap, PaCMAP, and UMAP all
stalled — a clear jump, not the flat 0.901 they returned — with the largest gains on the image datasets
where crowding was worst. kNN should rise too, since exact local neighborhoods help the 7-NN probe. The
honest *risk*, and where this rung could lose, is **continuity** and global structure, the direct
consequence of the KL asymmetry I computed: because KL(P||Q) barely penalizes placing originally-far
points near each other, t-SNE has no symmetric repulsion holding the global frame, so its continuity
could come in *at or below* UMAP's recovered level. I expect roughly level, because the PCA init and
early exaggeration supply enough global scaffolding to stand in for the missing repulsion; if continuity
instead *collapses* that would falsify the "PCA-init is enough global structure" assumption and vindicate
UMAP's explicit repulsion. Seed variance should stay tight like UMAP's, since the PCA init is
deterministic and only the solver's sampling carries noise. If trustworthiness does *not* break 0.90, the
whole thesis — that the cap was the global/local capacity trade — is wrong, and the ladder has no rung
left that beats UMAP on the metric that matters; so this rung either tops the ladder or shows the ceiling
was real. The ladder ends here because on this task's k=7 trustworthiness and kNN metrics, no published
neighbor-embedding method clearly beats a direct per-point KL match within the CPU/five-minute/sklearn
budget — the remaining published alternatives are the graph methods already on the ladder, which trade
local fidelity for scale or global structure and score lower on exactly these local metrics.
