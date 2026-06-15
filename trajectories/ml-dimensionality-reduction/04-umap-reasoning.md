PaCMAP's numbers came in almost exactly where I bet, and the small surprises sharpen the next move. On
MNIST kNN edged above TriMap, 0.832 → 0.853, and trustworthiness nudged from 0.890 to 0.901 — the
mid-near machinery bought a touch more local fidelity. Continuity dipped just as predicted, 0.958 →
0.957 on MNIST and 0.983 → 0.979 on Fashion-MNIST: phase-three repulsion carves cluster boundaries
harder and tears a few original-space neighbors apart, the signature of trading a little continuity for
cleaner separation. But the prediction that *failed* is the interesting one. I bet that putting the
global skeleton into the objective via mid-near pairs would tighten the seed variance relative to
TriMap's PCA-hostage layout. It did not: MNIST kNN std is 0.0206, even *wider* than TriMap's 0.0195,
with seed 42 at 0.882 and seeds 123/456 down at ~0.838. So the robustness I was buying did not show up
in the variance — PaCMAP's own randomness (the random mid-near and far sampling) is itself a source of
spread, and the three hand-tuned working-zone constants (10, 10000, 1) plus the hand-scheduled mid-near
anneal are an *empirical* construction whose exact behavior wobbles seed to seed. And on the headline
metric that matters most here, trustworthiness, PaCMAP plateaus at 0.901 on MNIST — short of the ~0.96 a
direct local-affinity match reaches. So two things are unsatisfying: the local ceiling is still capped,
and the whole construction rests on tuned constants and a scheduled pull rather than anything I can
*derive*. I want a method whose graph and whose objective both fall out of a principle, so the local
fidelity is not capped by a hand-chosen working zone and the result does not wobble with the schedule.

Where can a *principled* graph come from? Every method in this space, PaCMAP included, turns the cloud
into a weighted neighbor graph and lays it out; PaCMAP's graph is three empirically-chosen pair types.
Let me ask why a neighbor graph captures the manifold at all, because the honest justification tells me
what graph I am *allowed* to build and removes the guesswork. Cover the data with a ball around each
point. There is a classical construction — the Čech complex — that turns an open cover into a
combinatorial object (a node per set, an edge when two sets intersect, and so on), and the Nerve theorem
says this complex is homotopy-equivalent to the union of the cover, i.e. it faithfully captures the
topology, provided the cover is good. Keep only nodes and edges (the Vietoris–Rips truncation) and the
object is just a graph. So "build a neighbor graph and lay it out" is secretly "approximate the topology
of the manifold by a cover and embed the resulting complex." That is the principle PaCMAP's three pair
types only gestured at.

But picking the ball radius immediately hits a wall: too small and the cover shatters into disconnected
islands; too large and every ball overlaps and the topology drowns in giant simplices. A single radius
works only if the data is uniformly distributed on the manifold — and that same uniformity is exactly
what the clean spectral theory (Laplacian eigenmaps) needs for its convergence. Real data clumps and
thins, so no single radius is right. I cannot make the data uniform, so turn the problem on its head:
*assume* it is uniform and ask what that forces the geometry to be. If points look denser here and
sparser there but I insist they are uniform, the only reconciliation is that *distance itself* varies
across the manifold — space is stretched where points look sparse and compressed where they look dense.
A manifold can carry a custom Riemannian metric, so put one on it, point by point, chosen to make the
data uniform. Make this precise and it hands me the edge weights: demand that a fixed-volume ball around
every point contain the same number of points; that forces the local metric to be an isotropic scaling
by the neighborhood radius, under which geodesic distance from a point is just *ambient distance divided
by the distance to its k-th neighbor*. The whole construction collapses to: normalize each point's
distances by the scale of its own k-neighborhood. Every point gets its own bandwidth. And the
k-NN-graph hack falls out — connecting each point to its k nearest neighbors is choosing unit balls in
the per-point metric — with **k = n_neighbors = 15** now interpretable as the *resolution* at which I
approximate the manifold as flat, a far more natural knob than PaCMAP's separate neighbor/mid-near/far
counts.

Make membership fuzzy rather than binary, because a hard ball throws away the metric I just worked for.
The certainty that a neighbor is "in" decays smoothly: weight = exp(−d/σ_i), with σ_i the per-point
normalizer, set so the total fuzzy membership out of each point hits a fixed target (a one-dimensional
monotone equation, solved by binary search). The plain exponential still isolates points in high
dimensions, where distances concentrate and every weight is uniformly tiny — so bake in *local
connectivity*: let ρ_i be the distance to i's nearest neighbor and measure the kernel from ρ_i outward,
exp(−max(0, d − ρ_i)/σ_i). Now the nearest neighbor sits at weight exactly 1 (guaranteed connection)
and the kernel reads the *excess* over the nearest-neighbor distance — the relative spacing that
survives concentration. These directed weights are incompatible across points (i's view of edge (i,j)
differs from j's), so reconcile them by the principle that the weights are *edge-existence
probabilities*: the edge exists if at least one endpoint vouches for it, the probabilistic union
v_ij = v_{j|i} + v_{i|j} − v_{j|i}v_{i|j}. Not a heuristic — the fuzzy-set union once you accept the
probabilistic reading — and it softens the pure-kNN asymmetry by pulling in reverse neighbors. That is
the high-dimensional graph, every choice traced to "assume uniformity, infer the metric; demand local
connectivity; merge by probabilistic union" — exactly the derivation PaCMAP's tuned constants lacked.

Now the objective, and this is where the local ceiling lifts. Read both the high- and low-dimensional
graphs as vectors of edge-existence probabilities over the same edge set — Bernoulli parameters. The
right divergence between Bernoullis is not the KL-between-distributions t-SNE uses (these weights do not
sum to one); it is the edgewise **cross-entropy**,
sum over edges of [v log(v/w) + (1−v) log((1−v)/(1−w))]. Two matched terms per edge: the first,
v log(v/w), attracts (an edge that should exist but does not in the layout pulls the points together);
the second, (1−v) log((1−v)/(1−w)), repels (an edge that should not exist but does pushes them apart) —
and that second term is present for *every* pair, including originally-far ones. That is precisely the
symmetric global-structure pressure t-SNE's asymmetric KL lacks, and it is *derived* from the Bernoulli
reading rather than scheduled like PaCMAP's mid-near pull. Drop the v-only constant and I minimize
−sum_e [v log w + (1−v) log(1−w)], which has *no* all-pairs normalization (no partition function), so
the gradient decomposes edge by edge and plain SGD works: sample edges with probability v for the
attraction, negative-sample random vertices for the repulsion. Linear in the number of edges.

The embedding affinity w must be differentiable with w ≈ 1 out to **min_dist = 0.1** then a smooth
decay. Use w = (1 + a d^{2b})^{-1}, which at a = b = 1 recovers exactly t-SNE's Student-t — so this
family *generalizes* the heavy-tailed kernel that fixes crowding while gaining the freedom to encode
min_dist — with a, b fit so the curve matches an offset-exponential target carrying min_dist and a
spread. The two force coefficients follow by differentiation: attractive −2ab d^{2(b−1)}/(1 + a d^{2b}),
repulsive 2b/((ε + d^2)(1 + a d^{2b})) with ε flooring the 1/d^2 blow-up at coincidence; coordinates are
clipped for SGD stability and the learning rate decays linearly. And the one piece that gives the local
optimizer a global scaffold without any hand-tuned schedule: initialize from the bottom eigenvectors of
the graph's symmetric normalized Laplacian — the spectral embedding, a globally coherent set of manifold
coordinates from the *same* fuzzy graph (whose Laplacian approximates the Laplace–Beltrami operator). So
the graph serves twice: spectral init *and* cross-entropy target. This is structurally cleaner than
PaCMAP's PCA-init-plus-three-phase-schedule: the global frame comes from the graph's own spectrum, not a
separate initializer and a tuned weight ramp. The full module is in the answer; in the scaffold it lands
as `umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=...)`,
which builds the fuzzy simplicial graph, fits a,b, spectral-initializes, and runs the edge-/negative-
sampled cross-entropy SGD. (The harness exposes n_neighbors, min_dist, and the metric; the connectivity
offset, the union, the a/b fit, and the SGD schedule are internal.)

Now the falsifiable expectations against PaCMAP's numbers. The derived graph and symmetric-repulsion
cross-entropy should lift trustworthiness above PaCMAP's plateau toward ~0.96 on MNIST if the principled
construction really does raise the local ceiling — but I have to be honest that UMAP, like PaCMAP and
TriMap, spends real capacity on global structure (the (1−v) repulsion over all pairs, the spectral
init), so its *local* score may land *between* PaCMAP and a pure-local-affinity method rather than at
the very top. My sharpest prediction is actually on **continuity**, where UMAP's principled local
connectivity and spectral init should *recover* the continuity PaCMAP's hard repulsion gave up: MNIST
continuity from PaCMAP's 0.957 back up toward ~0.967, and newsgroups from 0.891 up toward ~0.91 — the
local connectivity guarantees no point is torn off, so original-space neighbors stay near. On
trustworthiness I expect a *modest* gain on MNIST (0.901 → ~0.901–0.91 range) rather than the full jump
to 0.96 — because the cross-entropy's symmetric repulsion, while better than PaCMAP's scheduled mid-near
pull for *global* structure, still trades against the very tightest local packing. kNN should hold near
PaCMAP's level (MNIST ~0.84, newsgroups ~0.67) since both preserve class neighborhoods comparably. The
seed variance should *finally* tighten relative to PaCMAP's wide spread, because the spectral init is a
deterministic global frame and only the SGD sampling carries randomness — if instead the variance stays
PaCMAP-wide, that would say the negative sampling is the dominant noise source. The one score this rung
should *not* reach is the ~0.96 MNIST trustworthiness of a method that matches per-point local neighbor
*distributions* directly under KL, paying the O(n^2) all-pairs cost to get the local structure exactly
right — that is precisely the remaining gap, and the strongest rung on the ladder is the one that pays
that cost.
