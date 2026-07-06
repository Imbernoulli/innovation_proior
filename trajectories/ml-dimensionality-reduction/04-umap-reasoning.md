PaCMAP's numbers came in almost exactly where I bet, and the small surprises sharpen the next move. On
MNIST kNN edged above TriMap, 0.832 → 0.853 (a lift of 0.021), and trustworthiness nudged from 0.890 to
0.901 — the mid-near machinery bought a touch more local fidelity. Continuity dipped just as predicted,
0.958 → 0.957 on MNIST and 0.983 → 0.979 on Fashion-MNIST: phase-three repulsion carves cluster
boundaries harder and tears a few original-space neighbors apart, the signature of trading a little
continuity for cleaner separation. But the prediction that *failed* is the interesting one, and I should
not paper over it. I bet that putting the global skeleton into the objective via mid-near pairs would
tighten the seed variance relative to TriMap's PCA-hostage layout. It did not: MNIST kNN std is 0.0206,
even *wider* than TriMap's 0.0195, with seed 42 at 0.882 and seeds 123/456 down at ~0.838 — a 0.044
gap between the best and worst seed on the headline dataset. Set against PCA's 0.007 probe-noise floor,
that is roughly six times the irreducible scoring noise, so it is emphatically the method, not the split.
The robustness I was buying did not show up in the variance. Let me own the mechanism I got wrong: I
reasoned that a deterministic PCA init was the *source* of TriMap's variance, but PaCMAP's frame is built
from *random* mid-near and far draws, so I traded a deterministic-init dependence for a
random-sampling dependence — and the three hand-tuned working-zone constants (10, 10000, 1) plus the
hand-scheduled mid-near anneal are an *empirical* construction whose exact behavior wobbles seed to seed
as those draws shift. The instability is not confined to MNIST kNN, either: on newsgroups PaCMAP's
*continuity* scatters across seeds with a std of 0.0061 (0.897 at seed 42 down to 0.883 at seed 456),
where PCA's continuity there was pinned to within 0.002 — so the same empirical construction that wobbles
the digit kNN also wobbles the newsgroups global structure, which is the fingerprint of an objective
whose behavior is set by which random pairs it drew rather than by a fixed rule. On the headline metric
that matters most here, trustworthiness, PaCMAP plateaus at 0.901 on MNIST — short of the ~0.96 a direct
local-affinity match reaches, and only 0.011 above TriMap's 0.890 despite all the extra machinery. So two things are unsatisfying: the local ceiling is still capped,
and the whole construction rests on tuned constants and a scheduled pull rather than anything I can
*derive*. I want a method whose graph and whose objective both fall out of a principle, so the local
fidelity is not capped by a hand-chosen working zone and the result does not wobble with the schedule.

The fork here is real. I could keep tuning PaCMAP — sweep the working-zone constants, resample more
mid-near pairs, lengthen phase one — but the variance is the tell that this is the wrong direction: an
empirical construction that moves 0.044 in kNN between seeds is telling me its behavior is not pinned by
principle but by which random pairs it happened to draw, and tuning constants on top of that only sets
the operating point of an unstable machine, it does not stabilize it. The other path is to ask where a
*principled* graph and a *principled* objective could come from, so that neither the graph's shape nor the
repulsion's schedule is a free parameter I guessed. Every method in this space, PaCMAP included, turns
the cloud into a weighted neighbor graph and lays it out; PaCMAP's graph is three empirically-chosen pair
types. Let me ask why a neighbor graph captures the manifold at all, because the honest justification
tells me what graph I am *allowed* to build and removes the guesswork. Cover the data with a ball around
each point. There is a classical construction — the Čech complex — that turns an open cover into a
combinatorial object (a node per set, an edge when two sets intersect, and so on), and the Nerve theorem
says this complex is homotopy-equivalent to the union of the cover, i.e. it faithfully captures the
topology, provided the cover is good. Keep only nodes and edges (the Vietoris–Rips truncation) and the
object is just a graph. So "build a neighbor graph and lay it out" is secretly "approximate the topology
of the manifold by a cover and embed the resulting complex." That is the principle PaCMAP's three pair
types only gestured at — and, crucially, it is a principle that tells me *which* edges belong in the
graph and with what weight, rather than leaving me to pick counts and constants.

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
by the distance to its k-th neighbor*. Let me sanity-check that the normalization does what I want with a
crude count. If a point sits in a dense patch, its k-th neighbor is close, say at ambient radius 1, so
dividing by ρ_k = 1 leaves distances roughly as-is; a point in a sparse patch has its k-th neighbor far
out, say at ρ_k = 10, so dividing by 10 shrinks all its distances by tenfold — which is exactly the
stretch that makes the sparse patch "look" as populated as the dense one when I count points inside a
unit ball of the *rescaled* metric. Both patches now hold about k points per unit ball, which was the
demand. The whole construction collapses to: normalize each point's distances by the scale of its own
k-neighborhood. Every point gets its own bandwidth. And the k-NN-graph hack falls out — connecting each
point to its k nearest neighbors is choosing unit balls in the per-point metric — with
**k = n_neighbors = 15** now interpretable as the *resolution* at which I approximate the manifold as
flat, a far more natural single knob than PaCMAP's three separate neighbor/mid-near/far counts.

Make membership fuzzy rather than binary, because a hard ball throws away the metric I just worked for.
The certainty that a neighbor is "in" decays smoothly: weight = exp(−d/σ_i), with σ_i the per-point
normalizer, set so the total fuzzy membership out of each point hits a fixed target. That target is a
smooth stand-in for "about k neighbors" — the sum of the fuzzy memberships is calibrated to log2(k), a
one-dimensional monotone equation in σ_i (raise σ_i and the sum rises, lower it and the sum falls), so a
binary search on σ_i nails it in a handful of steps per point. The log2 is deliberate and worth pausing
on: for k = 15 the target sum is log2(15) ≈ 3.9, not 15, so each point ends up with the fuzzy equivalent
of about four *strong* memberships and a long tail of weak ones, rather than fifteen equal edges. That is
the right shape — it lets the nearest handful of neighbors dominate the local geometry while the outer
neighbors contribute only faintly, so the resolution knob k sets how far out I *look* without forcing all
k neighbors to pull equally hard. The plain exponential still isolates
points in high dimensions, where distances concentrate and every weight is uniformly tiny — so bake in
*local connectivity*: let ρ_i be the distance to i's nearest neighbor and measure the kernel from ρ_i
outward, exp(−max(0, d − ρ_i)/σ_i). Now the nearest neighbor sits at weight exactly 1 — because at
d = ρ_i the exponent is max(0, 0) = 0 and exp(0) = 1, a *guaranteed* connection — and the kernel reads
the *excess* over the nearest-neighbor distance, the relative spacing that survives concentration. This
local-connectivity guarantee is the single most consequential design choice for the metric I care about
most, and I will come back to it in the predictions. These directed weights are incompatible across
points (i's view of edge (i,j) differs from j's), so reconcile them by the principle that the weights are
*edge-existence probabilities*: the edge exists if at least one endpoint vouches for it, the
probabilistic union v_ij = v_{j|i} + v_{i|j} − v_{j|i}v_{i|j}. Let me check that formula behaves like a
probability of "at least one vouches," because if it does not the whole Bernoulli reading downstream is
built on sand. If both endpoints are certain, v_{j|i} = v_{i|j} = 1, the union is 1 + 1 − 1 = 1 — certain,
good. If one is certain and the other says nothing, 1 + 0 − 0 = 1 — the edge exists because one vouched,
good. If neither vouches, 0 + 0 − 0 = 0 — no edge, good. And it is monotone in each argument. So it is
the fuzzy-set (probabilistic) union, not a heuristic, and it softens the pure-kNN asymmetry by pulling in
reverse neighbors. That is the high-dimensional graph, every choice traced to "assume uniformity, infer
the metric; demand local connectivity; merge by probabilistic union" — exactly the derivation PaCMAP's
tuned constants lacked.

Now the objective, and here I have to resist a tempting shortcut. I have a principled graph and I already
know its Laplacian spectrum carries a globally coherent embedding — so why not just *use* the spectral
embedding as the answer, the way classical Laplacian eigenmaps does, and skip the cross-entropy entirely?
Let me follow that through, because it fails for a reason that tells me what the objective must contain.
The spectral embedding minimizes sum over edges of w_ij ||y_i − y_j||^2 subject to a scale normalization —
a *purely attractive* quadratic. It has no repulsion at all: nothing pushes non-edges apart, so the only
thing stopping the whole map from collapsing to a point is the normalization constraint, and under that
constraint the crowding problem bites hard — a manifold of intrinsic dimension well above two gets
squashed into the plane with every neighborhood piled on top of the next, clusters overlapping, because a
quadratic attraction with no heavy tail cannot let a moderate similarity be represented by a large map
distance. That is exactly why the spectral embedding is superb as a *global frame* and poor as a final
*layout*: it places the clusters in the right coarse arrangement but cannot open the gaps between them.
So spectral belongs as the init, not the answer, and the answer needs two things the quadratic lacks — a
repulsion that acts on non-edges, and a heavy-tailed map kernel that tolerates large distances. Both fall
out of reading the graphs probabilistically. Read both the high- and low-dimensional graphs as vectors of
edge-existence probabilities over the same edge set — Bernoulli parameters. The right divergence between Bernoullis is not the KL-between-distributions a per-point match
uses (these weights do not sum to one); it is the edgewise **cross-entropy**,
sum over edges of [v log(v/w) + (1−v) log((1−v)/(1−w))]. Two matched terms per edge: the first,
v log(v/w), attracts (an edge that should exist but does not in the layout pulls the points together);
the second, (1−v) log((1−v)/(1−w)), repels (an edge that should not exist but does pushes them apart) —
and that second term is present for *every* pair, including originally-far ones. Let me confirm the
second term actually does global work rather than sitting inert. For a genuinely far pair v ≈ 0, so the
term reduces to log(1/(1 − w)), which is zero when w = 0 and grows without bound as w → 1: it is
minimized by driving the map affinity w to zero, i.e. by pushing the two points apart in the layout. So
every far pair carries a real, nonvanishing repulsive pressure — precisely the symmetric global-structure
pressure a per-point asymmetric objective lacks, and it is *derived* from the Bernoulli reading rather
than scheduled like PaCMAP's mid-near pull. Drop the v-only constant and I minimize
−sum_e [v log w + (1−v) log(1−w)], which has *no* all-pairs normalization (no partition function), so
the gradient decomposes edge by edge and plain SGD works: sample edges with probability v for the
attraction, negative-sample random vertices for the repulsion. The negative sampling is what dodges the
O(n^2) cost — instead of summing the (1−v) repulsion over all ~n^2/2 non-edges every step, I draw a small
fixed number of random vertices per edge update and repel only those, an unbiased stochastic estimate of
the full repulsive gradient. Linear in the number of edges — and at n_neighbors = 15 the fuzzy graph has
on the order of 15n directed edges, about 75,000 at n = 5000, so with a handful of negative samples each
a single epoch touches a few hundred thousand pairs rather than the 12.5 million distinct pairs a full
partition function would demand, and the whole optimization stays comfortably inside the five-minute
budget. That the negative sampling is *random* is also the caveat on my variance prediction below: the
init is deterministic, but the repulsion is estimated stochastically, so if that estimation noise is
large it could keep the seed spread wider than the deterministic init alone would suggest.

The embedding affinity w must be differentiable with w ≈ 1 out to **min_dist = 0.1** then a smooth
decay. Use w = (1 + a d^{2b})^{-1}, and check the family against something I already trust: at a = b = 1
it is exactly (1 + d^2)^{-1}, the Student-t with one degree of freedom that fixes crowding with its slow
inverse-square tail. So this family *generalizes* the heavy-tailed kernel while gaining the two free knobs
a, b that let me push the plateau out to min_dist before the decay begins; a and b are fit by least
squares so the curve matches an offset-exponential target Ψ(d) that equals 1 for d ≤ min_dist and decays
like exp(−(d − min_dist)) beyond it. The knobs earn their keep only if they can hold w near 1 across the
whole plateau: with min_dist = 0.1 the target is flat over [0, 0.1], and the fit chooses a, b so that
(1 + a d^{2b})^{-1} stays close to 1 there and only bends down past d = 0.1, which a plain Student-t
(pinned at a = b = 1, already at w = 1/(1 + 0.01) ≈ 0.99 at d = 0.1 but with the wrong curvature just
beyond) cannot match as tightly. That flat inner plateau is what lets points that *should* be close pack
to within min_dist of each other rather than being held apart by an ever-present attractive gradient —
the layout's local tightness knob, separate from the graph's resolution knob n_neighbors. The two
force coefficients follow by differentiation: attractive −2ab d^{2(b−1)}/(1 + a d^{2b}), repulsive
2b/((ε + d^2)(1 + a d^{2b})) with ε flooring the 1/d^2 blow-up at coincidence; coordinates are clipped for
SGD stability and the learning rate decays linearly. And the one piece that gives the local optimizer a
global scaffold without any hand-tuned schedule: initialize from the bottom eigenvectors of the graph's
symmetric normalized Laplacian L = I − D^{-1/2} W D^{-1/2} — the spectral embedding, a globally coherent
set of manifold coordinates from the *same* fuzzy graph (whose Laplacian approximates the
Laplace–Beltrami operator). The reason the *bottom* eigenvectors are the right coordinates is worth
stating: the Rayleigh quotient y^T L y measures how much a coordinate assignment y disagrees across
edges, so the eigenvectors of smallest eigenvalue are the *smoothest* non-constant functions on the graph
— they vary slowly, respecting the graph's large-scale connectivity, which is exactly a global manifold
coordinate. The very smallest eigenvector is the constant (eigenvalue 0, uninformative), so I take the
next two, and they give a frame in which connected regions of the manifold are laid out coherently before
any local force acts. Because this eigenproblem is a deterministic function of the fuzzy graph — which is
itself a deterministic function of the standardized data and n_neighbors — the starting frame does not
depend on any random draw at all, which is the whole point against PaCMAP's random-sampling-built frame. So the graph
serves twice: spectral init *and* cross-entropy target. This is structurally cleaner than PaCMAP's
PCA-init-plus-three-phase-schedule: the global frame comes from the graph's own spectrum, not a separate
initializer and a tuned weight ramp, and — this is the point of the whole exercise — a *deterministic*
spectral frame is exactly what a wobbling seed variance needs. The full module is in the answer; in the
scaffold it lands as `umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean",
random_state=...)`, which builds the fuzzy simplicial graph, fits a,b, spectral-initializes, and runs the
edge-/negative-sampled cross-entropy SGD. (The harness exposes n_neighbors, min_dist, and the metric; the
connectivity offset, the union, the a/b fit, and the SGD schedule are internal.)

Now the falsifiable expectations against PaCMAP's numbers. The derived graph and symmetric-repulsion
cross-entropy should lift trustworthiness above PaCMAP's plateau toward ~0.96 on MNIST if the principled
construction really does raise the local ceiling — but I have to be honest that UMAP, like PaCMAP and
TriMap, spends real capacity on global structure (the (1−v) repulsion over all pairs, the spectral init),
so its *local* score may land *between* PaCMAP and a pure-local-affinity method rather than at the very
top. My sharpest prediction is actually on **continuity**, and it follows directly from the
local-connectivity guarantee I flagged: because every point reaches its nearest neighbor at membership
exactly 1, no point can be torn off the graph, so original-space neighbors should stay near and the
continuity PaCMAP's hard repulsion gave up should *recover* — MNIST continuity from PaCMAP's 0.957 back
up toward ~0.967, and newsgroups from 0.891 up toward ~0.91. On trustworthiness I expect a *modest* gain
on MNIST (0.901 → ~0.901–0.91 range) rather than the full jump to 0.96 — because the cross-entropy's
symmetric repulsion, while better than PaCMAP's scheduled mid-near pull for *global* structure, still
trades against the very tightest local packing. kNN should hold near PaCMAP's level (MNIST ~0.84,
newsgroups ~0.67) since both preserve class neighborhoods comparably — and it would not shock me if MNIST
kNN even came in a *hair below* PaCMAP's 0.853, because the all-pairs (1−v) repulsion spends capacity on
holding the global frame that PaCMAP poured into the very tightest local packing. The seed variance should *finally*
tighten relative to PaCMAP's wide 0.0206 spread, because the spectral init is a deterministic global
frame and only the SGD edge/negative sampling carries randomness — this is the prediction I most want to
see land, since it is the direct fix for the mechanism I misdiagnosed at PaCMAP; if instead the variance
stays PaCMAP-wide, that would say the negative sampling is the dominant noise source and the
deterministic init does not buy the stability I am counting on. The one score this rung should *not*
reach is the ~0.96 MNIST trustworthiness of a method that matches per-point local neighbor
*distributions* directly under KL, paying the O(n^2) all-pairs cost to get the local structure exactly
right — that is precisely the remaining gap, and the strongest rung on the ladder is the one that pays
that cost.
