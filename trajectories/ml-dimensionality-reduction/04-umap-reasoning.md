PaCMAP came in almost where I bet, and the surprises sharpen the next move. On MNIST kNN edged above
TriMap, 0.832 → 0.853, and trustworthiness nudged from 0.890 to 0.901 — the mid-near machinery bought a
touch more local fidelity. Continuity dipped just as predicted, 0.958 → 0.957 on MNIST and 0.983 → 0.979
on Fashion: phase-three repulsion carves cluster boundaries harder and tears a few original-space
neighbors apart. But the prediction that *failed* is the interesting one. I bet the mid-near skeleton
would tighten the seed variance relative to TriMap's PCA-hostage layout. It did not: MNIST kNN std is
0.0206, even *wider* than TriMap's 0.0195, seed 42 at 0.882 against seeds 123/456 near 0.838 — roughly
six times PCA's 0.007 probe floor, emphatically the method. The mechanism I got wrong: I reasoned the
deterministic PCA init was the *source* of TriMap's variance, but PaCMAP's frame is built from *random*
mid-near and far draws, so I traded a deterministic-init dependence for a random-sampling dependence — and
the three hand-tuned working-zone constants plus the hand-scheduled anneal are an *empirical*
construction whose exact behavior wobbles seed to seed as those draws shift. The instability is not
confined to MNIST kNN: on newsgroups PaCMAP's *continuity* scatters across seeds with std 0.0061 (0.897
down to 0.883), where PCA's was pinned to within 0.002 — the fingerprint of an objective whose behavior
is set by which random pairs it drew rather than by a fixed rule. And on trustworthiness PaCMAP plateaus
at 0.901, only 0.011 above TriMap's 0.890 despite all the extra machinery. Two things are unsatisfying:
the local ceiling is still capped, and the whole construction rests on tuned constants and a scheduled
pull rather than anything I can *derive*.

The fork is real. I could keep tuning PaCMAP — sweep the working-zone constants, resample more mid-near
pairs — but the variance is the tell: an empirical construction that moves 0.044 in kNN between seeds is
telling me its behavior is not pinned by principle but by which random pairs it happened to draw, and
tuning constants on top of that only sets the operating point of an unstable machine. The other path is
to ask where a *principled* graph and a *principled* objective could come from, so neither the graph's
shape nor the repulsion's schedule is a free parameter I guessed. Every method here turns the cloud into
a weighted neighbor graph and lays it out; PaCMAP's is three empirically-chosen pair types. Ask why a
neighbor graph captures the manifold at all. Cover the data with a ball around each point. The Čech
complex turns an open cover into a combinatorial object — a node per set, an edge when two sets intersect
— and the Nerve theorem says this complex is homotopy-equivalent to the union of the cover, i.e. it
faithfully captures the topology, provided the cover is good. Keep only nodes and edges (the
Vietoris–Rips truncation) and the object is just a graph. So "build a neighbor graph and lay it out" is
secretly "approximate the topology of the manifold by a cover and embed the resulting complex" — a
principle that tells me *which* edges belong in the graph and with what weight, rather than leaving me to
pick counts and constants.

But picking the ball radius hits a wall: too small and the cover shatters into islands, too large and
every ball overlaps and the topology drowns. A single radius works only if the data is uniformly
distributed on the manifold — the same uniformity the clean spectral theory needs. Real data clumps and
thins, so no single radius is right. Turn it on its head: *assume* uniformity and ask what that forces
the geometry to be. If points look denser here and sparser there but I insist they are uniform, the only
reconciliation is that *distance itself* varies — space stretched where points look sparse, compressed
where dense. So put a custom Riemannian metric on the manifold, point by point, chosen to make the data
uniform: demand that a fixed-volume ball around every point contain the same number of points, which
forces the local metric to be an isotropic scaling by the neighborhood radius, under which geodesic
distance from a point is *ambient distance divided by the distance to its k-th neighbor*. A dense point's
k-th neighbor is close (ρ_k ≈ 1, distances left roughly as-is); a sparse point's is far (ρ_k ≈ 10,
distances shrunk tenfold) — so both patches hold about k points per unit ball of the *rescaled* metric,
which was the demand. The construction collapses to: normalize each point's distances by the scale of
its own k-neighborhood. And the k-NN-graph hack falls out — connecting each point to its k nearest
neighbors is choosing unit balls in the per-point metric — with **k = n_neighbors = 15** now
interpretable as the *resolution* at which I approximate the manifold as flat, a more natural single knob
than PaCMAP's three separate counts.

Make membership fuzzy rather than binary, because a hard ball throws away the metric I just worked for.
The certainty that a neighbor is "in" decays smoothly: weight = exp(−d/σ_i), with σ_i set so the total
fuzzy membership out of each point hits a fixed target. That target is a smooth stand-in for "about k
neighbors" — the sum of fuzzy memberships is calibrated to log2(k), a one-dimensional monotone equation
in σ_i solved by binary search in a handful of steps. The log2 is deliberate: for k = 15 the target sum
is log2(15) ≈ 3.9, not 15, so each point ends up with the fuzzy equivalent of about four *strong*
memberships and a long tail of weak ones — the right shape, letting the nearest handful dominate the
local geometry while outer neighbors contribute faintly, so k sets how far out I *look* without forcing
all k neighbors to pull equally. The plain exponential still isolates points in high dimensions, where
distances concentrate and every weight is uniformly tiny — so bake in *local connectivity*: let ρ_i be
the distance to i's nearest neighbor and measure from ρ_i outward, exp(−max(0, d − ρ_i)/σ_i). Now the
nearest neighbor sits at weight exactly 1 (at d = ρ_i the exponent is 0), a *guaranteed* connection, and
the kernel reads the *excess* over the nearest-neighbor distance, the relative spacing that survives
concentration. This local-connectivity guarantee is the single most consequential choice for the metric
I care about most, and I come back to it in the predictions. These directed weights are incompatible
across points, so reconcile them by reading the weights as *edge-existence probabilities*: the edge
exists if at least one endpoint vouches for it, the probabilistic union
v_ij = v_{j|i} + v_{i|j} − v_{j|i}v_{i|j} — the fuzzy-set union, monotone in each argument and equal to 1
whenever either endpoint is certain, which softens the pure-kNN asymmetry by pulling in reverse
neighbors. That is the high-dimensional graph, every choice traced to "assume uniformity, infer the
metric; demand local connectivity; merge by probabilistic union."

Now the objective, and I have to resist a tempting shortcut: I have a principled graph whose Laplacian
spectrum carries a globally coherent embedding, so why not just *use* the spectral embedding as the
answer, the way classical Laplacian eigenmaps does? Follow it through, because it fails for a reason that
tells me what the objective must contain. The spectral embedding minimizes sum over edges of
w_ij ||y_i − y_j||^2 subject to a scale normalization — a *purely attractive* quadratic with no repulsion,
so nothing pushes non-edges apart and the only thing stopping the map collapsing to a point is the
normalization. Under that constraint the crowding problem bites hard: a manifold of intrinsic dimension
well above two gets squashed into the plane with every neighborhood piled on the next, because a
quadratic attraction with no heavy tail cannot let a moderate similarity be represented by a large map
distance. That is why spectral is superb as a *global frame* and poor as a final *layout* — it places
clusters in the right coarse arrangement but cannot open the gaps. So spectral belongs as the init, and
the answer needs two things the quadratic lacks: a repulsion on non-edges and a heavy-tailed map kernel.
Both fall out of reading both graphs as vectors of edge-existence probabilities — Bernoulli parameters.
The right divergence is not the KL a per-point match uses (these weights do not sum to one) but the
edgewise **cross-entropy**, sum over edges of [v log(v/w) + (1−v) log((1−v)/(1−w))]. The first term
attracts (an edge that should exist but does not pulls the points together); the second repels — and it
is present for *every* pair, including originally-far ones. Confirm the second term does global work: for
a far pair v ≈ 0 it reduces to log(1/(1 − w)), zero when w = 0 and growing without bound as w → 1, so it
is minimized by driving the map affinity to zero, i.e. pushing the two points apart. So every far pair
carries a real, nonvanishing repulsive pressure — the symmetric global-structure pressure a per-point
asymmetric objective lacks, *derived* from the Bernoulli reading rather than scheduled like PaCMAP's
mid-near pull. Dropping the v-only constant leaves −sum_e [v log w + (1−v) log(1−w)], which has *no*
all-pairs normalization, so the gradient decomposes edge by edge and plain SGD works: sample edges with
probability v for attraction, negative-sample random vertices for repulsion — an unbiased stochastic
estimate of the full repulsive gradient that dodges the O(n^2) cost. At n_neighbors = 15 the fuzzy graph
has ~15n directed edges, about 75,000 at n = 5000, so with a handful of negative samples each a single
epoch touches a few hundred thousand pairs rather than the 12.5 million a full partition function demands
— comfortably inside budget. That the negative sampling is *random* is the caveat on my variance
prediction below: the init is deterministic, but the repulsion is estimated stochastically.

The embedding affinity w must be differentiable with w ≈ 1 out to **min_dist = 0.1** then a smooth decay.
Use w = (1 + a d^{2b})^{-1}, and check it against something I trust: at a = b = 1 it is (1 + d^2)^{-1},
the Student-t that fixes crowding with its slow inverse-square tail. So this family *generalizes* the
heavy-tailed kernel while gaining the two knobs a, b that push the plateau out to min_dist before the
decay begins; a and b are fit by least squares to an offset-exponential target that is flat over
[0, min_dist] and decays beyond it. That flat inner plateau is what lets points that *should* be close
pack to within min_dist of each other rather than being held apart by an ever-present attractive
gradient — the layout's local tightness knob, separate from the graph's resolution knob n_neighbors. The
force coefficients follow by differentiation, with an ε flooring the 1/d^2 blow-up at coincidence and
coordinates clipped for SGD stability. And the one piece that gives the local optimizer a global scaffold
without any hand-tuned schedule: initialize from the bottom eigenvectors of the graph's symmetric
normalized Laplacian L = I − D^{-1/2} W D^{-1/2}. The Rayleigh quotient y^T L y measures how much a
coordinate assignment disagrees across edges, so the eigenvectors of smallest eigenvalue are the
*smoothest* non-constant functions on the graph — they vary slowly, respecting large-scale connectivity,
exactly a global manifold coordinate. The very smallest is the constant (uninformative), so I take the
next two, a frame in which connected regions are laid out coherently before any local force acts. Because
this eigenproblem is a deterministic function of the fuzzy graph — itself a deterministic function of the
standardized data and n_neighbors — the starting frame does not depend on any random draw, which is the
whole point against PaCMAP's random-sampling-built frame. So the graph serves twice: spectral init *and*
cross-entropy target, structurally cleaner than PaCMAP's PCA-init-plus-three-phase-schedule. The full
module is in the answer; in the scaffold it lands as `umap.UMAP(n_components=2, n_neighbors=15,
min_dist=0.1, metric="euclidean", random_state=...)`.

Now the falsifiable expectations against PaCMAP's numbers. The derived graph and symmetric-repulsion
cross-entropy should lift trustworthiness above PaCMAP's plateau *if* the principled construction really
raises the local ceiling — but I have to be honest that UMAP, like PaCMAP and TriMap, spends real
capacity on global structure (the all-pairs (1−v) repulsion, the spectral init), so its *local* score may
land between PaCMAP and a pure-local-affinity method rather than at the top. My sharpest prediction is on
**continuity**, and it follows from the local-connectivity guarantee: because every point reaches its
nearest neighbor at membership exactly 1, no point can be torn off the graph, so the continuity PaCMAP's
hard repulsion gave up should *recover* — most on the datasets PaCMAP tore hardest. Trustworthiness I
expect to gain only modestly, because the symmetric repulsion, while better for *global* structure than
PaCMAP's scheduled pull, still trades against the tightest local packing; kNN should hold near PaCMAP's
level, and it would not shock me if MNIST kNN came in a *hair below* PaCMAP's 0.853, since the all-pairs
repulsion spends capacity on the frame that PaCMAP poured into the tightest local packing. The seed
variance should *finally* tighten relative to PaCMAP's wide 0.0206 spread, because the spectral init is a
deterministic global frame and only the SGD sampling carries randomness — this is the prediction I most
want to see land, the direct fix for the mechanism I misdiagnosed at PaCMAP; if the variance stays
PaCMAP-wide, the negative sampling is the dominant noise source and the deterministic init did not buy
the stability I am counting on. The one score this rung should *not* reach is the top-tier trustworthiness
of a method that matches per-point local neighbor *distributions* directly under KL, paying the O(n^2)
all-pairs cost to get local structure exactly right — that is the remaining gap, and the strongest rung
is the one that pays that cost.
