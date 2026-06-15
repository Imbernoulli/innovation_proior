Let me start from what actually goes wrong when I try to flatten high-dimensional data into a picture. I have `N` points in `R^n` — image pixels, gene-expression vectors, word embeddings — and I believe they don't fill `R^n` but sit on some curved low-dimensional manifold `M`. I want a map into `R^2` (or a handful of dimensions) that keeps the shape of `M` recognizable: who is near whom locally, and roughly how the clusters are arranged globally. PCA is the cheap thing I reach for first, and it's the wrong tool by construction: it can only find linear subspaces, so it flattens any fold of `M` and places points on opposite sides of a curl right next to each other. The whole point is the nonlinearity, and PCA throws exactly that away. So I need a genuinely nonlinear embedding, and I want it to scale — I have hundreds of thousands, maybe millions of points — and I'd like the design choices to rest on something firmer than "I tuned it until the picture looked nice," because this is unsupervised and there's no held-out accuracy to anchor on.

The shape of every method in this space is the same: turn the cloud into a weighted neighbor graph — nodes are points, edges connect nearby points with weights that decay with distance — and then lay that graph out in the plane. All the disagreement is in two places: how you build and weight the graph, and what objective the layout descends. So I want to think hard about both, and I want to know *why* each choice, not just *what*.

Why should a neighbor graph capture the manifold at all? The honest justification comes from topology, and I want to actually walk it because it tells me what I'm allowed to do later. Suppose I cover the data with a ball of some radius around each point. There's a classical construction — the Čech complex — that turns an open cover into a combinatorial object: a 0-simplex per set, a 1-simplex when two sets intersect, a 2-simplex when three share a common intersection, and so on. The Nerve theorem says this complex is homotopy-equivalent to the union of the cover, i.e. it faithfully captures the topology, provided the cover is good. That's the guarantee I want. And if I only keep the 0- and 1-simplices — the points and the edges, the Vietoris–Rips truncation — the object is just a graph. So "build a neighbor graph and lay it out" is secretly "approximate the topology of `M` by a cover and embed the resulting complex." If I then ask for a low-dimensional layout that's smooth with respect to the graph, I land on Laplacian eigenmaps: minimize `sum_ij w_ij ||y_i - y_j||^2` under a scale constraint, whose solution is the bottom eigenvectors of the graph Laplacian `L = D - W`, and those eigenvectors are the smoothest functions on the approximated manifold. Lovely and principled.

Except now I try to actually pick the radius and I hit a wall immediately. Too small a radius and the cover doesn't cover — the complex shatters into many disconnected components, each point an island or a tiny clique. Too large and every ball overlaps every other; I get giant high-dimensional simplices and the topology I wanted to read off is buried in mush. Is there a sweet spot? Only if the data is uniformly distributed on `M`: then the typical inter-point spacing is stable enough that one radius can cover everything cleanly with no gaps and no clumping. And here's the thing that should bother me: the *same* uniform-distribution assumption is exactly what Laplacian eigenmaps needs for its convergence guarantee — the graph Laplacian only tends to the Laplace–Beltrami operator of `M` in the limit of infinite, uniformly sampled data. So the clean theory I just leaned on, and the radius I can't pick, fail for the same reason: real data clumps and thins. Dense regions get over-covered, sparse regions get under-covered, and no single radius is right.

I could give up and just use `k` nearest neighbors with a hand-tuned bandwidth, which is what everyone does. But I don't want a hack; I want to understand why the hack is the right thing, because that's what will tell me how to weight the edges and what to do next. So let me turn the problem on its head. I can't make the data uniform. But what if I *assume* it's uniform and ask what that forces the manifold's geometry to be? If points look denser here and sparser there, and I insist they're uniformly distributed, then the only way to reconcile that is for the notion of distance itself to vary across `M` — space is stretched where points look sparse and compressed where they look dense. I'm allowed to do this: a manifold can carry a Riemannian metric that isn't the one inherited from the ambient space. So let me put a custom metric on `M`, point by point, chosen to make the data uniform with respect to it.

Let me make this precise, because the precise version hands me the edge weights. Take a point `p` and suppose, in a small neighborhood, the metric `g` is constant and diagonal in ambient coordinates. The volume of a ball `B` of ambient radius `r` under `g` is `integral over B of sqrt(det g) dx = sqrt(det g) * (pi^{n/2} r^n / Gamma(n/2 + 1))`, the Euclidean ball volume scaled by `sqrt(det g)`. Now I demand that *every* such ball, around every point, has the *same* fixed volume; a fixed-volume ball should always enclose about the same number of points. Fix the volume to `pi^{n/2}/Gamma(n/2 + 1)`, the unit Euclidean ball's volume. Then `sqrt(det g) * r^n = 1`, so `det g = 1/r^{2n}`. A diagonal metric with that determinant could be anisotropic, but the data I have is only a scalar neighbor distance, not a local covariance ellipsoid. The isotropic local choice is therefore the diagonal scaling `g_ij = (1/r^2) delta_ij`, which has exactly the required determinant and preserves directions while changing the local unit of length. Under this scalar metric, the geodesic distance from `p` to a nearby `q` is `inf over curves of integral sqrt(g(c', c')) dt = (1/r) * inf integral ||c'|| dt = (1/r) d_{R^n}(p, q)`. So: under the local isotropic metric that makes the data uniform at `p`, geodesic distance from `p` is just *ambient distance divided by `r`*, where `r` is the radius of that fixed-volume ball, and a fixed-volume ball around `p` is, in finite-data terms, the ball that contains exactly `p`'s `k` nearest neighbors. So `r` is the distance to the `k`-th neighbor. The whole construction collapses to: *normalize each point's distances by the scale of its own `k`-neighborhood.* Every point gets its own local metric, its own bandwidth. And the standard `k`-NN-graph hack falls out of it: a unit ball under the local metric reaches the `k`-th neighbor, so connecting each point to its `k` nearest neighbors is exactly choosing balls of radius one in the per-point metric. `k` is now interpretable as the scale at which I'm approximating `M` as flat, the resolution of the local-metric estimate. Small `k` chases fine detail; large `k` averages over bigger patches. That's a far more natural knob than an absolute radius, which I'd have to set by staring at the distance distribution of each dataset.

Now make the membership fuzzy rather than binary, because a hard ball throws away the metric information I just worked to get. The certainty that a neighbor is "in" the local ball should decay smoothly as it gets farther from `p`. The natural decay is exponential in the local-metric distance: weight `= exp(-(d(x_i, x_j))/sigma_i)`, where `sigma_i` is the per-point normalizer playing the role of `r`. (There's machinery — Spivak's adjoint functors between finite metric spaces and fuzzy simplicial sets — that makes this exact rather than hand-waved: converting a finite metric space to its fuzzy combinatorial representation literally sends distance to `exp(-distance)`. I'll lean on the intuition: edge weight = membership strength = "probability this edge exists.") How do I set `sigma_i`? By analogy to perplexity calibration but cleaner: pick `sigma_i` so the total fuzzy membership out of `i` hits a fixed target — `sum_j exp(-d_ij/sigma_i) = const`. Fix the cardinality of the fuzzy neighborhood. I'll need to choose that constant; I'll come back to it. It's a one-dimensional monotone equation in `sigma_i`, so a binary search solves it per point.

The plain exponential kernel still has a high-dimensional failure mode: many points can become effectively isolated, with all their edge weights tiny. That should not happen if `M` is locally connected. The culprit is the curse of dimensionality. In high `n`, distances concentrate; the nearest neighbor and the tenth nearest are almost the same distance away, and all of them are far in absolute terms. So `exp(-d_ij/sigma_i)` can be uniformly small, and the kernel reads the least informative part of the data: absolute distance. The information that survives is the *relative* spacing of the neighbors. What property of the manifold am I failing to encode? Local connectivity: at every point, some sufficiently small neighborhood is connected. No point should be completely cut off; each point must reach at least its nearest neighbor with full confidence. So let me bake that in. Let `rho_i` be the distance to `i`'s nearest nonzero-distance neighbor, and measure the kernel from `rho_i` outward instead of from zero: weight `= exp(-max(0, d(x_i, x_j) - rho_i)/sigma_i)`. Now the nearest neighbor sits at `d = rho_i`, the `max(0, ...)` is zero, and its weight is exactly 1: full membership, guaranteed connection. And the kernel now reads `d - rho_i`, the *excess* over the nearest-neighbor distance, which is precisely the relative spacing that survives in high dimensions. The same offset cures both the isolation and the concentration. With the offset in place, the calibration target becomes `sum_{j=1}^k exp(-max(0, d_ij - rho_i)/sigma_i) = target`. What target? The nearest neighbor already contributes 1; I want the remaining mass spread over the rest of the `k`-neighborhood so the effective fuzzy cardinality is meaningful. The implementation can include a bandwidth multiplier, but at the default bandwidth the chosen fuzzy cardinality is `log2(k)`, small enough that the tail doesn't dominate, growing slowly with `k`, and selected empirically. I'll take `target = log2(k)` by default and binary-search `sigma_i` to match it.

So far I have, for each `i`, a directed fuzzy edge to each neighbor `j` with weight `v_{j|i} = exp(-max(0, d_ij - rho_i)/sigma_i)`. But these are *directed*, and the local metrics are incompatible: from `i`'s vantage `v_{j|i}` might be 0.7, while from `j`'s vantage `v_{i|j}` is 0.2, because `i` and `j` have different `rho` and `sigma`. The graph layout wants a single undirected weight per edge. Which one is "right"? Neither — they're two local opinions and I need to reconcile them into one global graph. I could take the max, the min, the arithmetic or geometric mean — but those are arbitrary, and I promised myself principled choices. Here the "weight = probability the edge exists" reading pays off. If `v_{j|i}` is the probability that `i` thinks the edge exists and `v_{i|j}` the probability that `j` thinks so, then the natural combined statement is "the edge exists if *at least one* endpoint vouches for it" — the probabilistic union: `v_ij = v_{j|i} + v_{i|j} - v_{j|i} v_{i|j}`. That's `P(A or B) = P(A) + P(B) - P(A)P(B)` for independent existence. In matrix form, with `A` the directed weighted adjacency, the symmetric graph is `B = A + A^T - A ∘ A^T` (Hadamard product). This is a fuzzy-set union via the probabilistic t-conorm, and it's not a heuristic — it's *the* union operation once you accept the probabilistic semantics. It also has a nice side effect: it pulls "reverse" neighbors into the graph (if `i` is in `j`'s neighbor list but not vice versa, the union keeps the edge), which softens the pure-kNN asymmetry. That's the high-dimensional graph done: a single undirected fuzzy graph `G` whose edge weights are existence probabilities, every design choice traced back to "assume uniformity, infer the metric; demand local connectivity; merge by probabilistic union."

Now the layout. I need to build the *same* kind of fuzzy structure for the low-dimensional points `Y = {y_i} in R^d` and make it match `G`. Two simplifications fall out for free. First, in the embedding I'm not approximating an unknown manifold — I *chose* the target manifold to be plain `R^d` with the global Euclidean metric, so there's no per-point varying metric to estimate; the low-dim affinity is a single global function of `||y_i - y_j||`. Second, the local-connectivity offset `rho` was computed from the data; in the embedding I don't have data to compute it from, so I promote it to a hyperparameter `min_dist` — the distance below which points are allowed to be fully "together" in the layout. So the embedding affinity is some `w_ij = Phi(||y_i - y_j||)` with `Phi(0..min_dist) ≈ 1` and decaying after.

How do I measure "match" between the two fuzzy graphs, and what does it buy me over what's out there? Let me look hard at what the field does and where it stalls, because the gap is the whole contribution. t-SNE builds input affinities much like mine (Gaussian, perplexity-calibrated, symmetrized) and embedding affinities with a heavy-tailed Student-t `w_ij = (1 + ||y_i - y_j||^2)^{-1}` — the heavy tail there is a deliberate fix for the *crowding problem*, the fact that there isn't room in 2D to honor all distances, so moderate input distances are allowed to become large output distances. Then it normalizes *both* sides over all pairs into probability distributions — `q_ij = w_ij / sum_{k != l} w_kl` — and minimizes `KL(P || Q)`. Two things about this. The normalization of `q_ij` carries a partition function `sum_{k != l} w_kl` summing over *every* pair, so the gradient of the cost couples all `N^2` pairs; that's why t-SNE needs Barnes–Hut trees and still can't reach millions of points. And the KL is lopsided: `sum p_ij log(p_ij/q_ij)` blows up when `p_ij` is large and `q_ij` small — originally-near points placed far apart are punished hard — but when `p_ij ≈ 0`, the term `p_ij log(...) ≈ 0` regardless of `q_ij`, so originally-*far* points placed near each other are barely penalized. That asymmetry is exactly why t-SNE's clusters are locally crisp but globally float around arbitrarily; nothing in the cost insists that distinct things stay distinct.

LargeVis already saw the scaling half of this. Its move: drop the embedding-side normalization entirely. Work with the unnormalized `w_ij` and maximize `sum p_ij log w_ij + gamma sum log(1 - w_ij)` — first term attractive, second repulsive over all pairs, with `gamma` weighting them. No partition function, so plain SGD works: sample edges for the first term, sample random negatives for the second (the word2vec trick — approximate an all-pairs sum by a few random samples). That gets to millions of points. But it's assembled by hand: the first term pairs `p_ij` (an input affinity) with `log w_ij` (a raw output affinity) rather than with a matched quantity, and the attraction/repulsion balance is a free knob `gamma` with no principled value. I want the scaling *and* a cost whose attraction-repulsion balance is fixed by the comparison itself rather than by a separate likelihood term.

The fuzzy-graph framing gives me the cost. Both graphs, high and low, are fuzzy graphs over the *same* set of points and therefore the same reference set of possible 1-simplices `E`. Each edge `e` carries `v(e)` in the high-dimensional graph and `w(e)` in the low-dimensional graph, and I've been reading each as the probability that the edge exists, a Bernoulli parameter. So comparing the two graphs is comparing two vectors of Bernoulli probabilities over the same index set. The right divergence between Bernoullis is not KL-between-distributions, because these weights do not sum to one and are not a distribution over mutually exclusive outcomes. It is the **cross-entropy** summed edgewise:

  `C = sum_{e in E} [ v(e) log(v(e)/w(e)) + (1 - v(e)) log((1 - v(e))/(1 - w(e))) ].`

Stare at this and the asymmetry that crippled t-SNE is gone. There are two matched terms per edge. The first, `v log(v/w)`, is large when `v` is high but `w` is low — an edge that should exist but doesn't in the layout — so it *attracts*: minimizing it pushes `w` up, i.e. pulls the points together. The second, `(1 - v) log((1 - v)/(1 - w))`, is large when `v` is low but `w` is high — an edge that should *not* exist but does in the layout — so it *repels*: it pushes `w` down, pulls the points apart. Crucially the second term is present and weighted by `(1 - v)` for *every* pair, including the originally-far ones — exactly the penalty t-SNE was missing. So cross-entropy of edge existence pushes on both false absences and false presences, which is the global-structure pressure I wanted. And it's not a third heuristic dropped next to LargeVis: LargeVis's two terms are a hand-built shadow of this — its `gamma` is standing in for a repulsion balance that the cross-entropy supplies from `(1 - v)`. The canonical update can still expose a unit-default repulsion multiplier as a practical control, but the baseline objective no longer needs a separate LargeVis-style balance term.

Better still for scaling: split the cost into the parts that depend on `w` and the parts that don't. Expand,

  `C = sum_e [ v log v + (1 - v) log(1 - v) ]  -  sum_e [ v log w + (1 - v) log(1 - w) ].`

The first bracket is a function of `v` alone — the input graph, fixed during layout — so it's a constant I can drop. I only need to minimize `-sum_e [ v log w + (1 - v) log(1 - w) ]`. There's no normalization over all pairs anywhere — `w` appears only through `log w` and `log(1 - w)`, never through a partition function — so the gradient decomposes edge by edge and I can optimize by stochastic gradient descent the way LargeVis does, but now descending a *derived* cross-entropy instead of a hand-tuned likelihood. The `v log w` term I handle by sampling edges with probability `v` and pulling; the `(1 - v) log(1 - w)` term sums over all pairs, so I approximate it by negative sampling — for each positive edge I draw a few random vertices, treat them as non-edges (`v ≈ 0`), and push.

Now I need the embedding affinity `w` to be differentiable so SGD can run. I want `w(0) ≈ 1` out to `min_dist`, then a smooth decay. A clean family with a simple closed-form gradient is

  `Phi(y_i, y_j) = (1 + a (||y_i - y_j||^2)^b)^{-1},`

with two shape parameters `a, b`. Note that `a = b = 1` recovers exactly the Student-t `(1 + d^2)^{-1}` of t-SNE — so this family *generalizes* the heavy-tailed embedding kernel rather than discarding it, and I get the crowding-problem fix for free while gaining the freedom to match `min_dist`. How to set `a, b`? Fit them. The target curve `Psi` is the offset exponential that exactly encodes `min_dist`:

  `Psi(d) = 1` if `d <= min_dist`, else `exp(-(d - min_dist)/spread)`,

i.e. full membership inside `min_dist`, exponential decay beyond — the low-dim analogue of the `exp(-max(0, d - rho))` kernel I used for the input graph, with `min_dist` in the role of `rho` and `spread` setting the decay length. So I sample `d` over `0` to `3 * spread`, evaluate `Psi`, and do a nonlinear least-squares fit of `Phi` to `Psi` to get `a, b`. With the default `min_dist = 0.1` and `spread = 1` this gives roughly `a ≈ 1.929`, `b ≈ 0.7915`. `min_dist` is then a genuinely meaningful knob: small `min_dist` lets points pack tightly (faithful but cramped), large `min_dist` spreads clusters apart for legibility.

Let me actually derive the gradients, because the forces are the algorithm and I want them exact, not gestured at. Write `D = ||y_i - y_j||^2` so `w = (1 + a D^b)^{-1}`, and note `dD/dy_i = 2(y_i - y_j)`. For the attractive term I need the gradient of `v log w` with respect to `y_i`. First `d(log w)/dD`:

  `log w = -log(1 + a D^b)`, so `d(log w)/dD = -(a b D^{b-1})/(1 + a D^b).`

Chain through `dD/dy_i = 2(y_i - y_j)`:

  `grad_{y_i}(v log w) = v * [ -2 a b D^{b-1} / (1 + a D^b) ] (y_i - y_j).`

So the attractive force coefficient is `-2 a b D^{b-1}/(1 + a D^b)`, scaled by the edge weight `v` and applied along `(y_i - y_j)`. It's negative, so it moves `y_i` toward `y_j` — a pull, stronger for heavier edges, exactly what I want. Since `D = d^2`, the same coefficient is `-2ab d^{2(b-1)}/(1 + a d^{2b})`.

For the repulsive term, the gradient of `(1 - v) log(1 - w)` with respect to `y_i`. Now `1 - w = a D^b/(1 + a D^b)`, so

  `log(1 - w) = log(a D^b) - log(1 + a D^b)`,
  `d(log(1 - w))/dD = b/D - (a b D^{b-1})/(1 + a D^b) = (b/D) * [ 1 - a D^b/(1 + a D^b) ] = (b/D) * 1/(1 + a D^b) = b / (D (1 + a D^b)).`

Chain through `2(y_i - y_j)`:

  `grad_{y_i}((1 - v) log(1 - w)) = (1 - v) * [ 2 b / (D (1 + a D^b)) ] (y_i - y_j).`

Positive coefficient — it pushes `y_i` away from `y_j`, a repulsion, scaled by `(1 - v)`. One numerical hazard: as `D -> 0` the `1/D` in the repulsive coefficient blows up — two coincident points repel infinitely. So I floor the denominator with a small `eps`: `2 b / ((eps + D)(1 + a D^b))`, with `eps = 0.001`. And since SGD on a non-convex layout can throw large steps anyway (a tiny `D` still makes a big force), I clip each coordinate of the gradient to a fixed range like `[-4, 4]` before applying it — cheap insurance against a single update flinging a point across the embedding.

Now assemble the optimizer the way the scaling demands. I'm minimizing `-sum_e v log w - sum_e (1 - v) log(1 - w)`. The first sum is over the real edges weighted by `v`; rather than touch every edge every epoch, I sample edge `e` with frequency proportional to its weight `v(e)`. High-membership edges get pulled more often, which is the stochastic approximation of weighting the term by `v`. Concretely, if `max_v` is the largest edge weight, I want edge `e` sampled `n_epochs * v(e)/max_v` times over the whole run, so its wait time is `epochs_per_sample[e] = n_epochs / (n_epochs * v(e)/max_v) = max_v/v(e)` for positive weights; zero weights get no schedule. An edge is processed in epoch `m` whenever its running counter says it is due. For each processed edge I apply the attractive update to its endpoints. Then for the repulsive term, which is over all non-edges and far too many to enumerate, I negative-sample: for the just-pulled vertex I draw `n_neg_samples` random vertices, treat each as a non-edge (`v ≈ 0`), and apply the repulsive update. The exact negative-sample distribution from the cross-entropy can be approximated by uniform vertex samples for large `N`, matching the practical implementation. Default `n_neg_samples = 5`. The number of negative samples per positive edge is paced by a second counter, `epochs_per_negative_sample = epochs_per_sample / n_neg_samples`, so the repulsion budget stays proportional to the attraction budget. The learning rate decays linearly, `alpha = 1 - m/n_epochs`, simulated-annealing style, so the layout cools into a local minimum of the non-convex objective; `n_epochs` defaults to 500 for modest `N` and 200 for very large `N`. Total work scales with the number of edges, `O(k N)`, which is the linear-time training I needed.

One more piece, and it's the one that gives the local optimizer a global scaffold: initialization. Random init plus this local force-directed optimization would find *a* local minimum, but probably one where distant components can start in arbitrary relative positions. But I have the fuzzy graph `G`, and I argued at the very start that its normalized Laplacian approximates the Laplace–Beltrami operator of `M` under the usual sampling assumptions. So the bottom eigenvectors of `G`'s symmetric normalized Laplacian `L_sym = D^{-1/2}(D - A)D^{-1/2}` are a globally coherent set of manifold coordinates — exactly the Laplacian-eigenmaps embedding. Use them as the starting layout. This gives the SGD a starting point that already reflects the graph's coarse arrangement, so the local forces refine rather than invent it from random coordinates; it converges faster and more stably. Scale the coordinates into the optimizer's working range and add a sliver of noise to avoid a symmetric saddle. This is why the same fuzzy graph serves twice — it's the source of both the spectral init *and* the cross-entropy target.

The pipeline now has its missing pieces: the edge-weight rule, the directed-view reconciliation, the smooth embedding kernel, and the two update coefficients.

```python
import numpy as np
from scipy.sparse import coo_matrix, identity, spdiags
from scipy.sparse.linalg import eigsh
from scipy.optimize import curve_fit
from sklearn.neighbors import NearestNeighbors

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3


def smooth_knn_dist(distances, k, n_iter=64, local_connectivity=1.0, bandwidth=1.0):
    """Per-point rho_i (distance to nearest neighbor, for local connectivity) and
    sigma_i solving sum_j exp(-max(0, d_ij-rho_i)/sigma_i) = log2(k)*bandwidth."""
    target = np.log2(k) * bandwidth
    rho = np.zeros(distances.shape[0])
    sigma = np.zeros(distances.shape[0])
    mean_distances = np.mean(distances)
    for i in range(distances.shape[0]):
        ith = distances[i]
        nonzero = ith[ith > 0.0]
        if nonzero.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0:
                rho[i] = nonzero[index - 1]
                if interpolation > SMOOTH_K_TOLERANCE:
                    rho[i] += interpolation * (nonzero[index] - nonzero[index - 1])
            else:
                rho[i] = interpolation * nonzero[0]
        elif nonzero.shape[0] > 0:
            rho[i] = np.max(nonzero)
        lo, hi, mid = 0.0, np.inf, 1.0
        for _ in range(n_iter):                                 # binary search for sigma_i
            psum = 0.0
            for j in range(1, distances.shape[1]):
                d = distances[i, j] - rho[i]
                psum += np.exp(-d / mid) if d > 0 else 1.0
            if abs(psum - target) < SMOOTH_K_TOLERANCE:
                break
            if psum > target:
                hi = mid; mid = (lo + hi) / 2.0
            else:
                lo = mid; mid = mid * 2 if hi == np.inf else (lo + hi) / 2.0
        sigma[i] = mid
        # floor sigma so a point with a spread-out neighborhood can't collapse to ~0
        if rho[i] > 0.0:
            sigma[i] = max(sigma[i], MIN_K_DIST_SCALE * np.mean(ith))
        else:
            sigma[i] = max(sigma[i], MIN_K_DIST_SCALE * mean_distances)
    return sigma, rho


def spectral_layout(G, dim):
    """Bottom nontrivial eigenvectors of the symmetric normalized graph Laplacian."""
    N = G.shape[0]
    deg = np.asarray(G.sum(axis=0)).ravel()
    inv_sqrt_deg = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    D = spdiags(inv_sqrt_deg, 0, N, N)
    L = identity(N, dtype=np.float64) - D * G * D
    vals, vecs = eigsh(L.tocsc(), dim + 1, which="SM", v0=np.ones(N), maxiter=N * 5)
    order = np.argsort(vals)
    return vecs[:, order[1:dim + 1]]


def noisy_scale_coords(coords, rng, max_coord=10.0, noise=0.0001):
    coords = coords * (max_coord / np.abs(coords).max())
    return coords + rng.normal(scale=noise, size=coords.shape)


def fuzzy_simplicial_set(knn_indices, knn_dists, N, k):
    """High-dim fuzzy graph: membership v_{j|i} = exp(-max(0, d - rho_i)/sigma_i),
    then symmetrize by the probabilistic t-conorm  v_ij = v_{j|i} + v_{i|j} - v_{j|i} v_{i|j}."""
    sigma, rho = smooth_knn_dist(knn_dists, k)
    rows, cols, vals = [], [], []
    for i in range(N):
        for j in range(knn_indices.shape[1]):
            nb = knn_indices[i, j]
            if nb == i:
                val = 0.0
            elif knn_dists[i, j] - rho[i] <= 0.0 or sigma[i] == 0.0:
                val = 1.0                                        # nearest neighbor: membership 1
            else:
                val = np.exp(-((knn_dists[i, j] - rho[i]) / sigma[i]))
            rows.append(i); cols.append(nb); vals.append(val)
    A = coo_matrix((vals, (rows, cols)), shape=(N, N))
    A.eliminate_zeros()
    AT = A.transpose()
    prod = A.multiply(AT)                                        # A ∘ A^T
    G = A + AT - prod                                            # A + A^T - A∘A^T  (fuzzy union)
    G.eliminate_zeros()
    return G.tocoo()


def find_ab_params(spread, min_dist):
    """Fit a, b so Phi(d) = (1 + a d^{2b})^{-1} matches the offset-exponential target Psi."""
    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))
    xv = np.linspace(0, spread * 3, 300)
    yv = np.where(xv < min_dist, 1.0, np.exp(-(xv - min_dist) / spread))
    (a, b), _ = curve_fit(curve, xv, yv)
    return a, b


def clip(v):
    return np.clip(v, -4.0, 4.0)                                 # gradient clamp for SGD stability


def make_epochs_per_sample(weights, n_epochs):
    """Canonical edge-sampling schedule: stronger edges are visited more often."""
    result = np.full(weights.shape[0], -1.0, dtype=np.float64)
    n_samples = n_epochs * (weights / weights.max())
    positive = n_samples > 0
    result[positive] = float(n_epochs) / n_samples[positive]
    return result


def optimize_layout(head, tail, weights, Y, n_epochs, a, b,
                    n_neg_samples=5, gamma=1.0, rng=None):
    """Minimize  -sum_e v log w - sum_e (1-v) log(1-w)  by SGD with edge + negative sampling."""
    N = Y.shape[0]
    dim = Y.shape[1]
    rng = np.random.default_rng(rng)
    # edge e processed ~ proportional to its weight v(e): epochs to wait between samples
    epochs_per_sample = make_epochs_per_sample(weights, n_epochs)
    epochs_per_neg = epochs_per_sample / n_neg_samples
    next_sample = epochs_per_sample.copy()
    next_neg = epochs_per_neg.copy()
    alpha = 1.0
    for epoch in range(n_epochs):
        for e in range(head.shape[0]):
            if next_sample[e] > epoch:
                continue
            j, k_ = head[e], tail[e]
            cur, oth = Y[j], Y[k_]
            d2 = np.sum((cur - oth) ** 2)
            if d2 > 0.0:                                         # attractive: grad of v log w
                coeff = -2.0 * a * b * d2 ** (b - 1.0)
                coeff /= a * d2 ** b + 1.0
            else:
                coeff = 0.0
            grad = clip(coeff * (cur - oth))
            Y[j] += grad * alpha
            Y[k_] -= grad * alpha                               # both endpoints move
            next_sample[e] += epochs_per_sample[e]

            n_neg = int((epoch - next_neg[e]) / epochs_per_neg[e])
            for _ in range(n_neg):                              # repulsive: grad of (1-v) log(1-w)
                c = rng.integers(N)
                oth = Y[c]
                d2 = np.sum((cur - oth) ** 2)
                if d2 > 0.0:
                    coeff = 2.0 * gamma * b
                    coeff /= (0.001 + d2) * (a * d2 ** b + 1.0)  # eps floor against d2 -> 0
                elif j == c:
                    continue
                else:
                    coeff = 0.0
                grad = clip(coeff * (cur - oth)) if coeff > 0 else np.zeros(dim)
                Y[j] += grad * alpha
            next_neg[e] += n_neg * epochs_per_neg[e]
        alpha = 1.0 - epoch / n_epochs                          # linear (annealing) LR decay
    return Y


def fit_transform(X, n_neighbors=15, n_components=2, min_dist=0.1, spread=1.0,
                  n_epochs=None, random_state=None):
    rng = np.random.default_rng(random_state)
    N = X.shape[0]
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    knn_dists, knn_indices = nn.kneighbors(X)

    G = fuzzy_simplicial_set(knn_indices, knn_dists, N, float(n_neighbors))

    if n_epochs is None:
        n_epochs = 500 if N <= 10000 else 200
    # drop edges too weak to be sampled even once in n_epochs
    G.data[G.data < G.data.max() / float(n_epochs)] = 0.0
    G.eliminate_zeros()

    a, b = find_ab_params(spread, min_dist)

    # spectral initialization: bottom eigenvectors of the normalized Laplacian of G
    Y = spectral_layout(G, n_components)                       # Laplace-Beltrami starting layout
    Y = noisy_scale_coords(Y, rng).astype(np.float64)
    Y = 10.0 * (Y - np.min(Y, axis=0)) / (np.max(Y, axis=0) - np.min(Y, axis=0))

    Y = optimize_layout(G.row, G.col, G.data, Y.astype(np.float64),
                        n_epochs, a, b, rng=random_state)
    return Y
```

I now have the method I was trying to force out of the problem. PCA can't bend, MDS-family methods spend their budget on global distances and don't scale, Laplacian eigenmaps is principled but needs uniform sampling and is rigid, t-SNE has superb local structure but its all-pairs normalization makes it `O(N^2)` and its asymmetric KL lets global structure float, and LargeVis fixes scaling by dropping normalization but hand-balances the cost with a free `gamma`. The topological framing tells me a neighbor graph approximates the manifold through the Nerve theorem but needs uniform data. I can't make data uniform, so I assume uniformity and *infer* a per-point Riemannian metric, choosing the scalar local metric that turns fixed fuzzy cardinality into distance normalization by each point's own `k`-neighborhood scale. High-dimensional isolation forces the local-connectivity offset `rho_i`, which simultaneously fixes distance concentration by reading relative spacing. Incompatible directed weights force a principled merge, the probabilistic t-conorm `v_{j|i} + v_{i|j} - v_{j|i}v_{i|j}`, because the weights are edge-existence probabilities. Reading both graphs as Bernoulli existence probabilities over a shared edge set makes cross-entropy the natural divergence, and its `(1 - v) log(1 - w)` term supplies the symmetric repulsion t-SNE lacks and the LargeVis likelihood has to weight separately. Dropping the `v`-only constant leaves a normalization-free objective, hence plain SGD with edge sampling for attraction and negative sampling for repulsion, linear in the number of edges. A smooth `(1 + a d^{2b})^{-1}` kernel, generalizing t-SNE's Student-t and fit to an offset exponential carrying `min_dist` and `spread`, makes the cost differentiable, and its two derivatives are exactly the attractive coefficient `-2ab d^{2(b-1)}/(1 + a d^{2b})` and the repulsive coefficient `2b/((eps + d^2)(1 + a d^{2b}))`. Because the same fuzzy graph's normalized Laplacian approximates the Laplace–Beltrami operator under the usual sampling assumptions, its spectral embedding initializes the layout in a globally coherent coordinate system the local forces can refine. The geometric pieces, local metric normalization, local connectivity, fuzzy union, cross-entropy, and the smooth low-dimensional membership curve, carry the derivation; `log2(k)` is the empirical fuzzy-cardinality target, while `eps = 0.001`, coordinate clipping, and linear `alpha` decay are the numerical choices that make the stochastic optimizer behave.
