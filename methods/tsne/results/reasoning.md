I want to *look* at high-dimensional data. I have a few thousand handwritten digits, each a 784-pixel vector, or a pile of documents each a word-count vector with thousands of entries, and I want a single two-dimensional scatterplot that tells me the truth about them: that the 7s cluster together and the 1s cluster together, that within the 7s the ones with a crossbar drift to one side, that there are roughly ten blobs and they sit apart from each other. So the map has to do two jobs at once, and I should be honest that these pull in different directions. Locally, points that are close in 784 dimensions must land close in the plane — neighborhoods preserved. Globally, the blobs must sit at sensible places relative to each other and, crucially, there must be *gaps* between them so I can see them as separate clusters. And I am not allowed to use the class labels to place anything; labels, when they exist, only color the dots afterward. So this is unsupervised: out of nothing but the pairwise geometry of the data, manufacture a faithful planar picture.

Let me first feel where the existing tools break, because the failure modes are the design spec. The distance-matching family — classical scaling, PCA — minimizes squared error between high- and low-dimensional pairwise distances. But a sum of *squared* distance errors is dominated by the big distances, so these methods bend over backwards to keep faraway things faraway and barely care about the small distances that actually carry the local manifold structure. And PCA is linear, so it can't even follow a curved manifold. Sammon mapping tries to patch the bias by dividing each squared error by the original distance, C = (1/Σ||x_i−x_j||) Σ (||x_i−x_j|| − ||y_i−y_j||)² / ||x_i−x_j||, so small distances get up-weighted. But now the weight 1/||x_i−x_j|| blows up exactly where ||x_i−x_j|| is tiny, so the cost becomes obsessed with small *differences* between already-tiny distances — two points that happen to be nearly coincident dominate everything. I don't want a method whose attention is hostage to the most fragile pairs; I want roughly equal importance across all the small distances. Isomap and LLE and Laplacian Eigenmaps go spectral, but they have their own holes — Isomap short-circuits when one noisy edge bridges the graph, the spectral methods only avoid collapsing the whole map to a point by a covariance constraint that's cheaply gamed by a "curdled" map (everything piled at the center, a couple of outliers placed far out to supply variance), and none of them can show two genuinely separated submanifolds because the data doesn't give a connected graph.

The frame that actually feels right to me is the probabilistic one. Stop talking about distances and talk about neighbors-as-probabilities. Center a Gaussian on x_i and ask: if x_i picked one neighbor with probability falling off as that Gaussian, how likely would it pick x_j? That's

  p_{j|i} = exp(−||x_i − x_j||² / 2σ_i²) / Σ_{k≠i} exp(−||x_i − x_k||² / 2σ_i²),

with p_{i|i} set to zero. The lovely thing here is that the width σ_i is *per point*, and there's a principled way to choose it: fix the entropy of each neighbor distribution. Define the perplexity Perp(P_i) = 2^{H(P_i)} with H(P_i) = −Σ_j p_{j|i} log₂ p_{j|i} in bits. Perplexity reads as a smooth "effective number of neighbors," and because it climbs monotonically with σ_i I can binary-search σ_i for each point to hit a target perplexity — denser regions automatically get a smaller σ_i. Pick a perplexity somewhere in 5 to 50; the result is robust to the exact value. Let me sanity-check the monotonicity claim before I lean on the binary search, because if perplexity weren't monotone in σ_i the search would be meaningless. Take three neighbors at squared distances 1, 4, 9 from x_i and a small width, β = 1/(2σ_i²) = 1: the unnormalized weights are e^{−1}, e^{−4}, e^{−9} = 0.368, 0.0183, 0.00012, so after normalizing, p ≈ (0.952, 0.047, 0.0003), entropy H = −Σ p ln p ≈ 0.194 nats, perplexity e^{0.194} ≈ 1.21 — almost one effective neighbor, the nearest one dominates. Now widen to β = 0.1: weights e^{−0.1}, e^{−0.4}, e^{−0.9} = 0.905, 0.670, 0.407, normalized p ≈ (0.457, 0.338, 0.205), H ≈ 1.05 nats, perplexity ≈ 2.86 — almost three effective neighbors. So widening the Gaussian did raise the effective neighbor count, monotonically as claimed; the binary search is well-posed. The high-dimensional side I'll keep.

The low-dimensional side, in the predecessor I'm building on, mirrors it: put Gaussians of fixed width on the map points and define q_{j|i} = exp(−||y_i − y_j||²) / Σ_{k≠i} exp(−||y_i − y_k||²). (The 1/√2 width is just absorbed into the exponent; the overall scale of the map is free anyway, so fixing the low-D variance only fixes the scale.) If the map is faithful, q_{j|i} ≈ p_{j|i}. So measure unfaithfulness by the natural distance between two distributions, the Kullback-Leibler divergence, summed per point:

  C = Σ_i KL(P_i || Q_i) = Σ_i Σ_j p_{j|i} log(p_{j|i} / q_{j|i}),

and descend it. I actually like the *asymmetry* of KL here. KL(P||Q) punishes a small q modeling a large p far more than a large q modeling a small p — so it's expensive to place two map points far apart when their data points are close (small q, large p: a big log), but cheap to place two map points near each other when their data points are far (large q, small p: small cost). That asymmetry tilts the whole objective toward getting the *local* structure right, which is exactly what I want from a visualization. The gradient of this conditional KL works out to δC/δy_i = 2 Σ_j (p_{j|i} − q_{j|i} + p_{i|j} − q_{i|j})(y_i − y_j) — a sum of spring forces along (y_i − y_j), each spring pulling or pushing depending on whether the pair is modeled too close or too far.

But two things are wrong with this, and I have to be precise about them because they are what I have to fix. The first is optimization pain. Those conditional normalizations — a separate denominator per point i in both p and q — make the gradient awkward, and in practice the only way to get a decent map out is simulated annealing: add Gaussian noise to the map points each iteration, decay its variance slowly, babysit the momentum and step-size schedules, and even then rerun the whole thing several times to find settings that work. That's a lot of machinery for a visualization tool. I'd love a cost whose gradient is simpler and whose optimization doesn't need annealing.

The second is worse and more fundamental: crowding. Picture a manifold whose intrinsic dimension is well above two — say ten — embedded in the high-dimensional pixel space. I want to squash it into the plane. In ten dimensions the volume of a ball around point i grows like r^{10}, so the number of points at a *moderate* distance from i is large; there are far more "moderately far" neighbors than "near" ones. Now try to lay them out in 2-D at the right distances. The area available at a moderate radius in the plane grows only like r², nowhere near enough to seat all those moderately-distant points where they belong. So they get shoved too far out. Each one, being modeled by a Gaussian, still exerts a small attractive force back toward i; individually tiny, but there are so many of them that their sum crushes the whole map inward toward the center, and the gaps between the natural clusters never form. The clusters smear into one ball. Let me make sure I see *why* it's the Gaussian's fault: a Gaussian tail decays so fast that to represent "moderately dissimilar" as "moderately far" in the map, the map distance has to stay modest, and there just isn't room for modest distances times that many points. This crowding is not special to the conditional-KL method — it shows up in Sammon mapping too — but it's fatal for visualization, and indeed people have found that even strong methods can't cleanly separate the digit classes.

So I have a concrete to-do list: make the optimization easy (kill the annealing, simplify the gradient), and make room in the map so the clusters separate (kill the crowding). Let me take them in turn and see if one move can help both.

Start with the optimization. The conditional formulation has one normalization per point, which is the source of the asymmetry p_{j|i} ≠ p_{i|j} and the per-point KL sum. What if I match a *single* joint distribution over all pairs instead of one conditional distribution per point? Define a joint P over the high-dimensional pairs and a joint Q over the map pairs, each normalized once over all pairs, and minimize a single KL:

  C = KL(P || Q) = Σ_i Σ_j p_ij log(p_ij / q_ij),

with p_ii = q_ii = 0. With one global normalization, p_ij = p_ji and q_ij = q_ji by construction — call it symmetric SNE — and the gradient should collapse to something much cleaner because there's only one denominator to differentiate through.

Let me try the obvious joint, a single Gaussian over all high-dimensional pairs, p_ij = exp(−||x_i−x_j||²/2σ²) / Σ_{k≠l} exp(−||x_k−x_l||²/2σ²). And immediately I hit a wall: an outlier. Suppose x_i sits far from everything. Then every ||x_i − x_j||² is large, so every p_ij involving i is minuscule. Which means y_i barely appears in the cost at all — its position is essentially undetermined, it'll wander off and contribute nothing. With the per-point conditional version this didn't happen, because each p_{j|i} was normalized within point i's own row, so even an outlier had a proper distribution. I've traded the outlier robustness for the simpler gradient. I don't want to give up either. 

How do I get the single-KL simplicity *and* keep every point anchored? Go back to the conditionals I already trust — each row p_{j|i} is a genuine distribution — and *build* the joint out of them by symmetrizing:

  p_ij = (p_{j|i} + p_{i|j}) / (2n).

Now check the outlier. Σ_j p_ij = (1/2n) Σ_j (p_{j|i} + p_{i|j}) = (1/2n)(1 + Σ_j p_{i|j}) ≥ 1/(2n), because Σ_j p_{j|i} = 1 (it's a normalized row) regardless of how outlying i is. So every point, even a total outlier, contributes at least 1/(2n) of the total probability mass to the cost — its position is forced to matter. Let me confirm the bound is actually tight, i.e. that an outlier hits exactly 1/(2n) and not something larger that I've mis-derived. Take n = 5 and make point 0 a genuine outlier: every other point j gives it essentially zero conditional mass, p_{0|j} ≈ 0, so the column term Σ_j p_{0|j} ≈ 0 and the row sum should collapse to (1/2n)(1 + 0) = 1/10 = 0.1. Running the symmetrization on conditionals with p_{0|j} forced to 10^{−9} for j ≠ 0, the joint's row-0 sum comes out 0.1000000005 — exactly 1/(2n) up to the floor I injected. So the bound is tight precisely at the outlier, which is the case I cared about; the single-Gaussian joint, by contrast, would have driven that whole row to ~0 and let y_0 wander free. The symmetrized-conditional joint keeps the per-point Gaussian-perplexity machinery I like on the high-dimensional side, gives me a single joint P normalized over all pairs, and anchors outliers. On the map side, if I keep a Gaussian joint q_ij ∝ exp(−||y_i − y_j||²), the one-denominator calculation is now almost mechanical: write s_ij = ||y_i − y_j||² and Z = Σ_{k≠l} exp(−s_kl), so log q_ij = −s_ij − log Z. The map-dependent part of the cost is −Σ p_ij log q_ij = Σ p_ij s_ij + log Z (using Σ p_ij = 1). Differentiating in s_ij: the first sum gives p_ij, and (1/Z)∂Z/∂s_ij = −exp(−s_ij)/Z = −q_ij, so ∂C/∂s_ij = p_ij − q_ij. The two ordered distances s_ij and s_ji each carry that scalar out along (y_i − y_j) with a factor 2 (from ∂s_ij/∂y_i = 2(y_i − y_j)), so δC/δy_i = 4 Σ_j (p_ij − q_ij)(y_i − y_j) — no more of the four-term (p_{j|i} − q_{j|i} + p_{i|j} − q_{i|j}) business, just a single (p_ij − q_ij) per pair. That is the simpler gradient I was after from the symmetrization. But notice I have *not* touched crowding yet — if I keep a Gaussian on the map side, q_ij is still a Gaussian-derived joint and the same volume argument crushes the map. The symmetrization bought me simplicity, not room.

Now the room. Here is the leverage I didn't have before: I'm matching joint *probabilities* now, p_ij to q_ij, not distances to distances. That means the function I use to convert a map distance into a probability does *not* have to be the same function I used in the high-dimensional space. In the high-dimensional space a Gaussian is fine — it sets the local neighborhood scale via perplexity. But in the *map* I am free to choose any kernel I like that turns ||y_i − y_j|| into a similarity, and I should choose one that fixes crowding. What does fixing crowding require? It requires that a *moderate* high-dimensional similarity (moderate p_ij) be representable by a *larger* map distance than a Gaussian would allow — I need the map to be able to spread the moderately-dissimilar points far out without paying a huge attractive penalty, so the gaps can open. That is exactly what a *heavy-tailed* distribution does: a kernel whose tail decays slowly so that a large map distance still carries a non-vanishing similarity, meaning a moderate p_ij can be matched by a much larger ||y_i − y_j|| than under a Gaussian. The unwanted attraction from the moderately-dissimilar points evaporates, and the map is free to push them out and let clusters separate.

So I want a heavy-tailed kernel on the map. Which one? Let me think about what I actually need from the tail. Take a Student-t distribution with ν degrees of freedom; as ν → ∞ it's a Gaussian (no help), and as ν → 1 it's the Cauchy, the heaviest-tailed Student-t, with density ∝ (1 + d²)^{-1}. Let me try ν = 1 and define the map joint as

  q_ij = (1 + ||y_i − y_j||²)^{-1} / Σ_{k≠l} (1 + ||y_k − y_l||²)^{-1}.

Why precisely one degree of freedom, the Cauchy? Look at the tail: for large d, (1 + d²)^{-1} → d^{-2}, an inverse-square law. That inverse-square behavior is a gift I should pin down. If I rescale the whole map by a constant c — stretch every y by c — then for far-apart pairs the unnormalized similarity goes like (c·d)^{-2} = c^{-2} d^{-2}, the same c^{-2} for every far pair, so it should factor out of the comparison between two far pairs. Let me actually check that, because "approximately invariant" is the kind of phrase I'd otherwise be tempted to wave through. Put three well-separated centroids in the plane — at (0,0), (10,0), (0,7) — so the pair (0,1) is at squared distance 100 and the pair (0,2) at squared distance 49, and watch the ratio of their unnormalized Student-t similarities, sim(0,1)/sim(0,2), as I scale the whole map by c = 1, 2, 4, 8. The ratios come out 0.4950, 0.4913, 0.4903, 0.4901 — they barely move, converging to the inverse-square limit 49/100 = 0.49 (the far similarity over the near one tends to the inverse ratio of their squared distances) as c grows, exactly what the c^{−2} cancellation predicts. For contrast I run the same scaling on a Gaussian map kernel exp(−d²): the corresponding ratio goes 7×10^{−23}, 3×10^{−89}, then underflows to nan — the Gaussian's relative geometry of far pairs is completely destroyed by rescaling, while the Student-t's is essentially preserved. So the heavy tail really does make the map *approximately scale-invariant for the far-apart points* — approximately, not exactly, since the ratios do drift a hair before settling. That means I don't have to get the global scale of the map right, and — more importantly — a large cluster of points that's far away interacts with the rest of the map almost exactly as a single point would, because the inverse-square law makes the internal scale of the distant cluster irrelevant to its long-range pull. The optimization then behaves the same way at all scales except the very finest, which is precisely the property that lets it find good global organization without the simulated-annealing crutch SNE needed. There are two more reasons ν=1 is the natural pick: a Student-t is an infinite mixture of Gaussians, so it's a close cousin of the high-dimensional Gaussian rather than something exotic; and (1 + d²)^{-1} has no exponential in it, so it's *much* cheaper to evaluate than a Gaussian — and I'm evaluating it for all O(n²) pairs every iteration, so that matters.

Before I commit, let me make sure the heavy tail doesn't reintroduce a problem I saw with the uniform-background fix. That earlier attempt, UNI-SNE, added a constant uniform floor ρ to the low-dimensional model so far-apart points couldn't have zero similarity. The trouble was that two far-apart points got essentially *all* their q from the constant floor, so moving them changed q only by a vanishing *proportional* amount — no restoring force — and once two halves of a cluster drifted apart early, nothing pulled them back. Does the Student-t have that disease? No, and the difference is exactly that the Student-t's tail is a *function of the distance*, not a constant. (1 + d²)^{-1} keeps depending on d no matter how large d is, so there's always a real proportional change when the points move, hence always a restoring force. The heavy tail gives me strong repulsion of dissimilar points modeled too close — but the repulsion doesn't run off to infinity the way it would with a kernel that decays too slowly, and unlike the uniform floor it leaves long-range *attractive* forces alive, so two cluster-halves that separated early can still be pulled back together. The Student-t threads the needle the uniform background couldn't.

Now I have to actually derive the gradient of this Student-t-based KL, because the whole point of the symmetric formulation was a clean gradient and I need to verify the heavy tail doesn't wreck it. Let me set up the auxiliary variables so the factors cannot disappear. I will write d_ij for the Euclidean map distance, d_ij = ||y_i − y_j||, and Z = Σ_{k≠l} (1 + d_kl²)^{-1}, so q_ij = (1 + d_ij²)^{-1} / Z. The cost is C = Σ_{i≠j} p_ij log(p_ij / q_ij) = Σ p_ij log p_ij − Σ p_ij log q_ij; the first sum is constant in the map, so only −Σ p_ij log q_ij matters for the gradient. When I move y_i, the only ordered distances that change are d_ij and d_ji for all j, and ∂d_ij/∂y_i = (y_i − y_j)/d_ij, with the zero-distance case handled by continuity after the factors cancel. So the y_i derivative will be

  δC/δy_i = Σ_j [δC/δd_ij · (y_i − y_j)/d_ij + δC/δd_ji · (y_i − y_j)/d_ij].

Now compute the scalar derivative. Since log q_kl = −log(1 + d_kl²) − log Z,

  C = constant + Σ_{k≠l} p_kl log(1 + d_kl²) + log Z,

using Σ_{k≠l} p_kl = 1. Differentiating with respect to d_ij gives

  δC/δd_ij = 2p_ij d_ij/(1 + d_ij²) + (1/Z) δZ/δd_ij.

The normalization derivative is δZ/δd_ij = −2d_ij(1 + d_ij²)^{-2}. Because q_ij = (1 + d_ij²)^{-1}/Z, the second term is −2d_ij(1 + d_ij²)^{-2}/Z = −2q_ij d_ij/(1 + d_ij²). Therefore

  δC/δd_ij = 2(p_ij − q_ij)d_ij(1 + d_ij²)^{-1}.

Substitute that back into the ordered-pair derivative:

  δC/δy_i = 4 Σ_j (p_ij − q_ij)(1 + d_ij²)^{-1}(y_i − y_j)
          = 4 Σ_j (p_ij − q_ij)(1 + ||y_i − y_j||²)^{-1}(y_i − y_j).

So the gradient is

  δC/δy_i = 4 Σ_j (p_ij − q_ij)(y_i − y_j)(1 + ||y_i − y_j||²)^{-1},

barely more complicated than the symmetric-SNE gradient — just the extra (1 + ||y_i − y_j||²)^{-1} factor riding along. That derivation went through several hand steps (the δZ/δd_ij term, the cancellation that produces −q_ij, the factor-of-2-then-doubled bookkeeping for the ordered pairs), and any one of them could hide a sign error or a dropped factor, so I don't want to trust it on inspection. Let me check it against finite differences on a tiny instance: n = 4 points in 2-D with a fixed symmetric P, compute the analytic gradient from the formula above, then perturb each coordinate by ±10^{−6} and read off (C(y+ε) − C(y−ε))/2ε. Doing that, the analytic gradient is

  [[ 0.160689, −0.130198], [−0.128954, −0.022639], [ 0.130425, 0.120198], [−0.162160, 0.032639]]

and the finite-difference gradient agrees to every digit shown, with a maximum absolute discrepancy of 1.2×10^{−10} — round-off, not error. So the factor of 4, the (1 + d²)^{-1} weight, and the sign all check out; the formula is right. (And as a side benefit, if the factor of 4 had been wrong the discrepancy would have been order-1, not 10^{−10}, so this test is sharp enough to catch exactly the bookkeeping mistakes I was worried about.) The construction holds up end to end: symmetric joint for simplicity and outlier anchoring, Student-t for room, and a gradient that I've now verified rather than merely believed.

Let me read the negative gradient as forces, because the physics tells me whether it'll actually behave. It is a sum of pairwise pushes along the line between y_i and y_j, with magnitude scaled by |p_ij − q_ij|(1 + ||y_i − y_j||²)^{-1}. If p_ij > q_ij — the data says they're more similar than the map shows — the descent direction is −(y_i − y_j), so it pulls them together; if p_ij < q_ij — the map has them too close for how dissimilar they really are — the descent direction is +(y_i − y_j), so it pushes them apart. The extra (1 + ||y_i − y_j||²)^{-1} factor softens the force for far-apart map points. Crucially, when two dissimilar points are placed too close, the repulsion is strong, which is what carves the gaps; but because the whole thing is mediated by the Student-t and not a uniform floor, that repulsion is finite and the long-range attractions survive, so a cluster that got split early gets pulled back. This is why I can drop simulated annealing entirely: the long-range forces themselves do the global-organization job that annealing was hacking around.

Now let me settle the optimization loop and the few tricks that make it work in practice. The update is plain gradient descent with momentum,

  Y^{(t)} = Y^{(t−1)} − η · ∇_Y C + α(t) (Y^{(t−1)} − Y^{(t−2)}),

initialized either by sampling the map points from a tiny isotropic Gaussian near the origin (the basic algorithm uses N(0, 10^{-4} I), while the sklearn-style code below uses coordinate standard deviation 10^{-4}) or by taking a PCA projection and rescaling its first component to standard deviation 10^{-4}. The momentum α(t) should be small while the map is still a disorganized blob — I don't want to build up speed in a bad direction — and larger once the clusters have formed and I want to coast: so α = 0.5 early, α = 0.8 later. A reasonable learning rate is on the order of 100 to 200; for an automatic default I can scale it as max(n / early_exaggeration / 4, 50), which keeps the exaggerated first stage from taking steps that are too large on small data sets or too small on large ones. I can speed convergence with a per-coordinate adaptive learning rate (the Jacobs scheme): keep a gain for each coordinate, nudge it up where the descent direction is stable and shrink it where the direction flips, clipped at a small floor so it never dies. That's free smoothing of the descent and it replaces the hand-tuned schedules SNE needed.

Two more tricks earn their keep. The first I'll call early compression: at the very start, add a small L2 penalty proportional to Σ_i ||y_i||² that keeps all the map points near the origin. When the points are bunched, clusters can slide through one another cheaply, so it's easy to explore global arrangements; I remove the penalty once the map has had a chance to organize. The second, and the more important, is early exaggeration: for the first several dozen iterations, multiply *all* the p_ij by a constant — 4 is enough for the basic algorithm, and a larger implementation default such as 12 makes the same idea stronger — before computing the cost. Why does this help? With the p_ij inflated, the q_ij (which still sum to 1) are far too small to match their corresponding p_ij, so the only way to reduce the KL is to make the q_ij for genuinely-similar pairs as large as possible — which means pulling the members of each true cluster very tightly together. The clusters condense into tight, widely separated knots with lots of empty space between them, and *that* empty space is what lets the clusters move around relative to one another and find a good global layout. After the exaggeration phase I divide the p_ij back down and let the map relax to its real objective. Both tricks are robust to their exact settings; they're about giving the clusters room to rearrange early.

Let me also note how this extends if the data is too big for O(n²). The cost and memory are quadratic in n because of the all-pairs normalization Z and the all-pairs gradient, which caps the plain method around ten thousand points. I can keep the same map objective but compute the high-dimensional similarities from a random walk on a k-NN graph: build the neighborhood graph once, designate a subset of landmark points, run random walks from each landmark where the probability of stepping along an edge from x_i to x_j is ∝ exp(−||x_i − x_j||²), and let p_{j|i} be the fraction of walks started at landmark i that first arrive at landmark j. This integrates over *all* paths through the graph rather than a single shortest path, so it's far less sensitive to the short-circuits that wreck geodesic methods — one noisy bridge doesn't dominate. The map side and the optimization are unchanged. I'll keep this in my back pocket; the core method stands on the joint-KL plus Student-t.

So let me write the algorithm I'd actually ship, filling the empty slots in the embedding harness — the map-space similarity model, the cost, and the gradient — grounded in how a clean implementation does it, using the (p_ij − q_ij)·dist product to compute the gradient as a couple of matrix operations:

```python
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

MACHINE_EPSILON = np.finfo(np.double).eps


def compute_high_dim_affinities(X, perplexity):
    # per-point Gaussian neighbor distribution; sigma_i set by perplexity (effective #neighbors)
    D = squareform(pdist(X, "sqeuclidean"))
    n = X.shape[0]
    P = np.zeros((n, n))
    target = np.log(perplexity)                       # match entropy in nats
    for i in range(n):
        beta_lo, beta_hi, beta = -np.inf, np.inf, 1.0  # beta = 1 / (2 sigma_i^2)
        idx = np.arange(n) != i
        Di = D[i, idx]
        for _ in range(50):                            # binary search on sigma_i
            Pi = np.exp(-Di * beta)
            sumPi = max(Pi.sum(), MACHINE_EPSILON)
            H = np.log(sumPi) + beta * np.dot(Di, Pi) / sumPi
            if H > target:                             # entropy too high -> shrink sigma (raise beta)
                beta_lo = beta
                beta = beta * 2 if beta_hi == np.inf else (beta + beta_hi) / 2
            else:
                beta_hi = beta
                beta = beta / 2 if beta_lo == -np.inf else (beta + beta_lo) / 2
        Pi = np.exp(-Di * beta)
        P[i, idx] = Pi / max(Pi.sum(), MACHINE_EPSILON)  # conditional p_{j|i}
    # symmetrize into a joint and normalize: p_ij = (p_{j|i} + p_{i|j}) / (2n)
    P = P + P.T
    P = np.maximum(P / max(P.sum(), MACHINE_EPSILON), MACHINE_EPSILON)
    np.fill_diagonal(P, 0.0)
    return P


def cost_and_gradient(P, Y, degrees_of_freedom=1):
    n, d = Y.shape
    # Student-t similarities in the map; in 2-D, dof=1 gives (1 + ||y_i-y_j||^2)^-1.
    dist = pdist(Y, "sqeuclidean")
    dist /= degrees_of_freedom
    dist += 1.0
    dist **= (degrees_of_freedom + 1.0) / -2.0
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)  # normalize over all pairs -> q_ij

    Pf = squareform(P)                                  # condensed upper-triangle of P
    C = 2.0 * np.dot(Pf, np.log(np.maximum(Pf, MACHINE_EPSILON) / Q))  # KL(P||Q)

    # In 2-D this is dC/dy_i = 4 sum_j (p_ij-q_ij)(1+||y_i-y_j||^2)^-1 (y_i-y_j).
    PQd = squareform((Pf - Q) * dist)                   # sklearn's (P - Q) * dist trick
    grad = np.empty((n, d))
    for i in range(n):
        grad[i] = np.dot(PQd[i], Y[i] - Y)              # sum_j weight_ij * (y_i - y_j)
    grad = (2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom) * grad.ravel()
    return C, grad


def embed(X, perplexity=30.0, n_iter=1000, learning_rate="auto", n_components=2,
          early_exaggeration=12.0, init="pca", random_state=None):
    rng = np.random.RandomState(random_state)
    n = X.shape[0]
    eta = max(n / early_exaggeration / 4.0, 50.0) if learning_rate == "auto" else learning_rate
    P = compute_high_dim_affinities(X, perplexity)

    if init == "pca":
        Y = PCA(n_components=n_components, random_state=rng).fit_transform(X).astype(np.float32)
        Y = Y / np.std(Y[:, 0]) * 1e-4                   # sklearn rescales PC1 to std 1e-4
    else:
        Y = 1e-4 * rng.standard_normal((n, n_components)).astype(np.float32)
    Y = Y.ravel()
    degrees_of_freedom = max(n_components - 1, 1)        # equals 1 for the usual 2-D map

    def run_descent(Y, P_current, n_steps, momentum):
        update = np.zeros_like(Y)
        gains = np.ones_like(Y)                          # per-coordinate adaptive gains (Jacobs)
        for _ in range(n_steps):
            _, grad = cost_and_gradient(P_current, Y.reshape(n, n_components), degrees_of_freedom)
            inc = (update * grad) < 0.0                   # same descent direction as before -> larger gain
            gains[inc] += 0.2
            gains[~inc] *= 0.8
            np.clip(gains, 0.01, np.inf, out=gains)
            update = momentum * update - eta * gains * grad
            Y += update
        return Y

    explore = min(250, n_iter)
    P *= early_exaggeration                              # stage 1: exaggerated P, momentum 0.5
    Y = run_descent(Y, P, explore, momentum=0.5)
    P /= early_exaggeration
    if n_iter > explore:
        Y = run_descent(Y, P, n_iter - explore, momentum=0.8)
    return Y.reshape(n, n_components)
```

Let me trace the causal chain that got me here. I needed a single 2-D map that keeps similar points together, keeps dissimilar points apart, and leaves gaps so the clusters are visible — and I'm not allowed to use labels. Distance-matching methods (PCA, classical scaling) chase the large distances and lose local structure; Sammon's reweighting makes the cost hostage to the most fragile near-coincident pairs; spectral methods game their anti-collapse constraint or can't show separated submanifolds. The probabilistic-neighbor frame — Gaussian neighbor distributions with per-point bandwidth set by perplexity, matched under KL — gets the local structure right because KL's asymmetry punishes putting near points far apart, but its conditional formulation needs simulated annealing to optimize and, fatally, a Gaussian map kernel crowds: the huge number of moderately-distant points on a high-dimensional manifold can't be seated at moderate radius in the plane, so their summed attraction crushes the map inward and the clusters never separate. Symmetrizing the conditionals into a single joint p_ij = (p_{j|i} + p_{i|j})/(2n) gave a single-KL objective with a clean (p_ij − q_ij) gradient and kept outliers anchored (every Σ_j p_ij ≥ 1/2n). Because I was now matching joint probabilities rather than distances, the map kernel was free to differ from the high-dimensional one, so I replaced the Gaussian with a heavy-tailed Student-t of one degree of freedom: its slow (1 + d²)^{-1} tail lets a moderate similarity be represented by a much larger map distance, which removes the unwanted attraction and opens the gaps; its inverse-square far-field makes the map approximately scale-invariant and lets distant clusters act like single points; and being distance-dependent (not a uniform floor) it keeps real restoring forces so split clusters reunite — which is exactly why no annealing is needed. Differentiating the joint KL with d_ij = ||y_i − y_j|| and Z = Σ(1 + d_ij²)^{-1} gives ∇_{y_i} C = 4 Σ_j (p_ij − q_ij)(y_i − y_j)(1 + ||y_i − y_j||²)^{-1}, and the descent direction turns it into finite pairwise forces. The optimization is plain momentum gradient descent (α 0.5 then 0.8, learning_rate = max(n / early_exaggeration / 4, 50) by default, ~1000 iterations) from a tiny random or rescaled-PCA start, sped up by per-coordinate adaptive gains and helped early by exaggerating the p_ij so the true clusters condense with room to rearrange. And when n is too large for the O(n²) all-pairs computation, the same map objective runs on random-walk affinities from a k-NN graph, which integrate over all paths and so resist short-circuits.
