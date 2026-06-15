Let me start from what actually goes wrong when I try to visualize a high-dimensional dataset in 2D, because the failures are specific and I want to design against them rather than bolt together another method that happens to work. I have N points in R^P and I want to drop them onto a flat canvas so a person can read the structure. There are two things to read: the local structure — for each point, who its neighbors are and roughly in what order — and the global structure — how the neighborhoods sit relative to each other, whether this cluster is between those two, whether the data is one long curve or a hierarchy. The brutal fact underneath everything is capacity: 2D simply cannot hold all the pairwise relationships of a P-dimensional cloud, so I am not going to preserve everything, and the whole game is choosing *what* to keep and being honest that the rest is sacrificed.

What do the methods I have today do, and exactly where do they break? PCA and MDS are linear, so they give me a faithful coarse layout but flatten any curved manifold — local neighborhoods on a curved sheet get mangled. The local crowd — t-SNE, UMAP, and the manifold-learning ancestors — preserve neighborhoods beautifully and make crisp clusters, but I keep seeing their global layout be wrong: clusters scattered with no meaningful between-ness, sometimes false clusters that aren't really there. TriMap does better globally. But before I trust that, I want to know *why* each of these does what it does, because the literature is a pile of "we use this loss, this graph, this initialization, and empirically it's good," and I can't improve on a pile of empirical choices. I want principles.

Every one of these methods can be put into the same graph-based frame. From the high-dimensional data I construct a weighted graph — nodes are points, and the things I put in the loss are *graph components*: an edge (i,j), or a triplet (i,j,k). Then I minimize, over the low-dimensional positions Y, an objective of the form sum over chosen components of Weight^X(component) · Loss^Y(component), where the weight is fixed from the high-dimensional data and the loss is a function of the embedded positions. When I run gradient descent on this, the loss turns into *forces*: it pulls some pairs together (attraction) and pushes others apart (repulsion). So t-SNE, UMAP, TriMap differ in (a) which components they pick and (b) what loss/force they put on each. Good — that's two knobs I can reason about separately.

The losses themselves look nothing alike — KL of two probability distributions, a fuzzy cross-entropy, a triplet ratio — so comparing functional forms is hopeless. Let me compare the thing that actually drives the optimization: the forces. Consider a triplet of points (i,j,k) where i should attract j (they're neighbors) and repulse k (k is further). The optimizer is going to move j toward i and k away from i. So plot, over the two low-dimensional distances d_ij (horizontal) and d_ik (vertical), the loss value and the gradient direction/magnitude. Every triplet is a point on this plane; the arrows show which way the forces push. On this plane, t-SNE, UMAP, and TriMap keep producing a similar pattern, and plausible alternatives make the failure modes visible: log(1+e^{(d_ij^2-d_ik^2)/10}) can ignore useful near/far tradeoffs, (d_ij^2+1)/(d_ik^2+1) pulls hardest on neighbors that may already be hopeless, its negated inverse gives too little push to close intruders, and log(1+e^{d_ij^2}-e^{d_ik^2}) has the same over-pull problem on far neighbors. So the shared shape of the good ones is carrying real information. Let me write down what that shape *is*, as principles, in terms of the loss gradients.

Monotonicity first, the obvious one: as a neighbor j drifts away, loss should rise (∂Loss/∂d_ij ≥ 0); as a far point k drifts further, loss should fall (∂Loss/∂d_ik ≤ 0). Attraction on neighbors, repulsion on far points. That's principle one and it's necessary, but the bad losses all obey it too, so monotonicity is nowhere near sufficient — the *tradeoffs* are what matter, and the tradeoffs are about the relative magnitudes and where the gradient points across the plane.

Now the asymmetry. Neighbors and far points deserve different treatment, and the limited-capacity fact is the reason. Take the region where d_ik is not small — the far point is already comfortably far. Should I keep spending force pushing it even further? No — there's nothing to gain, and if I do, I'm wasting capacity and inviting crowding elsewhere. So in that region the gradient should point mainly *left*, toward shrinking d_ij; formally, for any fixed d_ij and any ε there's a threshold beyond which |(∂Loss/∂d_ik)/(∂Loss/∂d_ij)| < ε. That's principle two: once a far point is far enough, stop caring about it and focus on pulling neighbors in. The dual: in the *bottom* region where d_ik is small — a far point has crept too close — the gradient should point mainly *up*, push it away hard; for any fixed small d_ik, as d_ij grows, |(∂Loss/∂d_ij)/(∂Loss/∂d_ik)| → 0. Principle three.

Two more about magnitudes. Along the vertical axis, d_ij → 0: the neighbor is already right on top of i. The gradient magnitude here should be *small* — there's no point burning force to make an already-close neighbor closer; turn attention elsewhere. Principle four. Along the horizontal axis, d_ik small: a far point far too close. The gradient magnitude should be *large* — shove it out. Principle five. And the deepest one, principle six: the gradient as a function of d_ij should be largest for *moderately small* d_ij and fall off for large d_ij. Why fall off for large d_ij? Because of capacity again — if a neighbor has ended up very far away in the embedding, it usually means I genuinely cannot preserve it (too many conflicting constraints from i's other neighbors), and if I keep yanking it with a huge force I'll distort everyone else trying to save one neighbor I can't save anyway. Better to suffer a bounded penalty and let it go. So the attractive force should be unimodal in d_ij: gentle when very close (principle four), strong at moderate distance, gentle again when hopelessly far. That unimodal-with-vanishing-tails shape is the signature, and it's exactly what the bad losses get wrong — one of them has force *increasing* with d_ij (it fights hardest to save the neighbors it can't save, wrecking the ones it could), another has tiny gradient along the bottom (it never separates points that are too close, everything crowds).

This is a lot of principles to check one triplet at a time. Let me see if there's a structural shortcut — a simple condition on a loss that guarantees all six. Suppose I restrict to losses that are *separable*: the total loss is a sum of an attractive part that depends only on d_ij plus a repulsive part that depends only on d_ik,

  Loss = Σ_ij Loss_attr(d_ij) + Σ_ik Loss_rep(d_ik).

Define f(d_ij) = ∂Loss_attr/∂d_ij and g(d_ik) = -∂Loss_rep/∂d_ik (the negative so g is the *strength* of repulsion). Then for a single triplet (i,j,k) the contribution is Loss_attr(d_ij) + Loss_rep(d_ik), whose gradient components are f(d_ij) and -g(d_ik) — completely decoupled. Now ask what conditions on f and g give me the six principles. Claim: it's enough that f and g are both nonnegative, unimodal, and vanish at both ends, lim_{d→0} f = lim_{d→∞} f = 0 and the same for g. Let me actually verify each principle drops out, because if it does, I've reduced six geometric conditions to "two unimodal hump functions that die at 0 and ∞," which is something I can just *build*.

Principle one (monotonicity): f ≥ 0 means ∂Loss_attr/∂d_ij ≥ 0, loss increases as the neighbor recedes; g ≥ 0 means ∂Loss_rep/∂d_ik ≤ 0, loss decreases as the far point recedes. Done. Principle two: fix a d_ij where f(d_ij) is not zero; since lim_{d_ik→∞} g(d_ik) = 0, the ratio g(d_ik)/f(d_ij) → 0 as d_ik grows — gradient turns left. At the endpoints where f itself vanishes, the attractive component is already in its zero-force tail, so those boundary cases do not create a competing repulsive pull. Principle three is the dual: for a fixed d_ik with g(d_ik) nonzero, lim_{d_ij→∞} f(d_ij) = 0, so f/g → 0 as d_ij grows — gradient turns up. Principle four needs both coordinates: as d_ij → 0 the attractive part f goes to zero, and once d_ik is sufficiently large the repulsive part g is also arbitrarily small, so the whole gradient magnitude near the upper-left vertical axis is small. Principle five is not "infinite shove at exactly d_ik = 0" — g also vanishes at the endpoint — but unimodality gives an interior hump near the bottom and then non-increasing force after the hump, so too-close far points get the strong repulsive zone before the force dies away for already-far points. Principle six: f unimodal in d_ij gives the moderate-distance peak, and f → 0 at ∞ gives the give-up tail. Done. Every principle follows from separability plus two nonnegative unimodal forces with vanishing endpoints. So I never needed triplets to *couple* d_ij and d_ik — a separable loss obeys all the principles. That's a relief, because triplets are awkward and expensive, and it suggests the triplet machinery may be doing less than it looks. (Indeed if I take UMAP's force functions, f = const·d_ij^{2b-1}/(1+a d_ij^{2b}) and g = const·d_ik/((ε+d_ik^2)(1+a d_ik^{2b})), I can check they're nonnegative, vanish at 0 and ∞, and have a single interior extremum for b > 0.5 — the sign of f' is governed by -(a d_ij^{2b} - 2b + 1), positive for small d_ij and decreasing, so one crossing; similarly for g. So UMAP satisfies my sufficient condition, which is reassuring: the condition is descriptive of what already works.)

But now I have to stop myself, because there's a hole big enough to drive the whole problem through. Everything I just did is about *local* structure — the rainbow plane only ever talks about neighbors and far points and how to balance attraction against repulsion among them. None of it says a word about *global* structure, the arrangement of clusters. Let me check whether obeying the principles is even enough for global structure, with a thought experiment. Take a pure triplet loss like TriMap's, and to make it stark use the 0-1 triplet loss: l_ijk = 1 if j is further from i than k is (the wrong order), 0 otherwise. Now suppose — and this is the key — that *every* triplet in my set has at least one point very close to i serving as j. So my triplet set omits all triplets where both j and k are far from i. Can I drive this loss to zero? Yes: I just need every chosen near-j closer than its paired far-k, which is easy. But "every near point closer than its paired far point" says *nothing* about how the far points are arranged relative to each other. I can satisfy it while scrambling the global layout completely — fold a smooth curve into a tangled mess, and as long as locally each near neighbor beats its far partner, the loss is zero. So zero loss, perfect by its own measure, global structure destroyed. The same holds for an attract-neighbors/repulse-far loss with a 0-1 form: l_attract = 0 if i,j close, l_repulse = 0 if i,k far — drive it to zero, global structure unconstrained.

What does this tell me? The reason these losses are blind to global structure is that they only ever put forces between *neighbors* (attract) and *too-close far points* (repulse). For a point k that is genuinely far in low dimensions, the force is zero — and crucially, it's zero whether k is moderately far or extremely far. The loss cannot tell those apart, so the relative distances among far points never enter the objective, and relative distances among far things *are* the global structure. The conclusion is forced: to preserve global structure I must exert forces on *non-neighbors* — pairs that are neither nearest neighbors nor too-close intruders — in a way that distinguishes moderately distant from very distant. Nothing in the local principles does this.

Let me confirm this is really what t-SNE and UMAP suffer from, by computing it rather than asserting it, because it's the linchpin. t-SNE's gradient on i is 4 Σ_j (p_ij - q_ij)(y_i - y_j)(1 + d_ij^2)^{-1}, where d_ij = ||y_i - y_j||. Split the j-th term into the part with p (attraction) and the part with q (repulsion): F_attract = 4 p_ij d_ij (1+d_ij^2)^{-1} and F_repulse = 4 q_ij d_ij (1+d_ij^2)^{-1}, opposite directions along the unit vector e_ij, both nonnegative since p, q, d ≥ 0. Attraction: p_ij comes from exp(-||x_i-x_j||^2/2σ^2) in the *original* space — for a non-neighbor x_j that distance is large, so p_ij is minuscule (and modern implementations literally set it to 0 beyond 3·Perplexity neighbors). So non-neighbors get essentially no attraction. Repulsion: substitute q_ij = a_ij/(a_ij + B_ij) where a_ij = 1/(1+d_ij^2) and B_ij = Σ_{kl≠ij}(1+d_kl^2)^{-1}. Then F_repulse = 4 · a_ij/(B_ij + a_ij) · d_ij · a_ij, equivalently 4d_ij / [(1+d_ij^2)(B_ij(1+d_ij^2)+1)]. The product d_ij · a_ij = d_ij/(1+d_ij^2) is only an upper envelope because the extra factor a_ij/(B_ij+a_ij) also shrinks; with B_ij fixed at the current embedding, the full repulsive term decays like Θ(1/d_ij^3). So as two points get far apart in the embedding, repulsion dies even faster than the loose 1/d_ij envelope — and I can push further: the derivative of the repulsion with respect to d_ij is

  ∂F_repulse/∂d_ij = -4 · [3B_ij d_ij^4 + (2B_ij+1)d_ij^2 - B_ij - 1] / [(d_ij^2+1)^2 (B_ij d_ij^2 + B_ij + 1)^2],

and for large d_ij the numerator is dominated by 3B_ij d_ij^4 while the denominator is dominated by B_ij^2 d_ij^8, so the derivative goes like -12/(B_ij d_ij^4) → 0 — the force flattens out. So for any two points that are far apart in the embedding, both the force and its slope are near zero, *independent of how far they actually are in the original space*. t-SNE genuinely cannot distinguish among far points; that's the global-structure blindness, derived, not assumed. UMAP has the same character — its attractive and repulsive forces decay to nothing past a narrow working zone, and that zone is small compared to the spread of embedded distances, so most far pairs feel nothing.

And TriMap — the one that's globally better — let me see whether it escapes this or just hides it. About 90% of its triplets contain a nearest neighbor as j (50 of 55 per point), and the other 10% are random. So almost all its attraction is on neighbors, same as everyone. Its repulsion, written per far point, is the triplet loss viewed as a function of d_ik with d_ij fixed: d̃_ij/(d̃_ij + d̃_ik) where d̃ = d^2 + 1 — again strong only when the far point is close, near-flat otherwise, same near-sightedness. So where does its global structure come from? Either the 10% random triplets (which often pair two far points and so *could* carry global information) or the PCA initialization. The ablation answer is clear: remove the random triplets, almost no change; remove the PCA init, the global structure collapses. So TriMap's global structure is inherited from PCA, not produced by its triplets. And its weights, after the tempered-log transform, come out nearly all equal, so the elaborate weighting isn't doing much either. TriMap doesn't solve the problem; it borrows the answer from its initializer. I don't want a method whose global structure is a hostage to initialization — that's exactly the fragility I'm trying to kill.

The missing component is now precise: a third kind of pair, neither nearest-neighbor nor random-far, that I attract — pairs at *moderate* distance — so that the objective itself contains information about the mid-range arrangement and can build a global layout that doesn't depend on a lucky start. Call them mid-near pairs. The neighbors handle local structure, a repulsion handles crowding, and the mid-near pairs handle the global skeleton. Three pair types, all as plain edges — no triplets, because I proved separable edge losses already obey every local principle.

Now the design questions, one at a time. How do I pick mid-near pairs cheaply? I want, per point i, a pair at moderate distance — roughly a middling quantile of i's distance distribution, not the nearest (too local) and not a random far one (that's my repulsion set). Computing a full ranked distance list for every point is O(N log N) per point, too expensive. But I don't need the exact quantile; I need an approximate draw from the moderate region. Sample a handful of points uniformly — say six — and take the *second closest* of them. With six uniform draws, the closest tends to be a genuine near point, and the second-closest sits in the lower-middle of the distance distribution: moderate, not nearest. Six is small enough to be nearly free, and "second of six" is a cheap order statistic that lands where I want without any ranking. That's the mid-near construction.

How do I pick neighbors? Nearest neighbors by raw Euclidean distance, except neighborhoods in different parts of the space have different scales — a dense region and a sparse region shouldn't be compared on the same absolute ruler. So scale the distance per point: d²_select(i,j) = ||x_i - x_j||^2 / (σ_i σ_j), where σ_i is the average distance from i to its 4th-through-6th Euclidean neighbors (a local scale estimate that skips the very nearest, which can be degenerate, and stays in the genuinely-local band). I'll use this scaled distance only to *select* the n_NB neighbors, not in the loss — the loss works on low-dimensional positions, and the high-dimensional scaling is just to make "neighbor" mean the same thing everywhere. To reuse a fast k-NN routine, grab the min(n_NB + 50, N) Euclidean neighbors first, then re-rank that shortlist by the scaled distance and keep the top n_NB. Far pairs are the easiest: just sample non-neighbor points uniformly; that gives me the repulsion set.

Now the loss. It must be separable (so the principles hold) and built from two unimodal-vanishing force functions, and I want it dead simple. The transformed distance from t-SNE is the right building block: d̃ = ||y_a - y_b||^2 + 1 — the +1 keeps it bounded below, and its slow growth near zero is what gives me the small gradient for tiny d_ij that principle four wants. For neighbors I want attraction that's unimodal in d_ij, peaking at moderate distance and saturating for large d_ij. A saturating rational does exactly this:

  Loss_NB = d̃_ij / (10 + d̃_ij).

As d̃ grows this approaches 1 — it saturates, so the loss stops rising and the force dies for far-flung neighbors (give-up behavior, principle six's tail). Near zero d̃ ≈ 1, the function is small and slowly varying (principle four). The constant 10 sets where the transition happens — the width of the attractive working zone. For the repulsion I want strong force only when the far point is too close and almost nothing otherwise:

  Loss_FP = 1 / (1 + d̃_il),

which is large when d̃ is small (intruder close, big penalty) and decays to zero as the far point recedes — principles three and five. For the mid-near pairs I want the *same kind* of attractive shape but acting over a much *wider* range, because the mid-near pairs are supposed to organize structure across moderate distances, not just the local band. Same saturating form, far larger constant:

  Loss_MN = d̃_ik / (10000 + d̃_ik).

With the denominator at 10000, the function stays in its slowly-rising, essentially-linear regime out to much larger d̃ before saturating — so a mid-near pair feels a gentle, persistent pull across a broad distance range, exactly the "organize the global skeleton" behavior, instead of the narrow neighbor working zone. The constants 10, 10000, 1 are working-zone tuners, not sacred — many functions with the same unimodal-vanishing characteristics would do; what's load-bearing is that the loss is separable, the forces obey the principles, and the mid-near attraction is *separate* from the neighbor attraction and ranges wider.

Let me get the forces right by differentiating, because the signs and factors are what actually move points and I've burned myself on these before. Each loss is a function of d̃ = ||y_i - y_j||^2 + 1, and ∂d̃/∂y_i = 2(y_i - y_j). For the neighbor term, L = d̃/(10+d̃). dL/dd̃ = [(10+d̃) - d̃]/(10+d̃)^2 = 10/(10+d̃)^2. Chain rule: ∂L/∂y_i = 10/(10+d̃)^2 · 2(y_i - y_j) = [20/(10+d̃)^2](y_i - y_j). It's positive along (y_i - y_j), so the gradient on y_i points *away* from j; gradient descent steps against it, moving y_i *toward* j — attraction, good. So the attractive update coefficient is w_NB · 20/(10+d̃)^2, applied as +coeff·(y_i-y_j) to grad_i and -coeff·(y_i-y_j) to grad_j. For the mid-near term, identical algebra with 10000: dL/dd̃ = 10000/(10000+d̃)^2, force coefficient w_MN · 20000/(10000+d̃)^2 — also attractive. For the far term, L = 1/(1+d̃) = (1+d̃)^{-1}, dL/dd̃ = -(1+d̃)^{-2} = -1/(1+d̃)^2. Chain rule: ∂L/∂y_i = -1/(1+d̃)^2 · 2(y_i-y_j) = -[2/(1+d̃)^2](y_i - y_j). The minus means the gradient on y_i points *toward* j, so descent pushes y_i *away* — repulsion, good. So in the assembled gradient I *subtract* coeff·(y_i-y_j) from grad_i and add it to grad_j for the far pairs, opposite of the attractive terms, with coefficient w_FP · 2/(1+d̃)^2. Let me double-check the magnitude story matches the principles: the scalar coefficient for a neighbor is largest near d̃ = 1, but the vector force also multiplies by ||y_i-y_j||, so the actual attractive force is zero at distance 0, rises to an interior moderate-distance peak, and then decays like 1/d^3 for large distance — small for already-close neighbors and small again for hopelessly far ones. The far-pair coefficient is also largest near d̃ = 1, but its vector magnitude is zero at exact coincidence, peaks in a close-distance band, and then dies like 1/d^3, so a too-close intruder is pushed in the repulsive working zone while an already-far point is ignored. Exactly the shapes I wanted. Good.

Now the optimization, and here's where I have to think about local optima, because a sum of attractive and repulsive forces over chosen pairs is a nasty nonconvex landscape and the working zones are narrow — once two points leave each other's working zone, there's almost no force to bring them back. That last fact is double-edged. The danger: if early in optimization I fling a pair of true neighbors far apart, the attractive force between them has already decayed (the give-up tail), so they can never come back together — I get a false split, a phantom cluster. The opportunity: that same narrowness means once I've placed the global structure, the long-range forces are gentle and won't tear it up while I refine locally. So I should build the global structure *first*, with strong long-range pull, then freeze it in by relaxing those forces and refining local structure. That's a coarse-to-fine schedule, and it's the same spirit as t-SNE's early exaggeration (amplify early to organize, then relax) and simulated annealing (big moves first), except I'm exaggerating the *mid-near* pull, not the neighbor pull — because mid-near is what carries the global skeleton.

So a three-phase weight schedule on (w_NB, w_MN, w_FP). Phase one: pull the global structure into place. Crank w_MN very high — start it at 1000 — so the mid-near pairs dominate and organize the moderate-distance arrangement, with modest neighbor weight w_NB = 2 and constant repulsion w_FP = 1. But I don't want to leave w_MN at 1000, because once the skeleton is roughly right, an overwhelming mid-near pull would fight the local refinement and over-contract everything. So during the first 100 zero-indexed updates I anneal it by w_MN(itr) = (1 - itr/100)·1000 + (itr/100)·3: it starts at 1000, moves toward 3, and then phase two pins it exactly at 3. Phase two: stabilize. Hold w_MN at a small but nonzero 3, raise w_NB to 3 to start tightening neighborhoods while still gently maintaining the global structure, keep w_FP = 1. This interim keeps the global skeleton from drifting while local structure begins to form. Phase three: refine local structure and sharpen clusters. Set w_MN = 0 — the global structure is the structure I want to preserve now, and the narrow working zones help later local forces stay local — drop w_NB to 1, keep w_FP = 1, so attraction and repulsion on the local scale dominate, pulling neighbors tight and letting repulsion carve clean boundaries between clusters. The key failure modes this avoids are exactly the two I worried about: never neglecting forces on non-near points (phase one's mid-near pull is the whole point), and never flinging neighbors so far early that their attraction saturates and they can't return (phase one keeps w_NB modest and the structure coarse, so neighbors aren't yet being forced to specific spots). I'll spend 100 iterations in phase one, 100 in phase two, 250 in phase three — 450 total.

Initialization: I'll use PCA, scaled down small. But I want to be clear about *why* this is fine here when it was a crutch for TriMap. For TriMap, PCA was load-bearing — remove it and global structure dies — because TriMap has no mechanism to *create* global structure, only to inherit it. Here the mid-near pairs are supposed to create it during phase one, so PCA is just a head start that saves iterations, not the source of the answer; if a random initialization with coordinates 1e-4 times a standard normal cannot organize, then the graph itself is not carrying enough global information. And I scale the PCA init down (multiply by 0.01) because of that scale-sensitivity I noticed in all these methods: forces have a working zone in absolute distance, so if I start the points spread too wide, every force is in its dead zone and the optimizer freezes at step one; start them tightly clustered and the forces are live and can do their work. Small init keeps everything inside the working zones at the start.

For the optimizer itself I'll use Adam — per-parameter step sizes adapted from running estimates of the gradient and its square. I want it here specifically because my three force terms have wildly different magnitudes (w_MN starts at 1000), and those magnitudes change across phases; a single global step size would be wrong for one term or another, while Adam's per-coordinate adaptation keeps the effective step sane as the force scales shift. Standard betas (0.9, 0.999), eps = 1e-7; the bias-corrected step is lr·sqrt(1-beta2^t)/(1-beta1^t) on m/(sqrt(v)+eps). I can afford a fairly large base learning rate here (1.0) because the gradients are these bounded rational forces, not raw losses, and Adam normalizes per coordinate anyway.

Let me also fix the defaults from the structure. Number of neighbors n_NB = 10 — enough to define a neighborhood, small enough to stay local; for very large N I'd grow it slowly. Mid-near count as a ratio of neighbors, MN_ratio = 0.5, so n_MN = round(0.5·n_NB) = 5 — I need some global pairs but they're expensive-ish and a handful per point suffices to sketch the skeleton. Further count FP_ratio = 2.0, so n_FP = round(2·n_NB) = 20 — repulsion needs more samples than attraction to keep crowding at bay everywhere, and far pairs are the cheapest to sample. And if P is large (say > 100) I'll PCA-preprocess down to 100 dimensions first, both for speed and because the k-NN search is cleaner in a denoised space.

So let me assemble the whole thing as code I'd actually run. Neighbor selection by scaled distance, mid-near as second-of-six, far as random non-neighbors; the three rational force terms with their signs; the three-phase weight schedule; PCA init scaled small; Adam in the loop.

```python
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors


def _draw_excluding(n, size, rng, banned):
    banned = set(int(x) for x in banned)
    out = []
    while len(out) < size:
        k = int(rng.integers(n))
        if k not in banned:
            out.append(k)
            banned.add(k)
    return np.asarray(out, dtype=np.int64)


def select_components(X, n_neighbors, n_MN, n_FP, rng):
    n = X.shape[0]
    # --- neighbor pairs: nearest by SCALED distance d^2/(sig_i sig_j) ---
    n_extra = min(n_neighbors + 50, n - 1)            # over-fetch by Euclidean, then re-rank
    nn = NearestNeighbors(n_neighbors=n_extra + 1).fit(X)
    dist, nbrs = nn.kneighbors(X)
    dist, nbrs = dist[:, 1:], nbrs[:, 1:]             # drop self
    sig = np.maximum(dist[:, 3:6].mean(axis=1), 1e-10)        # local scale = avg dist to 4th-6th NN
    scaled = (dist ** 2) / (sig[:, None] * sig[nbrs])         # normalize neighborhoods to one ruler
    order = np.argsort(scaled, axis=1)[:, :n_neighbors]
    rows = np.repeat(np.arange(n), n_neighbors)
    nb = np.stack([rows, np.take_along_axis(nbrs, order, axis=1).ravel()], 1)

    # --- mid-near pairs: 2nd-closest of 6 uniform samples (a cheap moderate-distance draw) ---
    mn = np.empty((n * n_MN, 2), dtype=np.int64); t = 0
    for i in range(n):
        picked_for_i = []
        for _ in range(n_MN):
            s = _draw_excluding(n, 6, rng, [i] + picked_for_i)
            d = ((X[s] - X[i]) ** 2).sum(1)
            picked = int(s[np.argsort(d)[1]])            # second smallest = moderate
            picked_for_i.append(picked)
            mn[t] = (i, picked); t += 1
    mn = mn[:t]

    # --- further pairs: random non-neighbors (the repulsion set) ---
    nbr_of = nb[:, 1].reshape(n, n_neighbors)
    fp = np.empty((n * n_FP, 2), dtype=np.int64); t = 0
    for i in range(n):
        banned = set(nbr_of[i].tolist()) | {i}
        for _ in range(n_FP):
            k = int(_draw_excluding(n, 1, rng, banned)[0])
            banned.add(k)
            fp[t] = (i, k); t += 1
    return nb, mn, fp[:t]


def loss_and_forces(Y, nb, mn, fp, w_nb, w_mn, w_fp):
    grad = np.zeros_like(Y)
    # neighbor + mid-near: attraction, force coeff w * 2c/(c + d~)^2 along (y_i - y_j)
    for pairs, c, w in ((nb, 10.0, w_nb), (mn, 10000.0, w_mn)):
        i, j = pairs[:, 0], pairs[:, 1]
        diff = Y[i] - Y[j]
        dt = (diff ** 2).sum(1) + 1.0                  # d~ = ||y_i - y_j||^2 + 1
        coeff = w * (2.0 * c) / (c + dt) ** 2
        upd = coeff[:, None] * diff
        np.add.at(grad, i, upd); np.add.at(grad, j, -upd)   # descent pulls i toward j
    # further: repulsion, force coeff w * 2/(1 + d~)^2 -- OPPOSITE sign
    i, j = fp[:, 0], fp[:, 1]
    diff = Y[i] - Y[j]
    dt = (diff ** 2).sum(1) + 1.0
    coeff = w_fp * 2.0 / (1.0 + dt) ** 2
    upd = coeff[:, None] * diff
    np.add.at(grad, i, -upd); np.add.at(grad, j, upd)        # descent pushes i from j
    return grad


def weight_schedule(itr, p1, p2):
    if itr < p1:                                       # phase 1: build global structure
        return 2.0, (1 - itr / p1) * 1000.0 + (itr / p1) * 3.0, 1.0   # toward 3; phase 2 pins it
    if itr < p1 + p2:                                  # phase 2: stabilize
        return 3.0, 3.0, 1.0
    return 1.0, 0.0, 1.0                               # phase 3: refine local, w_MN = 0


def fit_transform(X, n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0,
                  num_iters=(100, 100, 250), lr=1.0, random_state=None):
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=np.float64)
    if X.shape[1] > 100:                               # denoise / speed up kNN
        X = X - X.mean(0)
        X = TruncatedSVD(n_components=100, random_state=random_state).fit_transform(X)
        pca_solution = True
        pca_init = None
    else:
        X = X - X.min()
        X = X / max(X.max(), 1e-12)
        X = X - X.mean(0)
        pca_solution = False
        pca_init = PCA(n_components=n_components, random_state=random_state).fit(X)
    n_MN = int(round(n_neighbors * MN_ratio))          # 5 with defaults
    n_FP = int(round(n_neighbors * FP_ratio))          # 20 with defaults
    nb, mn, fp = select_components(X.astype(np.float32), n_neighbors, n_MN, n_FP, rng)

    Y = (0.01 * X[:, :n_components] if pca_solution     # PCA init, scaled small (stay in working zone)
         else 0.01 * pca_init.transform(X))
    Y = np.ascontiguousarray(Y, dtype=np.float64)

    beta1, beta2, eps = 0.9, 0.999, 1e-7               # Adam: per-coord adaptive steps
    m = np.zeros_like(Y); v = np.zeros_like(Y)
    p1, p2, _ = num_iters
    for itr in range(sum(num_iters)):
        w_nb, w_mn, w_fp = weight_schedule(itr, p1, p2)
        g = loss_and_forces(Y, nb, mn, fp, w_nb, w_mn, w_fp)
        lr_t = lr * np.sqrt(1 - beta2 ** (itr + 1)) / (1 - beta1 ** (itr + 1))
        m += (1 - beta1) * (g - m)                     # m_t = beta1*m + (1-beta1)*g
        v += (1 - beta2) * (g ** 2 - v)                # v_t = beta2*v + (1-beta2)*g^2
        Y -= lr_t * m / (np.sqrt(v) + eps)             # bias-corrected step
    return Y
```

Let me trace the whole causal chain back to make sure it holds together. I started from the impossibility of preserving everything in 2D, so the question was *what* to keep. Comparing methods by their forces on the (d_ij, d_ik) plane gave me six principles that good local losses share — monotone, asymmetric attention to near vs far, unimodal-vanishing attractive and repulsive forces — and I proved that a *separable* edge loss with two nonnegative unimodal forces vanishing at 0 and ∞ satisfies all six, so triplets were never needed for local structure. But a thought experiment showed the principles only buy local structure: a 0-1 loss confined to near/too-close pairs can hit zero loss while scrambling the global layout, because no force ever acts on moderately-far pairs. Computing t-SNE's force decomposition confirmed it — attraction vanishes for non-neighbors (small p_ij) and repulsion has only a loose 1/d_ij envelope and in the normalized fixed-B expression decays like 1/d_ij^3, with a flat derivative, so far points are indistinguishable — and TriMap's better global structure turned out to be inherited from PCA, not produced by its triplets. That forced a third pair type: mid-near pairs at moderate distance, attracted, to put the global skeleton into the objective itself. I picked them cheaply as the second-closest of six random samples, picked neighbors by a scaled distance to normalize neighborhoods, and picked far pairs at random. The loss is three separable rational terms in d̃ = ||y_i-y_j||^2+1 — d̃/(10+d̃) for neighbors and d̃/(10000+d̃) for mid-near (same saturating shape, the larger constant giving a wider working zone for organizing global structure), 1/(1+d̃) for far points — whose gradients I worked out as 20/(10+d̃)^2, 20000/(10000+d̃)^2 (attractive) and 2/(1+d̃)^2 (repulsive), each with the working-zone shape the principles demand. The narrow working zones meant global structure, once placed, can stay put while local structure is refined — so a coarse-to-fine three-phase schedule: anneal w_MN from 1000 toward 3 during the first 100 updates to build global structure, hold it at 3 while tightening neighbors, then zero it and refine local structure with neighbors and repulsion, which also avoids flinging neighbors so far early that their attraction saturates. PCA init scaled small keeps everything inside the working zones at the start without asking the initializer to supply the global structure; Adam handles the swinging force magnitudes across phases. Three pair types, a separable principled loss, a three-phase schedule — local and global structure, robustly, from the data alone.
