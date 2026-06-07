OK, let me think about this from scratch. I have a weighted undirected graph, weights w_ij ≥ 0, and I want to split the vertices into two sides so that the total weight of edges crossing between the sides is as large as possible. Exact MAX CUT is NP-complete — it's on Karp's original list, and it stays hard even when all weights are 1 — so on the worst graph I am not going to find the optimum in polynomial time. What I can actually hope for is a *guarantee*: a polynomial-time algorithm that, on every graph, returns a cut of weight at least ρ·OPT, with ρ as large as I can prove.

So what's the state of play. The thing everyone reaches for first is the dumbest possible randomized algorithm: throw each vertex onto one side or the other by an independent fair coin. An edge (i, j) is cut exactly when its endpoints land on opposite sides, which happens with probability 1/2. By linearity of expectation the expected cut weight is (1/2)·Σ_{i<j} w_ij. And the optimum can't exceed the total weight, so expected cut ≥ (1/2)·OPT. There's the 1/2-approximation. Sahni and Gonzales gave a deterministic greedy version back in 1976 — walk the vertices, put each one wherever it helps the partial cut more — and it's essentially the derandomization of the coin flip, same 1/2.

And then... nothing, for twenty years. People shaved the constant up to 1/2 + 1/(2m), 1/2 + (n−1)/(4m), and so on, but every one of those is 1/2 plus a term that goes to zero as the graph grows. The worst-case constant never budged off 1/2. So let me stare at *why* it's stuck, because if I don't understand the wall I'll just keep hitting it.

Every one of these analyses compares the cut I produce against the **total weight** Σ w_ij. That's the denominator. Expected cut ≥ (1/2)Σ w_ij, and then Σ w_ij ≥ OPT closes the argument. If I try to prove a better guarantee by the same route, I would need a lower bound c·Σ w_ij with c > 1/2 that holds on every graph. But that cannot be true in general: in the complete graph K_n, the total weight is n(n−1)/2 while the maximum cut has floor(n²/4) edges, so OPT/Σ w_ij tends to 1/2. No algorithm can always certify more than roughly half the total edge weight as a cut, because the optimum itself can be only roughly half the total edge weight. So the ratio I can prove from a fraction-of-total-weight argument is pinned at 1/2. The denominator is the problem. If I want to break 1/2 I need a *tighter upper bound on OPT* than the total weight — something I can compute in polynomial time that hugs OPT much more closely, and then compare my cut against *that*.

So the real question turns into: is there a polynomial-time-computable upper bound on the maximum cut that's provably close to OPT? Let me write the cut down algebraically and see what structure I can exploit. Give vertex i a variable y_i ∈ {−1, +1}, side S = {i : y_i = +1}. The quantity ½(1 − y_i y_j) equals 1 when y_i ≠ y_j and 0 when y_i = y_j — exactly the indicator that edge (i,j) is cut. So

  cut weight = Σ_{i<j} ½ w_ij (1 − y_i y_j),

and MAX CUT is: maximize that over y ∈ {−1,+1}^n. An integer quadratic program. The integrality — y_i pinned to ±1 — is the whole difficulty; drop it carelessly and the bound goes slack, like the LP relaxations that have a bad integrality gap. On the triangle C3 the true max cut is 2, but a relaxation that forgets the global geometry over-counts and certifies something useless. So a *naive* continuous relaxation won't give me the tight denominator I need. I need to relax in a way that keeps enough geometry.

Let me look at the same data through a spectral lens, because that's where tight bounds on quadratic forms usually come from. Define the Laplacian L: off-diagonal L_ij = −w_ij, diagonal L_ii = Σ_j w_ij. There's a clean identity: for any x, x^T L x = Σ_{i<j} w_ij (x_i − x_j)^2. Check it — the diagonal contributes Σ_i x_i^2 Σ_j w_ij, the off-diagonal contributes −2 Σ_{i<j} w_ij x_i x_j, and Σ w_ij(x_i^2 + x_j^2 − 2 x_i x_j) regroups to exactly that. Good. Now plug in x = x_S ∈ {−1,+1}^n. Then (x_i − x_j)^2 is 4 on a cut edge and 0 otherwise, so

  x_S^T L x_S = 4·w(S, S̄).

So maximizing the cut is maximizing x^T L x over x ∈ {−1,+1}^n, up to the factor 4. Now I want an upper bound on a quadratic form over the cube. The Rayleigh principle gives me one for free: for any symmetric M, λ_max(M) ≥ x^T M x / x^T x for every x. With x = x_S, x_S^T x_S = n (all entries are ±1), so 4·mc(G) = x_S^T L x_S ≤ n·λ_max(L). There's an eigenvalue upper bound: mc(G) ≤ (n/4)λ_max(L).

But that bound is weak as written, and I can see the slack: I threw away the constraint that x_S^T x_S = n *exactly* and that entries are ±1 *individually*. Rayleigh only used the global norm. I can add a diagonal perturbation that's invisible on the cube but lowers the top eigenvalue. Pick any vector u with Σ_i u_i = 0, set U = diag(u). On any x_S, x_S^T U x_S = Σ_i u_i x_{S,i}^2 = Σ_i u_i = 0, because x_{S,i}^2 = 1. So adding U changes the quadratic form *not at all* on the cube:

  4·mc(G) = x_S^T (L + U) x_S ≤ n·λ_max(L + U),

for *every* such "correcting" u. So mc(G) ≤ (n/4)·min_{u: Σu_i=0} λ_max(L + diag(u)). Now I'm minimizing the top eigenvalue over a family of perturbations that don't touch the true value — squeezing the bound down toward OPT. And λ_max of a symmetric matrix is convex in the diagonal, the constraint Σu_i = 0 is linear, so this minimization is convex and I can solve it in polynomial time to arbitrary precision. This is exactly the kind of tight, computable upper bound I was missing. And empirically people found it *very* tight: the worst gap anyone could exhibit between mc(G) and this eigenvalue bound is the 5-cycle, ratio 32/(25 + 5√5) ≈ 0.88445 — but nobody could *prove* a worst-case ratio better than the trivial 0.5. So there's a tight denominator with an 0.88-ish gap dangling, and no algorithm that achieves it. That dangling gap is the target.

Now — the eigenvalue picture is one way to look at it, but it doesn't obviously hand me a *cut*; it hands me a number. Let me go back to the integer quadratic program and relax it in a way that's geometric, so that the relaxed solution carries enough structure to round back into an assignment. The constraint y_i ∈ {−1, +1} is the same as: y_i is a *unit vector in one dimension*. There are only two unit vectors on the line, +1 and −1, and y_i y_j is their inner product. So the integrality is "live on the unit sphere of R^1." The obvious relaxation: let each vertex's variable be a unit vector in a *higher*-dimensional space. Replace the scalar y_i by a vector u_i ∈ R^n with ‖u_i‖ = 1, and replace the product y_i y_j by the inner product u_i · u_j. The objective becomes

  maximize Σ_{i<j} ½ w_ij (1 − u_i · u_j),  subject to ‖u_i‖ = 1.

Is this actually a relaxation? Yes — any genuine ±1 assignment is the special case where all the u_i are collinear unit vectors in a 1-D subspace, and there the objective is identical to the integer program's. So the feasible set only grew; the optimum of this vector program is ≥ OPT. Call its optimal value Z*_P. Now I have a continuous problem with the geometry kept: each vertex is a point on the sphere, and the objective rewards making endpoints of heavy edges point in *opposite* directions (u_i · u_j near −1 makes ½(1 − u_i·u_j) near 1).

Can I solve it? It's a quadratic problem in vectors, which smells nonconvex, but watch what happens when I change variables to the inner products. Let Y be the matrix with Y_ij = u_i · u_j. Then Y is exactly the Gram matrix of the u_i, and a matrix is a Gram matrix of *some* set of vectors iff it's symmetric positive semidefinite; the constraint ‖u_i‖ = 1 is just Y_ii = 1. The objective Σ_{i<j} ½ w_ij(1 − Y_ij) is *linear* in Y. So the whole thing is:

  maximize  (linear function of Y)
  subject to  Y_ii = 1 for all i,  Y ⪰ 0.

That's a semidefinite program — linear objective, linear equality constraints, PSD-cone constraint. And SDPs are convex; the PSD cone is convex, and for any ε > 0 I can get within additive ε of the optimum in time polynomial in the input and log(1/ε), by the ellipsoid method or interior-point methods. So the relaxation is genuinely solvable. And once I have an (almost) optimal Y, I factor it — Cholesky, or eigendecomposition Y = QΛQ^T and U = QΛ^{1/2} — so row i of U is my vector u_i back. The nonconvex vector program and the convex SDP are the same problem; I optimize over Y and read off the geometry.

One more check makes this denominator much more credible: this SDP value Z*_P is the *same number* as the Delorme–Poljak eigenvalue bound. Let me check the dual instead of just trusting the resemblance. In scaled-Laplacian form the SDP is

  maximize  (L/4)·Y
  subject to  Y_ii = 1,  Y ⪰ 0.

Put dual variables z_i on the diagonal constraints. The Lagrangian is Σ_i z_i + (L/4 − diag(z))·Y. Since Y ranges over the PSD cone, the supremum is finite exactly when diag(z) − L/4 ⪰ 0, and then the dual objective is minimize Σ_i z_i. Now compare that to the eigenvalue form. If u is a correcting vector, Σ_i u_i = 0, and λ ≥ λ_max(L + diag(u)), then λI − L − diag(u) ⪰ 0. Setting z_i = (λ − u_i)/4 gives diag(z) − L/4 = (λI − diag(u) − L)/4 ⪰ 0 and Σ_i z_i = nλ/4. Conversely, from any feasible z, define λ = (4/n)Σ_i z_i and u_i = λ − 4z_i; then Σ_i u_i = 0 and L + diag(u) ⪯ λI. So minimizing the dual is exactly minimizing (n/4)λ_max(L + diag(u)) over correcting vectors. The relaxation I just built isn't some new looser thing — it *is* the tight, empirically-0.88 bound, now wearing a form I can round.

So I have vectors u_i on the unit sphere of R^n, and I need to collapse them to ±1. How? I want a rule where, the more "opposite" two vectors are (the more an edge wants to be cut), the more likely I separate them. The relaxation has a symmetry I should respect: its objective depends only on the *angles* between vectors — apply any rotation to all the u_i simultaneously and nothing changes. So whatever rounding I pick should be rotation-invariant too; otherwise I'd be privileging a coordinate frame the problem doesn't care about. What's a rotation-invariant way to assign a sign to a point on the sphere? Cut the sphere with a hyperplane through the origin and read off which side each point is on. To make it rotation-invariant, pick the hyperplane *at random*, uniformly: draw a random unit vector r as the normal, and set S = {i : u_i · r ≥ 0}. Every direction for r is equally likely, so no frame is special. A uniform r is easy to generate: draw each coordinate i.i.d. from N(0,1) and normalize — a spherical Gaussian is rotationally symmetric, so its direction is uniform on the sphere. And since I only test the *sign* of u_i · r, I don't even need to normalize r.

Now the payoff question: with this random hyperplane, what's the probability an edge (i, j) gets cut, i.e. that u_i and u_j land on opposite sides? Let θ_ij be the angle between u_i and u_j, so u_i · u_j = cos θ_ij. The event "opposite sides" is sgn(u_i · r) ≠ sgn(u_j · r). Only the component of r in the 2-dimensional plane spanned by u_i and u_j matters for the two signs — the orthogonal part of r contributes nothing to either inner product's sign relative to the other. And the projection of a uniform spherical r onto any fixed 2-plane points in a *uniform* direction on that plane's circle, by rotational symmetry. So I've reduced to a planar picture: u_i and u_j sit at angle θ in a plane, and a uniformly random line through the origin (the hyperplane's trace, with normal = projected r) separates them exactly when it falls into one of the two opposite angular wedges of width θ between them. Two wedges, each of angular width θ, out of the full 2π — so the probability is 2θ/(2π) = θ/π.

  Pr[(i,j) cut] = θ_ij / π = arccos(u_i · u_j) / π.

Beautiful — the cut probability is *exactly proportional to the angle*, which is exactly the geometric quantity the relaxation was pushing to be large. So the expected cut weight is

  E[W] = Σ_{i<j} w_ij · θ_ij/π.

Now I compare this, edge by edge, against what that edge contributed to the relaxation value Z*_P, which was ½ w_ij (1 − u_i·u_j) = ½ w_ij (1 − cos θ_ij). If I can show

  θ/π ≥ α · ½(1 − cos θ)

for every angle θ ∈ [0, π], with some constant α, then summing over edges (all weights nonnegative) gives E[W] ≥ α·Z*_P ≥ α·OPT, and α is my approximation ratio. So define

  α = min_{0 < θ ≤ π}  (θ/π) / (½(1 − cos θ))  =  min_{0 < θ ≤ π}  (2/π) · θ / (1 − cos θ).

What is this minimum? Let me look at the function g(θ) = (2/π)·θ/(1 − cos θ). As θ → 0, 1 − cos θ ≈ θ²/2, so g(θ) ≈ (2/π)·θ/(θ²/2) = (4/π)/θ → ∞ — no minimum there, the ratio blows up, good, small angles are very favorable. At θ = π/2, 1 − cos = 1, g = (2/π)(π/2) = 1. Actually let me check whether g stays above 1 on the whole first quadrant. I want 2θ/(π(1 − cos θ)) ≥ 1, i.e. 2θ ≥ π(1 − cos θ), for 0 < θ ≤ π/2. At θ = π/2 both sides equal π. For smaller θ the left side 2θ shrinks linearly while π(1 − cos θ) shrinks like πθ²/2, faster — so the inequality holds with room on (0, π/2). So the minimum, if it's below 1, lives in (π/2, π].

Let me find it. Set the derivative of θ/(1 − cos θ) to zero: d/dθ [θ/(1 − cos θ)] = [(1 − cos θ) − θ sin θ]/(1 − cos θ)² = 0 ⟹ 1 − cos θ − θ sin θ = 0, i.e. cos θ + θ sin θ = 1. The nonzero root of that is θ* = 2.331122... radians (about 133.6°). I should sanity-check it's a minimum of g and bound the value. The cleanest way to nail the bound without hand-waving the calculus: I already have 2θ/(π(1 − cos θ)) > 1 on (0, π/2]. On [π/2, π], use that h(θ) = 1 − cos θ is concave there (h'' = cos θ ≤ 0 for θ ≥ π/2), so for any anchor θ₀ in that range, h(θ) ≤ h(θ₀) + (θ − θ₀)h'(θ₀) = θ sin θ₀ + (1 − cos θ₀ − θ₀ sin θ₀). Choose θ₀ = 2.331122, the root above; there 1 − cos θ₀ − θ₀ sin θ₀ = 0, so the bound collapses to 1 − cos θ ≤ θ sin θ₀. Therefore

  g(θ) = (2/π)·θ/(1 − cos θ) ≥ (2/π)·θ/(θ sin θ₀) = 2/(π sin θ₀) ≈ 0.87856.

So α = 0.87856..., the minimum sitting at θ* ≈ 133.6°. Stitching it together: E[W] = Σ w_ij θ_ij/π ≥ α Σ ½ w_ij(1 − cos θ_ij) = α·Z*_P ≥ α·OPT. A 0.878-approximation. And to be honest about the "− ε": I can't solve the SDP exactly (the optimum can be irrational), only to within ε in polynomial time, and the rounding multiplies through, so the clean statement is a (0.878 − ε)-approximation for any ε > 0, with running time polynomial in the input and log(1/ε).

Let me also notice what this single constant is telling me geometrically. The worst edge — the one realizing the ratio α — is the one whose vectors sit at θ* ≈ 133.6°. That's an obtuse angle: the relaxation already mostly separates that pair, and the rounding probability θ*/π ≈ 0.74 is a hair below the relaxation's credit ½(1 − cos θ*) ≈ 0.844, ratio ≈ 0.878. Tiny angles (almost-parallel, edge doesn't want to be cut) and angles near π (antipodal, edge gets cut almost surely) are both handled near-perfectly; the loss concentrates at this one obtuse angle. And the 5-cycle, which realizes the eigenvalue bound's empirical gap ≈ 0.88446, sits just barely above α — so the analysis is essentially tight, and I shouldn't expect to squeeze a better constant out of *this* rounding without changing the scheme.

Now let me make it real, grounded in how this is actually built. I form the scaled Laplacian (¼ L, so that x^T(¼L)x = cut), declare a symmetric PSD matrix variable, constrain its diagonal to 1, and maximize the linear objective trace((¼L)·X). Because diag(X)=1, that trace is exactly Σ_{i<j} ½ w_ij(1 − X_ij). Solve the SDP. Then eigendecompose X and include the square roots of the positive eigenvalues so the rows really factor the Gram matrix. Draw a Gaussian r, take signs of the projections, and that's the cut.

```python
import numpy as np
import networkx as nx
import cvxpy as cvx

def max_cut(graph):
    # cut weight = x^T (L/4) x  for x in {-1,+1}^n, since x^T L x = 4*cut
    laplacian = np.array(0.25 * nx.laplacian_matrix(graph).todense())

    # --- SDP relaxation: y_i in {-1,1}  ->  unit vectors u_i, Y = Gram(u) ---
    # Y >= 0 (PSD) and diag(Y)=1  <=>  u_i are unit vectors with Y_ij = u_i . u_j
    X = cvx.Variable(laplacian.shape, PSD=True)               # the Gram matrix Y
    objective = cvx.Maximize(cvx.trace(laplacian @ X))        # linear relaxation objective
    constraints = [cvx.diag(X) == 1]                          # ||u_i|| = 1
    cvx.Problem(objective, constraints).solve()

    bound = float(np.trace(laplacian @ X.value))              # Z*_P >= OPT

    # --- recover the vectors u_i from the Gram matrix (factor Y) ---
    gram = np.array(X.value, dtype=float)
    gram = 0.5 * (gram + gram.T)                             # remove solver-level asymmetry
    evals, evects = np.linalg.eigh(gram)
    keep = evals > 1e-6
    sdp_vectors = evects[:, keep] @ np.diag(np.sqrt(evals[keep]))

    # --- random-hyperplane rounding: r uniform on the sphere via a Gaussian ---
    r = np.random.randn(sdp_vectors.shape[1])
    r /= np.linalg.norm(r)                                   # (normalization is optional; only sign matters)
    colors = np.sign(sdp_vectors @ r)                       # S = {i : u_i . r >= 0}
    colors[colors == 0] = 1

    score = float(colors @ laplacian @ colors.T)             # weight of the produced cut
    return colors, score, bound
```

So the whole chain, in one breath: the 1/2 barrier comes from measuring the cut as a fraction of the total weight, whose best universal fraction cannot exceed 1/2, so I needed a tighter denominator; encoding the cut as x^T L x and perturbing by a zero-on-the-cube diagonal gave a tight, polynomial-time eigenvalue upper bound; relaxing each ±1 to a unit *vector* and the product to an inner product turned MAX CUT into an SDP whose value is exactly that tight bound; cutting the resulting sphere with a uniformly random hyperplane cuts each edge with probability θ/π, exactly proportional to the angle the relaxation was maximizing; and the worst-angle ratio min_θ (2/π)θ/(1 − cos θ) = 0.87856 at θ* ≈ 133.6° is the approximation guarantee — the first jump past 1/2 in twenty years.
