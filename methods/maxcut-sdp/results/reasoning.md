OK, let me think about this from scratch. I have a weighted graph, weights w_ij ≥ 0, and I want to split the vertices into two sides to maximize the weight of edges that cross. Exact MAX-CUT is NP-hard, so I'm not going to find the optimum in polynomial time on every graph. What I actually want is a *guarantee*: a polynomial algorithm that, on the worst possible graph, still returns a cut of weight at least c·OPT, with c as large as I can prove.

Let me first nail down the baseline I have to beat, because it sets the bar. Flip an independent fair coin for each vertex: put i on side +1 or -1 with probability 1/2 each. An edge (i,j) is cut exactly when the two coins disagree, which happens with probability 1/2. So the expected cut weight is (1/2)·Σ w_ij, and since OPT is at most the total weight Σ w_ij, this is at least (1/2)·OPT. So I already have a 1/2-approximation for free, and it's deterministic if I derandomize with conditional expectations. The whole game is: can I provably beat 1/2?

Let me write the problem cleanly so I can see its shape. Assign each vertex a sign y_i ∈ {-1,+1}; i is on side S iff y_i = +1. Then edge (i,j) is cut iff y_i ≠ y_j iff y_i y_j = -1. The indicator "(i,j) is cut" is exactly (1 - y_i y_j)/2 — it's 0 when y_i y_j = +1 and 1 when y_i y_j = -1. Good. So

    maximize (1/2) Σ_{(i,j)∈E} w_ij (1 - y_i y_j)   over y ∈ {-1,+1}^n.

The only thing the objective ever touches is the product y_i y_j. That's worth remembering. As a continuous object this is a quadratic form in y with an indefinite matrix — a nonconvex integer quadratic program. That nonconvexity *is* the NP-hardness; I can't just relax y into a box and call it convex-friendly.

Let me try the obvious relaxation route anyway and watch it break, because the way it breaks should tell me what's missing. If I simply let y_i range over [-1,1] while keeping the products y_i y_j, I have not really obtained a useful linear relaxation; the product is still the wrong object. The LP move is to introduce a variable z_ij ∈ [0,1] for "is edge (i,j) cut" and maximize Σ w_ij z_ij over linear constraints that every cut satisfies. I want a relaxation whose optimum is an *upper bound* on OPT, and from which I can round back.

Now watch it fail on a triangle. Take K_3, three vertices, unit weights. Any real cut puts two vertices on one side and one on the other, so exactly two of the three edges cross: OPT = 2. But if I let the three "is this edge cut" indicators float independently in [0,1], nothing stops me from setting all three to 1 and getting objective 3. A triangle inequality can forbid this exact fractional point, but that is the lesson: I have to keep adding odd-cycle and cut-consistency inequalities that the independent scalar variables do not know on their own. On complete graphs, the same naive edge LP sets every edge to 1 while an integral cut crosses only floor(n^2/4) edges, so the integral/LP ratio tends to 1/2. Stronger linear lift-and-project relaxations also stay near 1/2 for a linear number of rounds. So the scalar LP route is not giving me a robust way to beat the coin flip. The relaxation is too weak; it can't see enough of the global parity structure.

So I'm stuck, and the wall is informative: a single real number per vertex, or a single number per edge, is not enough geometry to encode "these two cannot both be cut along with the third." I need the relaxed objects to carry more structure — enough that a convex constraint can enforce the global consistency that defeated the LP.

Let me go back to the one thing the objective sees: y_i y_j, the product of two ±1's. A ±1 is a unit vector in *one* dimension: y_i ∈ {-1,+1} is just a point on the 0-sphere, and y_i y_j is its inner product with y_j. What if I let each vertex be a unit vector in a *high*-dimensional space instead of a unit scalar? Replace the scalar y_i by a unit vector v_i ∈ R^n with ‖v_i‖ = 1, and replace the product y_i y_j by the inner product ⟨v_i, v_j⟩. The objective becomes

    (1/2) Σ_{(i,j)∈E} w_ij (1 - ⟨v_i, v_j⟩),   subject to ‖v_i‖ = 1.

Is this a relaxation — does its optimum upper-bound OPT? Yes: any genuine ±1 solution embeds by setting v_i = y_i · e_1, collinear unit vectors along one axis, and then ⟨v_i, v_j⟩ = y_i y_j reproduces the original objective exactly. So the vector problem ranges over a strictly larger feasible set; its optimum is ≥ OPT. The original is just the special case where all vectors are forced onto a single line.

And — crucially — is the relaxed problem tractable? The objective only depends on the inner products ⟨v_i, v_j⟩. Collect them into a matrix Y with Y_ij = ⟨v_i, v_j⟩. The set of matrices that arise as Gram matrices of vectors is *exactly* the set of positive semidefinite matrices. The constraint ‖v_i‖ = 1 is Y_ii = 1. The objective (1/2) Σ w_ij (1 - Y_ij) is linear in the entries of Y. So I'm maximizing a linear function of a symmetric matrix Y subject to Y ⪰ 0 and Y_ii = 1 — a semidefinite program. SDPs are convex and solvable to additive accuracy ε in polynomial time. The vector view and the matrix view are the same object: solve for Y, then factor Y = Q Q^T, or transpose the factor and use columns, and read off v_i as the i-th row of Q. The high-dimensional freedom is what lets the PSD constraint enforce a global consistency the scalar LP couldn't — the vectors live on a sphere and their pairwise angles are coupled through positive-semidefiniteness.

So I can compute, in polynomial time, an optimal set of unit vectors {v_i} on the sphere in R^n, with the SDP value SDP ≥ OPT. But the vectors are not a cut — they're points scattered on a sphere. I need to collapse them back to ±1, and I need to do it without throwing away the advantage I just bought.

How do I turn a configuration of unit vectors into a two-sided partition? I want vectors the SDP pushed *apart* to land on opposite sides, and vectors it pulled *together* to land on the same side. The SDP, by maximizing 1 - ⟨v_i,v_j⟩ on edges, tries to make endpoints of heavy edges nearly antipodal (⟨v_i,v_j⟩ near -1, angle near π). So a natural rounding: cut the sphere with a hyperplane through the origin and assign side by which half each vector falls in. Pick the hyperplane *at random* — its normal r drawn from a spherically symmetric distribution, the standard Gaussian N(0, I_n) is the obvious choice because it's spherically symmetric and trivial to sample. Then set

    y_i = sign(⟨v_i, r⟩).

Vectors on the same side of the random hyperplane get the same sign; vectors on opposite sides get split. Intuitively, the further apart two vectors are in angle, the more likely a random hyperplane slices between them. Let me make that precise, because the whole guarantee rides on it.

Fix an edge (i,j). Let θ_ij be the angle between v_i and v_j, so ⟨v_i,v_j⟩ = cos θ_ij, with θ_ij ∈ [0,π]. I want Pr[sign⟨v_i,r⟩ ≠ sign⟨v_j,r⟩]. Only the component of r in the plane P spanned by v_i and v_j affects the two signs ⟨v_i,r⟩ and ⟨v_j,r⟩ — the component orthogonal to P contributes nothing to either inner product. And the projection of a spherically symmetric Gaussian onto the 2-plane P is itself a 2D spherically symmetric Gaussian, whose *direction* is uniform on the circle. So the question collapses to a flat 2D one: I have two unit vectors in the plane separated by angle θ, and I draw a line through the origin (the trace of the hyperplane in P) with uniformly random orientation; what's the probability that the two vectors end up on opposite sides of the line?

A line through the origin separates v_i and v_j iff it passes through the angular wedge strictly between them. As I rotate the line's orientation from 0 to 2π, it separates them on two opposite arcs, each of angular width θ — one wedge between the vectors and the antipodal wedge between their negatives. Total favorable orientation measure is 2θ out of the full 2π. So

    Pr[(i,j) is cut] = 2θ_ij / (2π) = θ_ij / π.

That's clean, and it's *linear in the angle*. Now I can write the expected cut value:

    E[cut weight] = Σ_{(i,j)∈E} w_ij · θ_ij / π.

I want to compare this, edge by edge, to what the SDP "paid itself" for that edge, which was w_ij · (1 - ⟨v_i,v_j⟩)/2 = w_ij · (1 - cos θ_ij)/2. If I can show that for every angle the rounding probability θ/π is at least some fixed constant α times the SDP term (1 - cos θ)/2, then summing over edges gives E[cut] ≥ α·SDP ≥ α·OPT, and α is my approximation ratio. So define

    α = min_{0 < θ ≤ π}  (θ/π) / ((1 - cos θ)/2)  =  (2/π) · min_{0 < θ ≤ π}  θ / (1 - cos θ).

By construction, θ/π ≥ α·(1 - cos θ)/2 for every θ in (0,π], and α is the largest constant for which that holds. So the whole question — how much better than 1/2 — comes down to minimizing the single-variable function g(θ) = θ/(1 - cos θ) over (0,π].

Let me actually find that minimum, not gesture at it. At the endpoints: as θ → 0, g(θ) = θ/(1-cos θ) ≈ θ/(θ²/2) = 2/θ → ∞, so the boundary near 0 is not where the min is. At θ = π, g(π) = π/(1-cos π) = π/2 ≈ 1.5708, giving (2/π)·g(π) = 1. So at θ = π the ratio is exactly 1 — the rounding loses nothing on antipodal vectors, which makes sense: vectors at angle π are cut with probability 1, and the SDP term is also 1. The minimum is in the interior. Set the derivative to zero:

    g'(θ) = [ (1 - cos θ) − θ·sin θ ] / (1 - cos θ)² = 0  ⟹  (1 - cos θ) = θ sin θ.

This is a transcendental equation. Solving it numerically, the root is θ* ≈ 2.3311 radians ≈ 133.56°, where cos θ* ≈ -0.6892. Plugging back,

    α = (2/π) · θ*/(1 - cos θ*) = (2/π) · 2.3311 / 1.6892 ≈ 0.87856.

So α ≈ 0.87856 > 1/2. Stare at the two curves for a second: θ/π rises linearly from 0 to 1 as the angle goes 0 → π, while (1 - cos θ)/2 rises like sin²(θ/2). They meet at θ = 0, θ = π/2, and θ = π; before π/2 the rounding probability is larger than the SDP edge term, and after π/2 it falls below it. The worst *relative* gap — where the SDP's term most outruns the true cut probability — is at θ* ≈ 133.6°, and even there the rounding recovers a fraction α ≈ 0.878 of it. That's the worst case, so it bounds every edge, so it bounds the sum.

Putting the chain together:

    E[cut weight] = Σ w_ij · θ_ij/π
                 ≥ α · Σ w_ij · (1 - cos θ_ij)/2     (termwise, by definition of α)
                 = α · (SDP optimum)
                 ≥ α · OPT.

So a single random hyperplane gives, in expectation, at least 0.87856·OPT — comfortably past the 1/2 barrier that defeated the scalar LP route. Repeating the rounding and keeping the best cut is the practical way to spend the same SDP solution; derandomization can also recover a cut with value at least the expectation.

The choices line up with the obstacles I hit. The ±1 encoding (rather than 0/1) is what makes the product y_i y_j read off "same vs. different side" and the objective bilinear and clean. The leap from scalar to *vector* relaxation is the crux: a scalar per vertex collapses the structure and inherits the odd-cycle integrality gap; vectors on a sphere, coupled by the PSD constraint, carry enough geometry to enforce global consistency, and they make the relaxation convex (an SDP) at the same time. The dimension can be as large as n — n vectors live in at most n dimensions, and the SDP finds the best Gram matrix without my having to pick the dimension. The Gaussian r is chosen for spherical symmetry — that's exactly what makes the projection onto the 2-plane uniform in direction and hands me the clean θ/π identity; sampling r/‖r‖ would give the identical hyperplane, so the Gaussian is just the convenient sampler. The threshold is at 0 — the hyperplane must pass through the origin, because any nonzero offset would break the symmetry that gives θ/π and would no longer match the antipodal structure the SDP built. And I never need to materialize the v_i beyond a factor Q of Y: since Y ⪰ 0, write Y = Q Q^T, treat row i of Q as v_i, and sign(Q r)_i is exactly sign(⟨v_i, r⟩). One matrix square root and one Gaussian vector do the rounding.

Now let me write it as real code. Build the SDP over the Gram matrix X (this is Y above), enforce X ⪰ 0 and unit diagonal, maximize the cut objective, solve; then factor and round with one random hyperplane. I will write the common unweighted edge-list version, where a weighted version only multiplies each edge term by w_ij.

```python
import numpy as np
import cvxpy as cp
from scipy.linalg import sqrtm

def gw(n, edges):
    # ---- SDP relaxation over the Gram matrix X_ij = <v_i, v_j> ----
    X = cp.Variable((n, n), symmetric=True)
    constraints = [X >> 0]                      # X is PSD  <=> X is a Gram matrix of vectors
    constraints += [X[i, i] == 1 for i in range(n)]   # ||v_i|| = 1: unit vectors on the sphere
    # objective: (1/2) sum_{(i,j) in E} (1 - <v_i, v_j>), the cut weight in vector form
    objective = sum(0.5 * (1 - X[i, j]) for (i, j) in edges)
    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve()                                # SDP optimum >= OPT, solved to additive eps

    # ---- random-hyperplane rounding ----
    Q = sqrtm(X.value).real                     # X = Q Q^T; row i of Q is the vector v_i
    r = np.random.randn(n)                       # Gaussian normal -> spherically symmetric hyperplane
    x = np.sign(Q @ r)                           # y_i = sign(<v_i, r>): which side of the hyperplane
    return x                                      # x in {-1, +1}^n  (E[cut] >= 0.87856 * OPT)

def cut(x, edges):
    # edges whose endpoints landed on opposite sides
    return [(i, j) for (i, j) in edges if np.sign(x[i] * x[j]) < 0]
```

The causal chain in one breath: the coin flip caps out at 1/2 and the scalar LP route is stuck there too, because independent edge variables do not carry the odd-cycle and cut-consistency structure that bounds OPT. Lifting each vertex from a unit scalar to a unit vector and replacing y_i y_j by ⟨v_i,v_j⟩ gives a convex semidefinite relaxation (the Gram matrix is exactly a PSD matrix with unit diagonal) whose optimum upper-bounds OPT. Cutting the resulting sphere configuration with a single uniformly-random hyperplane through the origin cuts each edge with probability θ_ij/π, and since θ/π ≥ α·(1 - cos θ)/2 with α = (2/π)·min_θ θ/(1-cos θ) ≈ 0.87856 attained at θ* ≈ 133.6°, the expected cut is at least α·SDP ≥ α·OPT — a 0.878-approximation, decisively past 1/2.
