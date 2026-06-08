I have a convex function f, smooth, and I want its minimum — but not over all of space, over a constraint set D that is compact and convex. Polyhedron, a norm ball, the simplex, the set of doubly-stochastic matrices, whatever. All I can do with f is evaluate it and its gradient at a point. So the obvious thing is to walk downhill: from x, move in the direction −∇f(x). The trouble is immediate and stupid: ∇f points wherever it points, and −∇f(x) almost always shoots me straight out of D. After one honest gradient step I'm sitting at y = x − η∇f(x), which is infeasible. I've left the set I'm required to stay in.

The textbook fix is to project back. Take y, find the nearest feasible point, Π_D(y) = argmin_{z∈D} ‖z − y‖², and call that the next iterate. Projected gradient descent. And it works — for smooth f you get O(1/k), the non-expansiveness of the Euclidean projection makes the analysis go through. But stare at what the projection actually *is*. It's argmin over D of a quadratic. It is itself a constrained convex optimization problem, the same shape as the thing I'm trying to solve. On an Euclidean ball, fine, it's a one-line normalization. On the ℓ₁-ball, okay, there's a fast combinatorial routine. But the domains I actually care about? Project onto the trace-norm ball — that's a full singular-value decomposition, every single iteration. Project onto the Birkhoff polytope of doubly-stochastic matrices — that's a quadratic program over a set with n! vertices. Project onto some structured atomic norm ball — that can be exactly as hard as the original problem. So the entire per-iteration cost of my "first-order" method is dominated by an inner solve that I have no right to assume is cheap. I'm paying for a heavy projection just to undo the fact that the gradient stepped out of the set.

So let me back up and ask: what *can* I do cheaply on these awkward domains? Projection is hard. But on every one of those sets, there's a different operation that's easy. On the simplex, minimizing a *linear* function — find the smallest coordinate of a vector, pick that corner. On the trace-norm ball, minimizing a linear function over it is a single top singular-vector computation, not a whole SVD. On the Birkhoff polytope, minimizing a linear functional is the Hungarian assignment algorithm, polynomial time. On a submodular polyhedron, it's Edmonds' greedy algorithm in n log n. There's a pattern: **linear** optimization over D is cheap even when **quadratic** projection onto D is brutal. minimize ⟨s, c⟩ over s∈D — that's just linear programming when D is a polytope, and it has closed forms or fast oracles on the structured sets. Let me give it a name and lean on it: a linear-minimization oracle, s = argmin_{s∈D} ⟨s, c⟩.

Now, how do I turn "I can minimize linear functions over D" into "I can minimize f over D"? f isn't linear. But I have its gradient, and convexity hands me the one fact I always reach for: the tangent plane lies *under* the graph. f(y) ≥ f(x) + ⟨y − x, ∇f(x)⟩ for every y. So at my current point x, the linear function L_x(y) = f(x) + ⟨y − x, ∇f(x)⟩ is a global lower bound on f, and it agrees with f at x. If I want to make f small, a reasonable proxy is: make this linear lower model small over D. And minimizing L_x over D is, up to the constant f(x) and the −⟨x,∇f(x)⟩ piece, just minimizing ⟨s, ∇f(x)⟩ over s∈D — which is precisely the cheap oracle I just decided I have.

So: s = argmin_{s∈D} ⟨s, ∇f(x)⟩. This s is the point of D where the linear model promises the most improvement. And because I'm minimizing a linear functional over a compact convex set, I can choose an extreme-point minimizer — a vertex of D, a corner — whenever there is a whole face of ties. That's going to matter later, but for now I have a feasible point s, sitting at a corner, in the most-downhill direction the linear model can see.

Can I just jump to s? No — and here's the wall. The linear model is only a lower bound; f curves up away from its tangent plane. s might be way over on the far side of D, and f could be large there even though the *linear* approximation at x said "great direction." If I set x⁺ = s I'm trusting a linear model far outside the region where it's accurate. I'll overshoot. So I shouldn't go all the way to s; I should move *toward* it and stop short.

How far? Take a convex combination: x⁺ = (1 − γ)x + γs for some γ ∈ [0,1]. And now something lovely falls out for free. x is in D, s is in D, D is convex — so (1 − γ)x + γs is in D, automatically, for any γ in [0,1]. No projection. **Feasibility comes for free from convexity of the set**, because I'm only ever taking convex combinations of points I already know are feasible. The gradient never gets a chance to throw me out, because I never take a raw gradient step — I take a step toward a feasible vertex and blend. That's the whole trick that kills the projection: replace "step out, then project back in" with "stay in by construction." The price is that I gave up moving in the exact negative-gradient direction; I move toward s instead. But on these domains that's the trade I want — a cheap linear solve instead of a brutal projection.

Let me write down the loop so far. At x: compute ∇f(x); s = argmin_{s∈D}⟨s, ∇f(x)⟩; x⁺ = (1−γ)x + γs. Equivalently x⁺ = x + γ(s − x), a step of length γ along the direction (s − x). This is the conditional gradient idea — I'm doing steepest descent, but the descent direction is *conditioned* on the constraint set, picked among the feasible vertices rather than freely.

Before I worry about the step size, I want to notice something about the quantity I'm already computing. When I evaluate ⟨s, ∇f(x)⟩ I implicitly know ⟨x − s, ∇f(x)⟩ = max_{s'∈D} ⟨x − s', ∇f(x)⟩, since s minimizes ⟨s',∇f(x)⟩ over s'∈D. Call this g(x) := ⟨x − s, ∇f(x)⟩. It's nonnegative (taking s' = x in the max gives 0, so the max is ≥ 0). What is it? Go back to the convex lower bound, but evaluate it at the *optimum* x*: f(x*) ≥ f(x) + ⟨x* − x, ∇f(x)⟩. Rearrange: f(x) − f(x*) ≤ ⟨x − x*, ∇f(x)⟩. And x* is some point of D, so ⟨x − x*, ∇f(x)⟩ ≤ max_{s'∈D}⟨x − s', ∇f(x)⟩ = g(x). Chain them:

  f(x) − f(x*) ≤ g(x).

The true suboptimality, which I can't measure because I don't know f(x*), is *bounded above* by g(x), which I *do* compute — it's the by-product of the very oracle call that gives me s. So g(x) is a certificate: if g(x) ≤ ε then I'm provably within ε of optimal, and I can stop. A free, computable optimality gap, automatic from each step. (It's a surrogate duality gap; in fact it's exactly Fenchel duality with the dual variable chosen as the current gradient, but I don't need the machinery — the one-line convexity argument already gives me the bound.) And it certifies *any* feasible point, not just mine — it's a stopping criterion for any optimizer of this problem.

Now the step size and the rate. I need to control the overshoot, which is the amount f curves above its tangent plane. Let me define the worst-case curvature over the set in exactly the form the analysis will need. Moving from x toward s by γ, to y = x + γ(s − x), the deviation of f at y from its linearization at x is f(y) − f(x) − ⟨y − x, ∇f(x)⟩, which is ≥ 0 by convexity. I'll measure how big this can be, weighted by 1/γ² so it's a single constant independent of how far I step:

  C_f := sup_{x,s∈D, γ∈(0,1], y=x+γ(s−x)} (2/γ²) [ f(y) − f(x) − ⟨y − x, ∇f(x)⟩ ].

For a linear f this is 0. Why the 2/γ²? Because the deviation behaves like (γ²/2)·(second-order term), so dividing by γ²/2 strips the γ-dependence and leaves a pure "non-linearity of f over D" number. This is the natural constant here, and notice it's defined purely from f and D — no norm, no metric. If I do want to connect it to the familiar smoothness picture: suppose ∇f is L-Lipschitz in some norm, so by the descent lemma f(y) ≤ f(x) + ⟨y−x,∇f(x)⟩ + (L/2)‖y−x‖². Then f(y) − f(x) − ⟨y−x,∇f(x)⟩ ≤ (L/2)‖y−x‖², and since y − x = γ(s − x), (1/γ²)‖y−x‖² = ‖s−x‖², so

  C_f ≤ sup_{x,s∈D} L‖s − x‖² ≤ L · diam(D)².

So bounded curvature is just strong-smoothness in disguise, C_f ≤ L·diam², but the C_f form is cleaner because it's exactly the quantity the bound uses and it doesn't care which norm I picked.

With C_f in hand the per-step improvement is mechanical. Write x⁺ = x + γ(s − x). By the definition of C_f, taking this very (x, s, γ) as the supremand,

  f(x⁺) ≤ f(x) + γ⟨s − x, ∇f(x)⟩ + (γ²/2) C_f.

And ⟨s − x, ∇f(x)⟩ = −⟨x − s, ∇f(x)⟩ = −g(x), because s is the linear minimizer. Substitute:

  f(x⁺) ≤ f(x) − γ g(x) + (γ²/2) C_f.

This is the lemma the whole convergence rests on. The first-order term −γ g(x) is a genuine decrease proportional to the gap; the second-order term (γ²/2)C_f is the penalty for the linear model being wrong. I want γ small enough that the penalty doesn't eat the gain, but not so small that I crawl.

Subtract f(x*) and write h(x) := f(x) − f(x*) for the primal error. The −γ g(x) term carries the gap, but I want a recursion in the error h, and the certificate gives me g(x) ≥ h(x), so I can replace g(x) by h(x) and lose nothing:

  h(x⁺) ≤ h(x) − γ g(x) + (γ²/2)C_f ≤ h(x) − γ h(x) + (γ²/2)C_f = (1 − γ) h(x) + (γ²/2) C_f.

A contraction by (1 − γ) plus an additive γ² term. Let me name the constant C := C_f/2 to keep the algebra clean, so h(x⁺) ≤ (1 − γ)h(x) + γ²·2C... no — let me just carry (γ²/2)C_f explicitly and pick the schedule, then read off the constant.

What γ? It can't depend on knowing C_f or f(x*) (I don't), and it can't require a line-search (I want an open-loop, parameter-free rule). I need γ_k → 0 so the γ² penalty dies, but Σγ_k = ∞ so I actually traverse the set. The harmonic-ish γ_k ∼ 2/k does both. Let me *guess* the form γ_k = 2/(k+2) and see what rate the recursion produces — and whether the constants conspire.

Claim: h(x^(k)) ≤ 2C_f/(k+2) for k ≥ 1. Let me prove h(x^(k+1)) ≤ 4C/(k+1+2) by induction, where C := C_f/2, so 4C = 2C_f and the claim is the same statement. (I keep C around because it makes the algebra symmetric; at the end 4C = 2C_f.)

Base case, k = 0: γ^(0) = 2/(0+2) = 1, so x^(1) = (1−1)x^(0) + 1·s = s — the very first step jumps all the way to the vertex s. That's sensible: x^(0) is an arbitrary starting point I have no reason to trust, so discard it and land on a corner. With γ = 1 the recursion gives h(x^(1)) ≤ (1−1)h(x^(0)) + 1²·C = C. And the target at k = 0 is 4C/(0+1+2) = 4C/3 ≥ C. Base case holds.

Inductive step, k ≥ 1, assume h(x^(k)) ≤ 4C/(k+2). With γ = 2/(k+2):

  h(x^(k+1)) ≤ (1 − 2/(k+2)) h(x^(k)) + (2/(k+2))² C
            ≤ (1 − 2/(k+2)) · 4C/(k+2) + 4C/(k+2)²
            = (4C/(k+2)) [ (1 − 2/(k+2)) + 1/(k+2) ]
            = (4C/(k+2)) [ (k+2 − 2 + 1)/(k+2) ]
            = (4C/(k+2)) · (k+1)/(k+2)
            = 4C (k+1)/(k+2)².

Now I need this ≤ 4C/(k+3), i.e. (k+1)/(k+2)² ≤ 1/(k+3), i.e. (k+1)(k+3) ≤ (k+2)². Expand: k²+4k+3 ≤ k²+4k+4. True, by exactly 1. So h(x^(k+1)) ≤ 4C/(k+3) = 4C/((k+1)+2), closing the induction. And 4C = 2C_f, so

  f(x^(k)) − f(x*) ≤ 2C_f/(k+2).

There it is: O(C_f/k). And the constants weren't free — the "+2" and the factor "2" in γ_k = 2/(k+2) are exactly what makes (k+1)(k+3) ≤ (k+2)² fall out by a single unit. A different schedule c/(k+a) with the wrong c, a would leave a residual in this particular contraction proof. That's why γ_k = 2/(k+2) is the clean open-loop choice: it needs nothing but the iteration count, starts with γ_0 = 1, and makes the induction close without knowing C_f.

Two things I want to nail before I'm satisfied. First: the rate is on the primal error, but my *certificate* is the gap g. Does the gap I can actually measure also get small, or could it stay large while the error quietly shrinks? It can't stay large, and the reason is the same improvement lemma read backwards. From f(x^(k+1)) ≤ f(x^(k)) − γ g^(k) + (γ²/2)C_f, rearrange to γ g^(k) ≤ h^(k) − h^(k+1) + (γ²/2)C_f. If g^(k) stayed bigger than some threshold across a whole block of late iterations, summing these would force the telescoped primal error h to drop below zero — impossible, h ≥ 0.

Let K ≥ 2 and set D_K = K + 2. The primal theorem says h^(k) ≤ 2C_f/(k+2), so let C = 2C_f and write that as h^(k) ≤ C/(k+2). I want to prove that not all late gaps can exceed βC/D_K, because βC/D_K is exactly 2βC_f/(K+2). Pick a fraction μ in (0,1) and start the late block at k_min = ⌈μD_K⌉ − 2, so for every k from k_min to K I have μD_K ≤ k+2 ≤ D_K and there are at least (1−μ)D_K such steps. Suppose, toward a contradiction, that every one of those gaps satisfies g^(k) > βC/D_K. With γ_k = 2/(k+2), the improvement recursion becomes

  h^(k+1) < h^(k) − (2/(k+2))·(βC/D_K) + C/(k+2)²
          ≤ h^(k) − 2βC/D_K² + C/(μ²D_K²)
          = h^(k) − (C/D_K²)(2β − 1/μ²).

Summing from k_min to K telescopes the h terms, and the starting primal error is at most h^(k_min) ≤ C/(k_min+2) ≤ C/(μD_K). Therefore

  h^(K+1) < C/(μD_K) − (1−μ)D_K · (C/D_K²)(2β − 1/μ²)
          = (C/(μD_K)) [ 1 − (1−μ)(2μβ − 1/μ) ].

Now choose μ = 2/3. To make the bracket vanish I need (1/3)(2·(2/3)β − 3/2) = 1, which gives 4β/9 − 1/2 = 1 and therefore β = 27/8. With that choice the strict inequality says h^(K+1) < 0, contradicting h ≥ 0. So the assumption fails: at least one late iterate, and hence some k̂ in 1…K, has

  g(x^(k̂)) ≤ 2β C_f/(K+2) = O(C_f/K),

with β = 27/8. (If I'm willing to switch to a constant step γ = 2/(K+2) in the second half, the same argument sharpens β down to ≈ 2, giving g ≤ 2C_f/(K+2).) So the free certificate genuinely certifies optimality to O(1/K) — I don't just converge, I can *prove* I've converged, online, with no extra cost.

Second: the iterates. Each step adds exactly one new vertex s and convex-combines it with the old iterate. So if x^(0) is a vertex, x^(k) is a convex combination of at most k+1 vertices of D. Caratheodory says a generic point of D needs n+1 vertices — but I'm using only k+1, and the regime I care about is k ≪ n, so my iterate is *sparse* in its vertex representation. On the ℓ₁-ball or simplex that means literally a sparse vector (one new nonzero coordinate per step); on the trace-norm ball each vertex is a rank-one matrix, so x^(k) is a sum of ≤ k+1 rank-ones — low rank. The projection-free method doesn't just dodge the SVD, it *produces* exactly the low-rank object I wanted. And this sparsity is not wasteful: for f(x) = ‖x‖² on the simplex, any x supported on k coordinates has f(x) − f(x*) = ‖x‖² − 1/n ≥ 1/k − 1/n — Cauchy-Schwarz gives ‖x‖₁² ≤ k‖x‖₂² so on the simplex ‖x‖₂² ≥ 1/k — and the gap there is ≥ 2/k. So O(1/ε) atoms is the *best possible* for any method that adds one atom per step: sparsity and accuracy are tied at O(1/k), and I'm hitting the floor.

One more thing I notice, almost an accident of the construction. Everything in this method — the linear oracle, the convex-combination update, the curvature C_f — refers only to D and ⟨·, ∇f⟩, never to a chosen inner product or metric on the space. If I reparameterize D by any invertible linear map M, replacing f by f̂(x̂) = f(Mx̂), then ∇f̂ = Mᵀ∇f, the oracle argmin⟨ŝ, Mᵀ∇f⟩ picks the preimage of the same vertex, the convex combination transforms covariantly, and C_f is unchanged by its very definition. So the algorithm and its rate are *affine-invariant*: any preconditioning or distortion of the domain has literally no effect on the iterates. That's striking given how much projected/second-order methods depend on the conditioning of D — and it means there's nothing to precondition here, the method already doesn't see the distortion.

So the method is complete, and it landed exactly where the pain pointed. The pain was: projection is the expensive part. The escape was: never project — minimize the linear model over D to get a feasible vertex s via a cheap oracle, then stay feasible by convex-combining toward it with the open-loop step γ_k = 2/(k+2). Convexity of f makes the linear model a valid lower bound (so the direction is meaningful) and makes the gap g(x) = ⟨x − s, ∇f(x)⟩ a free upper bound on the true error (so I get a stopping certificate). Bounded curvature C_f turns one step's improvement into f(x⁺) ≤ f(x) − γ g(x) + (γ²/2)C_f, and the schedule γ_k = 2/(k+2) makes the induction close at f(x_k) − f(x*) ≤ 2C_f/(k+2), with the gap itself driven to O(C_f/K). And because only one vertex enters per step, the iterates are sparse / low-rank, optimally so. Let me write it down as it runs.

```python
import numpy as np

# The conditional-gradient (Frank-Wolfe) method:
#   minimize a smooth convex f over a compact convex set D,
#   using only a linear-minimization oracle over D -- no projection.

def frank_wolfe(grad, lmo, x0, max_iter, tol=1e-8):
    """
    grad(x) -> gradient of f at x.
    lmo(c)  -> argmin_{s in D} <s, c>   (linear-minimization oracle; tie-break to a vertex).
    x0      -> initial feasible point in D (ideally a vertex).
    """
    x = np.array(x0, dtype=float)
    gap = np.inf
    for k in range(max_iter):
        g = grad(x)                       # first-order oracle
        s = lmo(g)                         # minimize the LINEAR model over D -> a vertex
        gap = float(np.dot(g, x - s))      # duality gap  g(x) = <x - s, grad f(x)>, free certificate
        if gap <= tol:                     # gap >= f(x) - f(x*), so this proves epsilon-optimality
            break
        gamma = 2.0 / (k + 2.0)            # open-loop step: gamma_0 = 1 jumps to the first vertex
        x = (1.0 - gamma) * x + gamma * s  # convex combination -> stays in D for free (no projection)
        # x is a convex combination of <= k+1 vertices: sparse / low-rank by construction
    return x, gap
```

A concrete instance to see the oracle do its job: minimize f(x) = ½‖x − b‖² over the unit simplex Δ_n. Then ∇f(x) = x − b, and the linear oracle over the simplex is trivial — argmin_{s∈Δ_n}⟨s, c⟩ is the basis vector e_i with i = argmin_j c_j, the smallest gradient coordinate. No projection, each step touches one coordinate, the iterate stays a sparse convex combination of basis vectors.

```python
def lmo_simplex(c):
    s = np.zeros_like(c)
    s[np.argmin(c)] = 1.0          # vertex of the simplex in the most-descent coordinate
    return s

def solve_simplex_least_squares(b, max_iter=1000, tol=1e-10):
    b = np.asarray(b, dtype=float)
    x0 = lmo_simplex(np.zeros_like(b))       # arbitrary feasible vertex of the simplex
    grad = lambda x: x - b
    return frank_wolfe(grad, lmo_simplex, x0, max_iter, tol)
```
