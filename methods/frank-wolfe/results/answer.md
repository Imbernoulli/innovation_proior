# The Frank-Wolfe (conditional gradient) method

## Problem

Minimize a continuously differentiable convex function over a compact convex set:

  minimize f(x)  subject to  x ∈ D,   D ⊆ Rⁿ compact convex, f convex with gradient ∇f.

Access to f is first-order (value and gradient). The difficulty: a gradient step leaves D, and the
classical remedy — Euclidean projection Π_D(y) = argmin_{z∈D}‖z − y‖² — is itself a convex program
that can be as costly as the original problem (a full SVD on a trace-norm ball, a QP over n! vertices
on the Birkhoff polytope, an intractable solve on structured atomic norms).

## Key idea

Never project. At the current iterate x, minimize the **linear** model of f over D (cheap on
structured domains: an LP / a top-eigenvector / an assignment / a greedy solve), then move toward the
resulting vertex by a convex combination, which stays feasible automatically:

  s = argmin_{s∈D} ⟨s, ∇f(x)⟩    (linear-minimization oracle; choose a vertex on ties)
  x⁺ = (1 − γ_k) x + γ_k s,   γ_k = 2/(k+2).

Because x, s ∈ D and D is convex, x⁺ ∈ D — feasibility is free, with no projection. Convexity makes
the linear model f(x) + ⟨y − x, ∇f(x)⟩ a global underestimator of f, so s is a meaningful descent
target, and it makes the by-product quantity below a free optimality certificate.

## The duality gap (free certificate)

Define

  g(x) := max_{s'∈D} ⟨x − s', ∇f(x)⟩ = ⟨x − s, ∇f(x)⟩ ≥ 0,

computed at no extra cost from the same oracle call that produces s. Evaluating the convex lower
bound at the optimum x*, f(x*) ≥ f(x) + ⟨x* − x, ∇f(x)⟩, gives f(x) − f(x*) ≤ ⟨x − x*, ∇f(x)⟩ ≤ g(x).
Hence

  g(x) ≥ f(x) − f(x*),

so g(x) ≤ ε proves ε-optimality. It is a valid stopping certificate for any feasible point.

## Curvature constant

  C_f := sup_{x,s∈D, γ∈(0,1], y=x+γ(s−x)} (2/γ²) [ f(y) − f(x) − ⟨y − x, ∇f(x)⟩ ].

C_f = 0 for linear f. If ∇f is L-Lipschitz in a norm ‖·‖, then C_f ≤ L · diam_{‖·‖}(D)² (apply the
descent lemma f(y) ≤ f(x) + ⟨y−x,∇f(x)⟩ + (L/2)‖y−x‖² to the definition). C_f is affine-invariant and
norm-free.

## Algorithm

```
Input: x⁽⁰⁾ ∈ D
for k = 0, 1, … , K:
    s := argmin_{s∈D} ⟨s, ∇f(x⁽ᵏ⁾)⟩          # linear-minimization oracle
    g := ⟨x⁽ᵏ⁾ − s, ∇f(x⁽ᵏ⁾)⟩               # duality gap (certificate);  stop if g ≤ ε
    γ := 2/(k+2)
    x⁽ᵏ⁺¹⁾ := (1 − γ) x⁽ᵏ⁾ + γ s             # convex combination, stays in D
```

## Convergence theorem and proof

**Improvement lemma.** For x⁺ = x + γ(s − x), the definition of C_f gives
f(x⁺) ≤ f(x) + γ⟨s − x, ∇f(x)⟩ + (γ²/2)C_f, and ⟨s − x, ∇f(x)⟩ = −g(x), so

  f(x⁺) ≤ f(x) − γ g(x) + (γ²/2) C_f.

**Theorem 1 (primal convergence).** For each k ≥ 1 the iterates of the algorithm satisfy

  f(x⁽ᵏ⁾) − f(x*) ≤ 2C_f/(k+2).

*Proof.* Let h(x) = f(x) − f(x*) and use g(x) ≥ h(x) in the lemma:
h(x⁺) ≤ (1 − γ)h(x) + (γ²/2)C_f. Write C := C_f/2 (so 4C = 2C_f) and prove h(x⁽ᵏ⁺¹⁾) ≤ 4C/((k+1)+2)
by induction. Base k = 0: γ⁽⁰⁾ = 2/2 = 1, so x⁽¹⁾ = s and h(x⁽¹⁾) ≤ C ≤ 4C/3. Step k ≥ 1 with γ =
2/(k+2): h(x⁽ᵏ⁺¹⁾) ≤ (1 − 2/(k+2))·4C/(k+2) + 4C/(k+2)² = 4C(k+1)/(k+2)² ≤ 4C/(k+3), since
(k+1)(k+3) = k²+4k+3 ≤ k²+4k+4 = (k+2)². Re-indexing gives f(x⁽ᵏ⁾) − f(x*) ≤ 4C/(k+2) = 2C_f/(k+2). ∎

(With approximate linear subproblems of quality δ ≥ 0, the same proof gives 2C_f(1 + δ)/(k+2).)

**Theorem 2 (primal-dual / gap convergence).** Run for K ≥ 2 iterations. Then some iterate x⁽ᵏ̂⁾,
1 ≤ k̂ ≤ K, has

  g(x⁽ᵏ̂⁾) ≤ 2β C_f/(K+2),   β = 27/8 = 3.375.

*Proof.* Let C = 2C_f and D_K = K+2, so Theorem 1 gives h⁽ᵏ⁾ ≤ C/(k+2). Assume for contradiction
that g⁽ᵏ⁾ > βC/D_K for every k in the late block k_min = ⌈μD_K⌉−2, …, K. In this block
μD_K ≤ k+2 ≤ D_K and there are at least (1−μ)D_K steps. The recursion with γ_k = 2/(k+2) gives

  h⁽ᵏ⁺¹⁾ < h⁽ᵏ⁾ − 2βC/D_K² + C/(μ²D_K²)
          = h⁽ᵏ⁾ − (C/D_K²)(2β − 1/μ²).

Summing and using h⁽ᵏ_min⁾ ≤ C/(μD_K),

  h⁽ᴷ⁺¹⁾ < (C/(μD_K)) [1 − (1−μ)(2μβ − 1/μ)].

With μ = 2/3 and β = 27/8, the bracket is 0, so h⁽ᴷ⁺¹⁾ < 0, impossible. Hence some iterate has
g⁽ᵏ̂⁾ ≤ βC/D_K = 2βC_f/(K+2). ∎ (A constant step 2/(K+2) in the second half gives
g ≤ 2C_f/(K+2).)

## Properties

- **Projection-free:** the only oracle is linear minimization over D, often far cheaper than
  projection (simplex: pick a coordinate; trace-norm ball: one top singular-vector pair; Birkhoff
  polytope: Hungarian O(n³); submodular polyhedron: Edmonds greedy O(n log n)).
- **Sparse / low-rank iterates:** one vertex is added per step, so x⁽ᵏ⁾ is a convex combination of
  ≤ k+1 vertices — sparse on ℓ₁/simplex, low-rank on the trace-norm ball. This O(1/ε)-atom rate is
  worst-case optimal: for f(x) = ‖x‖² on Δ_n, any card-≤k point with k < n has f(x) − f(x*) ≥ 1/k − 1/n and gap
  ≥ 2/k, so no one-atom-per-step method beats O(1/k).
- **Affine-invariant and norm-free:** the iterates, the rate, and C_f are unchanged under any
  invertible linear reparameterization of D; there is no metric or preconditioner to choose.

## Python implementation

```python
import numpy as np

def frank_wolfe(grad, lmo, x0, max_iter, tol=1e-8):
    """
    grad(x) -> gradient of f at x.
    lmo(c)  -> argmin_{s in D} <s, c>  (linear-minimization oracle; tie-break to a vertex).
    x0      -> initial feasible point in D (ideally a vertex).
    Returns (x, gap).
    """
    x = np.array(x0, dtype=float)
    gap = np.inf
    for k in range(max_iter):
        g = grad(x)                          # first-order oracle
        s = lmo(g)                            # minimize the LINEAR model over D -> a vertex
        gap = float(np.dot(g, x - s))         # duality gap g(x) = <x - s, grad f(x)>
        if gap <= tol:                        # gap >= f(x) - f(x*): certified epsilon-optimal
            break
        gamma = 2.0 / (k + 2.0)               # open-loop step; gamma_0 = 1 jumps to first vertex
        x = (1.0 - gamma) * x + gamma * s     # convex combination -> feasible by construction
    return x, gap

# Example: minimize 0.5 * ||x - b||^2 over the unit simplex.
# Gradient is x - b; the simplex LMO is the basis vector at the smallest gradient coordinate.
def lmo_simplex(c):
    s = np.zeros_like(c)
    s[np.argmin(c)] = 1.0
    return s

def solve_simplex_least_squares(b, max_iter=1000, tol=1e-10):
    b = np.asarray(b, dtype=float)
    x0 = lmo_simplex(np.zeros_like(b))         # start at an arbitrary simplex vertex
    return frank_wolfe(lambda x: x - b, lmo_simplex, x0, max_iter, tol)
```
