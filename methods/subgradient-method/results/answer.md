# The subgradient method

## Problem

Minimize a **convex** function `f : R^n → R` that is **not necessarily differentiable**
(e.g. `max_i(a_iᵀx+b_i)`, `‖x‖₁`, a dual function, `λ_max(A(x))`). Gradient descent and second-order
methods need ∇f or ∇²f, which fail at the kinks where the minimizer often lives.

## Key idea

At any point a convex f has a **subgradient** `g ∈ ∂f(x)`: a vector with
`f(z) ≥ f(x) + gᵀ(z−x)` for all z (the slope of a supporting hyperplane to epi f). Step against it:

> `x_{k+1} = x_k − α_k g_k`,  `g_k ∈ ∂f(x_k)`.

A subgradient is **not** a descent direction — `f'(x;−g) = −inf_{h∈∂f(x)} hᵀg` can be positive, so
`f(x_k)` can increase. The right quantity to control is therefore not monotone decrease of the
function value but the **Euclidean distance to a minimizer** `‖x_k − x*‖`. Because the method is not
a descent method, report the best value seen, `f_best^{(k)} = min_{i≤k} f(x_i)`.

Step sizes are chosen by explicit rules rather than line search (which would need f to decrease along
`−g`):
- square-summable-not-summable `Σα_k=∞, Σα_k²<∞` (e.g. `α_k=a/k`) → convergence for any Lipschitz f;
- fixed-budget optimal `α_i=(R/G)/√K` for `i=1,…,K` → the optimal `RG/√K` rate;
- Polyak step `α_k=(f(x_k)−f*)/‖g_k‖²` (f* known), or
  `α_k=(f(x_k)−f_best^{(k)}+γ_k)/‖g_k‖²`,
  `γ_k→0, Σγ_k=∞` (f* estimated) → self-tuning, optimal rate, no horizon needed.

## Convergence theorem and proof

**Assumptions.** f convex with a minimizer x*, f* = f(x*); subgradients bounded `‖g_k‖ ≤ G`
(equivalently f is G-Lipschitz); `R ≥ ‖x_1 − x*‖`.

**Key inequality.** From `x_{k+1}=x_k−α_k g_k`,

```
‖x_{k+1}−x*‖² = ‖x_k−x*‖² − 2α_k g_kᵀ(x_k−x*) + α_k²‖g_k‖².
```

The subgradient inequality at x*, `f* ≥ f(x_k)+g_kᵀ(x*−x_k)`, gives `g_kᵀ(x_k−x*) ≥ f(x_k)−f* ≥ 0`,
hence

```
‖x_{k+1}−x*‖² ≤ ‖x_k−x*‖² − 2α_k(f(x_k)−f*) + α_k²‖g_k‖².        (★)
```

**Summed bound.** Telescoping (★) for i=1..k, using `‖x_{k+1}−x*‖²≥0` and `‖x_1−x*‖≤R`,

```
2 Σ_{i=1}^k α_i(f(x_i)−f*) ≤ R² + Σ_{i=1}^k α_i²‖g_i‖².
```

Since `Σ α_i(f(x_i)−f*) ≥ (Σα_i)·min_{i≤k}(f(x_i)−f*)` and `‖g_i‖≤G`,

```
f_best^{(k)} − f* ≤ ( R² + G² Σ_{i=1}^k α_i² ) / ( 2 Σ_{i=1}^k α_i ).     (B)
```

**Consequences.**
- `Σα_i=∞, Σα_i²<∞`: numerator → `R²+G²Σα²` (finite), denominator → ∞, so `f_best^{(k)} → f*`.
- Constant α: `(B)` → `R²/(2αk) + G²α/2`, converging to within `G²α/2` of f*.

**Optimal fixed step (O(1/√K) rate).** For a fixed horizon K, `(B)` with `‖g‖≤G` is a symmetric
convex function of `(α_1,…,α_K)`, so its minimizer takes all α_i = α, giving `(R²+G²Kα²)/(2Kα)`,
minimized at `α=(R/G)/√K`. Substituting back:

```
f_best^{(K)} − f* ≤ R G / √K.
```

So `K = (RG/ε)²` iterations suffice for ε-accuracy.

**Polyak step.** `α_i=(f(x_i)−f*)/‖g_i‖²` minimizes the RHS of (★) pointwise. Substituting into the
telescoped bound: `Σ_i (f(x_i)−f*)²/‖g_i‖² ≤ R²`, hence with `‖g_i‖≤G`,
`Σ_i (f(x_i)−f*)² ≤ R²G²`, so `f(x_i)→f*` and `f_best^{(k)}−f* ≤ RG/√k`.

## Matching lower bound (optimality)

For the black-box first-order oracle class (each query returns f and one adversarial subgradient;
`x_k ∈ x_0 + span{g_0,…,g_{k-1}}`), the worst function `f_{k+1}(x)=γ·max_{i≤k+1} x^{(i)} + (μ/2)‖x‖²`
reveals only one new coordinate per query, forcing, for any method and any
`0 ≤ k ≤ n−1`, on functions M-Lipschitz on `B(x*,R)`:

```
f(x_k) − f* ≥ M R / (2(2 + √(k+1)))  =  Ω(MR/√k).
```

This matches the `RG/√k` upper bound (M↔G), so **O(1/√k) is optimal** for nonsmooth Lipschitz convex
minimization: no first-order method does better than the subgradient method by more than a constant.

## Minimal implementation

```python
import numpy as np

def subgradient_method(f, subgrad, x0, num_iters, step,
                       f_star=None, G=None, R=None):
    """step in {'sqsum','sqrt','polyak','polyak_est'}."""
    x = np.array(x0, dtype=float)
    f_best, x_best = np.inf, x.copy()
    for k in range(1, num_iters + 1):
        fx = f(x)
        if fx < f_best:                       # not a descent method: keep best-so-far
            f_best, x_best = fx, x.copy()
        g = subgrad(x)                         # any subgradient at x
        if step == 'sqsum':                    # sum alpha=inf, sum alpha^2<inf
            alpha = 1.0 / k
        elif step == 'sqrt':                   # fixed-budget optimal -> RG/sqrt(num_iters)
            alpha = (R / G) / np.sqrt(num_iters)
        elif step == 'polyak':                 # f* known, optimal rate
            alpha = (fx - f_star) / g.dot(g)
        elif step == 'polyak_est':             # f* estimated as f_best - gamma_k
            gamma = 1.0 / k                    # gamma_k -> 0, sum gamma_k=inf
            alpha = (fx - f_best + gamma) / g.dot(g)
        x = x - alpha * g
    return x_best, f_best


def make_piecewise_linear(A, b):
    """f(x) = max_i (a_i^T x + b_i); subgradient = the active row a_j."""
    f = lambda x: np.max(A @ x + b)
    def subgrad(x):
        j = int(np.argmax(A @ x + b))
        return A[j].copy()
    return f, subgrad
```
