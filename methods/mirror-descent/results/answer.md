# Mirror Descent

## Problem

Minimize a convex, L_f-Lipschitz (possibly nonsmooth) function f over a closed convex set X ⊂ R^n,
accessing f only through a subgradient oracle, with iteration count almost independent of the dimension
n. Ordinary projected subgradient descent solves this but measures everything in the Euclidean norm; on
a non-Euclidean domain such as the probability simplex Δ_n this is the wrong geometry and injects a
hidden √n into the rate.

## Key idea

Projected subgradient descent is a linearized-proximal step whose penalty is the squared Euclidean
distance:
> x_{k+1} = argmin_{x∈X} { ⟨g_k, x⟩ + (1/2t_k)‖x − x_k‖_2² } = Π_X(x_k − t_k g_k).

That penalty is a modeling choice. Replace it with the Bregman divergence of a strongly convex potential
("mirror map") ψ, which encodes the geometry of X:
> B_ψ(x, y) = ψ(x) − ψ(y) − ⟨∇ψ(y), x − y⟩,   B_{½‖·‖²}(x,y) = ½‖x − y‖_2².

Because a subgradient is a dual object (a linear form), the additive step x − η g only type-checks in a
Euclidean geometry. The mirror map ∇ψ provides a primal→dual identification, with the conjugate giving
the inverse ∇ψ* = (∂ψ)^{-1}. The update reflects to the dual, steps, and reflects back, then
Bregman-projects onto X.

## The mirror-descent update

Choose a norm ‖·‖ and a potential ψ that is σ-strongly convex w.r.t. ‖·‖ on X. Then, with stepsize t_k:

> Mirror to dual:    ∇ψ(x̃_{k+1}) = ∇ψ(x_k) − t_k g_k,   g_k ∈ ∂f(x_k)
> Bregman project:   x_{k+1} = Π_X^ψ(x̃_{k+1}) = argmin_{x∈X} B_ψ(x, x̃_{k+1})

equivalently, the implicit/proximal form

> x_{k+1} = argmin_{x∈X} { t_k ⟨g_k, x⟩ + B_ψ(x, x_k) }.

For ψ = ½‖·‖_2² the mirror is the identity and this is exactly projected subgradient descent.

## Regret / efficiency bound

The proof uses convexity of f, the step's optimality condition, the Bregman three-point identity
> B_ψ(c, a) + B_ψ(a, b) − B_ψ(c, b) = ⟨∇ψ(b) − ∇ψ(a), c − a⟩,
to telescope B_ψ(x*, ·), and σ-strong convexity to bound the leftover term by t_k²‖g_k‖_*²/(2σ) (dual
norm). Summing over k = 1..s:

> Σ_{s=1}^k t_s (f(x_s) − f*) ≤ B_ψ(x*, x_1) + (2σ)^{-1} Σ_{s=1}^k t_s² ‖g_s‖_*²,

hence

> min_{1≤s≤k} f(x^s) − min_X f ≤ [ B_ψ(x*, x_1) + (2σ)^{-1} Σ_{s=1}^k t_s² ‖g_s‖_*² ] / Σ_{s=1}^k t_s.

With ‖g_s‖_* ≤ L_f and a known k-step horizon, choose the constant stepsize
t_s = √(2σ B_ψ(x*,x_1)) / (L_f √k) for s = 1,...,k. Then

> min_{1≤s≤k} f(x^s) − min_X f ≤ L_f · √( 2 B_ψ(x*, x_1) / σ ) · k^{-1/2}.

(Online form, constant step η: Regret_T ≤ B_ψ(x*,x_1)/η + η Σ_t ‖g_t‖_*²/(2σ); η = R/L_f · √(2σ/T)
gives Regret_T ≤ R L_f √(2T/σ), or average regret Regret_T/T ≤ R L_f √(2/(σT)), with R² the Bregman
radius.) The geometry enters only through the dual-norm gradient size ‖g‖_* and the Bregman radius
B_ψ(x*,x_1)/σ; both are tuned via (ψ, ‖·‖).

## Special cases

**Gradient descent (Ball setup): ψ = ½‖·‖_2², ‖·‖ = ℓ_2.** σ = 1, ∇ψ = id, B_ψ = ½‖·‖_2²,
the back-map is the Euclidean projection. Update y_{k+1} = x_k − t_k g_k, x_{k+1} = Π_X(y_{k+1});
rate L_f Diam(X)/√k — the classical Euclidean bound.

**Multiplicative weights (Simplex setup): ψ_e(x) = Σ_j x_j ln x_j on Δ_n, ‖·‖ = ℓ_1.**
∇ψ_e(x)_j = 1 + ln x_j, so the dual step is additive in log-space and the primal update is multiplicative;
B_{ψ_e} = KL divergence, and the Bregman projection onto Δ_n is ℓ_1 renormalization:

> x_{k+1,j} = x_{k,j} e^{−t_k g_{k,j}} / Σ_i x_{k,i} e^{−t_k g_{k,i}}   (Hedge / exponentiated gradient).

ψ_e is 1-strongly convex w.r.t. ‖·‖_1 (Pinsker: Σ_j (x_j−y_j) ln(x_j/y_j) ≥ ‖x−y‖_1²), so σ = 1 and
‖·‖_* = ℓ_∞. From x_1 = (1/n,…,1/n): B_{ψ_e}(x*, x_1) = ln n − H(x*) ≤ ln n. Hence

> min_{1≤s≤k} f(x^s) − min_Δ f ≤ √(2 ln n) · ‖f'‖_∞ · k^{-1/2},

using t_s = √(2 ln n)/(‖f'‖_∞√k) over the k-step run.

Versus √n · ‖f'‖_∞ / √k for Euclidean projected subgradient on Δ_n — an exponentially better dependence
on dimension, because the gradient is measured in ℓ_∞ (where it is O(1)) and the domain in entropy
(radius ln n, not √n).

## Implementation

```python
import numpy as np

# Mirror descent:  x_{k+1} = argmin_{x in X} { t_k <g_k, x> + B_psi(x, x_k) }
# = mirror to dual (grad psi) -> additive gradient step in dual -> mirror back (grad psi*)
#   -> Bregman projection onto X.

def gradient_descent_step(x, g, t, project_euclidean):
    """Euclidean mirror psi = 1/2||x||_2^2: flat mirror, Bregman proj = Euclidean projection."""
    return project_euclidean(x - t * g)

def entropy_mirror_step(x, g, t):
    """Entropy mirror psi_e on the simplex: multiplicative update + l1 renormalization."""
    y = x * np.exp(-t * g)              # x_j * exp(-t g_j)   (mirror back via grad psi*_e = exp)
    # x_{k+1,j} = x_{k,j} exp(-t g_j) / sum_i x_{k,i} exp(-t g_i)
    return y / y.sum()                  # KL/Bregman projection onto Delta_n = renormalization

def mirror_descent(f_oracle, mirror_step, x1, num_iters, L_f, bregman_radius, sigma=1.0):
    """Horizon-tuned stepsize for the stated k-step efficiency bound."""
    t = np.sqrt(2.0 * sigma * bregman_radius) / (L_f * np.sqrt(num_iters))
    x, best_x, best_val = x1, x1, f_oracle.value(x1)
    for k in range(1, num_iters + 1):
        g = f_oracle.subgrad(x)
        x = mirror_step(x, g, t)
        v = f_oracle.value(x)
        if v < best_val:
            best_x, best_val = x, v
    return best_x

# Simplex example: sigma=1, bregman_radius <= ln(n), ||g||_* = ||g||_inf
#   => min_s f(x^s) - f* <= sqrt(2 ln n) ||f'||_inf / sqrt(k).
```
