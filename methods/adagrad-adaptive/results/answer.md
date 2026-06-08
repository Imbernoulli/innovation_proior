# AdaGrad — adaptive subgradient methods

## The problem

Online / stochastic (sub)gradient learning on high-dimensional, sparse, heavy-tailed data
(text, ranking). A single global learning rate `η_t = η/√t` is structurally wrong: it gives
every coordinate the same schedule, so a rare-but-informative feature is hit with a tiny,
already-decayed step exactly when it finally fires, while common features keep jittering.
Worst-case regret of isotropic online gradient descent, `O(D₂√(Σ_t‖g_t‖₂²))`, is minimax-tight
(Abernethy et al. 2008), so improvement can only come from *adapting to the data geometry*.

## The key idea

Run a proximal / mirror-descent update with a **data-driven, per-coordinate Mahalanobis
metric** instead of a fixed one. Adapt the step *per coordinate* by dividing by the square root
of the accumulated sum of squared gradients: `η / √(Σ_τ g_{τ,i}²)`. This diagonal preconditioner
is not a heuristic — it is the **regret-minimizing diagonal metric**: minimizing the
gradient term of the mirror-descent regret bound over diagonal matrices subject to a trace
budget forces it to be proportional to `√(Σ_t g²)`. The causal running version costs only a
factor 2 in the online gradient sum, giving a regret that adapts to the geometry and can be
logarithmic in dimension on heavy-tailed sparse features instead of isotropic `√d` scaling.

## The update

Mirror-descent / composite form with `ψ_t(x) = ½⟨x, H_t x⟩`, `H_t = δI + diag(s_t)`,
`s_{t,i} = ‖g_{1:t,i}‖₂ = √(Σ_{τ≤t} g_{τ,i}²)`:

```
x_{t+1} = argmin_x { η⟨g_t, x⟩ + η φ(x) + B_{ψ_t}(x, x_t) }.
```

Unconstrained, `φ=0`, `δ=0`, this is the per-coordinate update

```
x_{t+1, i} = x_{t, i} − η · g_{t,i} / √( Σ_{τ≤t} g_{τ,i}² ).
```

A full-matrix variant uses `H_t = δI + G_t^{1/2}`, `G_t = Σ_{τ≤t} g_τ g_τ^T`, the matrix square
root of the gradient second-moment matrix; the diagonal version is its `O(d)` restriction.

## The adaptive regret theorem (diagonal)

**Optimal preconditioner.** For the hindsight problem
`min_s Σ_t Σ_i g_{t,i}²/s_i` s.t. `s⪰0, ⟨1,s⟩≤c`, write `A_i=Σ_t g_{t,i}²`. The Lagrangian
`Σ_i A_i/s_i − ⟨λ,s⟩ + θ(⟨1,s⟩−c)` gives `s_i ∝ √A_i` on coordinates with `A_i>0`, zero mass
on zero-gradient coordinates, and after normalization
`s_i = c·‖g_{1:T,i}‖₂ / Σ_j‖g_{1:T,j}‖₂`. The optimal value is
`(1/c)(Σ_i ‖g_{1:T,i}‖₂)²`, with value zero if all gradients are zero. So the post-hoc-optimal
diagonal metric is `diag(√(Σ_t g²))`.

**Causal factor-2 lemma (Auer–Gentile 2000).** With `s_{t,i} = ‖g_{1:t,i}‖₂`,
`Σ_{t} ⟨g_t, diag(s_t)† g_t⟩ ≤ 2 Σ_i ‖g_{1:T,i}‖₂`. *Proof:* it suffices that for any scalar
sequence, reading `0/0` as zero, `Σ_t a_t²/‖a_{1:t}‖₂ ≤ 2‖a_{1:T}‖₂`. Induction: if
`b_T = ‖a_{1:T}‖₂²` is zero the claim is trivial; otherwise concavity of `√·` gives
`√(b_T − a_T²) ≤ √(b_T) − a_T²/(2√(b_T))`, hence
`2‖a_{1:T-1}‖₂ + a_T²/‖a_{1:T}‖₂ ≤ 2‖a_{1:T}‖₂`. ∎

**Regret (Theorem, composite mirror descent).** With `X` compact, `D_∞ = max_t‖x*−x_t‖_∞`,
and `η = D_∞/√2`,

```
R(T) ≤ √2 · D_∞ · Σ_{i=1}^d ‖g_{1:T,i}‖₂ = √2 · D_∞ · Σ_{i=1}^d √( Σ_{t=1}^T g_{t,i}² ).
```

*Proof sketch.* Mirror-descent regret `≤ (1/η)B_{ψ_1}(x*,x_1) + (1/η)Σ_t[B_{ψ_{t+1}}(x*,x_{t+1})
− B_{ψ_t}(x*,x_{t+1})] + (η/2)Σ_t‖g_t‖²_{ψ_t*}`. The gradient term is bounded by the factor-2
lemma, so after multiplying by `η/2` it contributes at most `ηΣ_i‖g_{1:T,i}‖₂`. Because `s_t`
grows monotonically, each Bregman difference is
`½⟨x*−x_{t+1},diag(s_{t+1}−s_t)(x*−x_{t+1})⟩ ≤ ½D_∞²⟨s_{t+1}−s_t,1⟩`; together with
`B_{ψ_1}(x*,x_1)`, the distance side is at most
`½D_∞²⟨s_T,1⟩ = ½D_∞²Σ_i‖g_{1:T,i}‖₂`. Balancing
`(D_∞²/(2η))Σ_i‖g_{1:T,i}‖₂ + ηΣ_i‖g_{1:T,i}‖₂` gives the bound. ∎

**Optimality and graceful degradation.** Since
`Σ_i‖g_{1:T,i}‖₂ = √( d · inf_{s⪰0,⟨1,s⟩≤d} Σ_t⟨g_t,diag(s)^{-1}g_t⟩ )`, the bound is tied to
the best diagonal metric chosen in hindsight with the trace normalization explicit; `s=1` is
feasible in the infimum, so the expression reduces in the worst case to the usual Euclidean
`√(dΣ_t‖g_t‖₂²)` scaling for an `ℓ∞`-bounded domain. On `p_i = min{1,c i^{-α}}` sparse data,
`E Σ_i‖g_{1:T,i}‖₂ ≤ √T Σ_i√(p_i) = O(√T·log d)` for `α≥2` (Jensen), vs `O(√(dT))` for OGD.

## Full-matrix theorem

`min_S tr(S^{-1}G_T)` s.t. `S⪰0, tr(S)≤c` gives `S = c·G_T^{1/2}/tr(G_T^{1/2})`, value
`tr(G_T^{1/2})²/c` (using a pseudo-inverse/limit if `G_T` is singular). Using concavity of
`A↦tr(A^{1/2})` (matrix-concavity of `A^p`, `0≤p≤1`; first-order inequality
`tr(A^{1/2}) ≤ tr(B^{1/2}) + ½tr(B^{-1/2}(A−B))` with `B=G_t`, `A=G_{t-1}`) gives
`Σ_t⟨g_t,S_t^{†}g_t⟩ ≤ 2 tr(G_T^{1/2})`. For compact-set mirror descent, with
`D_2=max_t‖x*−x_t‖₂` and `η=D_2/√2`, this yields
`R(T) ≤ √2D_2 tr(G_T^{1/2})`; the dual-averaging form gives `2‖x*‖₂ tr(G_T^{1/2})` with
`η=‖x*‖₂`.

## Code

```python
import numpy as np

class AdaGrad:
    """Per-coordinate adaptive subgradient method:
    step = eta / sqrt(accumulated sum of squared gradients)."""

    def __init__(self, d, eta=1.0, eps=1e-10):
        self.eta = eta            # set from the diameter; not coupled to gradient scale
        self.eps = eps            # denominator floor before a coordinate fires
        self.state_sum = np.zeros(d)

    def step(self, x, g):
        self.state_sum += g * g
        return x - self.eta * g / (np.sqrt(self.state_sum) + self.eps)

# Sparse / dense gradient streams both work; for sparse g only the touched
# coordinates' accumulators and weights are updated, keeping per-step cost
# proportional to the support of g_t.
```

This mirrors the canonical optimizer form `state_sum += g²; x -= η·g/(√state_sum + ε)`.
