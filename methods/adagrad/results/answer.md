# AdaGrad, distilled

AdaGrad (Adaptive Subgradient method) is an online/stochastic subgradient method that
gives every coordinate its own effective step size, derived by treating the proximal
function of mirror descent as something to **learn from the data** rather than fix in
advance. Each coordinate's step is scaled inversely by the square root of its accumulated
sum of squared gradients, so frequently active coordinates take small steps and rarely
active ones take large steps. The choice is not a heuristic: it is the diagonal
preconditioner that minimizes the mirror-descent regret bound, and the online version is
provably within a factor of two of the best such preconditioner chosen in hindsight.

## Problem it solves

Online convex optimization (and its stochastic-optimization counterpart): predict `x_t`,
suffer convex `f_t`, read a subgradient `g_t ∈ ∂f_t(x_t)`, minimize regret
`R(T) = Σ_t f_t(x_t) − inf_x Σ_t f_t(x)`. On high-dimensional **sparse, heavy-tailed**
data (text, bag-of-words), a single global step size is structurally wrong — rare features
are informative yet get the same rate as constant ones. We want a principled per-coordinate
rate with a regret guarantee competitive with the best preconditioner in hindsight, at a
cost close to vanilla subgradient descent.

## Key idea

The mirror-descent regret for a fixed proximal function `ψ` is
`R_φ(T) ≤ (1/η) B_ψ(x*, x_1) + (η/2) Σ_t ||g_t||²_{ψ*}` — governed entirely by the
gradients' dual norms, hence by `ψ`. Make `ψ_t = ½⟨x, H_t x⟩` adaptive (and monotone,
`H_{t+1} ⪰ H_t`). The data term becomes `Σ_t ⟨g_t, H_t^{−1} g_t⟩`. Minimizing it over
diagonal `H = diag(s)` with a trace budget,

```
min_s  Σ_t Σ_i g²_{t,i} / s_i   s.t.  s ⪰ 0, ⟨1, s⟩ ≤ c,
```

has Lagrangian solution `s_i ∝ ||g_{1:T,i}||_2` (the accumulated ℓ2 norm of coordinate
`i`'s gradient history) and optimal value `(1/c)(Σ_i ||g_{1:T,i}||_2)²`. Online, use the
**causal** running norm `s_{t,i} = ||g_{1:t,i}||_2 = sqrt(Σ_{τ≤t} g²_{τ,i})`, i.e.
`H_t = δI + diag(G_t)^{1/2}` with `G_t = Σ_{τ≤t} g_τ g_τ^T`.

## Final algorithm (diagonal AdaGrad)

```
G_0 = 0  (per-coordinate accumulator);  x_1 given
for t = 1, 2, ...:
    suffer f_t(x_t), receive g_t ∈ ∂f_t(x_t)
    G_{t,i}   = G_{t-1,i} + g²_{t,i}              # accumulate sum of squared grads
    x_{t+1,i} = x_{t,i} - η · g_{t,i} / ( sqrt(G_{t,i}) + ε )      # (or δI floor)
```

Constrained / composite versions: project in the metric `diag(G_t)^{1/2}`, or for
`φ = λ||x||_1` use the per-coordinate soft-threshold (RDA form)
`x_{t+1,i} = sign(−ḡ_{t,i}) (ηt/H_{t,ii}) [|ḡ_{t,i}| − λ]_+`, which differs from vanilla
RDA's `η√t [|ḡ_{t,i}| − λ]_+` only by the per-coordinate denominator `H_{t,ii}`.

## Why these choices

- **Denominator = accumulated ℓ2 norm `sqrt(Σ g²)`** (not a count, not the current `|g|`,
  not a windowed average): it is the exact minimizer of the hindsight regret problem.
- **Accumulate the full history (no forgetting):** the optimal hindsight denominator is
  the full-history norm, and monotone `H_t` is what lets the divergence terms telescope;
  the resulting `1/√t`-style annealing is the correct schedule for convex regret.
- **Square root:** the doubling lemma and the concavity of `sqrt` (resp. `tr(·^{1/2})`)
  produce it; it is the conservative preconditioner (root of the outer-product matrix),
  stable on noisy subgradients.
- **`ε` / `δI` floor:** invertibility when a coordinate has zero accumulated mass
  (`sqrt(0) = 0`); small enough not to perturb active coordinates.
- **Diagonal, not full matrix:** the full matrix `S = c G_T^{1/2}/tr(G_T^{1/2})` is the
  trace-budget optimum (composite mirror-descent regret `√2 · D · tr(G_T^{1/2})` after
  balancing `η`) but `O(d²)` memory and matrix roots are
  infeasible in high dimension; the diagonal is linear time/space and keeps the
  per-coordinate adaptation that drives the win.
- **`η` is one global scalar:** the per-coordinate scaling already lives in `H_t`.

## Guarantee

Doubling lemma (scalar core, by induction + concavity of `sqrt`):
`Σ_{t≤T} a²_t/||a_{1:t}||_2 ≤ 2||a_{1:T}||_2`, hence
`Σ_t ⟨g_t, diag(s_t)^{−1} g_t⟩ ≤ 2 Σ_i ||g_{1:T,i}||_2`. Combined with the convexity bound
and `η = D_∞/√2`,

```
R_φ(T) ≤ √2 · D_∞ · Σ_{i=1}^{d} ||g_{1:T,i}||_2 = √2 · D_∞ · γ_T,
```

where `γ_T = Σ_i ||g_{1:T,i}||_2` is the hindsight optimum. Worst case `O(√T)`. On a
power-law sparse feature model `p_i = min{1, c i^{−α}}`, by Jensen
`E Σ_i ||g_{1:T,i}||_2 ≤ √T Σ_i sqrt(p_i)`, and `Σ_i i^{−α/2} = O(log d)` for `α ≥ 2`, so
the regret is `O((log d)√T)` — exponentially smaller in `d` than isotropic gradient
descent's `O(√(dT))`. The full-matrix version gives the analogous
`R ≤ √2 · D · tr(G_T^{1/2})` for balanced composite mirror descent
via the matrix doubling lemma (`2 tr((B − ν gg^T)^{1/2}) ≤ 2 tr(B^{1/2}) − ν tr(B^{−1/2}gg^T)`,
from concavity of `tr(A^{1/2})`).

## Relation to prior methods

- **Projected online gradient descent (Zinkevich):** the time-scaled isotropic corner
  `H_t = √t I`, giving `x_{t+1,i} = x_{t,i} − (η/√t)g_{t,i}`.
- **Regularized dual averaging (RDA) / primal-dual (Nesterov):** the dual-averaging form
  with the diagonal proximal term — one global `√t` rate replaced by the per-coordinate
  `t/H_{t,ii}`, with the `ℓ1` soft-thresholding (sparse iterates) intact.

## Working code

Filling the per-coordinate `step()` slot of the online-subgradient harness — one
accumulator per parameter block, a square root, and a divide. This is the unconstrained
diagonal AdaGrad (`X = R^d`, no composite `φ`):

```python
import torch


def get_hyperparameters(dim, sparsity, delta):
    # lr is a single global trust knob; the per-coordinate scaling lives in the
    # accumulated-norm denominator. eps floors the denominator (sqrt(0)=0).
    return {"lr": 0.01, "eps": 1e-6}


def init_state(u, v, hyperparameters):
    # Per-coordinate sum-of-squared-gradients accumulator == diag(G_t).
    # Its square root is the optimal-in-hindsight diagonal denominator.
    d = u.shape[0]
    return {
        "t": 0,
        "state_sum_u": torch.zeros(d, dtype=torch.float64),
        "state_sum_v": torch.zeros(d, dtype=torch.float64),
    }


def step(u, v, grad_u, grad_v, state, hyperparameters):
    lr = float(hyperparameters["lr"])
    eps = float(hyperparameters["eps"])
    # state_sum_{t,i} = sum_{tau<=t} g_{tau,i}^2 = ||g_{1:t,i}||_2^2
    state_sum_u = state["state_sum_u"] + grad_u * grad_u
    state_sum_v = state["state_sum_v"] + grad_v * grad_v
    # x_{t+1,i} = x_{t,i} - lr * g_{t,i} / ( sqrt(sum_{tau<=t} g_{tau,i}^2) + eps )
    std_u = torch.sqrt(state_sum_u) + eps
    std_v = torch.sqrt(state_sum_v) + eps
    u_new = u - lr * grad_u / std_u
    v_new = v - lr * grad_v / std_v
    return u_new, v_new, {
        "t": state["t"] + 1,
        "state_sum_u": state_sum_u,
        "state_sum_v": state_sum_v,
    }
```

General-harness form (per-coordinate accumulator, matching the canonical implementation
with `lr_decay = 0`):

```python
import torch


class AdaGrad:
    """Diagonal AdaGrad. Per-coordinate accumulator of squared gradients;
    step = lr * g / (sqrt(accumulator) + eps)."""

    def __init__(self, params, lr=1e-2, eps=1e-10, lr_decay=0.0,
                 initial_accumulator_value=0.0):
        self.params = list(params)
        self.lr, self.eps, self.lr_decay = lr, eps, lr_decay
        self.state = {id(p): {"step": 0,
                              "sum": torch.full_like(p, initial_accumulator_value)}
                      for p in self.params}

    @torch.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            st = self.state[id(p)]
            st["step"] += 1
            clr = self.lr / (1 + (st["step"] - 1) * self.lr_decay)  # optional lr decay
            st["sum"].addcmul_(g, g, value=1)            # sum += g*g
            std = st["sum"].sqrt().add_(self.eps)        # sqrt(sum) + eps
            p.addcdiv_(g, std, value=-clr)               # p -= clr * g / std
```
