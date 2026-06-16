# T-RHG (Truncated Reverse-mode Hyper-Gradient / K-RMD), distilled

T-RHG estimates the hypergradient of a bilevel problem by back-propagating through only the **last `K` of
`T`** inner gradient-descent steps. It runs the inner optimizer forward for the full horizon `T` (so the
inner solution `ŵ*` is good), but differentiates through only the last `K` transitions, storing the consecutive
state window `w_{T-K}, …, w_T`. This is the truncated form of Reverse-mode Hyper-Gradient (RHG): full RHG is the `K = T`
special case. The estimator is `h_{T-K}`; its bias decays exponentially in `K`, so a small `K` gives a
near-exact gradient at memory `O(MK)` instead of RHG's `O(MT)`.

## Problem it solves

Bilevel optimization `min_λ F(λ) = E_S[f_S(ŵ*(λ), λ)]` s.t. `ŵ*(λ)` = output of `T` steps of an iterative
inner optimizer on `g(w, λ)`, with both `λ ∈ R^N` (hyperparameters) and `w ∈ R^M` (parameters)
high-dimensional. The needed hypergradient is `d_λ f = ∇_λ f + ∇_λ ŵ*(λ) · ∇_{ŵ*} f`; the hard term is
`∇_λ ŵ*(λ) · ∇_{ŵ*} f`. Exact reverse-mode differentiation through the trajectory costs `O(MT)` memory,
which does not fit for large models and long horizons.

## Key idea

Unrolling the chain rule through the inner dynamical system `w_{t+1} = Ξ_{t+1}(w_t, λ)`, `ŵ* = w_T` gives

```
d_λ f = ∇_λ f + Σ_{t=0}^{T} B_t A_{t+1} ⋯ A_T ∇_{ŵ*} f,
A_t = ∇_{w_{t-1}} Ξ_t = I - γ ∇²_w g(w_{t-1}, λ),   B_t = ∇_λ Ξ_t = -γ ∇_{λ,w} g(w_{t-1}, λ).
```

Each term at index `t` is multiplied by `A_{t+1} ⋯ A_T`. When the last iterates lie in a region where `g`
is `α`-strongly convex and `β`-smooth and `γ ≤ 1/β`, each `‖A_t‖ ≤ 1 - γα < 1` is a contraction, so early
terms (small `t`) are suppressed by `(1 - γα)^{T-t}`. Drop them: keep only the last `K` terms,

```
h_{T-K} = ∇_λ f + Σ_{t=T-K+1}^{T} B_t A_{t+1} ⋯ A_T ∇_{ŵ*} f.
```

Reverse mode produces these terms in order (it sweeps `t` down from `T`), so `h_{T-K}` is the back-prop `h`
accumulator after `K` steps. The implementation stores `K+1` consecutive states because `K` transitions need
their left boundary state plus `w_T`; this is still `O(MK)` memory.

## Properties (each grounded in a derivation)

- **Exponentially small bias.** `e_K = d_λ f - h_{T-K} = (Σ_{t=0}^{T-K} B_t A_{t+1} ⋯ A_{T-K}) · (A_{T-K+1} ⋯ A_T ∇_{ŵ*} f)`.
  The right factor is `≤ (1 - γα)^K ‖∇_{ŵ*} f‖`. Globally strongly convex `g`:
  `‖e_K‖ ≤ ((1 - γα)^K / (γα)) ‖∇_{ŵ*} f‖ M_B` with `M_B = max_t ‖B_t‖`. Only locally strongly convex
  (nonconvex early): `‖e_K‖ ≤ 2^{T-K+1} (1 - γα)^K ‖∇_{ŵ*} f‖ M_B`.

- **Equivalence to implicit differentiation (Neumann/Taylor view).** At the fixed point, `A_∞ = I - γ ∇_{w,w} g`,
  `B_∞ = -γ ∇_{λ,w} g`; since `‖A_∞‖ < 1`, `(∇_{w,w} g)^{-1} = γ Σ_{k≥0} A_∞^k`, so
  `-∇_{λ,w} g (∇_{w,w} g)^{-1} = B_∞ Σ_{k≥0} A_∞^k`. Hence `d_λ f = ∇_λ f + B_∞ Σ_{k≥0} A_∞^k ∇_{ŵ*} f`, and
  `h_{T-K}` captures its first `K` terms — an order-`K` Taylor approximation of the inverse Hessian.
  Bias of `K`-truncation is `O((1 - 1/κ)^K)` vs. `K`-step CG's `O((1 - 1/√κ)^K)` (`κ = β/α`); CG is tighter
  if `w*` is available, but `h_{T-K}` needs only local strong convexity for the bias bound, has finite-run error
  control, and can differentiate the inner optimizer's own hyperparameters.

- **Sufficient descent.** If `g` is globally strongly convex, `∇_λ f = 0`, `g` is `C²`, and `B_t` has full
  column rank, then `h_{T-K}^T d_λ f ≥ c ‖∇_{ŵ*} f‖² = Ω(‖d_λ f‖²)` for `T` large and `γ` small — `-h_{T-K}`
  points downhill, even at `K = 1`. The proof bounds the cross terms via Lipschitz `A_t, B_t` and the linear
  convergence `‖w_t - w*‖ ≤ D e^{-αγ t}`, which makes the variation errors `O(e^{-αγ(T-1)})` plus a small
  `γ(β - α)/(1 - γ(β - α))` term.

- **Convergence.** With `‖h_{T-K} - d_λ f‖ ≤ ε` and outer SGD `λ_{τ+1} = λ_τ - η_τ h_{T-K}`, `η_τ = O(1/√τ)`:
  `E[Σ_τ η_τ ‖∇F(λ_τ)‖² / Σ_τ η_τ] ≤ Õ(ε + (ε² + 1)/√R)`, an `ε`-approximate stationary point with
  `ε = O((1 - γα)^K)`, so `K = O(log 1/ε)` suffices. Adding non-interference
  `∇_λ f^T(d_λ f + h_{T-K} - ∇_λ f) ≥ Ω(‖∇_λ f‖²)`, `C²` `g`, full-rank `B_t`, and a deterministic objective
  gives convergence to an **exact** stationary point for any `K ≥ 1`. Non-interference holds automatically
  when `∇_λ f = 0` (hyperparameter optimization, data hyper-cleaning, regularization learning).

- **Necessity of non-interference.** A scalar counterexample (`min_λ ½(ŵ*)² + φ(λ)`, inner `½(w - λ)²`)
  with `φ` chosen so stationary points have `ŵ* ≠ 0` makes `h_{T-1} = u - v ≠ 0` at every true stationary
  point, so outer SGD never settles — the assumption cannot be dropped.

## Defaults and why

- **`γ ≤ 1/β`** makes `A_t = I - γ∇²g` a contraction, which simultaneously gives linear inner convergence,
  the `(1 - γα)^K` bias decay, and a convergent Neumann series.
- **Full forward horizon `T`, truncated backward depth `K < T`.** `T` drives `ŵ*` into the locally strongly
  convex basin (the regime the bounds require); `K` controls only gradient accuracy. Forward is cheap and
  memoryless; backward is the `O(MK)`-memory cost. A practical setting such as `K = 100 < T = 500` buys a
  deliberate accuracy margin while remaining far below full-trajectory storage.
- **Decaying outer step `η_τ = O(1/√τ)`** is what the biased-SGD telescoping needs to vanish.

## Working code

Faithful to the canonical `hypergrad` reverse routine (`hg.reverse`) and the `K`-RMD outer loop.
`fp_map` is one differentiable inner GD step `Ξ(w, λ) = w - γ ∇_w g(w, λ)`; the outer loop runs `T` of them,
keeps the last `K+1` consecutive states, and back-propagates through those `K` transitions to build `h_{T-K}`.

```python
from collections import deque
import torch
from torch.autograd import grad as torch_grad


def grad_unused_zero(output, inputs, grad_outputs=None, retain_graph=False):
    grads = torch.autograd.grad(output, inputs, grad_outputs=grad_outputs,
                                allow_unused=True, retain_graph=retain_graph)
    return tuple(torch.zeros_like(var) if grad is None else grad
                 for grad, var in zip(grads, inputs))


def update_tensor_grads(hparams, grads):
    for h, g in zip(hparams, grads):
        if h.grad is None:
            h.grad = torch.zeros_like(h)
        h.grad += g


def fp_map(params, hparams, lr_inner, inner_loss_fn):
    """One differentiable inner step Ξ(w, λ) = w - γ ∇_w g(w, λ)."""
    loss = inner_loss_fn(params, hparams)                      # g(w, λ)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    return [p - lr_inner * g for p, g in zip(params, grads)]


def reverse(params_history, hparams, update_map_history, outer_loss, set_grad=True):
    """The hg.reverse recursion, truncated by the history passed to it.

    params_history: consecutive iterates [w_{T-K}, ..., w_T] (first to last).
    update_map_history: K callables, one for each retained transition.
    The backward sweep accumulates h_{t-1} = h_t + B_t alpha_t and
    alpha_{t-1} = A_t alpha_t. Memory is O(MK).
    """
    params_history = [[w.detach().requires_grad_(True) for w in ws]
                      for ws in params_history]

    o_loss = outer_loss(params_history[-1], hparams)
    alphas = grad_unused_zero(o_loss, params_history[-1], retain_graph=True)
    grad_outer_hparams = grad_unused_zero(o_loss, hparams, retain_graph=True)

    grads = [torch.zeros_like(h) for h in hparams]
    K = len(params_history) - 1
    for k in range(-2, -(K + 2), -1):                  # last K transitions, t = T down to T-K+1
        w_mapped = update_map_history[k + 1](params_history[k], hparams)
        # h_{t-1} += B_t alpha_t   (B_t = ∂Ξ/∂λ, applied to alpha via grad_outputs)
        bs = grad_unused_zero(w_mapped, hparams, grad_outputs=alphas, retain_graph=True)
        grads = [g + b for g, b in zip(grads, bs)]
        # alpha_{t-1} = A_t alpha_t   (A_t = ∂Ξ/∂w_{t-1})
        alphas = torch_grad(w_mapped, params_history[k], grad_outputs=alphas)

    grads = [g + v for g, v in zip(grads, grad_outer_hparams)]
    if set_grad:
        update_tensor_grads(hparams, grads)            # write h_{T-K} into λ.grad
    return grads


def bilevel_train(hparams, fresh_inner_params, inner_loss_fn, outer_loss_fn,
                  lr_inner, lr_outer, T, K, num_outer):
    """Outer loop: full T-step forward inner solve, K-step truncated reverse hypergradient."""
    if K > T:
        K = T                                          # K cannot exceed the horizon
    outer_opt = torch.optim.SGD(hparams, lr=lr_outer)  # decaying η in practice

    def step_fn(p, h):
        return fp_map(p, h, lr_inner, inner_loss_fn)

    for _ in range(num_outer):
        params = fresh_inner_params()                  # re-initialize the inner problem
        w = params
        history = deque([w], maxlen=K + 1)             # consecutive tail: w_{T-K}, ..., w_T
        update_maps = []
        for t in range(T):                             # full forward horizon: T GD steps on g
            w = step_fn(w, hparams)
            history.append(w)
            update_maps.append(step_fn)
            if len(update_maps) > K:
                update_maps.pop(0)
        outer_opt.zero_grad()
        reverse(list(history), hparams, update_maps, outer_loss_fn)
        outer_opt.step()                               # one outer update on λ
```

For the MNIST data-hyper-cleaning instance: `λ` is one weight `σ(λ_i)` per training example
(`|λ| = 5000`), `inner_loss_fn` is the weighted cross-entropy `Σ_i σ(λ_i) CE(w; x_i, y_i) + reg‖w‖²`,
`outer_loss_fn` is the clean-validation cross-entropy (so `∇_λ f = 0`, satisfying the non-interference
condition), and `fresh_inner_params` re-initializes the classifier each outer iteration. For the requested
truncated run, use `T = 500`, `K = 100`, `lr_inner = 0.1`, and `lr_outer = 0.001`.
