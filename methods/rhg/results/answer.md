# RHG (Reverse-mode Hyper-Gradient), distilled

RHG computes the gradient of a validation error with respect to (possibly very
high-dimensional) hyperparameters by treating the inner training procedure as a chain of
differentiable update steps and back-propagating through it — reverse-mode / iterative
differentiation of the unrolled optimizer. It is the reverse-mode dual of forward-mode
hypergradient computation, is structurally back-propagation through time over the optimizer's
steps, and generalizes earlier reverse-mode hyperparameter optimization without requiring the
inner dynamics to be invertible. Its truncated form, T-RHG (back-propagate only through the last
`K` of `T` inner steps), trades a geometrically small bias for `K/T` of the memory and compute.

## Problem it solves

Minimize an outer objective at the output of an inner optimizer,

```
min_lambda  f(lambda) = E(s_T(lambda)),   s_t = Phi_t(s_{t-1}, lambda),  t = 1, ..., T,
```

where `Phi_t` is one step of the inner optimizer (SGD / momentum / Adam) and `lambda` are the
hyperparameters (regularization weights, learning rates, per-example weights, a task-interaction
matrix, ...). The challenge is computing `df/dlambda` when `s_T(lambda)` is not closed-form and
`lambda` may have thousands or millions of coordinates.

## Key idea

By the chain rule, with `A_t = ∂Phi_t/∂s_{t-1}` (`d×d`) and `B_t = ∂Phi_t/∂lambda` (`d×m`),

```
∇f(lambda) = ∇E(s_T) · Σ_{t=1}^T ( Π_{s=t+1}^T A_s ) B_t.
```

Evaluate this product **from the left**, contracting the row vector `∇E(s_T)` backward through
the `A_t`'s, so only a `1×d` adjoint is ever carried (never the `d×d` or `d×m` matrices). This is
the reverse mode. Equivalently, it is the adjoint of the constrained problem `min E(s_T)` s.t.
`s_t = Phi_t(s_{t-1}, lambda)`: with Lagrangian

```
L = E(s_T) + Σ_t alpha_t (Phi_t(s_{t-1}, lambda) − s_t),
```

stationarity in `s_t` gives the backward (adjoint) recursion `alpha_T = ∇E(s_T)`,
`alpha_t = alpha_{t+1} A_{t+1}`, and `∂L/∂lambda = Σ_t alpha_t B_t = ∇f`. The Lagrangian view
makes the multipliers the adjoint state, hands over the recursion, and lets constraints on
`lambda` slot in directly.

Every backward operation is a transposed-Jacobian-vector product (`alpha A`, `alpha B`),
computed by reverse-mode AD in one step's time without forming any matrix.

## Final algorithm

```
Forward:   s_0 given; for t = 1..T:  s_t = Phi_t(s_{t-1}, lambda)        # keep iterates
Backward:  alpha <- ∇E(s_T);  g <- 0
           for j = T downto 1:
               g <- g + alpha · B_j                # = alpha_j · ∂Phi_j/∂lambda      (vjp)
               if j > 1:
                   alpha <- alpha · A_j            # alpha_{j-1} = alpha_j · ∂Phi_j/∂s_{j-1}
           ∇f(lambda) = g + ∂E/∂lambda             # last term often 0
```

- Time `O(T·g)` (no factor of `m`); space `O(T·d)` (the trajectory). The complementary
  **forward mode** carries `Z_t = A_t Z_{t-1} + B_t` (`d×m`) alongside training at time
  `O(T·m·g)`, space `O(g)` — preferable only when `m ≪ d`.

## Truncation (T-RHG)

Run the full `T`-step forward inner solve, but back-propagate through only the last `K`
transitions, storing only the `K+1` suffix iterates:

```
h_{T−K} = ∂E/∂lambda + Σ_{t=T−K+1}^T alpha_t · B_t.
```

If the inner map is gradient descent on a `beta`-smooth, `alpha`-strongly-convex objective with
step `gamma ≤ 1/beta`, then `A_t = I − gamma ∇²_w J` has `‖A_t‖ ≤ 1 − gamma·alpha < 1` (a
contraction), and the truncation bias decays geometrically in `K`:

```
‖h_{T−K} − ∇f‖ ≤ ( (1 − gamma·alpha)^K / (gamma·alpha) ) · ‖∇_w E(s_T)‖ · max_t ‖B_t‖.
```

So `K = O(log 1/epsilon)` suffices — a constant number of stored suffix iterates, independent of `T`.
With a decaying outer step `eta_tau = O(1/sqrt(tau))`, truncated RHG converges on-average to an
`epsilon`-stationary point whenever the gradient-estimate error is bounded by `epsilon`; the bound
above makes that error shrink as `O((1 − gamma·alpha)^K)`. Moreover, when the outer objective
depends on `lambda` only through the weights (`∂E/∂lambda = 0`, the non-interference case — true
for data hyper-cleaning), `−h_{T−K}` is a genuine descent direction and even `K = 1` can reach an
exact stationary point.

## Relationship to implicit differentiation

As the inner iterates converge (`A_t -> A_inf = I − gamma ∇²_w J(w*)`,
`B_t -> B_inf = −gamma ∇_{w,lambda}J(w*)`), the row-form full backward sum becomes
`∇_{w*}E Σ_{k=0}^∞ A_inf^k B_inf`. Since `‖A_inf‖ < 1`, the Neumann series
`Σ_k A_inf^k = (I − A_inf)^{-1} = (gamma ∇²_w J)^{-1}`, so the sum equals
`−∇_{w*}E (∇_{w,w}J)^{-1} ∇_{w,lambda}J`, equivalently the column-gradient implicit-function
hypergradient `−∇_{lambda,w}J (∇_{w,w}J)^{-1} ∇_{w*}E`. RHG is thus the trajectory-side view of
implicit differentiation, and the `K`-step truncation is exactly the first `K` terms of that
Neumann series. The unrolled-trajectory hypergradient `∇f_t` also converges to the true `∇f`
geometrically in the number of inner steps: `‖∇f_t − ∇f‖ ≤ (c_1 + c_2 t/q + c_3) q^t` with
`q < 1` the contraction factor.

Unlike implicit differentiation, RHG needs no inner optimum, no strong convexity, and no Hessian
linear solve, and it *can* differentiate hyperparameters that control the inner optimizer itself
(learning rate, momentum), because those appear inside `Phi_t`.

## Working code

Faithful to the canonical `hypergrad` reverse-mode implementation. `reverse` consumes the stored
inner iterates (`params_history`, first to last) and the per-step update maps, and writes the
hypergradient into `hparams.grad`. Passing only a suffix of `K+1` consecutive iterates gives
T-RHG for free.

```python
import torch
import torch.nn.functional as F


def reverse(params_history, hparams, update_map_history, outer_loss, set_grad=True):
    """Reverse-mode hypergradient by recomputing and back-propagating through each stored
    inner transition. Truncated RHG = pass K+1 consecutive suffix iterates and K update maps.

        params_history     : inner iterates [w_{T-K}, ..., w_T]  (first to last)
        hparams            : outer variables lambda (each requires_grad=True)
        update_map_history : inner update maps Phi applied along that suffix
        outer_loss         : (params, hparams) -> validation scalar
    """
    params_history = [[w.detach().requires_grad_(True) for w in ws] for ws in params_history]

    o_loss = outer_loss(params_history[-1], hparams)
    grad_outer_w, grad_outer_hparams = _outer_grads(o_loss, params_history[-1], hparams)

    alphas = grad_outer_w                                   # alpha_T = nabla_w E(s_T)
    grads = [torch.zeros_like(h) for h in hparams]
    K = len(params_history) - 1
    for k in range(-2, -(K + 2), -1):                       # walk the stored steps backward
        w_mapped = update_map_history[k + 1](params_history[k], hparams)   # re-apply Phi_{k+1}
        bs = _grad_or_zero(w_mapped, hparams, alphas, retain=True)         # alpha * B  (vjp)
        grads = [g + b for g, b in zip(grads, bs)]
        alphas = torch.autograd.grad(w_mapped, params_history[k], grad_outputs=alphas)  # alpha * A

    grads = [g + v for g, v in zip(grads, grad_outer_hparams)]             # direct dE/dlambda
    if set_grad:
        for h, g in zip(hparams, grads):
            h.grad = (torch.zeros_like(h) if h.grad is None else h.grad) + g
    return grads


def _outer_grads(o_loss, params, hparams):
    gw = _grad_or_zero(o_loss, params, None, retain=True)
    gh = _grad_or_zero(o_loss, hparams, None, retain=True)
    return gw, gh


def _grad_or_zero(output, inputs, grad_outputs, retain=False):
    gs = torch.autograd.grad(output, inputs, grad_outputs=grad_outputs,
                             allow_unused=True, retain_graph=retain)
    return [torch.zeros_like(x) if g is None else g for g, x in zip(gs, inputs)]


# --- bilevel wiring: data hyper-cleaning (per-example weights x as the hyperparameter) ---

def inner_update(params, hparams, lr_inner, data, dirty_target, reg=0.0):
    """One DIFFERENTIABLE GD step on the lambda-weighted training loss (the map Phi_t).
    create_graph=True makes the step differentiable, so autograd can form A and B as vjps."""
    x = hparams[0]
    per_example = F.cross_entropy(forward(params, data), dirty_target, reduction='none')
    loss = (torch.sigmoid(x) * per_example).mean() + reg * sum((p * p).sum() for p in params)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    return [p - lr_inner * g for p, g in zip(params, grads)]


def outer_loss(params, hparams, val_data, val_target):          # E(s_T): validation loss
    return F.cross_entropy(forward(params, val_data), val_target)


def hyperopt_loop(hparams, T, K, lr_inner, hyper_lr, num_outer_steps, project,
                  train_data, dirty_target, val_data, val_target):
    hyper_opt = torch.optim.Adam(hparams, lr=hyper_lr)
    phi = lambda ws, hp: inner_update(ws, hp, lr_inner, train_data, dirty_target)
    e   = lambda ws, hp: outer_loss(ws, hp, val_data, val_target)
    K = min(K, T)
    for _ in range(num_outer_steps):
        params = fresh_params()
        history = [params]
        for t in range(T):                                 # full forward inner solve
            params = phi(params, hparams)
            history.append(params)
            if len(history) > K + 1:                        # keep K transitions = K+1 iterates
                history.pop(0)
        hyper_opt.zero_grad()
        reverse(history, hparams, [phi] * (len(history) - 1), e, set_grad=True)
        hyper_opt.step()
        with torch.no_grad():
            for h in hparams:
                h.copy_(project(h))                        # project lambda onto its constraints
    return hparams
```

`K = T` is full RHG (the exact hypergradient of the `T`-step procedure); `K < T` is T-RHG.
