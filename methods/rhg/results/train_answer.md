Many learning problems are nested: there is an inner training procedure that, given hyperparameters $\lambda$, produces weights by minimizing a training loss $J(w,\lambda)$, and an outer validation error $E$ that I want small at the *result* of that inner minimization. The thing I actually want to minimize is $f(\lambda) = E(s_T(\lambda))$, where $s_T$ is the state the inner optimizer outputs after $T$ steps. If $\lambda$ were two or three numbers I would grid-search it, and the standard machinery — random search, Bayesian optimization, sequential model-based optimization — would be entirely adequate. But the cases I care about have $\lambda$ enormous: one weight per training example so I can down-weight the mislabeled ones, a whole dense task-interaction matrix, a per-parameter regularizer, a full learning-rate schedule. Black-box search dies here, because each completed training run reveals essentially one scalar, the validation loss, so the number of runs needed explodes with the dimension of $\lambda$ and these methods stall at a few hundred coordinates. The only escape is the gradient $\nabla f(\lambda)$: with it I can move all million coordinates at once. So the whole problem collapses to one question — how do I compute the gradient of a validation error with respect to $\lambda$ when $s_T(\lambda)$ is not a formula but the output of an iterative optimizer, and $\lambda$ may be huge?

The prior gradient-based attempts each leave something on the table. Differentiating a fixed number of optimization steps is the right instinct — differentiate the approximation you actually computed rather than waiting for an optimum — but the precise organization of the backward computation, what is stored and how cost scales, is left open. Reversing an exactly-reversed training run achieves cost independent of the number of hyperparameters, but it is welded to a specific invertible dynamics (momentum with decay $0 < \gamma < 1$) and needs fragile bit-exact reversal machinery to undo the finite-precision loss from multiplying the velocity by $\gamma$ each step; it does not extend to an arbitrary optimizer step. Implicit differentiation through the stationarity condition $\nabla_w J(w^*,\lambda)=0$ is elegant and needs no trajectory, but it presupposes you reach an exact, well-conditioned minimizer, demands a strong-convexity-inducing regularizer and a Hessian linear solve, and — decisively — is blind to any hyperparameter that controls the inner optimizer itself, since a learning rate or momentum factor leaves no fingerprint on $\nabla_w J(w^*,\lambda)=0$.

I propose RHG, Reverse-mode Hyper-Gradient: differentiate the *procedure you actually ran* by treating the inner optimizer as a chain of differentiable update steps and back-propagating through it. Write the inner state as $s_t$ — for plain gradient descent just the weights, for momentum the weights plus velocity — evolving by a smooth map $s_t = \Phi_t(s_{t-1},\lambda)$ for $t=1,\dots,T$, where the subscript on $\Phi_t$ lets it depend on minibatch $t$ and $\lambda$ is baked inside. Then $f(\lambda) = E(s_T(\lambda))$ is a composition of $T$ known smooth maps ending in a known scalar, exactly the object the chain rule chews through. Differentiating the recursion totally in $\lambda$, with the per-step Jacobians $A_t = \partial \Phi_t/\partial s_{t-1}$ ($d\times d$) and $B_t = \partial \Phi_t/\partial \lambda$ ($d\times m$), the total derivative $Z_t = ds_t/d\lambda$ obeys $Z_t = A_t Z_{t-1} + B_t$ with $Z_0 = 0$, and unrolling gives the exact hypergradient

$$\nabla f(\lambda) = \nabla E(s_T)\cdot \sum_{t=1}^{T}\Big(\prod_{s=t+1}^{T} A_s\Big) B_t .$$

This formula is the whole game, and the difficulty is purely computational: $A_t$ is $d\times d$ with $d$ in the millions, so it must never be materialized. The arithmetic can be ordered from either end. Contracting from the right means carrying $Z_t$ forward alongside training — beautiful $O(g)$ memory constant in $T$, but each step needs $m$ Jacobian-vector products to update the $d\times m$ matrix $Z$, for $O(T\cdot m\cdot g)$ time; fine when $m$ is a handful, fatal when $m$ is the dimension of my whole problem. So I flip it. The thing I ultimately want, $\nabla E(s_T)\cdot Z_T$, is a single $1\times m$ gradient, and a vector-times-Jacobian product costs $O(g)$ in reverse mode. Pushing the *row vector* $\nabla E(s_T)$ leftward through the chain of $A_t$'s, I only ever carry a $1\times d$ adjoint and the factor of $m$ vanishes. Define $\alpha_t = \nabla E(s_T)\cdot A_T A_{T-1}\cdots A_{t+1}$; then $\nabla f = \sum_t \alpha_t B_t$, and the $\alpha$'s satisfy the backward recursion $\alpha_T = \nabla E(s_T)$, $\alpha_t = \alpha_{t+1} A_{t+1}$, giving the hypergradient in $O(T\cdot g)$ with no $m$-penalty at all.

I want to trust this backward recursion rather than merely guess its grouping, so I derive it as the adjoint of a constrained problem — the same way back-propagation itself is derived. Cast it as $\min_{\lambda, s_1,\dots,s_T} E(s_T)$ subject to $s_t = \Phi_t(s_{t-1},\lambda)$, attach a row of multipliers $\alpha_t$ to each constraint, and form

$$L = E(s_T) + \sum_{t=1}^{T} \alpha_t\big(\Phi_t(s_{t-1},\lambda) - s_t\big).$$

Stationarity in $s_T$ gives $\partial L/\partial s_T = \nabla E(s_T) - \alpha_T = 0$, so $\alpha_T = \nabla E(s_T)$; for an interior $t$, $s_t$ appears both as $-\alpha_t s_t$ and inside $\Phi_{t+1}(s_t,\lambda)$, so $\partial L/\partial s_t = \alpha_{t+1} A_{t+1} - \alpha_t = 0$ recovers exactly $\alpha_t = \alpha_{t+1}A_{t+1}$. At a feasible point $\partial L/\partial\lambda = \sum_t \alpha_t B_t = \nabla f$. The two derivations compute the identical gradient, contracted from opposite ends — the consistency check I wanted. The Lagrangian view does real work: it identifies $\alpha_t$ as the adjoint state, the sensitivity of the final error to a perturbation of $s_t$; it hands over the recursion without guessing; and because it is a constrained formulation, constraints on $\lambda$ — a box, an $L_1$ budget, a symmetry — slot in directly. This is structurally back-propagation through time over the optimizer's own steps: a forward sweep computes and keeps the states $s_0,\dots,s_T$, a backward sweep carries the adjoint from the end to the beginning. Every backward operation, $\alpha A$ and $\alpha B$, is a transposed-Jacobian-vector product that reverse-mode AD computes in one step's time without forming any matrix.

The price is memory. The backward pass walks $t$ from $T$ down to $1$ and at each step needs $s_{t-1}$, the state going *into* step $t$, to recompute the local Jacobians, so the whole trajectory must be kept: $O(T\cdot d)$ space, the gigabyte-times-$T$ wall everyone hits. I refuse to pay for this with the fragile reverse-the-dynamics trick, which needs $\Phi_t$ invertible and the bit-bookkeeping; I want the clean, exact, optimizer-agnostic computation. So I truncate instead. Run the *full* $T$-step forward inner solve, so the final weights are properly converged, but in the backward pass walk back only $K$ transitions and stop, storing only the $K+1$ suffix iterates. The truncated hypergradient is $h_{T-K} = \partial E/\partial\lambda + \sum_{t=T-K+1}^{T}\alpha_t B_t$. The reason this costs almost nothing is contraction. If the inner map is gradient descent, $\Phi(w,\lambda) = w - \gamma\nabla_w J(w,\lambda)$, on a $\beta$-smooth, $\alpha$-strongly-convex objective with $\gamma \le 1/\beta$, then $A_t = I - \gamma\nabla^2_w J$ has every eigenvalue in $[0, 1-\gamma\alpha]$, so $\|A_t\| \le 1 - \gamma\alpha < 1$. The $t$-th term of the full sum carries $\prod_{s=t+1}^T A_s$, bounded by $(1-\gamma\alpha)^{T-t}$; truncation drops exactly the deepest terms, so the bias decays geometrically,

$$\|h_{T-K} - \nabla f\| \le \frac{(1-\gamma\alpha)^K}{\gamma\alpha}\,\|\nabla_w E(s_T)\|\cdot \max_t \|B_t\|.$$

Thus $K = O(\log 1/\epsilon)$ suffices — a *constant* number of stored iterates, independent of $T$. I run the inner optimizer as long as I like for good weights, but pay only $K$-worth of memory for the gradient, requiring only that the inner map contracts, which strong convexity already gives. With a decaying outer step $\eta_\tau = O(1/\sqrt{\tau})$, truncated RHG converges on-average to an $\epsilon$-stationary point; and in the non-interference case where $E$ depends on $\lambda$ only through the weights ($\partial E/\partial\lambda = 0$, true for data hyper-cleaning), $-h_{T-K}$ is a genuine descent direction and even $K=1$ can reach an exact stationary point.

There is a satisfying reconciliation with the implicit-differentiation route I set aside. As the iterates converge, $A_t \to A_\infty = I - \gamma\nabla^2_w J(w^*)$ and $B_t \to B_\infty = -\gamma\nabla_{w,\lambda}J(w^*)$, and the full backward sum becomes $\nabla_{w^*}E\sum_{k=0}^\infty A_\infty^k B_\infty$. Since $\|A_\infty\| < 1$, the Neumann series $\sum_k A_\infty^k = (I - A_\infty)^{-1} = (\gamma\nabla^2_w J)^{-1}$, so the sum equals $-\nabla_{w^*}E\,(\nabla^2_w J)^{-1}\nabla_{w,\lambda}J$ — exactly the implicit-function hypergradient, the $\gamma$'s cancelling. RHG is the trajectory-side view of implicit differentiation, and the $K$-step truncation is precisely the first $K$ terms of that Neumann series, with $K$ the order of the inverse-Hessian approximation. The unrolled hypergradient also converges to the true one geometrically in the number of inner steps. Unlike implicit differentiation, RHG needs no inner optimum, no strong convexity, and no Hessian solve, and it *can* differentiate hyperparameters that govern the inner optimizer itself, because those literally appear inside $\Phi_t$.

In code, the abstract $A_t, B_t$ become autograd operations. The inner update map is one gradient-descent step on the $\lambda$-weighted training loss, built with `create_graph=True` so the step is itself a graph node and autograd can take $\alpha A$ and $\alpha B$ as vector-Jacobian products through it; for data hyper-cleaning each example $i$ is weighted by $\mathrm{sigmoid}(x_i)$ with $x$ the hyperparameter vector. The `reverse` routine consumes the stored suffix of iterates (first to last) and the per-step maps, sets $\alpha \leftarrow \nabla_w E(s_T)$, and in the backward loop accumulates $g \mathrel{+}= \alpha B_{k+1}$ (a vjp against `hparams`) and pushes $\alpha \leftarrow \alpha A_{k+1}$ (a vjp against the input iterate), neither forming a matrix; afterward it adds the direct $\partial E/\partial\lambda$ term and writes into `hparams.grad`. Passing only $K+1$ suffix iterates gives the truncated version for free, and `K = T` recovers the exact unrolled hypergradient.

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
