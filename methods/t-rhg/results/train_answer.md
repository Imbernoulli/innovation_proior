Many learning problems are nested: I want to minimize an outer objective $F(\lambda) = \mathbb{E}_S[f_S(\hat w^*(\lambda), \lambda)]$ over a high-dimensional hyperparameter $\lambda \in \mathbb{R}^N$, but $\lambda$ enters only through the solution $\hat w^*(\lambda)$ of an inner problem $\min_w g(w, \lambda)$ over parameters $w \in \mathbb{R}^M$. Crucially $\hat w^*$ is not a closed-form argmin — it is literally the point where $T$ steps of gradient descent on $g$ land, which is what lets $\lambda$ control the inner optimizer's own dynamics. To move $\lambda$ by a first-order method I need the total derivative, the hypergradient $d_\lambda f = \nabla_\lambda f + \nabla_\lambda \hat w^*(\lambda)\cdot\nabla_{\hat w^*} f$. The two partials $\nabla_\lambda f$ and $\nabla_{\hat w^*} f$ are cheap — one backward pass through $f$ — and all the difficulty is the middle factor $\nabla_\lambda \hat w^*(\lambda)$, the sensitivity of the end of a long inner run to $\lambda$. The whole point is the regime where both $N$ and $M$ are in the thousands, and that is exactly where the existing tools break. Making the chain explicit, $w_{t+1} = \Xi_{t+1}(w_t, \lambda)$ with $\hat w^* = w_T$, unrolling the chain rule gives the exact hypergradient as a sum over the entire trajectory, $$d_\lambda f = \nabla_\lambda f + \sum_{t=0}^{T} B_t\, A_{t+1} A_{t+2}\cdots A_T\, \nabla_{\hat w^*} f,$$ where $A_t = \nabla_{w_{t-1}}\Xi_t = I - \gamma\nabla^2_w g(w_{t-1}, \lambda)$ is how the step reacts to a state perturbation and $B_t = \nabla_\lambda\Xi_t = -\gamma\nabla_{\lambda,w} g(w_{t-1}, \lambda)$ is how it reacts to $\lambda$ directly. Computing this by reverse mode is $O(MT)$ in memory — every intermediate iterate $w_1,\dots,w_T$ must be kept to evaluate $A_t, B_t$ on the backward sweep — and for a gigabyte-scale model across a long horizon that product simply does not fit. Forward mode propagates $Z_t = \nabla_\lambda w_t$ alongside the run at $O(MN)$ memory and $N$-times the time, fatal when $\lambda$ is high-dimensional. Exact reversal of the inner dynamics holds only $O(M)$ but breaks under finite precision and is wedded to a specific update. Checkpointing softens memory to $O(M\sqrt{T})$ but doubles compute and still grows with $T$. Implicit differentiation, $d_\lambda f = \nabla_\lambda f - \nabla_{\lambda,w} g\,(\nabla_{w,w} g)^{-1}\,\nabla_{\hat w^*} f$, is trajectory-free at $O(M)$ but assumes $\hat w^*$ is the *exact* minimizer — uncontrolled error for a finite run — and structurally cannot tune any hyperparameter living inside the inner optimizer.

So rather than hunt for a fifth trick I stare at the sum I already have and ask whether I need all of it. The term at index $t$ carries the factor $A_{t+1}\cdots A_T$, a product of $T-t$ inner-map Jacobians. Suppose the last stretch of the run sits in a region where $g$ is $\alpha$-strongly convex and $\beta$-smooth in $w$ — the regime where GD is actually converging — and pick the step size so $\gamma \le 1/\beta$. Then $\gamma\nabla^2_w g$ has eigenvalues in $[\gamma\alpha, 1]$, so $I - \gamma\nabla^2_w g$ has eigenvalues in $[0, 1-\gamma\alpha]$ and $\|A_t\| \le 1 - \gamma\alpha < 1$: each $A$ is a contraction. That changes everything, because the early terms get multiplied by *many* of these contractions and are killed by roughly $(1-\gamma\alpha)^{T-t}$, while the recent terms survive at full size. The inner optimization forgets how it got here — a perturbation to $\lambda$ made at the start is washed out by all the contractive steps that follow — so the sum is carried almost entirely by its recent steps and the ancient ones are a geometrically vanishing tail. I therefore propose **T-RHG** (Truncated Reverse-mode Hyper-Gradient, equivalently $K$-RMD): run the inner optimizer forward for the full horizon $T$ so $\hat w^* = w_T$ is a good solution deep in the strongly convex basin, but differentiate through only the last $K$ transitions, keeping the estimator $$h_{T-K} = \nabla_\lambda f + \sum_{t=T-K+1}^{T} B_t\, A_{t+1}\cdots A_T\, \nabla_{\hat w^*} f.$$ Full reverse-mode RHG is the $K=T$ special case. The estimator maps onto reverse mode for free: the backward recursion $h_{t-1} = h_t + B_t\alpha_t$, $\alpha_{t-1} = A_t\alpha_t$ (initialized $\alpha_T = \nabla_{\hat w^*} f$, $h_T = \nabla_\lambda f$) produces the terms in exactly this order as $t$ sweeps down from $T$, so $h_{T-K}$ is just the value of the $h$ accumulator after $K$ backward steps, each one only a Jacobian-vector product that autograd supplies for one inner-step cost. To compute those $K$ transitions I need the boundary state $w_{T-K}$ and its successors $w_{T-K+1},\dots,w_T$, i.e. a consecutive window of $K+1$ states — $O(MK)$ memory. The split is the whole bargain: forward depth $T$ for solution quality, where steps are cheap and memoryless and can be overwritten; backward depth $K \ll T$ for gradient quality.

I have to earn the truncation rather than assert it. The error is exactly the dropped early part, and factoring out the common contraction tail $A_{T-K+1}\cdots A_T$ that every dropped term shares gives $$e_K = d_\lambda f - h_{T-K} = \Big(\sum_{t=0}^{T-K} B_t A_{t+1}\cdots A_{T-K}\Big)\cdot\big(A_{T-K+1}\cdots A_T\,\nabla_{\hat w^*} f\big).$$ The right factor is the outer gradient pushed back through $K$ contractions, so $\|A_{T-K+1}\cdots A_T\nabla_{\hat w^*} f\| \le (1-\gamma\alpha)^K\|\nabla_{\hat w^*} f\|$, and the bias decays *exponentially in $K$*. When $g$ is globally strongly convex the left factor sums geometrically to $M_B/(\gamma\alpha)$ with $M_B = \max_t\|B_t\|$, giving $\|e_K\| \le \big((1-\gamma\alpha)^K/(\gamma\alpha)\big)\|\nabla_{\hat w^*} f\|\,M_B$; when $g$ is only locally strongly convex near the window but possibly nonconvex earlier, the worst case $\|A_t\| \le 1+\gamma\beta \le 2$ gives the honest $\|e_K\| \le 2^{T-K+1}(1-\gamma\alpha)^K\|\nabla_{\hat w^*} f\|M_B$ — the loose $2^{T-K}$ prefix is the price of early nonconvexity, dominated once the convex tail is long enough. The requirement is local: the run only has to be well-behaved near where it ends up. This also explains *what* $h_{T-K}$ approximates. In the convergent limit $A_t \to A_\infty = I - \gamma\nabla_{w,w} g$ and $B_t \to B_\infty = -\gamma\nabla_{\lambda,w} g$ at $(w^*,\lambda)$, and since $\|A_\infty\| < 1$ the Neumann series converges, $(\nabla_{w,w} g)^{-1} = \gamma\sum_{k\ge 0} A_\infty^k$, so the implicit-diff middle term is exactly $-\nabla_{\lambda,w} g\,(\nabla_{w,w} g)^{-1} = B_\infty\sum_{k\ge 0} A_\infty^k$. My truncated estimator captures the first $K$ terms of that very series — T-RHG is an order-$K$ Taylor approximation of the inverse Hessian in implicit differentiation. The two are the same object truncated two ways: implicit diff truncates the Neumann series of the inverse (and $K$-step conjugate gradient gets the tighter per-step rate $O((1-1/\sqrt\kappa)^K)$ vs. my $O((1-1/\kappa)^K)$, $\kappa = \beta/\alpha$), but it needs $w^*$ exactly and has no error control away from it, whereas my truncation needs only local strong convexity, controls its finite-run error, and can differentiate the inner optimizer's own knobs.

Small bias is reassuring but I want more — that $-h_{T-K}$ actually points downhill. Test the cheapest case $K=1$ in the clean setting where $g$ is globally strongly convex and $\nabla_\lambda f = 0$ (true for data hyper-cleaning, where the validation loss sees $\lambda$ only through $\hat w^*$). Then $h_{T-1} = B_T\nabla_{\hat w^*} f$ and $h_{T-1}^T d_\lambda f = \|h_{T-1}\|^2 + (\text{cross term})$; the self term $\|B_T\nabla_{\hat w^*} f\|^2$ is strictly positive when $B_T$ has full column rank, and the cross term against the dropped part is controlled by peeling off three Lipschitz error pieces (replacing $B_t$ by $B_T$, telescoping the $A$'s toward $A_T$, replacing $A_T$ by $(1-\gamma\alpha)I$), each multiplied by a state-distance $\|w_{T-1}-w_{t-1}\|$ that linear convergence $\|w_t - w^*\| \le D e^{-\alpha\gamma t}$ crushes. The $t$-dependence collapses because $e^{-\alpha\gamma(t-1)}e^{-\gamma\alpha(T-t)} = e^{-\alpha\gamma(T-1)}$, independent of $t$, so the errors are $O(T e^{-\alpha\gamma(T-1)})$ plus a small $\gamma(\beta-\alpha)/(1-\gamma(\beta-\alpha))$ term; for $T$ large and $\gamma$ small the positive self term wins and $h_{T-1}^T d_\lambda f \ge c\|\nabla_{\hat w^*} f\|^2 = \Omega(\|d_\lambda f\|^2)$ — a sufficient descent direction at $K=1$, and the same peeling works for $K>1$. Stitching this into the outer loop, with merely small bias $\|h_{T-K} - d_\lambda f\| \le \varepsilon$ and outer SGD at $\eta_\tau = O(1/\sqrt\tau)$, biased-SGD telescoping gives $\mathbb{E}[\sum_\tau \eta_\tau\|\nabla F(\lambda_\tau)\|^2 / \sum_\tau\eta_\tau] \le \tilde O(\varepsilon + (\varepsilon^2+1)/\sqrt R)$, an $\varepsilon$-approximate stationary point with $\varepsilon = O((1-\gamma\alpha)^K)$, so $K = O(\log 1/\varepsilon)$ back-prop steps buy arbitrary accuracy. And when the descent property holds — adding non-interference $\nabla_\lambda f^T(d_\lambda f + h_{T-K} - \nabla_\lambda f) \ge \Omega(\|\nabla_\lambda f\|^2)$, $C^2$ $g$, full-rank $B_t$, deterministic objective — the bias term dies entirely and outer SGD converges to an *exact* stationary point for any $K \ge 1$, which is why one-step heuristics had been working. Non-interference is not free: a scalar counterexample $\min_\lambda \tfrac12(\hat w^*)^2 + \phi(\lambda)$ with $\phi$ chosen so stationary points have $\hat w^* \ne 0$ makes $h_{T-1} = u - v \ne 0$ at every true stationary point, so the outer SGD never settles — but it holds automatically whenever $\nabla_\lambda f = 0$, exactly the hyperparameter-optimization, data-hyper-cleaning, and regularization-learning cases I care about. As for the design choices: $\gamma \le 1/\beta$ is one knob doing three jobs — it makes $A_t$ a contraction, which simultaneously gives linear inner convergence, the $(1-\gamma\alpha)^K$ bias decay, and a convergent Neumann series; running full $T$ forward but only $K < T$ backward is required because the bounds demand the *last* iterates sit in the locally strongly convex basin, which the full horizon delivers cheaply, while $K$ controls only gradient accuracy at the $O(MK)$ cost; and $\eta_\tau = O(1/\sqrt\tau)$ is precisely the schedule the telescoping needs to vanish. A practical setting such as $K = 100$ inside $T = 500$ buys a deliberate accuracy margin while staying far below full-trajectory storage.

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
