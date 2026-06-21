The constraint I am actually under is wall-clock time. Training a large convolutional network on ImageNet is slow on a single device, and the only knob that turns more hardware into less time without touching the model is the mini-batch. Data-parallel synchronous SGD splits the global batch of $B$ examples across $N$ workers, each worker computes the gradient on its $B/N$ chunk, the partials are summed, and one weight update is applied:
$$w_{t+1} = w_t - \lambda \cdot \frac{1}{B}\sum_{i=1}^{B}\nabla L(x_i, w_t).$$
To keep each worker busy enough to be efficient, its chunk has to be sizeable, so the way I add workers to go faster is by pushing $B$ up — more nodes, larger batch, fewer-but-heavier steps. The catch is that at a fixed epoch budget, multiplying $B$ by $k$ gives me $k$ times fewer weight updates, so each step must cover proportionally more ground, which forces the learning rate up. The natural scaling falls out of comparing two batch-$B$ sub-steps to one batch-$2B$ step: they line up only if $\lambda_2 = 2\lambda$ and only under the approximation $\nabla L(x_j, w_{t+1}) \approx \nabla L(x_j, w_t)$ — the weights barely moved within the step. That is the linear scaling rule, and its failure is built in: the approximation is exactly "the step is small," yet a large $\lambda$ makes the step large, so the rule licenses a big rate and the big rate then invalidates the rule. The gentler $\sqrt{B}$ scaling from gradient-noise reduction outruns its justification more slowly but leaves speed on the table, and warmup — ramping $\lambda$ from a small safe value over the first epochs — softens only the most sensitive initial phase. Run linear scaling plus warmup on AlexNet/ImageNet and it still dies around $B\!=\!2K$: a $B\!=\!256$ baseline near $58\%$ falls to the low-$50$s at $B\!=\!4K$ and the mid-$40$s at $B\!=\!8K$, diverging once the rate crosses a threshold even with warmup.

Swapping Local Response Normalization for Batch Normalization helps a great deal — large rates become usable, the band of good rates widens, and the $B\!=\!8K$ gap collapses to a couple of percent — but a stubborn residual remains, and I want to know what it is. The sharp-minima generalization hypothesis predicts a widening train–test loss gap at large batch; measuring that gap at small batch versus $B\!=\!8K$ shows no significant difference, so the residual is not a generalization problem but under-optimization. So why not simply crank the rate to optimize harder? Because the rate is exactly what diverges. Staring at the update $w_{t+1} = w_t - \lambda\nabla L(w_t)$ per layer, the meaningful quantity is the step size relative to that layer's own weight norm, $\|\lambda\nabla L(w)\| / \|w\|$: once it approaches $1$, a layer's parameters move as much as their own magnitude in a single step, and that is the blow-up. The safe condition is $\lambda \ll \|w\|/\|\nabla L(w)\|$. Tabulating $\|w\|/\|\nabla L(w)\|$ layer by layer after one iteration of the BN network shows it spans **orders of magnitude** — about $5.76$ for an early convolutional weight tensor, about $1345$ for a later fully connected one, with weights and biases differing within a layer. That is the whole problem in one table: a single global $\lambda$ must satisfy the *tightest* constraint or diverge, so it is pinned by the smallest-ratio layer, while every layer with a larger ratio is starved hundreds of times over. Raising the global rate either diverges the worst layer or, backed off to keep it safe, under-trains everyone else. Warmup is the right instinct applied through the wrong variable: it sneaks a single scalar up on a constraint that is really per-layer and moving.

I propose LARS — Layer-wise Adaptive Rate Scaling. The move is to drop the assumption that there must be one rate and give each layer $\ell$ its own local rate $\lambda^\ell$, with a single global $\gamma$ riding on top so I keep one knob for the overall schedule (warmup, decay). The per-layer direction is $\Delta w^\ell = \gamma \cdot \lambda^\ell \cdot \nabla L(w^\ell)$. The right choice for $\lambda^\ell$ is the one that makes the same safety condition hold for every layer at once — the step should be a fixed, safe fraction of that layer's weight norm regardless of its gradient magnitude. Since the quantity that varied was precisely $\|w^\ell\|/\|\nabla L(w^\ell)\|$, set the local rate to it:
$$\lambda^\ell = \eta \cdot \frac{\|w^\ell\|}{\|\nabla L(w^\ell)\|}, \qquad \eta < 1.$$
Then the gradient norm cancels in the step magnitude:
$$\|\Delta w^\ell\| = \gamma\,\eta \cdot \frac{\|w^\ell\|}{\|\nabla L(w^\ell)\|}\cdot\|\nabla L(w^\ell)\| = \gamma\,\eta\,\|w^\ell\|.$$
Every layer moves the same fixed fraction $\gamma\eta$ of its own weight norm, the same for all layers whether their gradient is huge or vanishing. The worst layer no longer pins the others, and the update becomes insensitive to exploding or vanishing gradient scales: a tiny gradient still yields a step proportional to the weights, a huge gradient does not overshoot. Here $\eta$ is simply how much I trust a layer to change in one update — the fraction of its own norm it may move — kept well below $1$ (in practice around $10^{-3}$, since $\gamma$ is large in the large-batch regime and the product $\gamma\eta$ is what actually sets the relative step).

This is genuinely different from per-coordinate adaptive optimizers, and worth pinning down so I know I am not reinventing them. Adam and RMSProp divide each individual weight's step by a running norm of that weight's own gradient history; LARS differs on both granularity and control variable. The granularity is per *layer*, not per weight — a single norm over a whole layer is a far more stable aggregate than per-coordinate statistics, and the divergence I am fighting is a layer-level phenomenon (a whole layer's update outrunning its own weights), so the layer is the correct, less noisy unit. The control variable is the weight norm in the numerator, not the gradient magnitude, so LARS controls the step *as a fraction of the weights* — the quantity that actually causes the blow-up — which the per-coordinate methods never look at. Framed structurally, it is a block-diagonal rescaling of the gradient with one block per layer, the simplest possible version where each block is a scalar.

Folding in what a real optimizer needs leaves two refinements. Weight decay $\beta$ means the direction I actually step along is $d = g + \beta w$, not $g$. Using only $\|g\|$ in the denominator would let a large decay term make the real step exceed the trust calculation, and using $\|g + \beta w\|$ directly would let cancellation between $g$ and $\beta w$ make the scale jump around. The stable, conservative choice is the triangle-inequality bound $\|g + \beta w\| \le \|g\| + \beta\|w\|$ in the denominator,
$$\lambda^\ell = \eta \cdot \frac{\|w^\ell\|}{\|\nabla L(w^\ell)\| + \beta\,\|w^\ell\|},$$
so the decayed step is governed by a bound, $\|\gamma\lambda^\ell(g^\ell + \beta w^\ell)\| \le \gamma\eta\|w^\ell\|$, rather than by wishful cancellation; this also gently shrinks the rate for layers with large weights, which is sensible. Momentum then accumulates the already-scaled, already-decayed direction into a heavy-ball velocity. Putting the full step together for layer $\ell$ under a global polynomial-decay schedule $\gamma_t = \gamma_0(1 - t/T)^2$, in the PyTorch convention where the momentum buffer stores the scaled direction and the global rate is applied once at the parameter update:
$$g_t^\ell = \nabla L(w_t^\ell), \quad r_t^\ell = \eta\,\frac{\|w_t^\ell\|}{\|g_t^\ell\| + \beta\|w_t^\ell\| + \varepsilon}, \quad u_t^\ell = r_t^\ell(g_t^\ell + \beta w_t^\ell),$$
$$v_{t+1}^\ell = m\,v_t^\ell + u_t^\ell, \qquad w_{t+1}^\ell = w_t^\ell - \gamma_t\,v_{t+1}^\ell.$$
Two degenerate cases need guarding at the very start: a tensor with zero weight norm has no meaningful fraction of its own norm, and a tensor with zero raw gradient norm would make the no-decay ratio blow up, so if either norm is zero I skip adaptation for that tensor and treat the multiplier as $1$. Warmup stays, but now only as a complement smoothing the first few steps where the ratios are still settling, not as the crutch carrying the whole cross-layer mismatch — that structural job belongs inside the optimizer.

```python
import torch
from torch.optim.optimizer import Optimizer

class Lars(Optimizer):
    """Layer-wise Adaptive Rate Scaling (LARS) on top of momentum SGD.

    For each parameter tensor (treated as a layer), scale its step by the
    trust ratio  eta * ||w|| / (||g|| + wd*||w||)  so the step is a fixed
    fraction of the weight norm, then apply momentum SGD with the global lr.
    Optional LARC `trust_clip` caps the local lr at the global lr.
    """
    def __init__(self, params, lr=1.0, momentum=0, dampening=0,
                 weight_decay=0.0, nesterov=False,
                 trust_coeff=0.001, eps=1e-8, trust_clip=False, always_adapt=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires momentum and zero dampening")
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        trust_coeff=trust_coeff, eps=eps,
                        trust_clip=trust_clip, always_adapt=always_adapt)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']     # beta
            momentum     = group['momentum']         # m
            dampening    = group['dampening']
            nesterov     = group['nesterov']
            trust_coeff  = group['trust_coeff']      # eta
            eps          = group['eps']
            lr           = group['lr']               # global gamma_t (set by scheduler)

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                # --- LARS per-layer trust ratio ---------------------------
                if weight_decay != 0 or group['always_adapt']:
                    w_norm = p.norm(2.0)
                    g_norm = grad.norm(2.0)
                    trust_ratio = trust_coeff * w_norm / (g_norm + w_norm * weight_decay + eps)
                    # no adaptation when either norm is zero
                    trust_ratio = torch.where(
                        w_norm > 0,
                        torch.where(g_norm > 0, trust_ratio, 1.0),
                        1.0,
                    )
                    if group['trust_clip']:
                        trust_ratio = torch.clamp(trust_ratio / lr, max=1.0)
                    grad.add_(p, alpha=weight_decay)         # g + beta*w
                    grad.mul_(trust_ratio)                   # scale by trust ratio

                # --- momentum SGD with the GLOBAL learning rate -----------
                if momentum != 0:
                    state = self.state[p]
                    buf = state.get('momentum_buffer')
                    if buf is None:
                        buf = state['momentum_buffer'] = torch.clone(grad).detach()
                    else:
                        buf.mul_(momentum).add_(grad, alpha=1.0 - dampening)
                    grad = grad.add(buf, alpha=momentum) if nesterov else buf

                p.add_(grad, alpha=-lr)
        return loss
```
