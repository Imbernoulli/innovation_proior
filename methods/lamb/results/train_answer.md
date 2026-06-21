Training a large language model or a deep image classifier is dominated by wall-clock time, and the one lever that turns more hardware into less time without touching the model is the mini-batch size: a batch of $b$ examples can be split across many devices, each computing part of the gradient $g = (1/b)\sum_{s} \nabla\ell(x,s)$ in parallel, and the partials summed. The catch is that stochastic gradient descent is inherently sequential — update $t+1$ waits on update $t$ — so more devices buy a bigger batch, not more steps. At a fixed number of epochs the number of optimizer updates $T = (\text{\#examples}\times\text{epochs})/b$ falls *linearly* in $b$, so going from batch $512$ to $32\text{k}$ leaves $64\times$ fewer steps, and each surviving step must accomplish proportionally more or training simply does not finish learning. The obvious compensation is to raise the learning rate as the batch grows — the variance argument says the standard deviation of $g$ falls like $1/\sqrt{b}$, suggesting $\sqrt{b}$ scaling, and empirically linear scaling works up to a point — but past some batch-size ceiling cranking $\eta$ does not merely slow down, it destabilizes training: large batches drift toward sharp minima that generalize worse, the early large steps are especially poisonous, and the patches that exist (a hand-tuned warmup that ramps $\eta$ from near-zero over the first epochs) do not transfer across tasks, where the ceiling and the right exponent move from problem to problem. So a fixed scaling rule plus warmup is a fragile per-problem hand-tune. Adam adds per-coordinate adaptivity and is strong on attention models, but it too is *globally* scaled — one $\eta$ for the whole network — so it inherits the same large-batch instability and diverges at the largest batches; the existing layerwise-trust-ratio momentum method (You et al., 2017) trains an image classifier at batch $32\text{k}$ but has no convergence theory and fails on language models.

The thing that keeps breaking is the single global learning rate, so I want to understand *why* it is the wrong object. A network is a stack of layers, and the layers are nothing alike: a normalization gain, an embedding table, and a deep weight matrix differ by orders of magnitude in their parameter norm $\|x^{(i)}\|$ and in the norm of the step the optimizer wants to take. A global $\eta$ applies the same multiplier to every layer. If I pick $\eta$ large enough that a layer whose desired-step-to-weight ratio is $0.001$ actually moves, then a layer whose ratio is near $1$ moves a large fraction of its own norm in a single step — and that is the divergence. One number cannot be simultaneously right for both layers; big batch just lets me crank $\eta$ high enough that the mismatch turns fatal. The fix is therefore not a cleverer global schedule but a per-layer effective step matched to each layer's own geometry.

I propose **LAMB** — Layerwise-Adaptive Moments for Batch training. The construction is first a *strategy* over any base optimizer $A$ that produces a layerwise raw direction $u_t$ (vanilla descent being $x \leftarrow x - \eta\,u_t$), and then a specific instantiation. The strategy is two moves applied per layer: first normalize each layer's update to a unit $\ell_2$ direction $u^{(i)}/\|u^{(i)}\|$, killing its arbitrary magnitude; second re-impose a magnitude tied to the layer by scaling the per-layer learning rate by $\phi(\|x^{(i)}\|)$ for some $\phi:\mathbb{R}^+\to\mathbb{R}^+$. The per-layer update is

$$x_{t+1}^{(i)} = x_t^{(i)} - \eta_t\,\phi(\|x_t^{(i)}\|)\,\frac{u_t^{(i)}}{\|u_t^{(i)}\|},$$

so with $\phi = \mathrm{id}$ the distance layer $i$ travels is $\eta\,\|x^{(i)}\|$, automatically proportional to how big that layer is, and the global $\eta$ is finally decoupled from per-layer scale. Why throw the magnitude away entirely? At small batch I would be nervous, because the gradient magnitude carries real information; but at large batch the gradient *direction* is estimated with low variance while its raw magnitude is exactly the quantity that varies wildly across layers and blows things up, so discarding it and re-imposing a layer-matched one is cheap in bias and buys robustness — an exploding-gradient layer and a plateaued layer both get the same unit-norm treatment. The multiplier $\phi(\|x^{(i)}\|)/\|u^{(i)}\|$ is not arbitrary either: with $\phi = \mathrm{id}$ it is $\|x^{(i)}\|/\|u^{(i)}\|$, which when $u^{(i)}$ is roughly the gradient is a cheap running estimate of the inverse local smoothness $1/L_i$ — the textbook safe gradient-descent step for an $L_i$-smooth block — so the scheme is doing per-layer second-order-ish step sizing with no Hessian. Pure identity is dangerous at the extremes, where huge weights or a near-zero update norm make the step explode, so the strategy admits a clipped identity $\phi(z) = \min(\max(z,\gamma_l),\gamma_u)$; the convergence theory only needs a positive lower and upper bound $\alpha_l \le \phi \le \alpha_u$, while the canonical Google/TensorFlow-Addons core uses the unclipped $\phi(z)=z$ with a zero-norm fallback.

Choosing the base optimizer $A$ fixes the algorithm. The natural first try is momentum SGD, $u_t = m_t$, which yields exactly the layerwise-trust-ratio momentum scheme — sound framework, but it trains attention/language models poorly and diverges at the biggest batches. That is a sharp clue: momentum SGD already underperforms on transformers, where what works is per-*coordinate* adaptivity. The trust ratio cures the *across-layer* scale mismatch, but within a transformer layer the coordinates themselves are wildly ill-conditioned, and a momentum base does nothing about that. The two adaptivities are orthogonal, and I want both, so I set $A = \text{Adam}$, whose raw direction is

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t,\quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2,\quad \hat m_t = \frac{m_t}{1-\beta_1^t},\quad \hat v_t = \frac{v_t}{1-\beta_2^t},\quad r_t = \frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.$$

This $r_t$ is the per-coordinate-normalized Adam update; wrapping it in the layerwise strategy gives two-fold adaptivity at once — $r_t$ rescales each dimension by $1/\sqrt{\hat v}$, and the trust ratio rescales each layer. The last piece is weight decay. Following the AdamW lesson I want decoupled decay rather than pushing $\lambda x$ through the adaptive denominator, but I must place it carefully relative to the normalization: if I added $\lambda x^{(i)}$ *after* normalizing, the trust ratio would never see the decay and its scale would float free of the layer step. So the decay rides *inside* the trust-ratio numerator and norm, giving the LAMB update

$$x_{t+1}^{(i)} = x_t^{(i)} - \eta_t\,\frac{\phi(\|x_t^{(i)}\|)}{\|r_t^{(i)} + \lambda x_t^{(i)}\|}\,\big(r_t^{(i)} + \lambda x_t^{(i)}\big),$$

so the decay magnitude stays commensurate with the layer's step. A sanity-check special case: with $\beta_1 = \beta_2 = 0$ we have $\hat m = g$, $\hat v = g^2$, and $r_j = g_j/(|g_j|+\epsilon)$, the sign of $g_j$, so a layer update becomes a normalized sign vector — signSGD scaled by $\sqrt{d_i}$, the square root of the layer dimension. Defaults are $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-6}$, $\lambda = 0.01$; across a batch-size sweep one uses $\sqrt{b}$ learning-rate scaling with linear-epoch warmup, $\ell_2$ as the layer norm (other norms move accuracy by $<0.1\%$), and because the Adam debiasing $\hat m = m/(1-\beta_1^t)$, $\hat v = v/(1-\beta_2^t)$ overlaps an explicit warmup, the correction can be dropped in practice when a warmup schedule is already running.

What the heuristic trust-ratio scheme never had is a convergence guarantee, and LAMB does. On the smooth nonconvex problem $\min_x f(x) = \mathbb{E}_s[\ell(x,s)] + (\lambda/2)\|x\|^2$ with per-layer $L_i$-smoothness, bounded per-layer gradient variance $\sigma_i^2$, and coordinatewise gradient bound $G$, the benchmark is SGD with $b=T$, whose rate $\mathbb{E}\|\nabla f\|^2 \le O((f(x_1)-f^*)L_\infty/T + \|\sigma\|^2/T)$ is hostage to the single worst-conditioned layer through $L_\infty = \max_i L_i$. The load-bearing mechanism in the proof is normalization: because every per-layer step has norm exactly $\eta_t\,\phi(\|x^{(i)}\|) \le \eta_t\alpha_u$ regardless of the gradient, the second-order term in the descent lemma cannot blow up with the gradient and is bounded by $(\eta_t^2\alpha_u^2/2)\|L\|_1$. The linear term, after adding and subtracting the true-gradient unit direction, splits into a clean descent piece $-\eta_t\phi(\|x^{(i)}\|)\|\nabla_i f\|$ and an error piece controlled by the reverse triangle inequality and Cauchy–Schwarz to at most $2\|\Delta_t^{(i)}\|$, with $\mathbb{E}\|\Delta_t^{(i)}\| \le \sigma_i/\sqrt{b}$. Telescoping, setting $b=T$ and balancing the step gives, for the trust-ratio scheme with $\beta_1 = \lambda = 0$,

$$\Big(\mathbb{E}\,\tfrac{1}{\sqrt h}\textstyle\sum_i \|\nabla_i f(x_a)\|\Big)^2 \le O\!\Big(\frac{(f(x_1)-f^*)\,L_{\text{avg}}}{T} + \frac{\|\sigma\|_1^2}{Th}\Big),$$

with $L_{\text{avg}} = (1/h)\sum_i L_i$ — the worst-layer constant traded for the average one. For LAMB the $\beta_2 = 0$ case reuses the same skeleton but, where Adam's $r$ appears, leans on two bounds — each coordinate satisfies $|r_j| \le 1/\sqrt{1-\beta_2}$ so $\|r_t^{(i)}\| \le \sqrt{d_i/(1-\beta_2)}$, and $\sqrt{\hat v_j} \le G$ — splitting the coordinate sum into sign-agreement and sign-disagreement sets and invoking the signSGD device $P(\mathrm{sign}\,g_j \ne \mathrm{sign}\,\nabla f_j) \le \sigma_{i,j}/(\sqrt b\,|\nabla_i f_j|)$, where the $|\nabla_i f_j|$ conveniently cancels; this yields the same clean $L_{\text{avg}}$-rather-than-$L_\infty$ shape. The general $\beta_2 > 0$ theorem is looser, $\mathbb{E}\|\nabla f(x_a)\|^2 \le O\!\big(\sqrt{G^2 d/(h(1-\beta_2))}\,[\sqrt{2(f(x_1)-f^*)\|L\|_1/T} + \|\tilde\sigma\|_1/\sqrt T]\big)$, but still a stationary-point guarantee. Whether $L_{\text{avg}}$ truly beats SGD's $L_\infty$ requires comparing like with like, since the convergence criterion changed; the signSGD-style density bookkeeping — writing each quantity as a density factor $\psi$ times a reference — rewrites the layerwise rate as $O((f(x_1)-f^*)L_\infty/T\cdot \psi_L/\psi_g^2 + \|\sigma\|^2/T\cdot\psi_\sigma^2/\psi_g^2)$, so LAMB wins exactly when $\psi_L \ll \psi_g^2$ and $\psi_\sigma \ll \psi_g^2$, i.e. when the gradient is *denser* than the curvature and the noise. That is the precise condition under which the layerwise trust ratio helps: when the signal is spread across many coordinates while curvature and stochasticity are concentrated. Empirically this is what lets a single $\sqrt b$-plus-warmup recipe scale BERT to batch $32\text{k}$ in about $100$ minutes, and about $76$ minutes with the $64\text{k}/32\text{k}$ mixed-batch schedule.

```python
import torch
from torch.optim import Optimizer

class Lamb(Optimizer):
    """LAMB: layerwise-adaptive Adam for large-batch training."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0.0, debias=False, adam=False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.debias = debias            # TFA/paper correction; Google BERT code omits it
        self.adam = adam                # trust_ratio := 1, for ablation
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad.data
                if g.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients.')
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)      # m
                    state['exp_avg_sq'] = torch.zeros_like(p.data)   # v
                m, v = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                # Adam moment EMAs
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                step_size = group['lr']
                if self.debias:
                    bc1 = 1 - beta1 ** state['step']
                    bc2 = 1 - beta2 ** state['step']
                    r = (m / bc1) / ((v / bc2).sqrt().add(group['eps']))
                else:
                    r = m / v.sqrt().add(group['eps'])
                if group['weight_decay'] != 0:
                    r = r.add(p.data, alpha=group['weight_decay'])

                # Canonical Google/TFA core uses phi(z)=z; some PyTorch variants clamp z.
                weight_norm = p.data.pow(2).sum().sqrt()
                r_norm = r.pow(2).sum().sqrt()
                if weight_norm == 0 or r_norm == 0:
                    trust_ratio = 1.0
                else:
                    trust_ratio = weight_norm / r_norm
                if self.adam:
                    trust_ratio = 1.0

                # per-layer step proportional to weight norm; global lr decoupled
                p.data.add_(r, alpha=-step_size * trust_ratio)
        return loss
```
