# Movement Pruning

## Problem

Magnitude pruning (keep weights with largest $|W|$) works well for models trained from scratch but is much less effective when **fine-tuning a pretrained model** (the transfer-learning regime), especially at high sparsity. The reason: fine-tuned weights stay close to their pretrained values, so which weights are large is decided by *pretraining*, not the end task — the prune/keep decision is essentially fixed before fine-tuning starts and is blind to the task.

## Key idea

Move the importance criterion from **0th order** (the weight's value) to **1st order** (the weight's *movement* during fine-tuning). Keep the weights that are moving *away from zero* and prune those moving *toward* it — so both small and large weights can be pruned, based on what fine-tuning does. Realize this by learning an importance-score matrix $\mathbf{S}$ jointly with $\mathbf{W}$, using a hard top-$v$ (or threshold) mask whose gradient is supplied by the **straight-through estimator**.

## Method

Score matrix $\mathbf{S}$, mask $\mathbf{M}$, inference $\mathbf{a}=(\mathbf{W}\odot\mathbf{M})\mathbf{x}$.

- **Hard movement pruning:** $\mathbf{M}=\mathrm{Top}_v(\mathbf{S})$. $\mathrm{Top}_v$ has zero gradient, so backward uses straight-through: the gradient passes to $\mathbf{S}$ as if the mask were identity, giving
$$\frac{\partial\mathcal{L}}{\partial S_{i,j}} = \frac{\partial\mathcal{L}}{\partial a_i}\,W_{i,j}\,x_j = \frac{\partial\mathcal{L}}{\partial W_{i,j}}\,W_{i,j}.$$
The last equality omits the binary gate in $\partial\mathcal{L}/\partial W_{i,j}$ for the movement comparison; the straight-through score gradient keeps the ungated first-order factor, so scores update even for masked weights and pruned weights can re-enter. With $S^{(0)}=0$,
$$S_{i,j}^{(T)} = -\alpha_S\sum_{t<T}\Big(\frac{\partial\mathcal{L}}{\partial W_{i,j}}\Big)^{(t)} W_{i,j}^{(t)},$$
an **accumulator of movement away from zero**: $S$ rises when $W>0$ grows or $W<0$ decreases further below zero, and falls when $W$ moves toward 0.

- **Soft movement pruning:** replace the top-$v$ cut with a global threshold $\mathbf{M}=(\mathbf{S}>\tau)$ and add a sparsity-inducing regularizer $R(\mathbf{S})=\sum_{i,j}\sigma(S_{i,j})$, optimizing $\mathcal{L}+\lambda_{\text{mvp}}R(\mathbf{S})$; $\lambda_{\text{mvp}}$ controls sparsity, which emerges non-uniformly across layers.

**Why the sign matters.** A first-order Taylor argument shows that when a swap occurs (a more important connection replaces an active one), the training loss decreases — for both $\mathrm{Top}_v$ and threshold masks. This guarantee provably *reverses* (loss increases on a swap) if one selects by $|S|$ instead of $S$; preserving the direction of movement is essential.

**Relation to $L_0$.** $L_0$ regularization's score gradient is $\frac{\partial\mathcal{L}}{\partial a_i}W_{i,j}x_j\cdot f(\overline{S}_{i,j})$ — the same first-order core times a hard-concrete density factor. Movement pruning is the same signal without the stochastic reparameterization.

**Wrappers:** automated gradual pruning with a cubic sparsity schedule (plus cool-down), embeddings frozen, and an optional orthogonal knowledge-distillation loss from a fine-tuned teacher.

## Code

```python
import torch, torch.nn as nn

class TopVSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, keep_ratio):
        mask = torch.zeros_like(scores)
        k = int(keep_ratio * scores.numel())
        idx = scores.flatten().argsort(descending=True)[:k]   # top-v% by score
        mask.flatten()[idx] = 1.0
        return mask
    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None          # straight-through: grad -> scores

class ThresholdSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, tau): return (scores > tau).float()
    @staticmethod
    def backward(ctx, grad_out):  return grad_out, None

class MaskedLinear(nn.Linear):
    def __init__(self, in_f, out_f, keep_ratio=1.0, soft=False, tau=0.0):
        super().__init__(in_f, out_f)
        self.score = nn.Parameter(torch.zeros_like(self.weight))   # learned, init 0
        self.keep_ratio, self.soft, self.tau = keep_ratio, soft, tau
    def forward(self, x):
        M = (ThresholdSTE.apply(self.score, self.tau) if self.soft
             else TopVSTE.apply(self.score, self.keep_ratio))
        return nn.functional.linear(x, self.weight * M, self.bias)

def sparsity(t, v_i, v_f, t_i, n_steps, delta_t, total_steps, t_f=0):
    if t < t_i:
        return v_i
    if t >= total_steps - t_f:
        return v_f
    progress = (t - t_i - t_f) / (n_steps * delta_t)
    return v_f + (v_i - v_f) * (1 - progress) ** 3

def fine_prune(model, loader, opt, T, sched, lam_mvp=0.0, teacher=None, soft=False):
    for t, (x, y) in zip(range(T), loader):
        if not soft: set_keep_ratio(model, 1 - sparsity(t, **sched))
        out = model(x); loss = task_loss(out, y)
        if teacher is not None:
            loss = 0.5 * loss + 0.5 * distill_loss(out, teacher(x))
        if soft:
            loss = loss + lam_mvp * sum(torch.sigmoid(m.score).sum()
                                        for m in masked_layers(model))
        opt.zero_grad(); loss.backward(); opt.step()
    return model
```
