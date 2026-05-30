# Context

## Research question

Practitioners are forced into a trade-off when picking an optimizer. Adaptive methods like Adam converge fast in the early phases of training and are the de facto default for hard-to-stabilize settings such as GANs — but on standard supervised tasks (especially CNNs) they generalize *worse* than plain SGD. SGD with momentum generalizes well at the end but is slow early and brittle in complex settings. So the field is split: fast-but-worse-generalizing adaptive methods versus slow-but-better-generalizing SGD. The question is whether a single first-order optimizer can have all three desirable properties at once — fast convergence like an adaptive method, good final generalization like SGD, and training stability robust enough for things like GANs — ideally as a tiny, drop-in modification of Adam that introduces no new hyperparameters.

## Background

The workhorse adaptive optimizer is Adam (Kingma & Ba 2014). It keeps two exponential moving averages: of the gradient, mₜ = β₁mₜ₋₁ + (1−β₁)gₜ, and of the squared gradient, vₜ = β₂vₜ₋₁ + (1−β₂)gₜ², bias-corrects both, and steps by θₜ = θₜ₋₁ − α·m̂ₜ/(√v̂ₜ + ε). The denominator √vₜ gives every coordinate its own effective learning rate set by the gradient's recent *magnitude*: directions with large gradients get small steps, directions with small gradients get large steps. AdaGrad (Duchi et al. 2011) is the ancestor of this idea (accumulated squared gradients in the denominator), but its accumulation never decays, so the effective rate shrinks to zero; Adam's EMA fixes that.

A large body of work tries to patch Adam's weaknesses. AMSGrad (Reddi et al. 2018) takes the running maximum of vₜ to fix a flaw in Adam's convergence proof. Yogi (Zaheer et al. 2018) controls how vₜ reacts to minibatch size. RAdam (Liu et al. 2019) rectifies the high variance of the adaptive learning rate in the first steps with a warmup-like correction. MSVAG (Balles & Hennig 2018) dissects Adam into a sign update times a variance-dependent magnitude. AdaBound and SWATS (Luo et al. 2019; Keskar & Socher 2017) interpolate or switch from Adam toward SGD to recover generalization. AdamW (Loshchilov & Hutter 2019) decouples weight decay from the gradient step. Despite all these, the consistent finding (Wilson et al. 2017) is that adaptive methods still generalize worse than well-tuned SGD on large datasets like ImageNet, and many of these variants are empirically unstable when training GANs.

The deeper diagnostic that motivates a different denominator is about *curvature*. An ideal optimizer should not simply take a large step where the gradient is large and a small step where it is small; it should take large steps in low-curvature directions and small steps in high-curvature directions (this is the second-order intuition). Adam's denominator √vₜ depends only on the gradient *magnitude*, so it cannot distinguish the two situations that motivate the whole construction:
- A *flat region* (small gradient, small curvature): the ideal step is large; SGD stalls (step ∝ gradient), but a small-vₜ denominator lets Adam move.
- A *steep, oscillating valley* (large gradient, large curvature): the ideal step is small; SGD overshoots, but a large-vₜ denominator damps Adam.
- A *large-gradient-but-small-curvature* region (large gradient, but the gradient is barely *changing* step to step): the ideal step is large; here Adam *fails* — its denominator √vₜ is large because the gradient magnitude is large, so it takes a needlessly small step, even though the loss surface is locally gentle. SGD, taking step ∝ gradient, does the right thing here.

So the magnitude-only denominator is right in the first two cases and wrong in the third, exactly the case where SGD's behavior is preferable. A 2D illustration sharpens this: on f(x,y) = |x| + |y|, where each gradient component is ±1, starting near the x-axis the trajectory keeps advancing in x (gradient always +1) while oscillating in y (gradient flips sign). One would want a *large* step in x (consistent progress) and a *small* step in y (oscillation). Adam computes vₓ ≈ E[gₓ²] = 1 and v_y ≈ E[g_y²] = 1 — equal, because squaring discards the sign — so Adam takes the *same* step in both directions. The quantity that separates the consistent direction from the oscillating one is not the magnitude but how much the gradient *deviates from its own running mean*: in x the gradient is constant so its deviation is ≈0, in y it flips so its deviation is large. That deviation is the gradient's variance, and it is the curvature-aware signal Adam is missing.

## Baselines

**Adam (Kingma & Ba 2014).** mₜ, vₜ EMAs of gₜ and gₜ²; bias-corrected; θ ← θ − α·m̂/(√v̂ + ε). Fast early convergence, default for GANs. Gap: magnitude-only denominator → wrong (too-small) step in large-gradient/low-curvature directions; generalizes worse than SGD on large datasets.

**SGD with momentum.** v ← βv + g; θ ← θ − αv. Step magnitude ∝ |momentum|. Generalizes best on many vision tasks; correctly takes large steps in the large-gradient/low-curvature case. Gap: no curvature adaptation, slow early, can overshoot in steep narrow valleys, unstable for GANs.

**AMSGrad (Reddi et al. 2018).** Uses max over time of vₜ in the denominator to guarantee non-increasing effective rates and fix Adam's convergence proof. Still magnitude-based; same curvature blind spot. (A natural robustness add-on, not a fix for the third case.)

**RAdam (Liu et al. 2019).** Rectifies the variance of the adaptive learning rate during early training via a closed-form term that effectively warms up the adaptive denominator. Addresses early-training instability, not the magnitude-vs-deviation issue.

**MSVAG (Balles & Hennig 2018).** Decomposes Adam into a sign component and a variance-adaptive magnitude. Closest in spirit in that it brings *variance* into the picture, but does not replace the denominator with a centered second moment.

**AdamW (Loshchilov & Hutter 2019).** Decoupled weight decay. Orthogonal improvement; combinable.

## Evaluation settings

Pre-existing yardsticks: image classification on CIFAR-10/100 (ResNet, DenseNet, VGG) and ImageNet (ResNet-18, top-1 accuracy), measuring both convergence speed (training loss/accuracy vs epochs) and final test accuracy to expose the convergence-vs-generalization trade-off; language modeling with LSTMs on Penn Treebank (1/2/3-layer, perplexity); and GAN training (DCGAN / a small WGAN on CIFAR-10) measured by FID and by whether training stays stable at all, since instability is the failure mode adaptive variants exhibit there. Standard data augmentation and learning-rate schedules. The optimizer hyperparameters are held at Adam's defaults (α = 1e-3, β = (0.9, 0.999), ε = 1e-8) so the comparison isolates the update rule.

## Code framework

The substrate is PyTorch's `torch.optim.Optimizer`: per-parameter `state`, `param_groups` carrying `lr`, `betas`, `eps`, `weight_decay`, and a `step()` that reads each parameter's `.grad`. An Adam-family optimizer keeps a first-moment buffer and a second-statistic buffer per parameter, does bias correction, and divides. The open slot is *which* second statistic goes in the denominator.

```python
import math
import torch
from torch.optim.optimizer import Optimizer


class AdaptiveDenomOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)       # first moment m_t
                    state['exp_avg_2'] = torch.zeros_like(p.data)     # the denominator statistic
                exp_avg, exp_avg_2 = state['exp_avg'], state['exp_avg_2']
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)       # m_t = EMA of gradient

                # TODO: form the second statistic that goes in the denominator,
                #       update exp_avg_2 as its EMA, and step
                #       theta <- theta - lr * mhat / (sqrt(2nd-stat-hat) + eps)
                pass
        return None
```
