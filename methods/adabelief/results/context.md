# Context

## Research question

Practitioners are forced into a trade-off when picking an optimizer. Adaptive methods like Adam converge fast in the early phases of training and are the de facto default for hard-to-stabilize settings such as GANs, but on standard supervised tasks, especially CNNs, they often generalize worse than plain SGD with momentum. SGD with momentum generalizes well at the end but is slow early and brittle in complex settings. The question is whether a single first-order optimizer can have all three desirable properties at once: fast convergence like an adaptive method, good final generalization like SGD, and training stability robust enough for volatile training dynamics, ideally as a small drop-in modification of Adam that introduces no new core hyperparameters.

## Background

The workhorse adaptive optimizer is Adam (Kingma & Ba 2014). It keeps two exponential moving averages: of the gradient, m_t = beta_1 m_{t-1} + (1 - beta_1) g_t, and of the squared gradient, v_t = beta_2 v_{t-1} + (1 - beta_2) g_t^2. It bias-corrects both and steps by theta_t = theta_{t-1} - alpha mhat_t / (sqrt(vhat_t) + epsilon). The denominator sqrt(v_t) gives every coordinate its own effective learning rate set by the gradient's recent magnitude: directions with large gradients get small steps, directions with small gradients get large steps. AdaGrad (Duchi et al. 2011) is the ancestor of this idea, with accumulated squared gradients in the denominator; its accumulation never decays, so the effective rate can shrink too aggressively, while Adam's exponential average keeps the statistic local in time.

A large body of work tries to patch Adam's weaknesses. AMSGrad (Reddi et al. 2018) takes the running maximum of v_t to fix a flaw in Adam's convergence proof. Yogi (Zaheer et al. 2018) controls how v_t reacts to minibatch size. RAdam (Liu et al. 2019) rectifies the high variance of the adaptive learning rate in the first steps with a warmup-like correction. MSVAG (Balles & Hennig 2018) dissects Adam into a sign update times a variance-dependent magnitude. AdaBound and SWATS (Luo et al. 2019; Keskar & Socher 2017) interpolate or switch from Adam toward SGD to recover generalization. AdamW (Loshchilov & Hutter 2019) decouples weight decay from the gradient step. Wilson et al. 2017 sharpen the concern that adaptive methods can converge quickly yet land at solutions that generalize worse than well-tuned SGD.

The deeper diagnostic is about curvature. An ideal optimizer should not simply take a large step where the gradient is large and a small step where it is small; it should take large steps in low-curvature directions and small steps in high-curvature directions. Adam's denominator sqrt(v_t) depends on gradient magnitude, so it cannot distinguish the cases that matter:
- A flat region, with small gradient and small curvature: the ideal step is large. SGD stalls because its step is proportional to the small gradient, while a small adaptive denominator lets Adam move.
- A steep, oscillating valley, with large gradient and large curvature: the ideal step is small. SGD overshoots, while a large adaptive denominator damps Adam.
- A large-gradient, small-curvature region, where the gradient is large but barely changing step to step: the ideal step is large. Adam's denominator is large because the gradient magnitude is large, so it takes a needlessly small step. SGD's step is proportional to the large gradient, so it behaves more appropriately here.

So the magnitude-only denominator is right in the first two cases and wrong in the third, exactly where SGD's behavior is preferable. A 2D illustration sharpens the point: on f(x,y) = |x| + |y|, where each gradient component is +/-1, starting near the x-axis makes the trajectory keep advancing in x while oscillating in y. One wants a large step in x and a small step in y. Adam computes v_x approximately E[g_x^2] = 1 and v_y approximately E[g_y^2] = 1, equal because squaring discards the sign, so it takes the same-size step in both directions. The quantity that separates the consistent direction from the oscillating one is how much the gradient deviates from its own running mean: near zero in x and large in y. That deviation is the gradient's variance, and it is the curvature-aware signal missing from a raw second-moment denominator.

## Baselines

**Adam (Kingma & Ba 2014).** Adam keeps EMAs of g_t and g_t^2, bias-corrects them, and updates theta by subtracting alpha mhat_t / (sqrt(vhat_t) + epsilon). It is fast early and widely used in unstable settings. Its gap is the magnitude-only denominator: it shrinks steps in large-gradient, low-curvature directions even when the gradient is stable and reliable.

**SGD with momentum.** Momentum keeps a velocity-like moving average of gradients and steps in that direction. It often generalizes well on vision tasks and correctly takes large steps in the large-gradient, low-curvature case. Its gap is the lack of per-coordinate curvature or variance adaptation, which makes it slow in flat regions and prone to overshooting in steep narrow valleys.

**AMSGrad (Reddi et al. 2018).** AMSGrad replaces the current second-moment denominator with a running maximum so the effective adaptive rate does not increase. This repairs an Adam convergence issue but remains tied to the squared-gradient magnitude.

**RAdam (Liu et al. 2019).** RAdam rectifies the early-stage variance of the adaptive denominator. When the estimated degrees of freedom are too low, it falls back toward a momentum-style update; when the statistic is reliable enough, it uses an adaptive denominator. This addresses early-training instability rather than the magnitude-versus-deviation issue.

**MSVAG (Balles & Hennig 2018).** MSVAG analyzes Adam as a sign component with a variance-adaptive magnitude. It is close in spirit because it points toward variance as the meaningful reliability signal, but it does not replace Adam's denominator with a centered second moment.

**AdamW (Loshchilov & Hutter 2019).** AdamW applies weight decay directly to the parameters instead of folding it into the gradient. This is orthogonal to the denominator choice and can be combined with any Adam-family update.

## Evaluation settings

Natural yardsticks include image classification on CIFAR-10/100 with ResNet, DenseNet, and VGG, ImageNet classification with ResNet-18 and top-1 accuracy, language modeling with LSTMs on Penn Treebank measured by perplexity, and GAN training on CIFAR-10 with WGAN or WGAN-GP measured by FID and training stability. These settings expose the convergence-versus-generalization trade-off and the instability of adversarial training. Standard data augmentation, learning-rate schedules, weight decay choices, and optimizer hyperparameter searches are part of the protocol.

## Code framework

The substrate is PyTorch's `torch.optim.Optimizer`: per-parameter `state`, `param_groups` carrying learning-rate and decay options, optional running maxima, optional rectified early-step behavior, and a `step()` method that reads each parameter's `.grad`. An Adam-family optimizer already has the first-moment buffer, bias correction, weight-decay placement, and optional RAdam/AMSGrad-style branches. The open slot is the denominator statistic and the update that uses it.

```python
import math
import torch
from torch.optim.optimizer import Optimizer


class AdaptiveFirstOrderOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16,
                 weight_decay=0, amsgrad=False, weight_decouple=True,
                 fixed_decay=False, rectify=True, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param_group in params:
                if 'betas' in param_group and param_group['betas'] != betas:
                    param_group['buffer'] = [[None, None, None] for _ in range(10)]

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            buffer=[[None, None, None] for _ in range(10)],
        )
        super().__init__(params, defaults)
        self.weight_decouple = weight_decouple
        self.fixed_decay = fixed_decay
        self.rectify = rectify
        self.degenerated_to_sgd = degenerated_to_sgd

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('adaptive denominator optimizers do not support sparse gradients')

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['denom_stat'] = torch.zeros_like(p.data)
                    if group['amsgrad']:
                        state['max_denom_stat'] = torch.zeros_like(p.data)

                if self.weight_decouple:
                    if not self.fixed_decay:
                        p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                    else:
                        p.data.mul_(1.0 - group['weight_decay'])
                elif group['weight_decay'] != 0:
                    grad.add_(p.data, alpha=group['weight_decay'])

                exp_avg = state['exp_avg']
                denom_stat = state['denom_stat']
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # TODO: fill the denominator-statistic slot, keep the same
                #       AMSGrad and rectification branches, and apply the step.
                pass

        return loss
```
