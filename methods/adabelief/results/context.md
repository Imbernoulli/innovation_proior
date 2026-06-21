# Context

## Research question

When choosing a first-order optimizer, practitioners weigh adaptive methods like Adam, which converge fast early and are widely used in hard-to-stabilize settings such as GANs, against SGD with momentum, which is the common default on standard supervised tasks such as CNN image classification. The question is how to set the per-parameter step size in an Adam-style update — what statistic of the gradient stream should drive the adaptive denominator.

## Background

The workhorse adaptive optimizer is Adam (Kingma & Ba 2014). It keeps two exponential moving averages: of the gradient, m_t = beta_1 m_{t-1} + (1 - beta_1) g_t, and of the squared gradient, v_t = beta_2 v_{t-1} + (1 - beta_2) g_t^2. It bias-corrects both and steps by theta_t = theta_{t-1} - alpha mhat_t / (sqrt(vhat_t) + epsilon). The denominator sqrt(v_t) gives every coordinate its own effective learning rate set by the gradient's recent magnitude: directions with large gradients get small steps, directions with small gradients get large steps. AdaGrad (Duchi et al. 2011) is the ancestor of this idea, with accumulated squared gradients in the denominator; its accumulation never decays, while Adam's exponential average keeps the statistic local in time.

A large body of work refines the Adam family. AMSGrad (Reddi et al. 2018) takes the running maximum of v_t in the convergence analysis. Yogi (Zaheer et al. 2018) controls how v_t reacts to minibatch size. RAdam (Liu et al. 2019) rectifies the variance of the adaptive learning rate in the first steps with a warmup-like correction. MSVAG (Balles & Hennig 2018) dissects Adam into a sign update times a variance-dependent magnitude. AdaBound and SWATS (Luo et al. 2019; Keskar & Socher 2017) interpolate or switch from Adam toward SGD. AdamW (Loshchilov & Hutter 2019) decouples weight decay from the gradient step. Wilson et al. 2017 study how adaptive methods and well-tuned SGD compare at the solutions they reach.

## Baselines

**Adam (Kingma & Ba 2014).** Adam keeps EMAs of g_t and g_t^2, bias-corrects them, and updates theta by subtracting alpha mhat_t / (sqrt(vhat_t) + epsilon). It is fast early and widely used in unstable settings.

**SGD with momentum.** Momentum keeps a velocity-like moving average of gradients and steps in that direction. It is a common default on vision tasks.

**AMSGrad (Reddi et al. 2018).** AMSGrad replaces the current second-moment denominator with a running maximum so the effective adaptive rate does not increase.

**RAdam (Liu et al. 2019).** RAdam rectifies the early-stage variance of the adaptive denominator. When the estimated degrees of freedom are too low, it falls back toward a momentum-style update; when the statistic is reliable enough, it uses an adaptive denominator.

**MSVAG (Balles & Hennig 2018).** MSVAG analyzes Adam as a sign component with a variance-adaptive magnitude, reweighting the update using variance estimates.

**AdamW (Loshchilov & Hutter 2019).** AdamW applies weight decay directly to the parameters instead of folding it into the gradient. It can be combined with any Adam-family update.

## Evaluation settings

Natural yardsticks include image classification on CIFAR-10/100 with ResNet, DenseNet, and VGG, ImageNet classification with ResNet-18 and top-1 accuracy, language modeling with LSTMs on Penn Treebank measured by perplexity, and GAN training on CIFAR-10 with WGAN or WGAN-GP measured by FID and training stability. Standard data augmentation, learning-rate schedules, weight decay choices, and optimizer hyperparameter searches are part of the protocol.

## Code framework

The substrate is PyTorch's `torch.optim.Optimizer`: per-parameter `state`, `param_groups` carrying learning-rate and decay options, optional running maxima, optional rectified early-step behavior, and a `step()` method that reads each parameter's `.grad`. An Adam-family optimizer already has the first-moment buffer, bias correction, weight-decay placement, and optional RAdam/AMSGrad-style branches. The slot to fill is the denominator statistic and the update that uses it.

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
