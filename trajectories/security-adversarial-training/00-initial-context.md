## Research question

A convolutional classifier can reach 99%+ clean accuracy on MNIST and 95% on CIFAR-10, yet an `L_inf` perturbation smaller than sensor noise flips its predictions with high confidence, and such a perturbation exists near almost every image. The task is to design the **adversarial training procedure**: the per-step rule that, given a clean batch, produces model weights whose predictions survive white-box `L_inf` attacks. Everything else in the pipeline is fixed.

## Prior art / Background / Baselines

Existing defenses fall into three families.

- **Gradient masking / obfuscation defenses (defensive distillation, bit-depth reduction, detector front-ends).** They hide or flatten the gradient that a particular attack relies on, making gradient-based attacks less effective.

- **FGSM adversarial training.** It augments each minibatch with the one-step sign perturbation `x + eps * sign(grad_x J)`.

- **Min-max robust optimization with one-step inner approximation.** It frames training as `min_theta E max_{||delta||_inf <= eps} L(theta, x+delta, y)` and approximates the inner maximization by a single linearized step.

## Fixed substrate / Code framework

The training loop, learning-rate schedule, model architectures (SmallCNN, PreActResNet-18, VGG-11-BN), data loading, and evaluation attacks are fixed. The optimizer is SGD with `lr`, `momentum`, and `weight_decay` already configured. Each scenario fixes its budget: `eps = 0.3` for SmallCNN on MNIST, `eps = 8/255` for the three CIFAR scenarios. The inner step size `alpha` and step count `attack_steps` are passed in. The loop calls one method, `train_step(images, labels, optimizer)`, once per minibatch and reads back its `'loss'`. `torch`, `torch.nn as nn`, and `torch.nn.functional as F` are imported at the top.

## Editable interface

Only the `AdversarialTrainer` class in `torchattacks/bench/custom_adv_train.py` is editable. It has two methods:

- `__init__(self, model, eps, alpha, attack_steps, num_classes, **kwargs)` — receives the `nn.Module`, the `L_inf` budget `eps`, the inner step size `alpha`, the number of inner steps `attack_steps`, and the label count. Set up any per-method state here.
- `train_step(self, images, labels, optimizer) -> dict` — receives clean `images` in `[0,1]`, shape `(N,C,H,W)`, labels `(N,)`, and the configured SGD optimizer. The method builds its adversarial batch, computes loss, calls `optimizer.zero_grad()/backward()/step()`, and returns `{'loss': float}`.

The default class body is plain cross-entropy training on clean images; any method replaces exactly this body and nothing else.

```python
"""Custom adversarial training method for MLS-Bench."""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════════
# EDITABLE — implement AdversarialTrainer below
# ═══════════════════════════════════════════════════════════════════
class AdversarialTrainer:
    """Default fill: standard (non-adversarial) training — the floor."""

    def __init__(self, model, eps, alpha, attack_steps, num_classes, **kwargs):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.attack_steps = attack_steps
        self.num_classes = num_classes

    def train_step(self, images, labels, optimizer):
        # Default: standard (non-adversarial) training — no inner attack.
        self.model.train()
        outputs = self.model(images)
        loss = F.cross_entropy(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {'loss': loss.item()}
# ═══════════════════════════════════════════════════════════════════
# END EDITABLE
# ═══════════════════════════════════════════════════════════════════
```

## Evaluation settings

Four scenarios: **SmallCNN on MNIST** (`eps = 0.3`), **PreActResNet-18 on CIFAR-10** (`eps = 8/255`), **VGG-11-BN on CIFAR-10** (`eps = 8/255`), and **PreActResNet-18 on CIFAR-100** (`eps = 8/255`, hidden). One seed, 42. After training, each model is scored three ways, all higher-is-better: clean accuracy, robust accuracy under one-step FGSM, and robust accuracy under a 50-step PGD attack. PGD-50 is the primary metric. Robustness and clean accuracy are in tension.
