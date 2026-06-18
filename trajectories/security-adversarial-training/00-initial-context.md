## Research question

A convolutional classifier can hit 99%+ on MNIST and 95% on CIFAR-10 clean, yet an `L_inf`
perturbation smaller than the sensor's own noise flips its prediction with confidence — and there is
one such perturbation next to essentially every image, transferable across architectures. The single
thing being designed here is the **adversarial training procedure**: the per-step rule that, given a
clean batch, produces the weights of a model whose predictions survive white-box `L_inf` attacks.
Everything else about the pipeline is fixed.

## Prior art before the first rung (the robustness lineage)

The first rung — PGD adversarial training — is the resolution of a line of defenses that came before
it, and each later rung on this ladder reacts to a specific failure of the one before. These are the
ancestors the ladder builds on.

- **Gradient masking / obfuscation defenses (defensive distillation, Papernot et al. 2016; input
  bit-depth squeezing; detector front-ends).** Each hardens the model against *one* attack by hiding or
  flattening the gradient the attacker uses, but the gradient is still there — a transfer or
  decision-based attack walks right past them. Gap: tuned against a single attack, no statement about
  attacks not yet tried; broken once the attacker adapts.
- **FGSM adversarial training (Goodfellow et al. 2015).** Augment training with one-step sign
  perturbations `x + eps*sign(grad_x J)`. Cheap and it raised FGSM robustness, but it linearizes the
  loss inside the ball, so it overfits to its own weak one-step adversary (the label-leaking
  pathology) and falls to iterative attacks. Gap: trains against a linearized adversary that a
  non-linear attacker bypasses.
- **The min-max view, pre-PGD (Huang et al. 2015; Shaham et al. 2015).** Framed defense as a saddle
  point `min_theta E max_{delta in S} L` but concluded the inner maximization over the non-concave
  loss was too hard, so they fell back to one-step linearization and only ever evaluated against FGSM.
  Gap: assumed the inner max intractable and never solved it, so the defense was never trained against
  a strong adversary.

The first rung resolves these by actually solving the inner maximization with multi-step PGD and
proving (via Danskin) that one ordinary SGD step on the worst-case point is the right outer descent.

## The fixed substrate

The training loop, the learning-rate schedule (cosine annealing), the model architectures
(SmallCNN, PreActResNet-18, VGG-11-BN), the data loading, and the evaluation attacks are all fixed and
must not be touched. The optimizer handed in is SGD with `lr`, `momentum`, and `weight_decay` already
configured. Each scenario fixes its budget: `eps = 0.3` for SmallCNN on MNIST, `eps = 8/255` for the
three CIFAR scenarios; the inner step size `alpha` and PGD-step count `attack_steps` are passed in.
The loop calls one method, `train_step(images, labels, optimizer)`, once per minibatch and reads back
its `'loss'`. Everything the method needs — `torch`, `torch.nn as nn`, `torch.nn.functional as F` — is
imported at module top.

## The editable interface

Exactly one region is editable — the `AdversarialTrainer` class in
`torchattacks/bench/custom_adv_train.py` (lines 10–54), with two methods. The contract:

- `__init__(self, model, eps, alpha, attack_steps, num_classes, **kwargs)` — `model` is the
  `nn.Module` to train; `eps` the `L_inf` budget; `alpha` the inner PGD step size; `attack_steps` the
  number of PGD steps; `num_classes` the label count (10 or 100). Any per-method state (regularizer
  weights, a proxy model) is set up here.
- `train_step(self, images, labels, optimizer) -> dict` — `images` are clean, shape `(N,C,H,W)`, in
  `[0,1]`; `labels` shape `(N,)`; `optimizer` is the externally configured SGD. The method crafts its
  adversarial batch, computes its loss, calls `optimizer.zero_grad()/backward()/step()` itself, and
  returns a dict with at least `'loss'` (float).

Every method on the ladder is a fill of this same contract. The starting point is the scaffold
default: **standard (non-adversarial) training** — no attack at all, plain cross-entropy on clean
images. Each later method replaces exactly this class body and nothing else.

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

Four scenarios spanning datasets and architectures: **SmallCNN on MNIST** (`eps = 0.3`),
**PreActResNet-18 on CIFAR-10** (`eps = 8/255`), **VGG-11-BN on CIFAR-10** (`eps = 8/255`), and
**PreActResNet-18 on CIFAR-100** (`eps = 8/255`, the hidden scenario). One seed, 42. After training,
each model is evaluated three ways, all higher-is-better: **clean accuracy** (unperturbed test
images), **robust accuracy under one-step FGSM**, and **robust accuracy under a 50-step PGD attack** —
PGD-50 is the primary metric. A method must lift PGD-50 robustness across all scenarios while keeping
clean accuracy reasonable; the standing tension is that pushing robustness up tends to drag clean
accuracy down.
