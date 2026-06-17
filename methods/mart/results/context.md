# Context: adversarial training for `L_inf` robustness (circa 2019)

## Research question

Deep image classifiers are extremely fragile to adversarial examples: for a correctly
classified natural image `x` with label `y`, there exists a tiny perturbation `delta` with
`||delta||_inf <= eps` (eps small enough that the image looks unchanged to a human) such that
the network's prediction flips off `y`. The most reliable known defense is adversarial
training — train the network on perturbed inputs rather than clean ones — but even the best
adversarially trained models leave a large gap between **natural accuracy** (test accuracy on
clean images) and **robust accuracy** (test accuracy under a strong white-box attack), even on
a dataset as simple as CIFAR-10. The goal is a training procedure that meaningfully closes
that gap: higher robust accuracy under strong white-box `L_inf` attacks at acceptable clean
accuracy, using the same compute budget as standard adversarial training (one inner attack +
one outer SGD step per minibatch), and tunable by a single scalar that trades off the natural
and robust objectives.

A solution has to confront a structural subtlety that the prevailing formulation glosses over.
An adversarial example is, by its textbook definition, only defined relative to a *correctly
classified* natural example — it is a perturbation that turns a right answer into a wrong one.
But during training, at any given epoch, some natural training examples are already
misclassified by the current model, with no perturbation at all. The dominant training recipe
treats every example identically: it perturbs all of them and applies the same loss to all of
them, ignoring whether the underlying natural example was correctly classified in the first
place. Whether that distinction matters for the final robustness — and if so, how to exploit
it — was an open and largely unexamined question.

## Background

By this time the field has converged on a clear way to *state* the robust-training problem,
and on a clear way to *attack* it.

**The min-max (saddle-point) view.** Robust training is cast as an outer minimization over
network parameters wrapped around an inner maximization over the perturbation
(Madry et al. 2018):

```
min_theta  E_(x,y) [ max_{||x' - x||_p <= eps}  ell(h_theta(x'), y) ]
```

where `h_theta` is the classifier, `ell` is a classification loss (usually cross-entropy), and
`B_eps(x) = {x' : ||x' - x||_p <= eps}` is the `L_p` ball. The inner max produces the worst-case
perturbed input; the outer min trains on it. With `L_inf` the inner problem is solved by
projected gradient descent — repeated signed-gradient steps with projection back into the
`eps`-ball, started from a random point in the ball — used as a strong "first-order adversary."

**Formal risk.** Against the 0-1 loss the adversarial risk on a dataset is

```
R(h_theta) = (1/n) sum_i  max_{x' in B_eps(x_i)}  1( h_theta(x') != y_i ),
```

with `h_theta(x) = argmax_k p_k(x, theta)` and `p_k` the softmax probability of class `k`.

**Known difficulties of robust training, established before any new method.** Training a robust
network is genuinely harder than training a clean one. A robust decision boundary is provably
more complex than a boundary that merely separates the clean data, and a model needs *larger
capacity* to be robust — a small network that is highly accurate on clean data tends to fail to
become robust at all (Madry et al. 2018; Nakkiran 2019). The sample complexity of robust
generalization is higher than clean generalization, so robust training tends to need more data,
labeled or unlabeled (Schmidt et al. 2018; Carmon et al. 2019; Uesato et al. 2019). And there
is evidence that robustness and clean accuracy are in tension, so a useful method must let a
practitioner *trade them off* with a knob rather than maximize one blindly (Tsipras et al. 2019;
Zhang et al. 2019).

**A diagnostic about which training examples drive robustness.** A proof-of-concept measurement
on CIFAR-10 (`L_inf`, `eps = 8/255`) isolates the role of misclassified natural examples. Train
a small CNN with standard 10-step-PGD adversarial training to a moderate (~87%) clean training
accuracy, then split the natural training set, by the current model's prediction, into a subset
of *misclassified* examples `S-` and an equal-size subset of *correctly classified* examples
`S+`, and re-train while manipulating only one subset at a time:

- Leaving `S-` *unperturbed* during training (still perturbing everything else) causes a large
  drop in final robustness; leaving `S+` unperturbed barely changes it. The misclassified subset
  carries most of the robustness.
- Using only a *weak* one-step attack (FGSM) on `S-` in the inner maximization barely changes
  final robustness; using a weak attack on `S+` degrades it. The choice of attack strength on
  the misclassified subset is nearly irrelevant.
- Changing the *outer* (minimization) loss on `S-` — adding a consistency/KL regularizer between
  the clean and perturbed outputs — significantly improves final robustness; the same change on
  `S+` helps far less.

Read together: misclassified natural examples dominate the final robustness; what is done to
them in the *inner maximization* matters little, while what is done to them in the *outer
minimization* matters a great deal. These are facts about how existing adversarial training
behaves, measurable before any new objective exists.

**Margin-style and consistency-style losses already in circulation.** Two ingredients exist in
the literature as separate ideas. First, *margin* losses for attacks/defenses: instead of (or in
addition to) cross-entropy, penalize the gap between the true-class score and the best competing
class, which directly widens the classification margin (Carlini & Wagner 2017). Second,
*output-consistency / stability* losses: encourage the network's output distribution on a
perturbed input to match its output on the clean input, used to make predictions stable to small
input changes (Zheng et al. 2016).

## Baselines

The prior methods a new training procedure would be measured against and would react to. All
share the inner CE-PGD attack unless noted.

**Standard PGD adversarial training (Madry et al. 2018).** Solve the saddle point directly:
generate `x'` by random-start PGD that maximizes `CE(p(x'), y)` inside the `eps`-ball, then take
an SGD step minimizing `CE(p(x'), y)` on the perturbed input. PGD is treated as the strongest
practical first-order attack, and this remains the only method shown to train moderately robust
nets without being fully broken. **Gap (observed):** a single cross-entropy loss is applied to
the perturbed input for *every* example, identically, with no distinction between examples whose
clean version the model already gets wrong and those it gets right; the robust-vs-natural
accuracy gap stays large.

**Adversarial / clean logit pairing, ALP/CLP (Kannan et al. 2018).** Add an `L2` regularizer
that pulls the perturbed-input output toward the clean-input output:

```
CE(p(x'), y)  +  lambda * || p(x') - p(x) ||_2^2      (ALP)
CE(p(x),  y)  +  lambda * || p(x') - p(x) ||_2^2      (CLP)
```

**Gap (observed):** an `L2` pairing of outputs with a single global weight applied uniformly to
all examples, and without an accuracy/robustness decomposition behind it.

**TRADES (Zhang et al. 2019).** Decompose the robust error into a *natural* error plus a
*boundary* error and minimize a surrogate for that decomposition:

```
CE(p(x), y)  +  (1/lambda) * max_{x' in B_eps(x)} KL( p(x) || p(x') ).
```

The first term fits the clean label; the second pushes the decision boundary away from each
example by minimizing the worst-case clean-vs-perturbed output mismatch (a KL divergence). Here
the inner maximization maximizes the *KL* term rather than CE, and `1/lambda` is the single
natural-vs-robust trade-off knob (a typical setting puts the robustness weight around 6).
**Gap (observed):** the KL regularizer is applied with the *same weight to every example*,
whether the model already classifies its clean version correctly or not; nothing in the loss
treats an already-misclassified example differently from a confidently-correct one.

**Max-margin adversarial training, MMA (Ding et al. 2018).** Use a per-example perturbation
budget and a *hard* split: for examples it deems correctly classified, apply margin
maximization with an example-dependent `eps`; for examples it deems misclassified, apply
cross-entropy on the *clean* input. **Gap (observed):** the correctly-classified-vs-misclassified
decision is a *hard* threshold that is not itself optimized and cannot be learned jointly with
the network, and the two branches use different perturbation limits and different losses chosen
by hand.

## Evaluation settings

The natural yardsticks already in use:

- **Datasets / threat model.** CIFAR-10 (and CIFAR-100, SVHN, MNIST), `L_inf` perturbations.
  CIFAR `eps = 8/255`; MNIST `eps = 0.3`. Images normalized to `[0, 1]`; standard augmentation
  (4-pixel-pad random crop, random horizontal flip).
- **Architectures.** ResNet-18 / WideResNet-34-10 on CIFAR; smaller CNNs on MNIST.
- **Optimizer / schedule.** SGD, momentum 0.9, weight decay `2e-4` (or `5e-4`), initial
  learning rate 0.1 divided by 10 at the 75th and 90th epoch, ~100-120 epochs total.
- **Training attack.** 10-step PGD with random start, step size `eps/4`.
- **Test attacks (the yardstick).** White-box PGD-20 (random start, step size `eps/10`) as the
  primary robustness metric, plus one-step FGSM, C&W-style attacks, and black-box transfer.
  Metrics reported: clean accuracy and robust accuracy under each attack.
- **Optional data regime.** A semi-supervised setting that adds ~500K unlabeled tiny-images,
  pseudo-labeled by a clean model, weighting the unlabeled loss by a coefficient `gamma`.

## Code framework

The training procedure plugs into the standard adversarial-training harness: an outer loop
draws a minibatch, an inner routine produces perturbed inputs, and an outer SGD step updates
the parameters. The pieces that already exist are the model, the SGD optimizer with its cosine
or step schedule, the softmax / cross-entropy / KL primitives, and the projected-gradient
inner-attack pattern. The one unsettled slot is the minibatch loss minimized after the existing
attack has produced perturbed inputs.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdversarialTrainer:
    """Generic adversarial-training step. Owns the model and the L_inf attack
    hyperparameters; produces a per-minibatch training loss to backprop."""

    def __init__(self, model, eps, alpha, attack_steps, num_classes, **kwargs):
        self.model = model
        self.eps = eps                 # L_inf perturbation budget
        self.alpha = alpha             # inner-attack step size
        self.attack_steps = attack_steps
        self.num_classes = num_classes

    def _generate_adv(self, images, labels):
        # Standard L_inf PGD inner attack on the clean images.
        self.model.eval()
        adv = images.detach() + 0.001 * torch.randn_like(images)
        adv = torch.clamp(adv, 0.0, 1.0)
        for _ in range(self.attack_steps):
            adv.requires_grad_(True)
            loss = F.cross_entropy(self.model(adv), labels)
            grad = torch.autograd.grad(loss, adv)[0]
            adv = adv.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv - images, min=-self.eps, max=self.eps)
            adv = torch.clamp(images + delta, 0.0, 1.0).detach()
        return adv

    def train_step(self, images, labels, optimizer):
        adv_images = self._generate_adv(images, labels)
        self.model.train()
        optimizer.zero_grad()
        # TODO: define the minibatch loss to minimize.
        loss = None
        loss.backward()
        optimizer.step()
        return {'loss': loss.item()}
```

The inner attack and the optimizer step are fixed; the empty slot is the loss computed from the
clean and perturbed outputs.
