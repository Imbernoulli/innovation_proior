# Context: defending neural-network classifiers against bounded adversarial perturbations (circa 2018)

## Research question

Deep image classifiers achieve excellent accuracy on natural test data yet can be flipped to
a wrong label by a perturbation so small a human cannot see it. Formally, for a labeled point
`(x, y)` with `x in [0,1]^d`, an adversary is allowed any `x'` inside an `l_inf` ball
`B(x, eps) = { x' : ||x' - x||_inf <= eps }`, and succeeds if the classifier's prediction on
`x'` differs from `y`. The goal is to train a classifier that is accurate not only on clean
inputs but on the *worst* input inside every such ball — i.e. one with high accuracy under
white-box attacks that can read the model's gradients.

Two pressures pull against each other and both have to be respected at once. First, the
worst-case (robust) objective is brutal: even *defining* the training loss requires a maximum
over the perturbation ball, and the natural worst-case 0-1 loss `max_{x' in B(x,eps)} 1{c(x') != y}`
is NP-hard to optimize and non-differentiable. Second — and this is the harder pressure — there
is mounting evidence that pushing for robustness *costs* clean accuracy: a model tuned hard
against perturbations can end up classifying clean data worse than a standardly-trained one.
So the question is not only "how do we make a model robust at all," but "how do we make it
robust *with an explicit, controllable trade-off against clean accuracy*," and do it through a
loss that is differentiable, trainable at the scale of ResNets on CIFAR-10, and not just
robust to one particular attack.

## Background

By 2018 adversarial examples were a well-established phenomenon. Szegedy et al. (2013)
observed that imperceptible perturbations reliably flip deep-network predictions, and that the
same perturbation often transfers across models. Goodfellow, Shlens & Szegedy (2015) gave the
"linear" explanation: in high dimensions, a locally near-linear network accumulates many small
coordinate-wise changes into a large change in the logits, so the *steepest* perturbation
under an `l_inf` budget is the sign of the input gradient — the basis of the one-step
Fast Gradient Sign Method.

Two facts about the world frame the whole problem.

The first is geometric. An adversarial perturbation succeeds by moving an input across the
classifier's decision surface, so the location of that surface relative to the data matters as
much as the clean labels do. A classifier can have low clean error and still place complicated
or nearby boundaries around many data points, making small `l_inf` moves dangerous.

The second is that robustness and clean accuracy can genuinely conflict. Prior theoretical
examples show distributions on which the classifier that minimizes standard error and the
classifier that minimizes worst-case perturbation error are different. A defense that simply
drives clean error down can therefore remain fragile, while a defense that simply minimizes a
worst-case loss can give up clean accuracy without any explicit way to control the exchange.

The standard relaxation used to make the intractable 0-1 problem trainable is a *surrogate
loss*: replace the indicator with a margin loss `phi(f(x) y)` (hinge, logistic, exponential,
sigmoid), and learn by minimizing `R_phi(f) = E phi(f(X) Y)`. The bridge from surrogate back
to true error is the calibration theory of Bartlett, Jordan & McAuliffe (2006). A surrogate is
*classification-calibrated* if forcing the score to disagree in sign with the Bayes rule
strictly increases the surrogate risk; formally, with the conditional risk
`C_eta(alpha) = eta phi(alpha) + (1-eta) phi(-alpha)`, `H(eta) = inf_alpha C_eta(alpha)`, and
`H^-(eta) = inf_{alpha(2eta-1) <= 0} C_eta(alpha)`, calibration is `H^-(eta) > H(eta)` for
`eta != 1/2`. From these they build the `psi`-transform — `psi = (psi~)^{**}` (the convex hull),
where `psi~(theta) = H^-((1+theta)/2) - H((1+theta)/2)` — and prove that `psi` is non-decreasing,
continuous, convex on `[0,1]`, with `psi(0)=0`, and that excess 0-1 risk is controlled by excess
surrogate risk:

```
R_nat(f) - R_nat^*  <=  psi^{-1}( R_phi(f) - R_phi^* ).
```

For hinge loss `psi` is the identity. This machinery is exactly what makes surrogate
minimization *principled* for the clean classification problem — but it is a statement about
natural error only. It says nothing about what happens near the boundary under perturbation.

## Baselines

**FGSM adversarial training (Goodfellow, Shlens & Szegedy, 2015).** Augment training with
one-step adversarial examples `x' = x + eps * sign(grad_x L(f(x), y))` and train on them.
Cheap — one extra gradient per step. **Gap:** a single linear step is a weak adversary; models
trained this way remain vulnerable to stronger, iterative attacks, and can learn to mask
gradients rather than become genuinely robust.

**PGD adversarial training / robust optimization (Madry et al., 2018).** Cast defense as a
saddle-point (min-max) problem:

```
min_theta  E_{(x,y)}  max_{||delta||_inf <= eps}  L( f_theta(x + delta), y ),
```

and approximately solve the inner maximization with multi-step projected gradient descent from
a random start inside the ball:

```
x^{t+1} = Pi_{B(x,eps)} ( x^t + alpha * sign( grad_x L(f_theta(x^t), y) ) ).
```

Train the parameters on these strong adversarial examples with SGD. This is the strongest and
most widely-used empirical defense of its time, and PGD is treated as a near-"universal"
first-order adversary. The objective `E max_{x'} phi(f(x') y)` is an upper bound on the robust
error. **Gap:** that upper bound is loose in complex problems — it need not be a tight surrogate
for the robust 0-1 error — and, decisively, the formulation collapses *everything* into a single
worst-case-vs-label loss. It carries no separate knob that says how much clean accuracy one is
willing to spend for robustness; the clean-accuracy drop it incurs is whatever falls out, not
something the user can set.

**Regularization-based adversarial training (Kurakin et al. 2017; Ross & Doshi-Velez 2018;
Zheng et al. 2016; Miyato et al., VAT).** Instead of (or in addition to) training on
adversarial inputs, add a regularizer encouraging stability. Kurakin et al. pair the
adversarial prediction with the *label*; Ross & Doshi-Velez penalize the input-gradient norm;
Zheng et al. ("stability training") penalize the discrepancy between the clean prediction and
the prediction on a *Gaussian-noised* copy of the input; Miyato et al.'s virtual adversarial
training penalizes the divergence between the clean output distribution and the output on a
perturbation found *without* labels. **Gaps:** none comes with a guarantee tying the regularizer
to the robust error; Zheng et al. perturb with random Gaussian noise rather than a
worst-case-found point; the Kurakin/Ross regularizers compare the perturbed prediction to the
*label* or penalize a gradient norm rather than controlling worst-case prediction stability
under the perturbation model.

**Adversarial Logit Pairing (Kannan et al., 2018).** Generate `x'` by an iterated FGSM attack
on the label loss, then minimize
`alpha phi(f(x') y) + (1-alpha) phi(f(x) y) + ||f(x) - f(x')||_2 / lambda` — i.e. an `l_2`
penalty pulling the adversarial logits toward the clean logits. **Gap:** the logit-pairing term
is heuristic, with no calibrated-loss justification, and the adversarial point is found by
attacking the *label* loss.

**Relaxation / certified defenses (Wong & Kolter 2018; Raghunathan et al. 2018).** Optimize a
provable upper bound on the robust loss via convex relaxation or semidefinite programming,
yielding certificates. **Gap:** they target the worst-case bound and ignore performance on
non-adversarial inputs, leaving the robustness/accuracy trade-off untreated, and they scale
poorly to large networks.

## Evaluation settings

The natural yardsticks already in use for `l_inf` robustness:

- **Datasets / architectures.** MNIST with `eps = 0.3`, using a small convolutional network
  (e.g. the four-conv-plus-three-fc network of Carlini & Wagner 2017); CIFAR-10 with
  `eps = 0.031 ~ 8/255`, using a wide residual network (WRN-34-10, Zagoruyko & Komodakis 2016)
  or a ResNet-18 (He et al. 2016). Inputs are scaled to `[0,1]`.
- **Metrics.** Natural accuracy `A_nat = 1 - R_nat` on clean test data, and robust accuracy
  `A_adv = 1 - R_adv` under attack.
- **White-box attacks** (attacker has the model and its gradients): iterated FGSM, i.e. PGD,
  with several step counts (e.g. 20, 40, up to 1000 iterations), plus C&W, DeepFool, MI-FGSM,
  and L-BFGS attacks. Typical PGD step sizes: `0.003` for CIFAR-10, `0.01` for MNIST.
- **Black-box attacks**: transfer adversarial examples crafted on a separate source model (a
  naturally-trained model, or a PGD-trained model) onto the defended model.
- **Training hyperparameters in scope** (set by the existing harness): SGD with momentum and
  weight decay, learning rate `eta_2` (e.g. `0.1` on CIFAR-10, `0.01` on MNIST), batch size
  `128`, a cosine or step learning-rate schedule, the inner-attack step size `eta_1` and step
  count `K`, and the perturbation budget `eps`.
- **Sanity protocol:** because several published defenses were later shown to "work" only by
  obfuscating gradients (Athalye et al. 2018), a defense is checked under strong, many-step,
  multiple attacks rather than a single weak one.

## Code framework

The method plugs into a standard adversarial-training harness. The outer training loop, the
SGD optimizer (with its learning-rate schedule, momentum, and weight decay), the model
architecture, and the data pipeline already exist; what is *not* settled is the per-step
adversarial-training procedure itself — how to construct the perturbed input and what objective
to descend on. That single procedure is the empty slot.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def adversarial_training_loss(model, x_natural, y, optimizer,
                              step_size=0.003, epsilon=0.031,
                              perturb_steps=10, beta=1.0,
                              distance='l_inf'):
    # x_natural: minibatch images in [0, 1]; y: integer labels.
    # TODO: construct the perturbed inputs, compute the minibatch loss,
    # and return it to the outer SGD loop.
    pass


def train(model, data_loader, optimizer, scheduler):
    for images, labels in data_loader:
        optimizer.zero_grad()
        loss = adversarial_training_loss(model, images, labels, optimizer)
        loss.backward()
        optimizer.step()
        scheduler.step()
```

Standard primitives are available: `F.cross_entropy`, `F.kl_div`, `F.log_softmax`,
`F.softmax`, elementwise `sign`, clamping into the `l_inf` box around an input, and clamping
into the valid image range `[0,1]`. The inner sign-gradient ascent and box projection of a PGD
attack are off-the-shelf. The undetermined piece is what the inner attack should maximize and
what outer loss the parameters should minimize.
