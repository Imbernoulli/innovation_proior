# Adversarial training and the robust generalization gap

## Research question

Adversarially trained image classifiers do not generalize their robustness. A
PreAct ResNet-18 trained on CIFAR-10 with standard PGD adversarial training for 200 epochs
reaches roughly 84% robust accuracy on the *training* set under a 10-step PGD attack, yet
only about 43% under the same attack on the *test* set — a robust generalization gap near
41%. Standard (non-robust) training on the same architecture keeps its train/test accuracy
gap under 10%. Worse, the gap is not static: in adversarial training the best test
robustness is reached early (around the first learning-rate decay), and from then on test
robustness *falls* as training robustness keeps rising. This is robust overfitting, and it
is the dominant obstacle to robustness; early stopping alone recovers much of the apparent
benefit of several stronger training recipes.

The precise goal is a training procedure that narrows the robust generalization gap
*without* giving up training robustness. Early stopping narrows the gap only by halting
before the model has learned enough, so its test robustness is capped by its lower training
robustness. A useful answer should preserve the existing adversarial-training interface:
plain PGD-AT, the KL-regularized variant, the misclassification-aware variant, and
semi-supervised variants already define natural losses and attacks, so the missing piece
should remain compatible with the existing training stack.

## Background

The standard formulation of adversarial training is a saddle-point (min-max) problem
(Madry et al., 2018, arXiv:1706.06083):

```
min_w  rho(w),    rho(w) = (1/n) sum_i  max_{ ||x'_i - x_i||_p <= eps }  ell( f_w(x'_i), y_i ),
```

where `f_w` is the network, `ell` the classification loss (cross-entropy), and `rho(w)` is
called the adversarial loss. The inner maximization is solved with projected gradient descent
(PGD): starting from a randomly perturbed point, repeatedly take a signed-gradient step and
project back into the `eps`-ball,

```
x'  <-  Pi_eps( x' + eta_1 * sign( grad_{x'} ell(f_w(x'), y) ) ).
```

Geometrically, training on these worst-case inputs flattens the **input loss landscape** —
the loss varies little as the input is perturbed inside the ball. Madry's PGD adversary has
held up as a reliable first-order attack, and adversarial training remains the most effective
defense that has not been broken by adaptive attacks.

The diagnostic facts that frame the problem are empirical and about *existing* systems:

- **Robust overfitting is real and large.** Across SVHN, CIFAR-10, CIFAR-100, and ImageNet,
  and across `L_inf` and `L_2` threat models, adversarially trained networks overfit the
  training set badly; test robustness peaks early and then degrades while training robustness
  keeps improving (Rice, Wong & Kolter, ICML 2020, arXiv:2002.11569). Early stopping recovers
  most of the apparent gains of fancier methods, but at the cost of low training robustness.

- **In standard training, flatness predicts generalization.** A network that sits in a
  *flat* region of its loss-vs-weight surface generalizes better than one in a *sharp* region
  (Keskar et al. 2017; Neyshabur et al. 2017; Li et al. 2018). The surface here is the
  **weight loss landscape**: how the loss changes as the *weights* move, not as the input
  moves. Whether the same flatness/generalization link holds under adversarial training had
  not been established — earlier attempts (Prabhu et al. 2019; Yu et al. 2018) used a fixed
  set of adversarial examples pre-generated on the unperturbed model to probe the surface,
  which underestimates the adversarial loss because the examples were crafted for a different
  (unperturbed) model than the one being evaluated.

- **Comparing weight loss landscapes requires removing scale invariance.** A ReLU network is
  scale-invariant: multiply one layer's weights by `c` and divide the next layer's by `c` and
  the function is unchanged, but a naive random perturbation of fixed size means something
  completely different for the two scalings. Li et al. (2018, arXiv:1712.09913) fix this with
  **filter normalization**: to plot `g(alpha) = rho(w + alpha*d)` along a random direction
  `d ~ N(0, I)`, rescale each filter of the direction to match the norm of the corresponding
  filter of the weights, `d_{l,j} <- (d_{l,j} / ||d_{l,j}||_F) * ||w_{l,j}||_F`. Only then are
  two networks' landscapes comparable.

There is one more background frame that turns flatness from a heuristic into a bound. The
PAC-Bayes analysis of Neyshabur et al. (2017, arXiv:1706.08947) bounds the expected error of
a randomized predictor whose weights are `w + nu`. With probability at least `1 - delta`
over the training draw,

```
E_nu[ L(f_{w+nu}) ]  <=  Lhat(f_w)
                      +  { E_nu[ Lhat(f_{w+nu}) ] - Lhat(f_w) }
                      +  4 * sqrt( ( KL(Q || P) + ln(2n/delta) ) / n ),
```

where `n` is the training-set size, `Lhat` is the empirical loss, `Q` is the posterior
distribution induced by the perturbation around `w`, and `P` is a data-independent prior. The
middle braced term is the **expected sharpness**: how much the empirical loss rises, on
average, when the weights are jittered. If `P = N(0, sigma^2 I)` and
`Q = N(w, sigma^2 I)`, then `KL(Q || P) = ||w||^2 / (2*sigma^2)`. Writing the variance as
`sigma^2 = a*||w||^2` fixes this KL contribution at `1/(2a)`; `a` is the variance-to-squared-
norm ratio, not an extra learned parameter. Once that relative scale is fixed, the remaining
data-dependent quantity in the bound is the expected sharpness of the weight loss landscape.

## Baselines

These are the prior adversarial-training methods a new method would be measured against and
would build on. Each shares the saddle-point skeleton above and differs only in the loss `ell`
and how the inner adversarial example is crafted.

**Vanilla PGD adversarial training (Madry et al., 2018).** Inner PGD attack on the
cross-entropy loss; outer SGD on the cross-entropy of the resulting adversarial examples. It
flattens the input loss landscape and gives the strongest broken-attack-free robustness of
its time. **Gap:** it leaves an enormous robust generalization gap and overfits — test
robustness peaks early then declines while training robustness keeps climbing.

**TRADES (Zhang et al., ICML 2019, arXiv:1901.08573).** Decomposes robust error into natural
error plus boundary error and minimizes a differentiable surrogate that trades the two off
through a regularizer:

```
min_w  (1/n) sum_i  { CE( f_w(x_i), y_i )  +  beta * max_{ ||x'_i - x_i|| <= eps }  KL( f_w(x_i) || f_w(x'_i) ) }.
```

The inner example is crafted by maximizing `KL(f_w(x) || f_w(x'))`, the divergence from the
clean prediction distribution to the perturbed prediction distribution,
`x' <- Pi_eps( x' + eta_1 * sign( grad_{x'} KL(f_w(x) || f_w(x')) ) )`, and `beta`
(the usual implementation name for the inverse trade-off coefficient, default 6) controls the
accuracy/robustness trade-off.
TRADES decouples keeping clean accuracy (the CE term) from pushing the decision boundary away
from data (the KL term). **Gap:** it is still an input-space regularizer; it improves the
trade-off but does not act directly on the weight surface, and it too overfits as training
proceeds.

**MART (Wang et al., ICLR 2020).** Adds an explicit emphasis on misclassified examples:

```
(1/n) sum_i  { BCE( f_w(x'_i), y_i )  +  lambda * KL( f_w(x_i) || f_w(x'_i) ) * ( 1 - [f_w(x_i)]_{y_i} ) },
```

where the boosted-CE term and the KL term are reweighted by how confidently the *clean* input
is classified, `lambda` default 5. **Gap:** another input-space loss design; the same robust
overfitting persists.

**Semi-supervised robust training / RST (Carmon et al., NeurIPS 2019).** Generate pseudo-labels
for extra unlabeled data with a natural model, then apply an adversarial loss (TRADES-style) on
labeled plus pseudo-labeled data, `rho^SSL(w) = rho^labeled(w) + lambda * rho^unlabeled(w)`.
**Gap:** improves robustness through *more data*, not through the optimization geometry; it
needs an external unlabeled corpus, and it likewise overfits the robust objective over a long
schedule.

**Random weight perturbation (He et al., CVPR 2019; flipout-style noise injection).** Inject a
*random* direction into the weights during training, `w + nu` with `nu` sampled from a fixed
distribution, in the spirit of the PAC-Bayes expectation `E_nu[rho(w+nu)]`. **Gap:** a random
direction is an inefficient probe in a high-dimensional surface; much of the noise can land in
directions that the loss barely uses, while large noise can make the optimization problem itself
harder.

Across these, the common limitation is that every method shapes the *input* side or adds
*data*. None directly controls the geometry of the robust objective around the learned
weights, and each one still overfits the robust objective over a long training schedule.

## Evaluation settings

The natural yardsticks were:

- **Datasets / architectures.** SVHN, CIFAR-10, CIFAR-100 (and ImageNet for the overfitting
  study). Architectures: PreAct ResNet-18 (the standard for ablations), VGG-19, and
  WideResNet-34-10 / WideResNet-28-10 (the standard for benchmark numbers).
- **Threat models.** `L_inf` with `eps = 8/255` and `L_2` with `eps = 128/255` on CIFAR-style
  data; `eps = 0.3` `L_inf` on MNIST. Inner training attack PGD-10 with step size `2/255`
  (`1/255` for SVHN).
- **Attacks for evaluation.** Clean accuracy; FGSM (one-step); PGD-20 / PGD-50 / PGD-100;
  the CW-loss attack optimized by PGD; the query-based black-box SPSA attack; and the
  parameter-free AutoAttack ensemble (APGD-CE, APGD-DLR, FAB, Square). The headline metric is
  test robust accuracy under a strong multi-step PGD / AutoAttack; "best" (peak over
  checkpoints) and "last" (final epoch) are both reported, the gap between them measuring
  overfitting.
- **Diagnostic probe.** The 1-D weight loss landscape `g(alpha) = rho(w + alpha*d)` with `d`
  filter-normalized and the adversarial loss recomputed *on-the-fly* for each perturbed model
  `f_{w+alpha*d}` (PGD examples regenerated for the perturbed weights, not reused from the
  center). Repeated over several random directions to check stability.
- **Optimizer / schedule.** SGD, momentum 0.9, weight decay `5e-4`, 200 epochs, initial
  learning rate 0.1 with a piece-wise (divide by 10 at epochs 100 and 150) or cosine schedule;
  batch size 128; `32x32` random crop with 4-pixel padding and horizontal flip.

## Code framework

The per-step adversarial-training procedure plugs into a fixed outer harness: the data
pipeline, the model, the learning-rate schedule, and the SGD optimizer (with `lr`,
`momentum`, and `weight_decay` already configured) all live outside and call into a single
`train_step`. The substrate already exists for the KL-regularized adversarial loss: craft
`x_adv` by maximizing the divergence between clean and perturbed predictions, then descend on
clean cross-entropy plus the same divergence term. What is unsettled is how the minibatch
update should be modified so that reducing the robust loss on the training set does not simply
sharpen the solution and widen the robust generalization gap.

```python
import torch
import torch.nn.functional as F


def perturb_input(model, images, eps, step_size, perturb_steps):
    """KL-based PGD inner adversary: random start, signed-gradient ascent on
    KL(clean prediction || perturbed prediction), then projection into the L_inf eps-ball."""
    model.eval()
    x_adv = torch.clamp(images.detach() + 0.001 * torch.randn_like(images), 0.0, 1.0)
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        loss = F.kl_div(F.log_softmax(model(x_adv), dim=1),
                        F.softmax(model(images), dim=1),
                        reduction='sum')
        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, images - eps), images + eps)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()
    return x_adv.detach()


class UpdateHelper:
    """A helper that may prepare state around each SGD step. The outer loop owns
    the model and the SGD optimizer (lr/momentum/weight_decay preset). The helper
    details are the open slot."""

    def __init__(self, model, **kwargs):
        self.model = model
        # TODO: any state the update procedure needs.

    def before_step(self, x_adv, images, labels):
        # TODO: prepare the minibatch update.
        pass

    def after_step(self, state):
        # TODO: finish any helper work after the optimizer step.
        pass


def train_step(model, images, labels, optimizer, helper,
               eps, step_size, perturb_steps, beta):
    # 1) craft adversarial examples with the existing KL-based attack
    x_adv = perturb_input(model, images, eps, step_size, perturb_steps)

    model.train()
    # 2) TODO: prepare the minibatch update (the procedure we design)
    state = helper.before_step(x_adv, images, labels)

    # 3) compute the existing KL-regularized loss and take the SGD step
    optimizer.zero_grad()
    logits_adv = model(x_adv)
    loss_robust = F.kl_div(F.log_softmax(logits_adv, dim=1),
                           F.softmax(model(images), dim=1),
                           reduction='batchmean')
    logits = model(images)
    loss_natural = F.cross_entropy(logits, labels)
    loss = loss_natural + beta * loss_robust
    loss.backward()
    optimizer.step()

    # 4) TODO: finish any helper work after the optimizer step
    helper.after_step(state)
    return {'loss': loss.item()}
```
