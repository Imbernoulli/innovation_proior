# Context: evaluating adversarial robustness when gradients are unavailable or untrustworthy

## Research question

We want to measure how robust an image classifier really is to small, worst-case input
perturbations. The standard tool is to *attack* the model: search for a perturbation `delta`
with `||delta||_inf <= eps` that flips the prediction, and report the fraction of inputs that
survive. The strongest, cheapest attacks rely on `nabla_x` of the model's loss, computed by
backpropagation. Many proposed defenses insert a non-differentiable preprocessing step (JPEG
compression, quantization, a discrete purification loop) before the network, so backprop cannot
be run through the whole pipeline. Even when an analytic gradient exists, a model can
intentionally or accidentally produce gradients that are tiny, noisy, or point the wrong way
("gradient masking"), so a gradient-based attacker may report a high robust accuracy that does
not reflect the true vulnerability.

The question is how to mount a strong attack that needs **no access to model gradients or
weights** — only the ability to feed inputs forward and read out the model's logits — on the
`L_inf` ball in high input dimension, against models whose outputs may be stochastic.

## Background

By the time this problem is posed, adversarial examples are well established: imperceptible
`L_inf`-bounded perturbations reliably fool deep classifiers (Szegedy et al. 2013; Goodfellow
et al. 2014). The dominant *white-box* attack is projected gradient descent (PGD): take a
gradient step on a per-input loss, project back into the `eps`-ball, repeat (Kurakin et al.
2016; Madry et al. 2017). A wave of defenses then reports high robustness — input
transformations (Guo et al. 2017), density-based purification (Song et al. 2017),
autoencoder denoisers (Liao et al. 2017), randomized resizing (Xie et al. 2017). A recurring
worry, documented as **gradient masking**, is that several of these may not actually remove
adversarial inputs; they may only make the *gradient signal* that PGD relies on uninformative,
so the attack fails for the wrong reason (Papernot et al. 2017; the cleverhans blog).
Distinguishing "this model has few adversarial examples" from "this model merely hides them
from the gradient" is the live empirical question, and it cannot be answered with an attack
that itself depends on the gradient.

Three pieces of background are load-bearing.

**Stochastic approximation.** When we want to minimize a scalar `L(theta)` and can only obtain
*noisy measurements* of `L`, the classical machinery is stochastic approximation. The
Robbins–Monro recursion (1951) descends with a noisy gradient estimate and a decaying step,
`theta_{k+1} = theta_k - a_k * ghat_k(theta_k)`, and converges to a stationary point under mild
conditions if `a_k` shrinks neither too fast nor too slowly. The Robbins–Monro framework
presumes access to a (noisy) *gradient* `ghat`. When even that is unavailable — we can measure
the function value `L` but have no formula for its derivative — the classical fallback is the
Kiefer–Wolfowitz finite-difference scheme (1952): probe each coordinate separately,

```
ghat_{k,i} = ( L(theta + c * e_i) - L(theta - c * e_i) ) / (2c),
```

with `e_i` the i-th basis vector and `c` a small probe radius. This is just the textbook
definition of a partial derivative, made numerical, and it requires on the order of `2D`
function evaluations to assemble one full gradient in `D` dimensions.

**The margin / logit objective.** For finding adversarial examples, the cross-entropy loss is
a poor objective once the model is confident: the softmax saturates, so its gradient (and its
*value*) barely move as the input changes, giving a flat, uninformative landscape. Carlini &
Wagner (2017) showed that a **margin read off the logits** behaves much better, because logits
stay roughly linear in the input where probabilities are pinned at 0 or 1. For an untargeted
attack on true label `y0`, the correct-class advantage is

```
J(x) = m(x)_{y0} - max_{j != y0} m(x)_j,
```

where `m(x)_j` is the logit of class `j`. `J(x) < 0` exactly when `x` is misclassified, and `J`
keeps a usable slope well past the point where cross-entropy has flattened. Its negative,
`max_{j != y0} m(x)_j - m(x)_{y0}`, is the misclassification margin used by common code
interfaces; it is positive after a successful untargeted attack.

**Adam as the inner descent step.** Adam (Kingma & Ba 2014) maintains per-coordinate
exponential moving averages of the gradient and its square and steps by their ratio, giving a
self-normalizing, per-coordinate-scaled update that copes well with noisy and unevenly-scaled
gradients. It is the standard drop-in replacement for a raw gradient step when the gradient
estimates are noisy, which is exactly the regime here.

## Baselines

These are the attacks a new black-box attack is measured against and reacts to.

**Projected gradient descent (PGD), white-box (Kurakin et al. 2016; Madry et al. 2017).**
Iterate `x <- Pi_{N_eps(x0)}( x - alpha * nabla_x J(x) )`, where `Pi` projects onto the
`L_inf` ball and `J` is the correct-class advantage; in practice the raw gradient step is
replaced by Adam and `x` is randomly initialized inside the ball. It needs `nabla_x J`,
computed by backprop.

**Transfer-based attacks (Papernot et al. 2017; Szegedy et al. 2013).** Craft adversarial
examples on a *surrogate* model the attacker can differentiate, then feed them to the target.
Success depends on how similar the surrogate is to the target model.

**Coordinatewise finite-difference / zeroth-order (the Kiefer–Wolfowitz route; ZOO, Chen et
al. 2017).** Estimate the gradient one coordinate at a time with two-sided differences and
descend (ZOO pairs this with coordinatewise Adam and importance sampling). This is a genuine
black-box attack — it queries only `L`. Assembling a full gradient costs on the order of `2D`
queries, where `D` is the number of input pixels times channels.

**Natural evolution strategies (NES; Wierstra et al. 2008; Ilyas et al. 2017).** Estimate a
*smoothed* gradient of `E_{u}[L(x + sigma u)]` by sampling Gaussian directions `u ~ N(0, I)`
and forming `(1/sigma) * E[ u * (antithetic loss difference) ]`; descend on the estimate. This
is an efficient black-box random-search family rather than a coordinate sweep.

## Evaluation settings

The natural yardsticks are:

- **Datasets / models.** CIFAR-10 (32×32 color, 10 classes; Krizhevsky 2009) and ImageNet
  (Deng et al. 2009), attacking standard convolutional classifiers (VGG-style nets, ResNet-50)
  and the defended pipelines listed above (input transformations, density purification,
  denoising autoencoders, randomized resizing, adversarially trained nets).
- **Threat model.** `L_inf` perturbation with a fixed budget `eps` (e.g. CIFAR `eps = 8/255`,
  ImageNet `eps = 2/255`); inputs constrained to the valid pixel range `[0, 1]`. The attacker
  sees only forward outputs (logits / scores), never gradients or weights.
- **Metrics.** Adversarial accuracy `1 - hat{L}(theta, f)` of the model under the attack
  (lower means a stronger attack); equivalently attack success rate. The number of model
  queries (forward evaluations) consumed is the cost budget — a function of iterations × samples
  per step — and is the natural efficiency axis.
- **Protocol.** Attacks run for a capped number of iterations, optionally stopping early once
  the margin objective crosses a misclassification threshold; perturbations are
  random-initialized inside the `eps`-ball and re-projected after each step. Comparisons sweep
  the per-step sample budget to trace accuracy versus total queries.

## Code framework

The attack plugs into a fixed black-box evaluation harness. The harness owns the model wrapper
(it returns logits and counts every forward call against the query budget), the data, the
`eps` and per-sample query budget, and the validity checks (`L_inf` and `[0,1]`). The single
empty slot is the attack algorithm itself: given only forward access to the model, produce an
adversarial image inside the ball. Everything the attack is allowed to lean on already exists —
a forward-only model call, a margin loss read off the logits, a generic per-coordinate descent
optimizer (Adam) over the perturbation, and a projection back into the feasible set. The
unfilled part is the rule that chooses each update from forward-query results alone.

```python
import torch
import torch.nn as nn


def margin_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Untargeted logit margin, per sample: max_{j != y} logit_j - logit_y.
    Positive iff misclassified; stays informative where cross-entropy saturates."""
    top2_vals, top2_idx = torch.topk(logits, 2, dim=-1)
    target = logits[torch.arange(logits.shape[0]), labels]
    max_other = torch.where(top2_idx[..., 0] == labels, top2_vals[..., 1], top2_vals[..., 0])
    return max_other - target


def linf_project(dx: torch.Tensor, x: torch.Tensor, eps: float) -> torch.Tensor:
    """Project the perturbation back into the L_inf ball and keep x+dx in [0,1]."""
    dx = torch.clamp(dx, min=-eps, max=eps)
    x_adv = torch.clamp(x + dx, min=0.0, max=1.0)
    return x_adv - x


def run_attack(model, images, labels, eps, num_steps, device, n_classes):
    """Black-box L_inf attack. Only forward access: each model(x) call returns logits
    and consumes x.shape[0] queries. Must return adv images in [0,1] inside the eps-ball.

    The descent variable is the perturbation dx (same shape as one image). We have a
    margin objective and an Adam optimizer ready to take steps on dx, and a projection
    to keep dx feasible. The update body is intentionally left blank.
    """
    model.eval()
    dx = torch.zeros_like(images)
    optimizer = torch.optim.Adam([dx], lr=0.01)

    for _ in range(num_steps):
        optimizer.zero_grad()
        # TODO: estimate a descent direction for the attack objective using only
        #       forward queries to `model`, and write it into dx.grad.
        dx.grad = ...   # the object we will design
        optimizer.step()
        dx.data = linf_project(dx.data, images, eps)

    return torch.clamp(images + dx, 0.0, 1.0).detach()
```

The harness supplies the model, the budget, and the feasibility machinery; the attack must fill
in how to turn forward-only queries into a usable step on `dx`.
