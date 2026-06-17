# Context: white-box Lp evasion attacks and the reliability of robustness evaluation (circa 2018-2019)

## Research question

A classifier `g: [0,1]^d -> R^K` decides by `argmax_k g_k(x)`. Given a correctly classified
point `x_orig` of class `c`, a metric `d(.,.)` and a budget `eps>0`, an *adversarial example*
is a point `z` with `d(x_orig, z) <= eps`, `z in [0,1]^d`, but `argmax_k g_k(z) != c`. The
attacker's job is to *find* such a `z` inside the feasible ball; equivalently, to solve
`max_{z in S} L(g(z), c)` for a surrogate loss `L`, with `S = {z : ||z - x_orig||_p <= eps} ∩
[0,1]^d`. For `p in {2, infinity}` the projection onto `S` is available in closed form, so the
problem is a constrained first-order maximization.

The deeper problem is not finding *an* attack — it is finding a *reliable* one. The field is in
a state where a newly proposed defense reports some robust accuracy, and there is no way to
trust that number without an independent re-evaluation, because the standard evaluation tools
overestimate robustness: a defense looks robust only because the attack used to test it was too
weak or was silently disabled by the defense. A trustworthy evaluation would have to (1) not
depend on a step size or other knobs the user must tune per model — tuning is exactly where
evaluations go wrong; (2) not be defeatable by a defense that leaves the *decision* unchanged
but reshapes the loss surface the attack climbs; (3) be cheap enough to run as a default test;
and (4) not fail silently on a single failure mode, since different defenses break different
attacks. Closing the gap between *reported* and *true* robustness, autonomously, is the problem.

## Background

Adversarial examples — small, often imperceptible input perturbations that flip a classifier's
decision — are a safety problem for deployed models. Many defenses have been proposed and most
were later broken once a more powerful or *adapted* attack was applied (Carlini & Wagner 2017;
Athalye, Carlini & Wagner, ICML 2018; Mosbach et al. 2018). Adversarial training (Madry et al.
ICLR 2018) is one of the few that has held up, alongside variants using other losses (TRADES,
Zhang et al. 2019) and provable/certified defenses, though the latter are not yet competitive
with empirical robustness on CIFAR-10 at large `eps`.

The load-bearing concepts:

- **The threat model and the surrogate-maximization view.** Misclassification inside the ball
  is achieved by maximizing a surrogate `L(g(z), c)` over `S`. The choice of `L` and of the
  optimizer for this constrained problem are the two levers an attack has.

- **Steepest ascent under an `l_p` norm.** For `l_infinity`, the steepest-ascent direction of a
  linear model is the *sign* of the gradient (each coordinate moved to its box boundary); for
  `l_2` it is the normalized gradient. This is why `l_infinity` attacks step along
  `sign(grad)` and `l_2` attacks along `grad/||grad||`.

- **Projection onto `S`.** After a gradient step the iterate is projected back: for
  `l_infinity`, clip to `[x_orig - eps, x_orig + eps]` and then to `[0,1]`; for `l_2`, rescale
  the perturbation to norm `eps`.

- **Diagnostic finding: cross-entropy is scale-variant, and that is exploitable.** With logits
  `z` and softmax probabilities `p_i = e^{z_i}/sum_j e^{z_j}`, the CE loss `CE(x,y) = -log p_y =
  -z_y + log(sum_j e^{z_j})` is invariant to *shifts* of the logits but not to *rescaling*. Its
  input-gradient is `grad_x CE(x,y) = (-1 + p_y) grad_x z_y + sum_{i!=y} p_i grad_x z_i`. When
  `p_y ≈ 1` (and `p_i ≈ 0` for `i != y`), this is `≈ 0`; in single precision, where only
  exponents roughly in `[-127, 127]` are representable, it becomes *exactly* zero, so even
  `sign(grad)` is zero and the attack gets no ascent direction. Crucially, a defender can force
  `p_y ≈ 1` for *any* model by using the *equivalent* classifier `h = alpha·g` (same decision
  for every `x`) with a large constant `alpha > 0`: a measured experiment dividing a CIFAR-10
  model's logits by `alpha in {1, 10, 100, 1000}` shows the fraction of zero gradient entries
  and the (falsely high) robust accuracy of a CE-PGD attack swing dramatically with `alpha`.
  This is a known mechanism of *gradient masking* — the model is no more robust, but the attack
  is blinded. (Carlini & Wagner 2017 first noted CE-gradient vanishing.)

- **Diagnostic finding: a fixed step size makes the iteration budget-agnostic and trend-blind.**
  Run PGD (with or without a momentum term) for 1000 iterations at several fixed step sizes
  `eps/t`, `t in {0.5,1,2,4,10,25,100}`, on robust MNIST/CIFAR-10 models. The best-so-far loss
  *plateaus* after a few iterations except at extremely small step sizes — which themselves do
  not translate into better final loss. So the number of iterations is a poor proxy for attack
  strength, and the right step size differs by model. The optimizer also never reacts to whether
  it is actually making progress.

- **Guidelines exist but are not enough.** Community guidelines for evaluating robustness
  (Carlini et al. 2019) help, yet even papers that try to follow them still mis-evaluate,
  because there is no *protocol* that works autonomously without per-model fine-tuning.

## Baselines

- **FGSM (Goodfellow, Shlens & Szegedy, ICLR 2015).** One step: `delta = eps·sign(grad_x
  CE(x,y))`, relying on a linear approximation of the loss over the ball. *Limitation:* a single
  linearized step is weak; it leaves large headroom and badly overestimates robustness on
  anything but the most fragile models.

- **PGD attack (Kurakin, Goodfellow & Bengio, ICLR-W 2017; Madry et al., ICLR 2018).** The
  dominant white-box attack. Iterate, for `k = 1..N`,
  `x^{(k+1)} = P_S(x^{(k)} + eta·sign(grad f(x^{(k)})))` with a *fixed* step `eta`, from a
  random start `x^{(0)} = x_orig + zeta`, optimizing the CE loss. Cheap and strong in many
  cases. *Limitations:* (i) the fixed `eta` has no convergence guarantee even on convex
  problems and its value strongly determines the outcome; (ii) the loss plateaus well before the
  iteration budget is spent and the scheme cannot tell whether it is still improving; (iii) on
  the CE loss it inherits the scale-variance failure above, so a rescaled (decision-identical)
  model masks its gradient and PGD reports false robustness.

- **CW loss (Carlini & Wagner, IEEE S&P 2017).** Replace CE with a decision-aligned margin,
  `CW(x,y) = -z_y + max_{i!=y} z_i`. Its global maximum is positive exactly when an adversarial
  example exists, and it is *shift*-invariant. *Limitation:* it is **not** scale-invariant — an
  extreme logit rescaling can in principle still induce gradient masking — and its magnitude is
  unbounded across models, so it behaves inconsistently when used as the ascent objective.

- **FAB (Croce & Hein, 2019).** A white-box *minimum-norm* attack: instead of maximizing a loss
  inside a fixed ball, it iteratively linearizes the decision boundary and steps toward the
  closest boundary, finding the smallest perturbation that misclassifies. It is observed to
  remain effective on some models affected by gradient masking. *Limitation:* the untargeted
  version computes the full classifier Jacobian each iteration (cost scaling with the number of
  classes `K`), so it is prohibitive for CIFAR-100/ImageNet; and as a single attack it does not
  by itself give a reliable budget-bounded success rate.

- **Square Attack (Andriushchenko, Croce, Flammarion & Hein, ECCV 2020).** A *score-based
  black-box* attack: random search that proposes square-shaped, norm-bounded perturbations and
  keeps a proposal if it increases a margin loss, using only the model's output scores — no
  gradient at all. Query-efficient and competitive with white-box attacks. *Limitation:* being
  black-box it can need many queries and, alone, is not a complete robustness protocol; but its
  not using gradients is precisely what makes it immune to gradient masking.

The common thread across the white-box baselines: each is either tied to a user-chosen step
size, or to a scale-fragile loss, or to a single failure mode that some defense can exploit.

## Evaluation settings

The natural yardsticks already in use, all pre-existing:

- **Datasets / threat models.** MNIST (`l_infinity`, `eps = 0.3`), CIFAR-10 (`l_infinity`,
  `eps = 8/255` and `0.031`; `l_2`, `eps = 0.5`), CIFAR-100 (`l_infinity`, `eps = 8/255`),
  ImageNet (`l_infinity`, `eps = 4/255`; `l_2`, `eps = 3`). Inputs in `[0,1]^d`.
- **Models.** A large pool of published "robust" classifiers from recent top venues — adversarially
  trained models, TRADES, certified/provable defenses, and *randomized* defenses (whose output
  is stochastic, so robust accuracy must be estimated over several runs).
- **Metrics.** Clean accuracy; robust accuracy (fraction of points for which *no* attack in the
  test finds an adversarial example within the ball); equivalently attack success rate
  `1 - robust_acc`. For randomized defenses, robust accuracy averaged/worst-cased over repeated
  runs on the same adversarial batch.
- **Protocol.** A point counts as defended only if it survives the attack within the `l_p`-ball
  with the perturbation validity and `[0,1]` range enforced; the natural comparison is between
  an attack's reported robust accuracy and the value claimed in each defense paper. For
  multi-attack testing the worst case over the attacks is taken per point.

## Code framework

The attack plugs into the standard white-box harness that already exists: a model with a
differentiable forward pass, an `l_p` projection onto the feasible ball, autograd to get the
input-gradient of a scalar objective, and an outer loop that, for each still-defended point,
runs an attack routine and marks the point broken if the routine returns a valid in-ball point
whose prediction differs from the true label. Nothing about *which* objective to climb, *how* to
schedule the optimization, or *how many complementary routines* to run and combine is settled —
those are exactly what is to be designed. The substrate is only the generic machinery:

```python
import torch
import torch.nn as nn


def project_linf(x_adv, x_orig, eps):
    """Project onto the Linf ball of radius eps around x_orig, then onto [0,1]."""
    x_adv = torch.min(torch.max(x_adv, x_orig - eps), x_orig + eps)
    return x_adv.clamp(0.0, 1.0)


def attack_objective(logits, y):
    """Scalar (per-example) objective whose ascent should produce misclassification.
    Which objective to maximize is intentionally left open."""
    # TODO: choose the attack objective.
    raise NotImplementedError


def optimize_in_ball(model, x_orig, y, eps, n_iter):
    """Maximize attack_objective over the Linf ball around x_orig using first-order
    steps with projection. The update rule and any step-size schedule are open."""
    x_adv = x_orig.clone()
    for k in range(n_iter):
        x_adv.requires_grad_(True)
        loss = attack_objective(model(x_adv), y).sum()
        grad = torch.autograd.grad(loss, [x_adv])[0]
        with torch.no_grad():
            # TODO: choose the projected update and any step-size scheduling.
            pass
    return x_adv


def run_attack(model, images, labels, eps, device, n_classes):
    """Return adversarial images in [0,1] within the Linf ball of radius eps.
    For each input, decide what to optimize and which routine(s) to apply."""
    model.eval()
    # TODO: choose the attack routine or composition of routines.
    raise NotImplementedError
```

The outer harness supplies the model, the projection, and autograd; the empty slots are the
objective, the step schedule, and how routines are assembled into the final attack.
