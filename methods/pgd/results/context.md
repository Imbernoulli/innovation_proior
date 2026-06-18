## Research question

A classifier `f_theta` maps an input image `x in [0,1]^d` to a label. The attacker has full
white-box access: the parameters `theta`, the architecture, and — crucially — the ability to
backpropagate, so the input-gradient `∇_x L(theta, x, y)` of any differentiable loss `L` is
available. The attacker is handed a single example `x` that the model classifies correctly,
and a perturbation budget `eps`, and must produce `x_adv` with

```
||x_adv - x||_inf <= eps,   x_adv in [0,1]^d
```

that the model misclassifies. The L_inf constraint encodes a perceptual-similarity proxy: no
single pixel may move by more than `eps`, so for small `eps` the perturbed image looks
identical to a human. The precise goal is to maximize the loss the model incurs on `x_adv`
inside that box — equivalently, to *solve the constrained maximization*

```
max_{||delta||_inf <= eps}  L(theta, x + delta, y),   subject to x + delta in [0,1]^d.
```

Why it matters: how confidently we can *find* a worst-case input inside the box determines
both how dangerous an attacker is and how meaningfully we can *measure* a model's robustness.
A weak attack that leaves loss on the table reports a model as more robust than it really is.
At small budgets chosen to avoid trivial saturation, the *quality* of the attack becomes the
thing under test: a stronger maximizer flips examples that a weaker one leaves standing. The
pain point is that this inner maximization is over a high-dimensional non-concave loss
surface, so there is no closed-form maximizer and no guarantee a given procedure finds the
most adversarial point in the box.

## Background

The phenomenon that sets the stage: state-of-the-art neural networks misclassify inputs that
are only imperceptibly different from correctly classified ones (Szegedy et al. 2014), and the
*same* perturbed input often fools many independently trained models. The standard reading of
why this happens is the **linear view** (Goodfellow et al. 2015): modern nets are built out of
deliberately near-linear pieces (ReLU, maxout, the linear regime of saturating units) to make
them easy to optimize. For a locally linear response with weight vector `w`, the change in
pre-activation from a perturbation `eta` is `w^T eta`, and under the constraint `||eta||_inf <=
eps` this is maximized by `eta = eps * sign(w)`, giving a change of `eps * sum_i |w_i| = eps *
||w||_1`. The key observation is that `||eta||_inf` does not grow with the dimension `d`, but
the induced change `eps * ||w||_1` grows linearly with `d`. So in high dimension, many
coordinate-wise changes each smaller than the sensor precision *add up* to a large swing in
the output. Adversarial examples are then not a quirk of extreme nonlinearity but a
consequence of high-dimensional (near-)linearity — and that is exactly what makes them cheap
to compute from a single gradient.

A second background fact concerns the *geometry of the constraint*. The natural attack metric
here is L_inf, and the relevant primitive is steepest ascent under an L_inf bound. For a fixed
gradient `g`, the direction inside the unit L_inf ball that maximizes the linear gain `g^T v`
is `v = sign(g)`: because the constraint decouples across coordinates (`|v_i| <= 1`
independently), each coordinate is independently pushed to `+-1` with the sign matching `g_i`,
giving gain `sum_i |g_i| = ||g||_1`. (This is the L_1 / L_inf duality: the dual norm of L_inf
is L_1, and `||g||_1` is the support-function value of the L_inf ball.) So under an L_inf
budget the "right" step direction is the *sign* of the gradient, not the raw gradient — the
raw gradient is the steepest-ascent direction only under the L_2 metric.

A third background fact is about the *local loss surface*. Empirically the loss as a function
of the input, restricted to the eps-ball around a clean point, is not benign: there are sharp
curvature artifacts localized right next to the data point that can mask the true direction of
steepest ascent, so the gradient evaluated exactly at `x` can be a misleading local cue
(Tramer et al. 2017). This leaves an open diagnostic question for any first-order attack:
whether a local procedure can consistently reach high-loss points in the box, or whether the
non-concavity hides substantially worse points from the search.

Finally, the whole setting has a name outside deep learning: **robust optimization** (Wald
1945; Ben-Tal, El Ghaoui & Nemirovski 2009), where one optimizes against a worst-case
perturbation drawn from an uncertainty set. The min-max / saddle-point template `min_theta E[
max_{delta in S} L(theta, x+delta, y) ]` is the robust-optimization view of training; the
attacker's job is exactly the inner `max` for fixed `theta`.

## Baselines

**Box-constrained L-BFGS (Szegedy et al. 2014).** The original construction: solve `min
||delta||` subject to the model misclassifying `x + delta` and `x + delta in [0,1]^d`, via a
box-constrained quasi-Newton solver. It reliably finds small perturbations. Core idea — treat
adversarial example generation as a constrained optimization and solve it accurately. **Gap:**
the constrained quasi-Newton solve is expensive — an inner optimization loop per example —
which is impractical when an attack (or a defense that uses the attack at every training step)
must run on large datasets at scale.

**FGSM — Fast Gradient Sign Method (Goodfellow et al. 2015).** Linearize the loss around the
clean input and take the single L_inf-steepest-ascent step:

```
x_adv = x + eps * sign( ∇_x L(theta, x, y) ).
```

One gradient, one step, lands at a corner of the eps-box. Cheap enough to run inside a
training loop. Core idea — the steepest-ascent direction under L_inf is `sign(grad)`, and one
full-budget step of the *linearized* loss is a fast adversary. **Gap:** the linearization is
exact only infinitesimally; the true loss is curved, so a single full-budget sign-step jumps
to a box corner that need not be anywhere near the actual maximizer inside the box. It leaves
loss on the table, and slightly more careful adversaries find substantially higher-loss points
in the same box.

**BIM / iterative FGSM (Kurakin et al. 2017).** Apply FGSM repeatedly with a small step `alpha`
and clip back into the eps-box after every step:

```
x_0 = x,    x_{N+1} = Clip_{x, eps} { x_N + alpha * sign( ∇_x L(theta, x_N, y) ) },
```

where `Clip_{x,eps}` clamps each coordinate of its argument to `[x_i - eps, x_i + eps]`
(and the result is kept in `[0,1]`). Because each step re-evaluates the gradient at the current
iterate rather than only at `x`, it follows the curved surface instead of trusting one
linearization, and in practice it produces more harmful examples as the iteration budget
increases. Typical settings take a unit step (`alpha = 1` in pixel units) for
`min(eps+4, 1.25*eps)` iterations.
Core idea — many small gradient-sign steps with re-projection beat one big step. **Gap:** the
iteration starts deterministically at `x` and follows a single trajectory; if that launch
gradient is distorted by the local curvature around the clean point, the whole trajectory can
inherit the mistake.

**R+FGSM (Tramer et al. 2017).** A single FGSM step *preceded by a small random step* to
escape the non-smooth neighborhood of the data point before linearizing:

```
x' = x + alpha * sign( N(0, I) ),    x_adv = x' + (eps - alpha) * sign( ∇_x L(theta, x', y) ),    alpha < eps.
```

Core idea — the gradient at the exact data point can be distorted by sharp local curvature
artifacts that mask the true ascent direction; a small random pre-step jumps off that
non-smooth point so the subsequent linearization sees a more honest gradient. **Gap:** after
the random jump it is still a *single* linearized step, so it inherits FGSM's one-shot
weakness and does not follow the curved surface.

**Carlini-Wagner (Carlini & Wagner 2017).** A strong optimization-based attack that replaces
cross-entropy with a margin objective — directly driving the logit of the true class below the
largest other logit (a difference-of-logits loss, optionally with a confidence margin) — and
solves the resulting box-constrained problem with a gradient optimizer. Core idea — the
attack's *loss function* matters: a margin loss targets the decision boundary more directly
than a probability-space objective and can demand a confidence margin after the boundary has
already been crossed. **Gap:** as originally posed it is a per-example optimization with its
own constant-search loop; it establishes that the inner loss can matter, but it is not by
itself a single fixed-budget L_inf procedure.

## Evaluation settings

- **Datasets / models:** image classifiers on standard small-image benchmarks, especially
  MNIST and CIFAR-10 (Krizhevsky 2009), with inputs scaled to `[0,1]`.
- **Threat model:** white-box, L_inf-bounded. `||x_adv - x||_inf <= eps`, `x_adv in [0,1]^d`.
  Budgets are chosen in the small-per-pixel regime where attack strength can be differentiated
  rather than saturated.
- **Protocol / metric:** take a pool of examples the model classifies correctly; run the
  attack on each; verify the L_inf bound and the `[0,1]` range; report attack success rate
  `ASR = 1 - (robust accuracy)`, the fraction of initially-correct examples whose label is
  flipped. An output that violates the norm, escapes `[0,1]`, or is non-finite counts as a
  failed attack on that example.

## Code framework

The attack plugs into a generic white-box evaluation harness that already exists. The harness
owns the pretrained model, the data pipeline, the gradient machinery (autodiff), and the
validity checks; the single empty slot is the per-example attack procedure itself — given the
model, a batch of clean images and labels, the budget `eps`, and the number of classes, return
a batch of perturbed images. Nothing about *how* the perturbation is searched for is fixed
here; that is exactly what is to be designed.

```python
import torch
import torch.nn as nn


def run_attack(
    model: nn.Module,
    images: torch.Tensor,   # (N, C, H, W), values in [0, 1]
    labels: torch.Tensor,   # (N,)
    eps: float,             # L_inf budget: ||x_adv - x||_inf <= eps
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    """Return adv_images, same shape as images, in [0, 1], with
    ||adv_images - images||_inf <= eps. White-box: model params and
    input-gradients are available via backprop."""
    model.eval()
    x = images.detach()
    # TODO: fill in the per-example search procedure.
    pass


# existing evaluation loop the attack plugs into
def evaluate(model, attack_fn, loader, eps, device, n_classes):
    correct_clean = 0
    still_correct = 0
    for images, labels in loader:                  # batch of clean, correctly-classified inputs
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            clean_pred = model(images).argmax(1)
        keep = clean_pred == labels                # ASR denominator: initially-correct only
        images, labels = images[keep], labels[keep]
        correct_clean += keep.sum().item()

        adv = attack_fn(model, images, labels, eps, device, n_classes)
        assert (adv - images).abs().amax() <= eps + 1e-6   # L_inf validity
        assert adv.min() >= 0.0 and adv.max() <= 1.0       # range validity

        with torch.no_grad():
            adv_pred = model(adv).argmax(1)
        still_correct += (adv_pred == labels).sum().item()
    robust_acc = still_correct / correct_clean
    return {"robust_acc": robust_acc, "asr": 1.0 - robust_acc}
```

The harness supplies clean inputs and the gradient machinery; `run_attack` is where the search
for the worst-case in-budget perturbation will live.
