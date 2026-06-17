# Context: computing sparse adversarial perturbations for deep image classifiers (circa 2018)

## Research question

A deep image classifier `f: R^n -> R^c` maps an image `x` (here `n` is the number of input
values — channels times spatial pixels) to a predicted label `k(x) = argmax_k f_k(x)`. It is by
now well established that an imperceptibly small additive perturbation `r` can flip that label.
The classical adversarial perturbation is the *minimal* one in an `L_p` sense,

```
min_r ||r||_p   s.t.   k(x + r) != k(x).
```

For `p = 2` or `p = infinity` the solution spreads tiny noise across almost every pixel. The
question here is the *sparse* regime: instead of distributing a small budget over all pixels,
change as *few spatial pixels as possible* — minimize `||r||_0` — while each touched pixel may
move arbitrarily far (within the valid dynamic range). This matters because sparse perturbations
model a different and very physical threat: a few raindrops glinting on a STOP sign, a handful of
bright flowers in a crop field, a sticker on a lens — a small set of locations changed a lot,
rather than a faint film over the whole frame. A solution would have to (1) actually minimize the
number of perturbed pixels, (2) run fast and scale to high-dimensional inputs like ImageNet, and
(3) return an image whose pixel values stay inside the valid box `[l, u]` (e.g. `[0, 255]` or
`[0, 1]`), because a sparse perturbation concentrates large magnitude on few pixels and is
therefore the regime most likely to drive those pixels out of range. No method available at the
time meets all three at once; closing that gap is the problem.

## Background

By this time the vulnerability of deep networks to adversarial perturbations is firmly
established (Szegedy et al. 2013; Goodfellow, Shlens & Szegedy 2015), and a family of efficient
`L_2` / `L_infinity` attacks exists. The relevant facts that bear on the sparse problem:

- **`L_0` minimization is NP-hard.** Finding the truly minimal-cardinality perturbation that
  flips the label is combinatorial, and reaching a global optimum cannot be guaranteed in general
  (Blumensath & Davies 2008; Natarajan 1995; Patrascu & Necoara 2015). The standard escape in
  compressed sensing is the convex `L_1` relaxation: minimizing `||r||_1` under linear
  constraints approximates the `||r||_0` solution, and under restricted-isometry-type conditions
  the `L_1` solution is provably the sparsest one (Candès & Tao 2005; Donoho 2006; Donoho & Elad
  2003). So the natural plan for a *linearized* version of the sparse-attack problem is to solve
  its `L_1` form rather than the intractable `L_0` one.

- **A geometric fact about deep decision boundaries: low mean curvature.** Diagnostic studies of
  trained image classifiers found that, in the neighborhood of a natural data sample, the
  decision boundary is *only slightly curved* — its mean curvature near data points is small
  (Fawzi, Moosavi-Dezfooli & Frossard 2017; Fawzi et al. 2018; Jetley, Lord & Torr 2018). In
  other words, for a datapoint `x` and its minimal `L_2` adversarial perturbation `v`, the
  boundary in the vicinity of `x` is well approximated by a single hyperplane passing through the
  boundary point `x_B = x + v`, with some normal vector `w`. The same studies also report that
  this flatness is *local*: it holds in a neighborhood of `x` but degrades as one moves away, so
  the hyperplane model is trustworthy only near the data point.

- **A diagnostic failure of the naive `L_1` route: ignoring the box destroys the attack.** When a
  perturbation is dense and small (`L_2` / `L_infinity`), almost no pixel leaves the valid range,
  so most attacks simply ignore the box constraint and clip at the end with negligible effect.
  The sparse regime is the opposite. Because the budget is concentrated, each touched pixel
  carries large magnitude and routinely exceeds the dynamic range, so the final clip removes most
  of the perturbation's effect. Concretely, on a VGG-16 ImageNet model, an `L_1`-based projection
  attack reaches almost a 100% fooling rate while perturbing only about 0.037% of the pixels —
  but clipping the resulting adversarial image to `[0, 255]` collapses the fooling rate to about
  13%, and folding the clip into the iteration does not recover it. A sparse `L_1` attack that
  does not *natively* respect the box is therefore not a usable sparse attack.

## Baselines

The prior sparse attacks a new method would be measured against, and the dense `L_2` attack it
leans on.

**DeepFool (Moosavi-Dezfooli, Fawzi & Frossard, CVPR 2016).** An efficient iterative attack for
the *dense* minimal-`L_p` problem, built on linearizing the *classifier*. For an affine binary
classifier `f(x) = w^T x + b`, the minimal `L_2` perturbation is the orthogonal projection onto
the separating hyperplane, with the closed form `r* = -(f(x0) / ||w||_2^2) w`. For a general
differentiable classifier it iterates: linearize `f` around the current iterate `x_i`, take the
closed-form projection step `r_i = -(f(x_i) / ||grad f(x_i)||_2^2) grad f(x_i)`, update
`x_{i+1} = x_i + r_i`, and stop when the label flips; because the iterate tends to land *on* the
boundary, the final perturbation is scaled by `(1 + eta)` with a small overshoot `eta = 0.02` to
push it across. The multiclass one-vs-all version, at iteration `i`, forms for every class
`k != k(x)` the difference gradient `w'_k = grad f_k(x_i) - grad f_{k(x)}(x_i)` and difference
value `f'_k = f_k(x_i) - f_{k(x)}(x_i)`, picks the closest boundary
`l_hat = argmin_{k != k(x)} |f'_k| / ||w'_k||_2`, and steps
`r_i = (|f'_{l_hat}| / ||w'_{l_hat}||_2^2) w'_{l_hat}`. The framework extends to `L_p` norms by
Holder duality with `q = p / (p - 1)`: the closest-hyperplane test uses `||w'_k||_q`, and for
`p > 1` the step becomes
`r_i = (|f'_{l_hat}| / ||w'_{l_hat}||_q^q) |w'_{l_hat}|^{q-1} (.) sign(w'_{l_hat})`. The limiting
`p = 1` case uses `q = infinity`, so the denominator is `||w'||_infinity` and the optimizer puts
all mass on a largest-`|w'|` coordinate. **Gap for the sparse problem:** DeepFool was designed to
*measure robustness* with dense `L_2` / `L_infinity` perturbations; it linearizes the classifier
and never considers the validity box, and its `L_2` output spreads mass across all coordinates
rather than concentrating it.

**`L_1`-DeepFool (the `p = 1` limit of the above).** Setting `p = 1` gives `q = infinity`, so
the closest-hyperplane test uses `||w'_k||_infinity`. If
`d = argmax_j |w'_{l_hat,j}|`, the projection step is supported on that coordinate:
`r_i = (|f'_{l_hat}| / |w'_{l_hat,d}|) sign(w'_{l_hat,d}) e_d`. It is therefore a perturbation
concentrated on a single coordinate per step, hence sparse. **Gap:** as
the diagnostic above shows, it computes sparse perturbations but does not respect the box; its
few high-magnitude pixels leave the valid range and clipping destroys the attack (about 100%
fooling before clipping, about 13% after, on VGG-16/ImageNet). It also linearizes the
*classifier* globally, which is accurate only very close to the boundary.

**JSMA — Jacobian Saliency Map Attack (Papernot, McDaniel, Jha, Fredrikson, Celik & Swami,
2016).** A targeted sparse attack that scores each pixel by a *saliency map* built from the
Jacobian of the class logits with respect to the input, then greedily perturbs the
highest-scoring pixels (typically to an extremum) toward a target class, repeating until
misclassification. **Gap:** it searches over *pairs* of candidate pixels at each step, so its
cost grows sharply with input dimension — it is reported to be impractical on ImageNet — and as a
targeted method it must be adapted to the untargeted setting.

**One-pixel attack (Su, Vargas & Sakurai, 2019).** A black-box attack that uses differential
evolution to search for an extremely small set of pixels (often a single pixel) and their RGB
values that cause misclassification, querying only the model's outputs. **Gap:** the
evolutionary search needs many model queries and is very slow; it does not scale to
high-dimensional inputs and is run by sweeping the number of perturbed pixels `kappa` upward
until an adversarial example is found.

**Greedy local search (Narodytska & Kasiviswanathan, 2017).** A black-box method that perturbs a
small set of pixels chosen by a greedy local search over the image. **Gap:** black-box and
greedy, again with high query/computational cost and limited scalability.

Across these sparse baselines the common limitations are: high computational complexity, poor
scaling to large images, and perturbations made of high-magnitude pixels that are perceptible
and often out of the valid range.

## Evaluation settings

The natural yardsticks already in use for adversarial attacks:

- **Datasets / models.** MNIST (a LeNet convolutional network), CIFAR-10 (e.g. VGG-19 and
  ResNet-18), and ImageNet / ILSVRC-2012 (pretrained VGG-16, ResNet-101, DenseNet-161,
  Inception-v3). For CIFAR-10, the natural robust-model yardstick is an adversarially trained
  network (e.g. `L_infinity` or `L_2` adversarial training), since on an undefended model a
  strong sparse attack saturates trivially.
- **Metrics.** *Fooling rate* — the fraction of samples whose label changes,
  `|{x : k(x + r_x) != k(x)}| / |D|`; *median perturbation percentage* — the median fraction of
  pixels perturbed per fooled sample (the operational `L_0` budget, counting a spatial pixel as
  modified if any of its channels changes); and *average execution time* per sample. A sparse
  attack is also naturally scored under a fixed `L_0` budget (a maximum number of modified
  spatial pixels), where the output must respect both the budget and the `[0, 1]` range to count.
- **Protocol.** Restrict to samples that are initially classified correctly; run the attack;
  check `L_0` validity and that the adversarial image lies in the valid box; report fooling
  rate / robust accuracy and time. Targeted baselines (JSMA) are evaluated in an untargeted
  variant by accepting any label change; the one-pixel attack is run by increasing the pixel
  count until it succeeds.

## Code framework

The attack plugs into the standard evaluation harness: a pretrained, frozen classifier `model`,
a batch of clean images in `[0, 1]`, their true labels, an `L_0` pixel budget, and a single
function `run_attack` that returns adversarial images of the same shape, also in `[0, 1]`. The
substrate that already exists is just the model, automatic differentiation (so per-class logit
gradients with respect to the input are available), and a box-projection clamp. How the sparse
perturbation itself is constructed is exactly what is to be designed, so the body is an empty
slot.

```python
import torch
import torch.nn as nn


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Predicted labels argmax_k f_k(x)."""
    return model(x).argmax(dim=1)


def clip_to_box(x: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
    """Project pixel values back into the valid box [lb, ub]."""
    return torch.max(torch.min(x, ub), lb)


def class_logit_grad(model: nn.Module, x: torch.Tensor, k: int) -> torch.Tensor:
    """Gradient of the k-th class logit f_k w.r.t. the input x (one row of the Jacobian)."""
    x = x.clone().detach().requires_grad_(True)
    fk = model(x).flatten()[k]
    (grad,) = torch.autograd.grad(fk, x)
    return grad.detach()


def craft_sparse_perturbation(model: nn.Module, x: torch.Tensor, label: torch.Tensor,
                              lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
    """Return an adversarial image for x (label-flipping), changing as few spatial pixels
    as possible while staying inside [lb, ub]. This is the slot the method fills in."""
    # TODO: the construction we will design.
    pass


def run_attack(model, images, labels, pixels, device, n_classes):
    model.eval()
    lb = torch.zeros_like(images)
    ub = torch.ones_like(images)
    adv = []
    for i in range(len(images)):
        x = images[i : i + 1].to(device)
        y = labels[i : i + 1].to(device)
        adv.append(craft_sparse_perturbation(model, x, y, lb[i : i + 1], ub[i : i + 1]))
    return torch.cat(adv, dim=0)
```

The outer loop hands one image at a time to `craft_sparse_perturbation`; that function is where
the sparse-attack rule will live, with `class_logit_grad` and `clip_to_box` as the only
primitives it may assume.
