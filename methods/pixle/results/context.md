# Context: black-box sparse adversarial attacks on image classifiers (circa 2020-2021)

## Research question

A neural image classifier `f: x in [0,1]^(3,w,h) -> R^c` maps an image to a vector of `c`
class probabilities and predicts `arg max_i f_i(x)`. It is by now well established that such
classifiers are fragile: tiny, carefully chosen changes to `x` flip the prediction. The
precise problem here is to craft, for a correctly-classified image `x` of true class `y`, an
adversarial image `xbar` with `arg max_i f_i(xbar) != y` (untargeted) or `= ybar` for a chosen
`ybar != y` (targeted), under two simultaneous restrictions that make it both realistic and hard:

- **Black-box, query-limited.** The attacker has no access to the model's weights, architecture,
  training data, or gradients. The only thing observable is the output, and in the hardest
  realistic setting only the predicted **probabilities** for a queried input. Each query is one
  forward pass, and real deployed models cap the number of queries, so the attack must succeed in
  *few* queries.

- **Sparse (L0).** The difference between `x` and `xbar` is measured by `||x - xbar||_0` — the
  *number of modified spatial pixels*, with each modified pixel allowed to change by any amount in
  `[0,1]`. A pixel counts as modified if any of its channels changes. The budget is a small integer
  `epsilon`: only a handful of pixels may be touched. This is a different threat model from the
  much-studied `L_inf`/`L2` attacks, which spread a small change across *every* pixel.

A solution must therefore find, using only probability queries and a strict pixel budget, a small
set of pixels and what to do to them so that the prediction breaks — and do so fast enough that the
query count stays low even as images grow from `32x32` to full ImageNet resolution.

## Background

The field state. The vulnerability of neural networks to adversarial examples was established by
Szegedy et al. (2014, *Intriguing properties of neural networks*) and Goodfellow et al. (2014,
*Explaining and harnessing adversarial examples*), who showed that imperceptible perturbations
crafted from model gradients reliably fool classifiers. This split the literature along two axes.

First, the **threat model**: *white-box* attacks assume full access to the model (weights,
gradients) and are the most studied; *black-box* attacks assume access only to inputs and outputs
and are closer to a real adversary who can merely query a deployed API. White-box attacks reveal
the existence of the vulnerability; black-box attacks measure the real-world risk.

Second, the **norm** constraining `x - xbar`. Most work bounds `L_inf` (largest per-pixel change)
or `L2` (Euclidean), and the resulting perturbation is *dense*: every pixel is nudged a little.
`L1` attacks bound the absolute sum. The `L0` (sparse) regime — change few pixels, each by an
arbitrary amount — is the least studied and is awkward for gradient-based methods: a gradient step
naturally distributes the perturbation over many coordinates, the opposite of what L0 wants, so
sparse attacks need extra machinery (saliency, boundary geometry, or combinatorial search) to pick
*which* few pixels.

Several facts about the design space are load-bearing before a new attack is built:

- **Decision boundaries near natural images are mostly flat with few sensitive directions**
  (curvature analyses in the DeepFool/SparseFool line). The space is curved and sensitive only along
  a small number of directions, which is *why* perturbing a few well-chosen pixels can suffice and
  why universal/structured perturbations beat random ones.
- **Differential Evolution (DE)** (Storn & Price, 1997) is a population-based, gradient-free
  optimizer for hard multi-modal problems: it keeps a population of candidate solutions and, each
  generation, forms children `x_i(g+1) = x_{r1}(g) + F*(x_{r2}(g) - x_{r3}(g))` from random members
  (scale `F`), keeping a child only if it is fitter than its parent. It needs only to *evaluate* the
  objective, never differentiate it, so it works in black-box settings — at the cost of evaluating a
  whole population every generation.
- **Random search** (Rastrigin, 1963) is the even simpler gradient-free template: hold a current
  iterate, sample a random update, **accept it only if it improves the objective**, otherwise discard
  and resample. Despite its simplicity it is competitive on non-convex objectives and spends only one
  objective evaluation per step. It has been observed that for adversarial search, *localized*
  contiguous updates (a few neighboring pixels changed together) are far more effective per query than
  diffuse ones, because convolutional networks are especially sensitive to localized high-contrast
  structure.

## Baselines

These are the prior attacks a new sparse black-box method would be compared against and reacts to.

**JSMA — Jacobian Saliency Map Attack (Papernot et al., 2016, arXiv:1511.07528).** A white-box L0
attack. It computes the forward derivative (Jacobian) of the network outputs with respect to the
input, builds a *saliency map* ranking how strongly each input pixel pushes the logits toward a
chosen target class, and greedily fixes a small number of the most salient pixels to extreme values,
re-computing the saliency after each pick. **Limitation:** it requires gradients (the Jacobian) and
full model access — a white-box method that does not apply when only output probabilities are
visible.

**SparseFool (Modas et al., CVPR 2019, arXiv:1811.02248).** A white-box L0 attack that exploits the
low curvature of decision boundaries: it repeatedly linearizes the boundary (DeepFool-style) to find
the minimal perturbation toward it, then sparsifies that perturbation onto few coordinates.
**Limitation:** again white-box and gradient-dependent — it needs to query the boundary geometry of
the model, unavailable to a black-box attacker.

**One-Pixel attack (Su et al., 2019, arXiv:1710.08864).** The first black-box L0 attack, and the
closest ancestor. It encodes a perturbation as a fixed-length *candidate solution* — an array of
`(x, y, R, G, B)` five-tuples, one per modified pixel — and searches over both *which* pixels and
*what new RGB values* to write into them using Differential Evolution. The fitness is exactly the
quantity a black-box attacker can read: the predicted probability of the true class (untargeted) or
the target class (targeted). With a population on the order of 400 candidates and an early stop when
the prediction flips, it can fool small images by changing as few as one pixel. **Limitation:** the
candidate solution mixes discrete pixel *positions* with *continuous color values* over `[0,255]^3`
per pixel, so the search space is large; DE must evaluate an entire population each generation, and
on larger images it needs many thousands of queries and its success rate collapses — it works on
`32x32` but fails as images grow.

**ScratchThat (Jere et al., 2019).** A black-box L0 attack, also DE-based, that draws colored lines
(Bezier curves) across the image. **Limitation:** the perturbation is large and visible, and the DE
search is again query-heavy.

**PatchAttack (Yang et al., 2020).** Uses reinforcement learning to place pre-computed texture
patches over the image. **Limitation:** the patches are large and easily detected.

**Square Attack (Andriushchenko et al., ECCV 2020, arXiv:1912.00049).** A black-box `L_inf`/`L2`
attack built on random search: hold the current adversarial iterate, sample a random *localized
square* update that lies on the boundary of the norm ball, accept it if the margin loss decreases,
else discard; stop as soon as the image is adversarial. It is highly query-efficient and needs only
the model's output. **Limitation for this problem:** it is a *dense* `L_inf`/`L2` attack — its
square updates inject new pixel values across a contiguous region and it is not formulated for the
`L0` sparse budget — but its accept-if-improves random-search core and its preference for localized
updates are the algorithmic template a sparse method can borrow.

## Evaluation settings

The natural yardsticks in place at the time:

- **Datasets / models.** CIFAR-10 (`32x32`), TinyImageNet, and full ImageNet, attacking standard
  classifiers — ResNet (e.g. ResNet18/20/50, He et al. 2016) and VGG (VGG11/16, Simonyan & Zisserman
  2014), trained with SGD (lr `0.01`, momentum `0.9`) and standard augmentation (random horizontal
  flip, padded random crop), or pretrained ImageNet models used as-is. For the robust-model variant
  of this threat model, adversarially-trained `L2`-robust networks from a public model zoo are used,
  since a strong `L0` attack trivially saturates undefended models.
- **Sample selection.** Attack only images the model already classifies *correctly* (so success is
  attributable to the attack), a fixed number per class to balance the set.
- **Both untargeted and targeted** attacks (targeted: force a specific wrong class).
- **Metrics.** *Success rate* — fraction of attacked images whose prediction is flipped (to any wrong
  class, or to the target). *Iterations / queries* — number of forward passes (model interrogations)
  used per image; lower is better and is capped in realistic settings. *L0 norm* — number of pixels
  actually modified. The attack is run with an early-stopping callback that halts the moment the
  current candidate is misclassified (or hits the target), so query count and `L0` are kept small.

## Code framework

The attack plugs into a standard black-box harness: a frozen, already-trained classifier exposing
only a forward pass to probabilities, a batch of clean images in `[0,1]`, their true labels, a pixel
budget, and a function that must return adversarial images of the same shape, still in `[0,1]`,
modifying at most the budgeted number of spatial pixels. The substrate is only the generic
query-only search machinery that already exists: a way to evaluate the classifier's probability for
an image, a loss read off that probability, a misclassification check, and a search loop that
proposes a candidate, scores it, and keeps it if it helps.

```python
import torch
import torch.nn as nn
from torch.nn.functional import softmax


def run_attack(
    model: nn.Module,
    images: torch.Tensor,   # (N, C, H, W), values in [0, 1]
    labels: torch.Tensor,   # (N,)
    pixels: int,            # L0 budget: max number of modified spatial pixels
    device: torch.device,
    n_classes: int,
) -> torch.Tensor:
    """Return adversarial images, same shape as `images`, in [0, 1], modifying
    at most `pixels` spatial pixels per sample. Black-box: only probabilities."""
    model.eval()
    adv_images = []
    for idx in range(images.size(0)):
        image, label = images[idx:idx + 1].to(device), labels[idx:idx + 1].to(device)
        adv_images.append(_attack_one(model, image, label, pixels, device))
    return torch.cat(adv_images)


@torch.no_grad()
def _prob_true_class(model, image, label):
    """The one quantity a black-box attacker can read: P(true class)."""
    p = softmax(model(image), dim=1)
    return p[0, label].item()


@torch.no_grad()
def _is_misclassified(model, image, label):
    p = softmax(model(image), dim=1)
    return int(p.argmax(dim=1).item()) != int(label.item())


def _propose_candidate(image, state):
    # TODO: the candidate-construction rule we will design.
    #       Given the clean image (and any search state we keep), produce a new
    #       candidate adversarial image that touches only a few pixels.
    pass


def _attack_one(model, image, label, pixels, device):
    # TODO: the search procedure we will design, built on the query-only
    #       primitives above: repeatedly propose a candidate, score it by the
    #       loss, keep it if it helps, and stop when the prediction flips.
    pass
```

The harness supplies the frozen model, the probability read-out, and the loop scaffold. Candidate
construction and search organization remain as empty slots.
