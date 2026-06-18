## Research question

A classifier trained on a private set leaks who was in that set. The leak is concrete and
single-signal: an adversary hands the trained model a sample and asks "was this in your training
data?", and on these benchmarks the adversary needs only one number to answer — the model's
confidence on the sample (here the max softmax probability). Training crushes that confidence to ~1
on members and leaves it lower and more scattered on non-members, so a threshold separates the two
populations. The single thing being designed is the **training-time loss** — the scalar objective the
fixed loop minimizes per minibatch — so that the member and non-member confidence distributions
become hard to tell apart (low membership-inference AUC) *without* giving up test accuracy.
Everything else — architecture, optimizer, schedule, data split, the attack itself — is frozen.

## Prior art before the first rung (output-regularization lineage)

The losses on this ladder are all answers to one question: how do I stop the model from being so
sharply confident on its training set that a threshold on that confidence reveals membership? The
lineage the first rung reacts to:

- **Plain cross-entropy / ERM.** Minimize `−log p(y)` against a one-hot target. The optimum sits at an
  *infinite* correct-vs-rest logit gap, and the logit gradient `p − y` only vanishes as `p → y`, so
  nothing stops the optimizer from driving member losses to ~0 and member confidence to ~1. The
  generalization gap — and with it the membership-advantage of a loss/confidence threshold — is
  manufactured by the training procedure itself. Gap: no pressure on the output distribution at all;
  maximal leakage by construction.
- **Label smoothing (Szegedy et al., 2016).** Bleed a fraction `ε` of target mass onto a uniform
  prior, `q'(k) = (1−ε)δ_{k,y} + ε/K`, so an infinite logit gap becomes infinitely costly and member
  confidence is bounded below 1. It shifts every output toward uniform by roughly the *same* amount —
  members and non-members alike — so it relocates the confidence distributions but barely changes how
  *separable* they are. Gap: a rigid mean-shift; the two populations slide together without
  overlapping.
- **Confidence penalty (Pereyra et al., 2017).** Subtract the predictive entropy, `L = L_CE − β·H(p)`,
  whose logit gradient `p_i(−log p_i − H)` adaptively pushes hardest on the classes the model is most
  confident about. More surgical than smoothing — it concentrates on over-confident predictions rather
  than flattening every class uniformly — but it is still a *mean*-control regularizer: it lifts the
  confidence floor without deliberately spreading the member distribution to overlap the non-member
  one, and pushed too hard it can destabilize training on the hardest dataset. Gap: adaptive but still
  mean-shifting, with no variance lever.

The fixed substrate below is the harness all four losses plug into.

## The fixed substrate

A standard supervised-classification loop is frozen and must not be touched. It loads the full
training set, shuffles by seed, and splits it 50/50 by index into a **member** subset (used for
training) and a **non-member** subset (never trained on); the held-out test set is separate. It builds
the architecture for the benchmark — ResNet-20 (CIFAR-10), VGG-16-BN (CIFAR-100), MobileNetV2
(FashionMNIST), all ReLU, Kaiming-initialized — and trains for **300 epochs** with **SGD** (lr 0.1,
momentum 0.9, weight-decay 1e-4) under a **MultiStepLR** schedule (decay ×0.1 at epochs 150 and 225),
batch size 128. Crucially, **train-time augmentation is OFF**: the loop deliberately lets the model
overfit, because without overfit there is no measurable membership signal to defend against.

After training, the loop runs a **confidence-based membership inference attack**: it computes the max
softmax probability on every member and every non-member example and reports the AUC of that score as
a member/non-member discriminator (Mann–Whitney U, so 0.5 is a coin flip). It also reports test
accuracy, the mean member-minus-non-member confidence gap, and the composite privacy score. The loop
calls exactly one method per minibatch — `compute_loss(logits, labels, epoch)` — and uses whatever
scalar it returns as the loss to backpropagate. The attack, the split, the optimizer, the schedule,
and the architectures are all fixed.

## The editable interface

Exactly one region is editable — the `MembershipDefense` class in `custom_membership_defense.py`. Every
method on the ladder is a fill of one method: `compute_loss(self, logits, labels, epoch)` returns the
scalar loss the frozen loop minimizes. `logits` are the raw model outputs `(B, K)`; `labels` are the
ground-truth class indices `(B,)`; `epoch` is the current 0-indexed training epoch (available so a
method can alternate behavior by epoch). The loop's only contract is "return a scalar loss"; the
method may read `logits.size(1)` to learn the class count `K` and adapt per dataset.

The starting point is the scaffold default: **plain cross-entropy** (ERM). Each later method replaces
exactly this method body and nothing else.

```python
# EDITABLE region of custom_membership_defense.py — default fill (plain cross-entropy / ERM)
import torch
import torch.nn.functional as F


class MembershipDefense:
    """Training-time regularizer for privacy-utility tradeoffs.

    compute_loss replaces nn.CrossEntropyLoss() in the fixed training loop.
    Design a loss that reduces membership-inference leakage (lower MIA AUC)
    while preserving test accuracy.
    """

    def __init__(self):
        pass

    def compute_loss(self, logits, labels, epoch):
        return F.cross_entropy(logits, labels)
```

## Evaluation settings

Three benchmarks spanning architecture and class count — **resnet20-cifar10** (10 classes),
**vgg16bn-cifar100** (100 classes, the hardest: low test accuracy, easiest to overfit and to leak),
and **mobilenetv2-fmnist** (10 classes, the easiest: already low leakage under plain training) — each
on the fixed 300-epoch recipe at seed 42. Four reported metrics: `test_acc` (higher better),
`mia_auc` (the attack's AUC; 0.5 is ideal, higher is worse leakage), `privacy_gap` (mean member −
non-member confidence; smaller is better), and the primary composite

`privacy_score = test_acc − max(mia_auc − 0.5, 0)`  (higher is better),

which rewards accuracy and charges only for attack AUC above the coin-flip floor — so a defense is
only worth its accuracy cost if it actually pushes the attack toward 0.5.
