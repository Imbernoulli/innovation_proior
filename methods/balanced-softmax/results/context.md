## Research question

Real-world image data is long-tailed: a handful of head classes account for most of the
observations while a long list of tail classes are seen only a few times each — in instance
segmentation datasets like LVIS the most frequent class can outnumber the rarest by tens of
thousands to one, and across vision datasets the per-class counts fall off roughly geometrically
from head to tail. A classifier is trained by empirical risk minimization on this skewed training
split `{(x_i, y_i)}`, with per-class counts `n_1, ..., n_k` (`Σ_j n_j = n`). But it is *graded* on a
**balanced** test set — uniform over classes, scored by top-1 accuracy or mean per-class accuracy
(equivalently, *balanced error*, the average of the per-class error rates). The training label
distribution `p̂(y)` and the test label distribution `p(y)` therefore disagree: for a tail class `j`,
`p̂(y=j) ≪ p(y=j)`. A model that minimizes ordinary cross-entropy on the training split fits the
*training* class proportions and systematically under-predicts the tail; on the balanced metric it
loses most of its score on exactly the classes that are rare in training. The precise goal is a
training procedure that produces a classifier whose decision rule is correct for the *balanced*
test distribution, that holds up across architectures and across imbalance ratios from mild to
extreme, and that does so without paying the instability that the existing rebalancing fixes incur.

## Background

The standard classifier is multinomial (Softmax) regression. A feature extractor `f` and a linear
head with per-class weights `θ_j` produce logits `η_j = θ_j^T f(x)`, and the Softmax maps them to a
conditional probability

```
φ_j = e^{η_j} / Σ_{i=1}^k e^{η_i},   Σ_j φ_j = 1,
```

trained by the cross-entropy / negative-log-likelihood loss `l_y(θ) = -log φ_y`. Read through Bayes'
rule, `φ_j = p(y=j|x) = p(x|y=j) p(y=j) / p(x)`: the Softmax posterior factors into a
*class-conditional likelihood* `p(x|y=j)` and a *class prior* `p(y=j)`. The likelihood is a property
of how class `j` looks and is shared between training and testing (same images, same generative
process); the prior and the evidence `p(x)` are what differ between a skewed training split and a
uniform test split. This is the standard exponential-family parameterization of the multinomial,
which also supplies the *canonical link* `η_j = log(φ_j / φ_k)` — the inverse of the Softmax,
writing each logit as a log-ratio of probabilities. These two facts — the Bayes factorization of the
Softmax posterior, and the link function that inverts the Softmax — are the load-bearing pieces of
machinery available before any long-tail-specific method exists.

Several empirical facts about *existing* systems frame the problem. First, plain ERM under
long tails produces a marginal predicted-label distribution that collapses toward the head: as the
imbalance factor grows, the mass the model puts on tail classes at test time shrinks, and the
balanced accuracy degrades steeply on the rare classes. Second, the two classical cures — resampling
the data to be balanced, and reweighting the loss per class — are each observed to break: oversampling
the tail overfits its few images, undersampling the head throws away the head's variation, and loss
reweighting with large per-class weights is observed to produce abnormally large, unstable gradients
precisely when the imbalance (and hence the weights) is most severe. Third, there is a theoretical
reason reweighting may not even help asymptotically: on separable data, unregularized logistic
regression converges to the max-margin solution, and that solution is *unchanged* by importance
weights (Soudry et al. 2018; Byrd & Lipton 2019) — so simply scaling each class's loss by a constant
need not move the learned classifier at all once it can fit the training set. Fourth, a diagnostic
decomposition of where the imbalance damage lives: when training is split into representation
learning and classifier learning, the *features* learned under ordinary instance-balanced training
are already good, and it is mainly the *classifier head* whose decision boundary is skewed toward the
head classes (Kang et al. 2020). Margin theory supplies the last piece: generalization error bounds
for a classifier scale inversely with the training margin, and with far fewer samples a tail class has
a worse bound — so the classes most in need of a large margin are exactly the rare ones.

## Baselines

These are the prior approaches a new method would be measured against and reacts to.

**Inverse-frequency reweighting (cost-sensitive learning; Huang et al. 2016; Wang et al. 2017).**
Multiply each class's cross-entropy term by a weight inversely proportional to its frequency,
`w_c ∝ 1/n_c` (or the milder `w_c ∝ 1/√n_c`), so the rare classes contribute more loss. The weight
is a per-class scalar that multiplies the whole gradient of that example. *Limitation:* the weight
depends only on the class count, never on the model's current output, so it rescales gradients
uniformly rather than correcting *where* the decision boundary sits; and when the imbalance is severe
the tail weights become very large, producing abnormal, unstable gradients. On
separable data the rescaling can leave the converged classifier unchanged.

**Class-Balanced / Effective Number of samples (Cui et al. 2019, CVPR; arXiv:1901.05555).** Argues
that raw `1/n_c` over-credits frequent classes because additional samples of a class become
near-duplicates under data augmentation, so the *effective* number of samples saturates. Modeling
sampling as random covering of a per-class volume `N` gives the effective number
`E_n = (1 - β^n)/(1 - β)` with `β = (N-1)/N`, and the class-balanced weight `w_c ∝ (1 - β)/(1 -
β^{n_c})`, interpolating between no reweighting (`β=0`) and inverse frequency (`β→1`); `β` is tuned in
`{0.9, 0.99, 0.999, 0.9999}`. *Limitation:* it is still a static, model-oblivious per-class scalar
weight on the loss — a smoother member of the same reweighting family — and it does not address the
mismatch between the training and testing posterior; `β` is an extra hyperparameter to set per
dataset.

**Label-Distribution-Aware Margin loss (LDAM; Cao et al. 2019, NeurIPS; arXiv:1906.07413).** From
margin theory: minimizing the balanced bound `(1/k) Σ_j [(1/γ_j)√(C/n_j) + (log n)/√n_j]` under a
class-wise margin budget `Σ_j γ_j = β` gives, by Cauchy-Schwarz, an optimal per-class margin
`γ_j ∝ n_j^{-1/4}`, i.e. rarer classes deserve larger margins. LDAM enforces this with a smooth
cross-entropy that *subtracts a class-dependent margin from the true class's logit*,

```
L_LDAM = -log( e^{η_y - Δ_y} / ( e^{η_y - Δ_y} + Σ_{j≠y} e^{η_j} ) ),   Δ_j = C / n_j^{1/4}.
```

*Limitation:* the margin/trade-off is derived for *binary* classification using the hinge loss and
extended to multi-class heuristically; the offset is applied only to the ground-truth logit (an
enforced margin), not as a correction to the full posterior over all classes; `C` must be tuned, and
the method is usually paired with a deferred-reweighting training schedule to optimize stably.

**Equalization Loss (EQL / SEQL; Tan et al. 2020, CVPR; arXiv:2003.05176).** Observes that for a rare
class the *discouraging* (negative) gradients — flowing through it whenever it is the wrong answer for
some other class's sample — vastly outnumber its encouraging gradients and suppress it. EQL randomly
zeroes those negative-gradient terms for rare classes by gating the Softmax denominator,
`p̃_j = e^{η_j} / Σ_k w̃_k e^{η_k}`, `w̃_k = 1 - β T_λ(f_k)(1 - y_k)`, where `T_λ` thresholds on
class frequency `f_k` and `β` is a Bernoulli switch. *Limitation:* it is a gradient-gating heuristic
with a frequency threshold `λ` and a drop probability `β`, drawing a hard rare/non-rare line, rather
than a principled correction to the training/test distribution shift.

**Decoupled training (Kang et al. 2020, ICLR; arXiv:1910.09217).** Since the diagnostic above says
features are fine and only the head is skewed, learn the representation with ordinary
instance-balanced sampling (`p_j ∝ n_j`), then in a second stage adjust *only the classifier*: retrain
it with class-balanced sampling (cRT), rescale the per-class weight norms (LWS / τ-normalization), or
use a nearest-class-mean head. *Limitation:* it is a two-stage classifier repair rather than a single
training objective, and it does not directly model the mismatch between the training and testing
label distributions; in extreme-vocabulary settings, fixed classifier-only recipes still leave the
rare classes with very low sampling frequency. It is largely orthogonal to changes in the loss itself.

## Evaluation settings

The natural yardsticks, all pre-existing:

- **CIFAR-10-LT and CIFAR-100-LT** — long-tailed versions of CIFAR built by exponentially decaying
  per-class counts, `n_y · μ^y` with `μ ∈ (0,1)`, to hit imbalance factors of 10, 100, 200 (the test
  set kept full and balanced). Backbone ResNet-32. The standard small-scale long-tail benchmark; small
  enough that runs are repeated and mean ± standard error reported.
- **ImageNet-LT** (1000 classes, imbalance factor 256, Pareto-sampled) and **Places-LT** (365 classes,
  factor 996), with the original balanced validation sets as the test set. Backbones ResNet-10 and an
  ImageNet-pretrained ResNet-152.
- **LVIS** instance segmentation (1230 classes, imbalance factor ~26,148), the extreme-imbalance
  stress test, with Mask R-CNN (ResNet-50) and COCO-style AP, reported overall and on frequent /
  common / rare class splits.
- Metric throughout: top-1 accuracy (or AP) on the **balanced** test set, optionally broken into
  many-shot (>100 images), medium-shot (20–100), and few-shot (<20) splits. The training optimizer is
  the usual SGD-with-momentum + cosine schedule; the protocol fixes the data pipeline, sampler, model,
  and optimizer so that only the per-class treatment of the loss varies.

## Code framework

The loss plugs into the standard classification training loop already used for the baselines: a
model emits logits, a loss module turns (logits, labels) into a scalar, and the loop backpropagates
and steps the optimizer. The long-tail setting also makes available the vector of per-class training
counts `n_1, ..., n_k`, computed once from the training split. Whether any of that split-level
information should enter the scalar loss is the single open slot; everything else — the
cross-entropy primitive, the optimizer, the loop — already exists.

```python
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class LongTailLoss(_Loss):
    """A classification loss that is allowed to see the per-class training
    counts. Takes raw logits and integer labels, returns a scalar loss.
    The standard cross-entropy primitive F.cross_entropy already exists; the
    per-class counts n_1..n_k are known. How (or whether) the counts enter the
    loss is the slot to be designed."""

    def __init__(self, sample_per_class):
        super().__init__()
        # n_1, ..., n_k : training samples per class (a fixed property of the split)
        self.sample_per_class = torch.as_tensor(sample_per_class)

    def forward(self, logits, labels, reduction='mean'):
        # logits: [batch, k]   labels: [batch]
        # TODO: fill in the count-aware loss computation.
        loss = F.cross_entropy(logits, labels, reduction=reduction)  # placeholder: ignores counts
        return loss


# existing classification training loop the loss plugs into
def train(model, loss_fn, data_loader, optimizer):
    for inputs, targets in data_loader:          # draw a minibatch from the long-tailed split
        optimizer.zero_grad()
        logits = model(inputs)                    # forward through the existing backbone + head
        loss = loss_fn(logits, targets)           # the loss module above
        loss.backward()                           # backprop
        optimizer.step()


# at test time the same model is evaluated on the BALANCED test set:
@torch.no_grad()
def predict(model, x):
    logits = model(x)
    return logits.argmax(dim=-1)                  # decision rule on the test set
```

The loop draws minibatches from the long-tailed split, the model produces logits, the loss module
(with access to the per-class counts) returns a scalar, and at test time the same model's argmax is
scored on the balanced set. The `forward` body is the unresolved design point.
