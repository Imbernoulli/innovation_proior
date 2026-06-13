# Class-Balanced Loss via the Effective Number of Samples, distilled

The class-balanced loss reweights a per-example loss by a per-class factor that is inversely
proportional to the *effective number of samples* of that class, rather than to the raw class
count. The effective number `E_n = (1 - β^n)/(1 - β)` measures how much non-redundant feature
space a class of `n` samples covers, building in diminishing returns from data overlap. A single
hyperparameter `β ∈ [0, 1)` slides the raw weighting rule from no reweighting (`β = 0`) to ordinary
inverse-class-frequency reweighting (`β → 1`), then the weights are normalized to sum to `C`. The
weight depends only on the class counts, so it is model- and loss-agnostic and drops into softmax
cross-entropy, sigmoid cross-entropy, or focal loss.

## Problem it solves

Long-tailed image classification: a few head classes dominate, most tail classes have very few
samples, and plain cross-entropy biases the model toward the head. Per-class loss reweighting is
the fix, but inverse class frequency (`w_c ∝ 1/n_c`) over-amplifies the noisy, near-duplicate tail
and performs poorly under heavy imbalance, while the field's smoothed patch (`w_c ∝ 1/sqrt(n_c)`)
is a bare heuristic with no principle and no tunable knob. The goal is a principled per-class
weight controlled by one interpretable parameter that interpolates between those extremes and
adapts to the dataset.

## Key idea

Stop treating data value as linear in the count. Real images of a class overlap in information
(near-duplicates plus augmentation), so the marginal value of each new sample diminishes. Model a
class as a feature region of volume `N` (the number of *unique prototypes*) and each sample as a
unit-volume subset that may overlap others; the *effective number* `E_n` is the expected covered
volume after `n` random draws.

Under the simplifying assumption that a new sample lands either entirely inside the already-covered
region (probability `p = E_{n-1}/N`) or entirely outside it, the expected covered volume satisfies

```
E_n = p·E_{n-1} + (1-p)·(E_{n-1}+1) = 1 + ((N-1)/N)·E_{n-1},    E_1 = 1.
```

Writing `β = (N-1)/N ∈ [0,1)` and solving (by induction) gives the closed form

```
E_n = (1 - β^n)/(1 - β) = Σ_{j=1}^{n} β^{j-1},
```

so the `j`-th sample contributes `β^{j-1}` (geometrically diminishing). As `n → ∞`,
`E_n → 1/(1-β) = N`, consistent with `β = (N-1)/N`.

**Asymptotics.** `β = 0` (`N = 1`): `E_n = 1` for all `n` (one prototype, everything redundant).
`β → 1` (`N → ∞`): by L'Hôpital on `(1-β^n)/(1-β)`, `lim_{β→1} E_n = lim_{β→1} (-nβ^{n-1})/(-1) = n`
(no overlap, every sample unique).

## The class-balanced loss

Weight class `i` inversely to its effective number, then normalize so the weights sum to `C`
(keeps the total loss on the unweighted scale, so the optimizer's effective step size is
unchanged):

```
r_i = 1/E_{n_i} = (1 - β)/(1 - β^{n_i}),
α_i = r_i / Σ_{j=1}^{C} r_j · C,    so    Σ_{i=1}^{C} α_i = C.
```

For a sample with label `y` (class size `n_y`) and any base loss `L`,

```
CB(p, y) = α_y · L(p, y).
```

- **β = 0** ⇒ every weight is 1: no reweighting.
- **β → 1** ⇒ `E_{n_y} → n_y`, so `α_y` is the sum-to-`C` normalized inverse-frequency weight.
- Intermediate **β** is a principled interpolation; smaller `β` (smaller `N`) suits fine-grained
  classes with fewer unique prototypes, larger `β` suits coarse classes.

Instantiated on the three standard losses (`z` = logits, `p_i^t` = sigmoid of the signed logit,
`γ` = focal focusing parameter):

```
CB_softmax(z, y) = -α_y · log( exp(z_y) / Σ_j exp(z_j) )
CB_sigmoid(z, y) = -α_y · Σ_i log( sigmoid(z_i^t) )
CB_focal(z, y)   = -α_y · Σ_i (1 - p_i^t)^γ · log(p_i^t)
```

The focal case is exactly the α-balanced focal loss with `α_t = α_y`, where `α_y` is the normalized
effective-number weight whose raw factor is `(1-β)/(1-β^{n_y})`: the effective number supplies the
per-class weight that focal loss had left as a free hyperparameter, and it composes orthogonally
with focal's difficulty term `(1-p_t)^γ`.

## Working code

The per-class weight computation (pure, sees only the counts), matching the canonical
implementation:

```python
import numpy as np
import torch


def class_balanced_weights(samples_per_cls, num_classes, beta):
    """Per-class weights from the effective number of samples.

    effective_num[c] = 1 - beta**n_c              # 1 - beta^{n_c}
    weights[c]       = (1 - beta) / effective_num # (1-beta)/(1-beta^{n_c})  proportional to 1/E_{n_c}
    weights          = weights / sum(weights) * C # normalize so the weights sum to num_classes
    """
    samples_per_cls = np.asarray(samples_per_cls, dtype=np.float64)
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * num_classes
    return torch.tensor(weights).float()
```

Applying it to a base loss (softmax / sigmoid / focal), gathering the per-class weight onto each
sample's label:

```python
import torch
import torch.nn.functional as F


def focal_loss(labels, logits, alpha, gamma):
    """Focal loss = -alpha_t * (1 - p_t)^gamma * log(p_t), using one-vs-all signed logits."""
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    if gamma == 0.0:
        modulator = 1.0
    else:
        # numerically stable (1 - p_t)^gamma
        modulator = torch.exp(-gamma * labels * logits
                              - gamma * torch.log1p(torch.exp(-logits)))
    loss = modulator * bce
    weighted = alpha * loss
    return weighted.sum() / labels.sum()


def cb_loss(labels, logits, samples_per_cls, num_classes, loss_type, beta, gamma):
    """Class-balanced loss: ((1-beta)/(1-beta^n)) * L(labels, logits), normalized to sum to C."""
    weights = class_balanced_weights(samples_per_cls, num_classes, beta).to(logits.device)

    labels_one_hot = F.one_hot(labels, num_classes).float()
    # gather each sample's class weight, broadcast across the C logits
    w = weights.unsqueeze(0).repeat(labels_one_hot.shape[0], 1) * labels_one_hot
    w = w.sum(1).unsqueeze(1).repeat(1, num_classes)

    if loss_type == "focal":
        return focal_loss(labels_one_hot, logits, w, gamma)
    if loss_type == "sigmoid":
        return F.binary_cross_entropy_with_logits(logits, labels_one_hot, weight=w)
    if loss_type == "softmax":
        pred = logits.softmax(dim=1)
        return F.binary_cross_entropy(pred, labels_one_hot, weight=w)
    raise ValueError(loss_type)
```

`β` is the dataset-level overlap knob: larger `β` means a larger assumed prototype volume and
weights closer to inverse frequency; smaller `β` means stronger assumed redundancy and smoother
weights across classes.
