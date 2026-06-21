Long-tailed image classification breaks ordinary cross-entropy training because the loss is dominated by a few head classes. A network that minimizes average loss finds it cheaper to perfect the frequent classes and let the rare ones collapse. Resampling is the traditional alternative, but in deep feature learning it is costly: oversampling the tail repeats the same scarce images and invites overfitting, while undersampling the head discards data the representation needs. The cleaner lever is to leave the data unchanged and reweight the loss per class, so that mistakes on rare classes matter more. Cost-sensitive decision theory licenses this—class weights shift the operating threshold the way resampling would—but it leaves the magnitude of the weights unspecified. Inverse class frequency is the natural magnitude, yet on real long-tailed data it fails: it assumes that data value is linear in count, so a class with five images receives hundreds of times the per-sample weight of a class with thousands, even though those five images are near-duplicates and noisy. The resulting gradients dominate training and hurt accuracy. Practitioners have retreated to inverse-square-root frequency, which empirically helps but has no derivation and no tunable knob; it is simply a heuristic frozen between no reweighting and full inverse frequency. What is needed is a single interpretable parameter that interpolates between those two extremes and adapts to how much redundant overlap the data actually contains.

The right replacement for raw count is the effective number of samples. The intuition is that real images of a class overlap in information: the first image of a bird species teaches a lot, the tenth is largely redundant, and the thousandth adds almost nothing once augmentation is taken into account. So instead of asking how many samples a class has, ask how much non-redundant feature volume it covers. Model the class as a region of volume N, the number of unique prototypes, and each sample as a unit-volume subset placed at random. The effective number E_n is the expected covered volume after n draws. If overlap is simplified so that a new sample either lands entirely inside the already-covered region or entirely outside it, then the recurrence is E_n = 1 + ((N-1)/N) * E_{n-1} with E_1 = 1. Writing β = (N-1)/N gives the closed form E_n = (1 - β^n)/(1 - β) = 1 + β + β^2 + ... + β^{n-1}. Each successive sample contributes geometrically less, which is exactly diminishing returns made explicit. As n grows large, E_n saturates toward N = 1/(1-β). At β = 0 the region has one prototype and every sample is redundant, so E_n = 1 for all n. At β → 1 the prototypes are infinite and no two samples overlap, so E_n → n. These are the two limits that map to the endpoints of reweighting: no reweighting and inverse class frequency.

The method is Class-Balanced Loss. It weights each class inversely to its effective number of samples rather than to its raw count. For class i with n_i training samples, the raw weight is r_i = 1/E_{n_i} = (1 - β)/(1 - β^{n_i}). These raw weights are then normalized so that their sum equals the number of classes C, which keeps the total loss on the same scale as unweighted cross-entropy and prevents the knob β from silently rescaling the optimizer step size. The final per-class factor is α_i = r_i / (Σ_j r_j) * C. The loss for a sample with label y is then CB(p, y) = α_y * L(p, y), where L is any base loss. Because the weight depends only on the class count, it is model-agnostic and loss-agnostic: it plugs into softmax cross-entropy, sigmoid cross-entropy, and focal loss without modification. In the focal case it supplies the per-class α_t that focal loss normally leaves as a free hyperparameter, while focal's difficulty term (1 - p_t)^γ remains orthogonal. The single parameter β is interpretable as the assumed prototype volume through β = (N-1)/N. Smaller β corresponds to fine-grained classes with few distinct prototypes and stronger saturation, producing smoother weights; larger β corresponds to coarse, varied classes and approaches inverse frequency. This turns the old hardcoded square-root heuristic into a principled, tunable interpolation.

```python
import numpy as np
import torch
import torch.nn.functional as F


def class_balanced_weights(samples_per_cls, num_classes, beta):
    """Per-class weights from the effective number of samples.

    E_n = (1 - beta**n) / (1 - beta); the (1 - beta) cancels in 1/E_n.
    Returns a length-num_classes tensor whose entries sum to num_classes.
    """
    samples_per_cls = np.asarray(samples_per_cls, dtype=np.float64)
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def cb_loss(labels, logits, samples_per_cls, num_classes, loss_type, beta, gamma=2.0):
    """Class-Balanced Loss: ((1-beta)/(1-beta^n)) * base_loss, normalized to sum to C."""
    weights = class_balanced_weights(samples_per_cls, num_classes, beta).to(logits.device)
    labels_one_hot = F.one_hot(labels, num_classes).float()

    # Gather the per-sample class weight and broadcast over classes.
    w = weights.unsqueeze(0).repeat(labels_one_hot.shape[0], 1) * labels_one_hot
    w = w.sum(1).unsqueeze(1).repeat(1, num_classes)

    if loss_type == "softmax":
        pred = logits.softmax(dim=1)
        return F.binary_cross_entropy(pred, labels_one_hot, weight=w)
    if loss_type == "sigmoid":
        return F.binary_cross_entropy_with_logits(logits, labels_one_hot, weight=w)
    if loss_type == "focal":
        bce = F.binary_cross_entropy_with_logits(logits, labels_one_hot, reduction="none")
        modulator = torch.exp(
            -gamma * labels_one_hot * logits
            - gamma * torch.log1p(torch.exp(-logits))
        )
        return (w * modulator * bce).sum() / labels_one_hot.sum()
    raise ValueError(f"Unknown loss_type: {loss_type}")
```
