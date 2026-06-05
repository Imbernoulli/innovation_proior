# Focal Loss and RetinaNet

## Problem

One-stage (dense) object detectors classify a regular grid of ~10^4–10^5 candidate boxes (anchors) per image in a single pass. Only a few of those boxes contain objects, so training faces an extreme foreground-to-background class imbalance (~1:1000). Under standard cross entropy, the vast number of *easy negatives* — background patches the model already classifies correctly — each contributes a small but non-zero loss, and their sum dominates the total loss and gradient, driving the model toward a degenerate solution. This is the central reason one-stage detectors trail two-stage detectors (which sidestep imbalance via a proposal cascade and biased sampling) in accuracy.

## Key idea

Reshape cross entropy so that well-classified examples are down-weighted, focusing training on hard examples — without any sampling or hard-example mining. With p_t the probability assigned to the correct class, standard cross entropy is CE(p_t) = −log p_t. The **focal loss** adds a modulating factor:

    FL(p_t) = −(1 − p_t)^γ · log(p_t)

- When an example is hard or misclassified (p_t → 0), the factor (1 − p_t)^γ → 1 and the loss is essentially unchanged.
- When an example is easy (p_t → 1), the factor → 0 and the loss is sharply suppressed.
- γ ≥ 0 is the **focusing parameter**; γ = 0 recovers CE. For γ = 2, an example at p_t = 0.9 has 100× lower loss than CE, and at p_t ≈ 0.968 it has 1000× lower loss, while hard examples (p_t ≤ 0.5) are down-weighted by at most 4×.

The **α-balanced** form, used in practice, adds class weighting (α_t = α for positives, 1−α for negatives) on top of the modulating factor:

    FL(p_t) = −α_t · (1 − p_t)^γ · log(p_t)

α should decrease as γ increases; α = 0.25, γ = 2 is the default pairing. Unlike a robust loss (e.g. Huber) that down-weights hard outliers, the focal loss down-weights easy inliers.

**Gradient (w.r.t. the logit x, with x_t = yx, p_t = σ(x_t), before α-balancing):**

    dFL/dx = y · (1 − p_t)^γ · (γ · p_t · log(p_t) + p_t − 1)

which reduces to the CE gradient y(p_t − 1) when γ = 0. The α-balanced form multiplies this derivative by α_t.

**Prior initialization for stability.** A default-initialized classifier predicts foreground-probability ≈ 0.5 everywhere, so the first-iteration loss from the sea of background anchors is huge and training diverges. Initialize the final classification conv's bias to a foreground prior π (e.g. π = 0.01): b = −log((1 − π)/π), so σ(b) = π. This is a change to initialization, not to the loss, and it stabilizes both CE and focal-loss training.

## RetinaNet

A deliberately simple one-stage detector so that accuracy gains come from the loss, not the architecture:

- **Backbone:** Feature Pyramid Network (top-down pathway + lateral connections) on ResNet, pyramid levels P3–P7, 256 channels each. P3–P5 from ResNet stages C3–C5; P6 = 3×3 stride-2 conv on C5; P7 = ReLU then 3×3 stride-2 conv on P6.
- **Anchors:** A = 9 per level — 3 aspect ratios {1:2, 1:1, 2:1} × 3 scales {2^0, 2^{1/3}, 2^{2/3}}, areas 32²–512². IoU ≥ 0.5 → foreground; [0, 0.4) → background; [0.4, 0.5) → ignored.
- **Two subnets, shared across levels, separate parameters.** Classification subnet: four 3×3 conv (256, ReLU) → 3×3 conv with K·A outputs → sigmoid (K independent binary predictions per anchor). Box subnet: identical FCN ending in 4·A linear outputs, class-agnostic.
- **Loss:** focal loss applied to **all** ~100k anchors per image, summed and normalized by the number of anchors assigned to a ground-truth box (the positives; the implementation can keep a moving average of this count for scale stability), plus smooth-L1 box regression on positives. Trained with SGD; prior-π bias init on the final cls conv.

## Code

The dense head, the sigmoid focal loss, and the shared positive-anchor normalization:

```python
import math
import torch
from torch import nn
from torch.nn import functional as F


class DenseHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes, conv_dims=None, initial_prob=0.01):
        super().__init__()
        conv_dims = conv_dims or [in_channels] * 4
        cls_layers = []
        bbox_layers = []
        prev_channels = in_channels
        for out_channels in conv_dims:
            cls_layers += [nn.Conv2d(prev_channels, out_channels, 3, padding=1), nn.ReLU()]
            bbox_layers += [nn.Conv2d(prev_channels, out_channels, 3, padding=1), nn.ReLU()]
            prev_channels = out_channels

        self.cls_subnet = nn.Sequential(*cls_layers)
        self.bbox_subnet = nn.Sequential(*bbox_layers)
        self.cls_score = nn.Conv2d(prev_channels, num_anchors * num_classes, 3, padding=1)
        self.bbox_pred = nn.Conv2d(prev_channels, num_anchors * 4, 3, padding=1)

        for module in [self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred]:
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, mean=0, std=0.01)
                    nn.init.constant_(layer.bias, 0)

        bias = -math.log((1 - initial_prob) / initial_prob)
        nn.init.constant_(self.cls_score.bias, bias)

    def forward(self, features):
        logits = []
        bbox_reg = []
        for feature in features:
            logits.append(self.cls_score(self.cls_subnet(feature)))
            bbox_reg.append(self.bbox_pred(self.bbox_subnet(feature)))
        return logits, bbox_reg


def dense_binary_loss(inputs, targets, loss_params, reduction="sum"):
    inputs = inputs.float()
    targets = targets.float()
    loss_params = {"alpha": 0.25, "gamma": 2.0} if loss_params is None else loss_params
    alpha = loss_params.get("alpha", -1)
    gamma = loss_params.get("gamma", 2.0)

    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


class PositiveAnchorNormalizer:
    def __init__(self, momentum=100):
        self.momentum = momentum
        self.value = None

    def update(self, num_pos):
        num_pos = max(float(num_pos), 1.0)
        if self.value is None:
            self.value = num_pos
        else:
            self.value = self.value * (self.momentum - 1) / self.momentum + num_pos / self.momentum
        return self.value


def classification_loss(pred_logits, gt_labels, num_classes, loss_normalizer, loss_params):
    logits = torch.cat(pred_logits, dim=1) if isinstance(pred_logits, (list, tuple)) else pred_logits
    gt_labels = torch.stack(gt_labels) if isinstance(gt_labels, (list, tuple)) else gt_labels

    valid_mask = gt_labels >= 0
    pos_mask = valid_mask & (gt_labels != num_classes)
    normalizer_value = loss_normalizer.update(pos_mask.sum().item())

    targets = F.one_hot(gt_labels[valid_mask], num_classes + 1)[:, :-1]
    loss_cls = dense_binary_loss(
        logits[valid_mask], targets.to(logits.dtype), loss_params, reduction="sum"
    )
    return loss_cls / normalizer_value, normalizer_value, pos_mask


def detector_losses(anchors, pred_logits, pred_deltas, gt_labels, gt_boxes,
                    num_classes, loss_normalizer, loss_params):
    loss_cls, normalizer_value, pos_mask = classification_loss(
        pred_logits, gt_labels, num_classes, loss_normalizer, loss_params
    )
    deltas = torch.cat(pred_deltas, dim=1) if isinstance(pred_deltas, (list, tuple)) else pred_deltas
    target_deltas = encode_boxes(anchors, gt_boxes)

    if pos_mask.any():
        loss_box = smooth_l1_loss(deltas[pos_mask], target_deltas[pos_mask], reduction="sum")
    else:
        loss_box = deltas.sum() * 0

    return {"loss_cls": loss_cls, "loss_box_reg": loss_box / normalizer_value}
```

## Alternate form (FL*)

The exact form is not crucial. With x_t = y·x and p_t = σ(x_t), define p_t* = σ(γ·x_t + β) and FL* = −log(p_t*)/γ. Parameters γ (steepness) and β (shift) control where the loss falls off, and dFL*/dx = y(p_t* − 1). The useful property is the same: the gradient is small once x_t > 0 is confidently correct and large when the example remains wrong or ambiguous.
