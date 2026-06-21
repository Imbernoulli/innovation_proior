Dense, one-stage object detectors classify a regular grid of roughly 10^4 to 10^5 candidate boxes per image in a single forward pass. Because only a handful of those boxes actually contain objects, training encounters an extreme foreground-to-background imbalance on the order of one to a thousand. Under ordinary cross entropy, each well-classified background box still pays a small but non-zero loss, and the sum over the entire dense grid dwarfs the contribution of the rare foreground examples. The gradient is then dominated by easy negatives, so the model drifts toward a degenerate solution that is excellent at saying "background" and poor at detecting objects. This imbalance is the main reason one-stage detectors have historically trailed two-stage detectors by a wide margin on benchmarks like COCO.

Existing approaches address the symptom rather than the loss itself. Two-stage detectors sidestep the problem architecturally: a proposal stage filters out most background before the classifier ever sees it, and the second stage trains on minibatches sampled to a roughly fixed positive-to-negative ratio. That works, but it sacrifices the simplicity and speed of a single dense pass. Hard example mining methods such as OHEM keep only the highest-loss examples for each minibatch, which does focus on hard negatives, but it discards easy examples entirely and introduces extra machinery and hyperparameters. Class weighting can rebalance the positive and negative classes, yet it treats every negative equally and cannot distinguish an ambiguous hard negative from a trivially easy one. What is needed is a loss that continuously down-weights easy examples and leaves hard ones untouched, without any sampling step.

The method is Focal Loss. It starts from the standard cross-entropy written in terms of the probability p_t assigned to the correct class: CE(p_t) = -log p_t. The focal loss multiplies this by a modulating factor (1 - p_t)^gamma, giving FL(p_t) = -(1 - p_t)^gamma log p_t. When an example is hard or misclassified, p_t is small, the factor is close to one, and the loss is nearly the same as cross entropy. When an example is easy and correctly classified, p_t is close to one, the factor shrinks toward zero, and the loss is sharply suppressed. With gamma equal to 2, an example at p_t = 0.9 is down-weighted by a factor of one hundred, while a hard example at p_t = 0.5 is down-weighted by only four. The focusing parameter gamma smoothly interpolates between cross entropy at gamma = 0 and stronger focusing as gamma grows.

In practice the alpha-balanced form is used: FL(p_t) = -alpha_t (1 - p_t)^gamma log p_t, where alpha_t is alpha for the positive class and 1 - alpha for the negative class. This handles the separate positive-versus-negative imbalance, while the modulating factor handles the easy-versus-hard imbalance within each class. The two knobs are coupled: because gamma already quiets the flood of easy negatives, alpha can be smaller than it would be for plain weighted cross entropy. A typical pairing is alpha = 0.25 and gamma = 2. A final practical detail is initialization. A default classifier outputs foreground probability near 0.5 everywhere, which makes the first-iteration loss from the sea of background anchors explode. Initializing the final classification layer bias so that the initial foreground probability is a small prior pi, for example pi = 0.01, stabilizes training. The bias is set to -log((1 - pi) / pi), which is approximately -4.6 for pi = 0.01.

Focal loss is usually paired with a deliberately plain one-stage detector called RetinaNet so that the accuracy gains can be attributed to the loss rather than architecture. RetinaNet uses a Feature Pyramid Network built on ResNet, with pyramid levels P3 through P7 and 256 channels each. At every level it places nine anchors per spatial location, combining three aspect ratios and three scales, with anchor areas spanning 32^2 to 512^2. Two small fully-convolutional subnets, sharing weights across levels but separate from each other, predict class scores and box offsets. The classification subnet ends in a sigmoid producing K independent binary predictions per anchor, and the box subnet outputs class-agnostic offsets. The focal loss is applied to all anchors in the image, summed, and normalized by the number of anchors assigned to a ground-truth box, plus standard smooth-L1 box regression on positives.

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
