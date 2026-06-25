# YOLO: unified real-time object detection

## Method

Map an entire image to a fixed detection tensor in one CNN evaluation.

- Divide the image into an `S x S` grid. The cell containing an object's center is responsible for that object.
- Each cell predicts `B` boxes. A box has `(x, y, sqrt(w), sqrt(h), confidence)`, where `x,y` are offsets inside the cell and `w,h` are normalized by image size. At decode time, square the root-size channels to recover `w,h`.
- Each cell predicts one shared set of `C` conditional class scores, `Pr(Class_i | Object)`, not one class vector per box.
- Box confidence means `Pr(Object) * IOU(pred, truth)`. In an empty cell the target is `0`; for the responsible predictor in an object cell the target is its IOU with the ground-truth box.
- At test time, each class-specific box score is `Pr(Class_i | Object) * confidence = Pr(Class_i) * IOU`.

For PASCAL VOC, `S=7`, `B=2`, `C=20`, so the output is `7 x 7 x 30 = 1470` values and 98 candidate boxes.

## Loss

Let `1_i^obj` mark cells containing an object, `1_ij^obj` mark the one responsible predictor in such a cell, and `1_ij^noobj = 1 - 1_ij^obj` mark every non-responsible predictor, including all predictors in empty cells and the unused predictor in an object cell. With `lambda_coord = 5` and `lambda_noobj = 0.5`:

```text
L =
  lambda_coord * sum_{i=1}^{S^2} sum_{j=1}^{B} 1_ij^obj
      [(x_i - xhat_ij)^2 + (y_i - yhat_ij)^2]

+ lambda_coord * sum_{i=1}^{S^2} sum_{j=1}^{B} 1_ij^obj
      [(sqrt(w_i) - sqrt(w_hat_ij))^2 + (sqrt(h_i) - sqrt(h_hat_ij))^2]

+ sum_{i=1}^{S^2} sum_{j=1}^{B} 1_ij^obj
      (IOU(b_hat_ij, b_i) - C_hat_ij)^2

+ lambda_noobj * sum_{i=1}^{S^2} sum_{j=1}^{B} 1_ij^noobj
      (0 - C_hat_ij)^2

+ sum_{i=1}^{S^2} 1_i^obj sum_{c=1}^{C}
      (p_i(c) - p_hat_i(c))^2
```

Training cases:

- Empty cell: no coordinate loss, no class loss, both box confidences pushed to `0` with `lambda_noobj`.
- Object cell, responsible box: coordinate loss, confidence target `IOU(pred, truth)`, and the cell's class loss.
- Object cell, non-responsible box: no coordinate or class loss, confidence pushed to `0` with `lambda_noobj`.
- Responsibility: choose the box with highest current IOU to the truth; the Darknet code falls back to smallest coordinate RMSE when all IOUs are zero.

## Reference Code

This PyTorch sketch follows the Darknet `cfg/yolo.cfg` and `detection_layer.c` semantics: no BatchNorm, `B=2`, output `1470`, linear final layer, root-size box channels, IOU-rescored object confidence, and no-object loss on all non-responsible boxes.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


architecture_config = [
    (7, 64, 2, 3), "M",
    (3, 192, 1, 1), "M",
    (1, 128, 1, 0), (3, 256, 1, 1), (1, 256, 1, 0), (3, 512, 1, 1), "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0), (3, 1024, 1, 1), "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1), (3, 1024, 2, 1),
    (3, 1024, 1, 1), (3, 1024, 1, 1),
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=True, **kwargs)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(self.conv(x))


class YoloV1(nn.Module):
    def __init__(self, in_channels=3, split_size=7, num_boxes=2, num_classes=20):
        super().__init__()
        self.in_channels = in_channels
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes
        self.darknet = self._create_conv_layers(architecture_config)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * self.S * self.S, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, self.S * self.S * (self.C + self.B * 5)),
        )

    def forward(self, x):
        return self.head(self.darknet(x))

    def _create_conv_layers(self, arch):
        layers = []
        in_c = self.in_channels
        for item in arch:
            if item == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif isinstance(item, tuple):
                k, out_c, stride, pad = item
                layers.append(CNNBlock(in_c, out_c, kernel_size=k, stride=stride, padding=pad))
                in_c = out_c
            else:
                conv1, conv2, repeats = item
                for _ in range(repeats):
                    k, out_c, stride, pad = conv1
                    layers.append(CNNBlock(in_c, out_c, kernel_size=k, stride=stride, padding=pad))
                    in_c = out_c
                    k, out_c, stride, pad = conv2
                    layers.append(CNNBlock(in_c, out_c, kernel_size=k, stride=stride, padding=pad))
                    in_c = out_c
        return nn.Sequential(*layers)


def box_iou_center(boxes1, boxes2, eps=1e-6):
    b1_xy1 = boxes1[..., :2] - boxes1[..., 2:4] / 2
    b1_xy2 = boxes1[..., :2] + boxes1[..., 2:4] / 2
    b2_xy1 = boxes2[..., :2] - boxes2[..., 2:4] / 2
    b2_xy2 = boxes2[..., :2] + boxes2[..., 2:4] / 2

    inter_xy1 = torch.maximum(b1_xy1, b2_xy1)
    inter_xy2 = torch.minimum(b1_xy2, b2_xy2)
    inter_wh = (inter_xy2 - inter_xy1).clamp_min(0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]

    area1 = (b1_xy2[..., 0] - b1_xy1[..., 0]).clamp_min(0) * (
        b1_xy2[..., 1] - b1_xy1[..., 1]
    ).clamp_min(0)
    area2 = (b2_xy2[..., 0] - b2_xy1[..., 0]).clamp_min(0) * (
        b2_xy2[..., 1] - b2_xy1[..., 1]
    ).clamp_min(0)
    return inter / (area1 + area2 - inter).clamp_min(eps)


class YoloV1Loss(nn.Module):
    """Darknet-layout loss.

    predictions: (N, S*S*(C + B + 4B)), laid out as all class scores,
    then all B confidences, then all B boxes.

    target: (N, S, S, 1 + C + 4), laid out as objectness, one-hot class,
    and one ground-truth box (x_cell, y_cell, w, h) for the object-centered cell.
    """

    def __init__(self, S=7, B=2, C=20, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, target):
        n = predictions.shape[0]
        predictions = predictions.reshape(n, -1)
        locations = self.S * self.S

        class_scores = predictions[:, : locations * self.C].reshape(n, self.S, self.S, self.C)
        conf_start = locations * self.C
        box_start = locations * (self.C + self.B)
        confidences = predictions[:, conf_start:box_start].reshape(n, self.S, self.S, self.B)
        boxes = predictions[:, box_start:].reshape(n, self.S, self.S, self.B, 4)

        exists = target[..., 0].float()
        target_classes = target[..., 1 : 1 + self.C]
        target_boxes = target[..., 1 + self.C : 1 + self.C + 4]

        # For matching within a cell, adding the same cell offset to prediction
        # and truth would only translate both boxes. Darknet scales local x/y by S.
        pred_for_match = boxes.clone()
        pred_for_match[..., :2] = pred_for_match[..., :2] / self.S
        pred_for_match[..., 2:4] = pred_for_match[..., 2:4].pow(2)

        truth_for_match = target_boxes.clone()
        truth_for_match[..., :2] = truth_for_match[..., :2] / self.S
        ious = box_iou_center(pred_for_match, truth_for_match.unsqueeze(3))

        best_iou, iou_idx = ious.max(dim=-1)
        rmse = (pred_for_match - truth_for_match.unsqueeze(3)).pow(2).sum(dim=-1)
        _, rmse_idx = rmse.min(dim=-1)
        best_idx = torch.where(best_iou > 0, iou_idx, rmse_idx)
        responsible = F.one_hot(best_idx, self.B).float() * exists.unsqueeze(-1)
        no_object = 1.0 - responsible

        target_coord = target_boxes.clone()
        target_coord[..., 2:4] = torch.sqrt(target_coord[..., 2:4].clamp_min(0))
        coord_loss = ((boxes - target_coord.unsqueeze(3)).pow(2) * responsible.unsqueeze(-1)).sum()

        object_loss = ((confidences - best_iou.detach().unsqueeze(-1)).pow(2) * responsible).sum()
        no_object_loss = (confidences.pow(2) * no_object).sum()
        class_loss = ((class_scores - target_classes).pow(2) * exists.unsqueeze(-1)).sum()

        return (
            self.lambda_coord * coord_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )
```

## Training And Inference

Pretrain the first 20 convolutional layers for ImageNet classification at `224 x 224`. Convert to detection by adding 4 convolutional layers and 2 fully connected layers, then train at `448 x 448` for about 135 epochs on VOC 2007+2012 with batch 64, momentum `0.9`, weight decay `5e-4`, dropout `0.5`, random scale/translation up to 20%, and HSV exposure/saturation jitter up to `1.5x`. The schedule warms from `1e-3` to `1e-2`, then uses `1e-2` for 75 epochs, `1e-3` for 30, and `1e-4` for 30.

Inference decodes each box as:

```text
cx = (x_hat + cell_col) / S
cy = (y_hat + cell_row) / S
w  = sqrt_w_hat^2
h  = sqrt_h_hat^2
score_c = C_hat * p_hat_c
```

Then threshold class-specific scores and run light NMS.

## Limits

The fixed grid is the price of speed. A cell can emit only `B=2` boxes and one class distribution, so crowded small objects in the same cell are structurally hard. Coarse final features make small objects and unusual aspect ratios difficult. The square-root width/height trick improves size sensitivity, but squared error remains only a proxy for IOU and average precision.
