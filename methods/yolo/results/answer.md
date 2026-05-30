# YOLO — You Only Look Once: Unified, Real-Time Object Detection

## Problem

Detect every object in an image (a tight bounding box + class for each), fast enough for real-time video, as a single model trained end-to-end against detection — unlike the contemporary detectors (R-CNN family, DPM) that are slow pipelines of separately trained stages and classify isolated regions without global context.

## Key idea

Frame detection as a **single regression** from image pixels straight to box coordinates and class probabilities, computed in **one forward pass** of one CNN over the full image.

- Divide the image into an **S×S grid**. The cell containing an object's center is **responsible** for detecting it. This gives a fixed-size output, divides spatial labor between predictors, and suppresses duplicate detections by construction.
- Each cell predicts **B bounding boxes** (each is `x, y, w, h, confidence`) and **one shared set of C conditional class probabilities** `Pr(Class_i | Object)`.
  - `(x, y)` = box center as an offset within the cell (∈[0,1]); `w, h` = normalized by image size (∈[0,1]).
  - **confidence ≡ Pr(Object) · IOU(pred, truth)** — zero if no object, otherwise the box's IOU with ground truth, so the network reports its own localization quality.
- Output is an **S×S×(B·5 + C)** tensor. For PASCAL VOC: S=7, B=2, C=20 → **7×7×30**, i.e. 98 boxes per image.
- Test-time class-specific score per box: `Pr(Class_i | Object) · Pr(Object) · IOU = Pr(Class_i) · IOU`. Threshold, then light NMS (adds ~2–3% mAP, not load-bearing).

## Network

24 convolutional layers + 2 fully-connected layers, GoogLeNet-inspired but with **1×1 reduction layers** (NIN/inception trick) before 3×3 convs instead of inception modules. Leaky ReLU (slope 0.1) everywhere except a linear final layer. Conv layers pretrained on ImageNet at 224×224, then detection at **448×448** (fine detail), with 4 extra conv + 2 FC layers added on top.

## Loss

Sum-squared error, reshaped to respect detection, with three fixes for its misalignment with mAP:

- **λ_coord = 5** upweights coordinate error (localization matters more; also amplifies the rare object cells).
- **λ_noobj = 0.5** downweights the confidence error of empty cells (most cells are empty; otherwise their "confidence→0" gradient swamps the object cells and training diverges).
- Predict **√w, √h** instead of w, h (a fixed absolute size error costs less on large boxes than small ones).

Only the **responsible** predictor (highest current IOU with the truth) in a cell gets coordinate + object-confidence gradient; class error is applied only to cells that contain an object.

$$
\begin{aligned}
\mathcal{L} =\; & \lambda_\text{coord} \sum_{i=0}^{S^2}\sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} \big[(x_i-\hat{x}_i)^2 + (y_i-\hat{y}_i)^2\big] \\
+\; & \lambda_\text{coord} \sum_{i=0}^{S^2}\sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} \big[(\sqrt{w_i}-\sqrt{\hat{w}_i})^2 + (\sqrt{h_i}-\sqrt{\hat{h}_i})^2\big] \\
+\; & \sum_{i=0}^{S^2}\sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} (C_i-\hat{C}_i)^2
   + \lambda_\text{noobj} \sum_{i=0}^{S^2}\sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{noobj}} (C_i-\hat{C}_i)^2 \\
+\; & \sum_{i=0}^{S^2} \mathbb{1}_{i}^{\text{obj}} \sum_{c \in \text{classes}} (p_i(c)-\hat{p}_i(c))^2
\end{aligned}
$$

where $\mathbb{1}_{i}^{\text{obj}}$ marks cells containing an object and $\mathbb{1}_{ij}^{\text{obj}}$ marks the responsible predictor; the confidence target $C_i$ for a responsible box is its IOU with the ground truth.

**Training:** ~135 epochs on VOC 2007+2012; batch 64, momentum 0.9, weight decay 5e-4; LR warmup 1e-3→1e-2, then 1e-2 (75 ep) → 1e-3 (30 ep) → 1e-4 (30 ep); dropout 0.5 after the first FC; random scale/translate ±20% and HSV exposure/saturation jitter ×1.5.

## Limitations

Each cell predicts only B=2 boxes and one class → struggles with groups of small objects (e.g. flocks of birds); coarse features hurt unusual aspect ratios; the SSE loss (even with √) under-penalizes errors on small boxes, so localization is the dominant error source.

## Code

```python
import torch
import torch.nn as nn

# (kernel, out_channels, stride, padding); "M" = 2x2 maxpool; [block, n] = repeat n times.
architecture_config = [
    (7, 64, 2, 3), "M",
    (3, 192, 1, 1), "M",
    (1, 128, 1, 0), (3, 256, 1, 1), (1, 256, 1, 0), (3, 512, 1, 1), "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4], (1, 512, 1, 0), (3, 1024, 1, 1), "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2], (3, 1024, 1, 1), (3, 1024, 2, 1),
    (3, 1024, 1, 1), (3, 1024, 1, 1),
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Yolov1(nn.Module):
    def __init__(self, in_channels=3, split_size=7, num_boxes=2, num_classes=20):
        super().__init__()
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(architecture_config)
        self.fcs = self._create_fcs(split_size, num_boxes, num_classes)
    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))
    def _create_conv_layers(self, arch):
        layers, in_c = [], self.in_channels
        for x in arch:
            if type(x) == tuple:
                layers += [CNNBlock(in_c, x[1], kernel_size=x[0], stride=x[2], padding=x[3])]
                in_c = x[1]
            elif x == "M":
                layers += [nn.MaxPool2d(2, 2)]
            elif type(x) == list:
                c1, c2, reps = x[0], x[1], x[2]
                for _ in range(reps):
                    layers += [CNNBlock(in_c, c1[1], kernel_size=c1[0], stride=c1[2], padding=c1[3])]
                    layers += [CNNBlock(c1[1], c2[1], kernel_size=c2[0], stride=c2[2], padding=c2[3])]
                    in_c = c2[1]
        return nn.Sequential(*layers)
    def _create_fcs(self, S, B, C):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (C + B * 5)),
        )


class YoloLoss(nn.Module):
    # Per-cell output layout: [ C class scores | box1: conf,x,y,w,h | box2: conf,x,y,w,h ].
    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S, self.B, self.C = S, B, C
        self.lambda_coord = 5
        self.lambda_noobj = 0.5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        _, bestbox = torch.max(ious, dim=0)          # responsible box (higher IOU)
        exists_box = target[..., 20].unsqueeze(3)    # 1 if an object is in this cell

        # coordinate loss (responsible box only), sqrt on w,h
        box_predictions = exists_box * (
            bestbox * predictions[..., 26:30] + (1 - bestbox) * predictions[..., 21:25]
        )
        box_targets = exists_box * target[..., 21:25]
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # object confidence loss (target = IOU of the responsible box)
        pred_box = bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21]),
        )

        # no-object confidence loss (both boxes, target 0)
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1),
        )

        # class loss (only cells with an object)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2),
        )

        return (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )
```
