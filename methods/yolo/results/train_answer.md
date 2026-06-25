# YOLO: unified real-time object detection

The detection task itself is simple to state — given an image, return every object with a class label and a tight box — but the systems that do it well do not look like that simple mapping at all; they look like pipelines, and that is the problem I want to fix. DPM scores a sliding window over a HOG pyramid with a root filter, part filters, and deformation costs, then predicts boxes and runs non-max suppression: a real spatial model, but with features, classification, geometry correction, and post-processing all as separate pieces. R-CNN swaps exhaustive windows for Selective Search proposals, warps each of roughly two thousand crops, runs a CNN, scores with per-class SVMs, refines boxes with a regressor, and suppresses duplicates — tens of seconds per image, and trained as disjoint stages. Fast R-CNN shares one convolutional pass and trains the proposal classifier and box regressor jointly, but the proposals still come from Selective Search, whose two-second-per-image cost pins it near half a frame per second. Faster R-CNN learns the proposals with a region proposal network, yet the design is still propose first, classify and refine second, and at its accurate setting still below real-time rates.

Two pressures recur. The first is speed: generating thousands of regions, or making a separate decision per crop, forecloses real-time video even when convolution is shared, because a proposal stage plus a per-region stage still costs latency. The second is trainability: a pipeline optimizes proposal recall, classification, box regression, and duplicate suppression against four separate proxy objectives, so I cannot simply backpropagate through the whole detector as one function of the image and the final targets. There is also a structural context failure. A region classifier sees only a cropped patch; if that patch locally resembles an object it has no access to the surrounding scene that would reveal it as background. This is not that any one model must fail on any one crop — it is the consequence of deciding locally after the proposals have already been cut out. Score every box from features of the full image instead, and the decision can depend on both local appearance and global context.

So I want one network, one image-sized forward pass, and one detection objective. I propose YOLO. The immediate obstacle is that a detector must emit a variable-size set of objects while a network head naturally produces a fixed-size tensor; allocating a fixed list of boxes turns into an assignment problem where two slots chase the same object, another object is dropped, and the symmetry between slots gives the optimizer no clean division of labor. I make the division of labor spatial: divide the image into an $S \times S$ grid, and assign each object to the single cell that contains its center. That fixes the set of output locations and breaks the symmetry — a cell owns only objects whose centers fall inside it, neighboring cells do not all chase the same object, and because the owner is unique, duplicates are suppressed before any post-processing runs. Within a cell, one box predictor is too rigid, since the same cell position must over the training set represent a tall person, a wide car, or a squat object, and a single regressor would average incompatible shapes; letting the cell predict $B$ boxes lets the predictors specialize, and for VOC the minimal useful choice is $B=2$ — enough to separate by shape, not enough to rebuild a proposal set.

Specialization needs a responsibility rule. For an object cell I select the box predictor whose current box has the highest IOU with the ground truth, and only that predictor receives the coordinate gradient for that object. This is self-reinforcing: a predictor that is already better on tall objects gets chosen, improves on tall objects, and stays the tall-object specialist. Early in training all candidate boxes may have zero overlap, so I need a deterministic tie-break, and choosing the smallest coordinate RMSE keeps exactly one predictor responsible. The confidence channel must mean more than "an object is somewhere in this cell"; a good detection score should be high only when a box both contains an object and localizes it well, so I define confidence as $\Pr(\text{Object}) \cdot \mathrm{IOU}(\text{pred}, \text{truth})$. In an empty cell $\Pr(\text{Object})=0$, so every box's confidence target is $0$; in an object cell the responsible box should predict its current IOU with the truth rather than a flat $1$, so a poorly placed box around a real object honestly reports a low score. Class prediction is not duplicated per box, because the two boxes in a cell are competing geometric hypotheses for the same cell-owned object, not two independent objects with independent labels; I predict one shared set of conditional class probabilities $\Pr(\text{Class}_i \mid \text{Object})$ per cell. At test time the multiplication is then clean: the cell's class-conditional probability times each box's confidence gives $\Pr(\text{Class}_i) \cdot \mathrm{IOU}$, a class-specific score combining label belief with localization quality. For VOC this makes the output $S \times S \times (B \cdot 5 + C) = 7 \times 7 \times 30 = 1470$ values and just $98$ candidate boxes — a severe compression against thousands of proposals, which is exactly why the model can be fast.

The backbone is a strong classifier feature extractor, but detection data is too scarce to learn everything from scratch, so I pretrain the convolutional layers on ImageNet at $224 \times 224$ and then switch to $448 \times 448$ for detection because localization needs finer spatial detail. Following GoogLeNet's compute-saving pattern I place $1 \times 1$ channel reductions before the expensive $3 \times 3$ convolutions, but I do not need full inception modules: a stack of $1 \times 1$ and $3 \times 3$ convolutions followed by two fully connected layers produces the fixed tensor, with leaky ReLU of slope $0.1$ in the hidden layers and a linear final layer.

The loss is where the design becomes precise. Plain sum-squared error over the whole tensor is easy to optimize but badly balanced: it weights localization and classification equally even though a detector lives or dies by its boxes, and it sees far more empty cells than object cells, so letting every empty-cell confidence term push hard toward zero makes the trivial "predict low confidence everywhere" solution swamp the rare positive signal. I introduce two weights — $\lambda_{\text{coord}}=5$ multiplies the coordinate terms so localization gets a larger gradient, and $\lambda_{\text{noobj}}=0.5$ multiplies the confidence terms for boxes not responsible for an object so empty and non-responsible boxes do not dominate. A third mismatch in squared error is that the same absolute width error is far worse on a small object than a large one — ten pixels can destroy a bird's IOU but barely move a bus's — so I have the network predict $\sqrt{w}$ and $\sqrt{h}$ rather than $w$ and $h$, and the coordinate loss compares against $\sqrt{w^*}$ and $\sqrt{h^*}$; at decode time those channels are squared back to recover the actual size. With $\mathbb{1}_i^{\text{obj}}$ marking object cells, $\mathbb{1}_{ij}^{\text{obj}}$ the one responsible predictor in such a cell, and $\mathbb{1}_{ij}^{\text{noobj}} = 1 - \mathbb{1}_{ij}^{\text{obj}}$ every non-responsible predictor (both boxes in empty cells and the unused box in an object cell), the objective is

$$
\begin{aligned}
L = \;& \lambda_{\text{coord}} \sum_{i=1}^{S^2} \sum_{j=1}^{B} \mathbb{1}_{ij}^{\text{obj}} \big[(x_i - \hat{x}_{ij})^2 + (y_i - \hat{y}_{ij})^2\big] \\
+ \;& \lambda_{\text{coord}} \sum_{i=1}^{S^2} \sum_{j=1}^{B} \mathbb{1}_{ij}^{\text{obj}} \big[(\sqrt{w_i} - \sqrt{\hat{w}_{ij}})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_{ij}})^2\big] \\
+ \;& \sum_{i=1}^{S^2} \sum_{j=1}^{B} \mathbb{1}_{ij}^{\text{obj}} \big(\mathrm{IOU}(\hat{b}_{ij}, b_i) - \hat{C}_{ij}\big)^2 \\
+ \;& \lambda_{\text{noobj}} \sum_{i=1}^{S^2} \sum_{j=1}^{B} \mathbb{1}_{ij}^{\text{noobj}} \big(0 - \hat{C}_{ij}\big)^2 \\
+ \;& \sum_{i=1}^{S^2} \mathbb{1}_i^{\text{obj}} \sum_{c=1}^{C} \big(p_i(c) - \hat{p}_i(c)\big)^2 .
\end{aligned}
$$

The case structure is exactly what the indicators dictate. An empty cell gets no coordinate and no class loss, with both confidences pushed toward $0$ under $\lambda_{\text{noobj}}$. The responsible box in an object cell gets coordinate loss, a confidence target equal to its IOU with the truth, and the cell's class loss. The non-responsible box in an object cell gets no coordinate or class loss but still has its confidence pushed toward $0$ under $\lambda_{\text{noobj}}$, and the class loss is applied once per cell because the class distribution is conditional on an object being present. This tells the implementation what it must not do: it cannot treat only empty cells as no-object cases, or the unused box in an object cell would go untrained on confidence; it cannot set the responsible box's confidence target to $1$, because that target is the box's current IOU; and the width and height output channels must already live in square-root coordinates, not be square-rooted later inside the loss.

At inference the procedure is intentionally small. One forward pass gives the fixed tensor; for each cell and box I decode the center as $c_x = (\hat{x} + \text{col})/S$ and $c_y = (\hat{y} + \text{row})/S$, square the root-size channels to get $w$ and $h$, multiply box confidence by the cell's class probabilities to get class-specific scores, threshold those, and run one light NMS pass. The grid does most of the duplicate prevention; NMS only cleans up large or boundary-straddling objects. The same choices set the limits — a cell carries only two boxes and one class distribution, so several small objects sharing a cell center cannot all be represented, the coarse final grid makes small objects and unusual aspect ratios hard, and the square-root size trick improves size sensitivity but does not make squared error identical to IOU or average precision. Those are the costs of compressing the detector into a single fixed tensor from one full-image pass.

The implementation follows the original Darknet `cfg/yolo.cfg` and `detection_layer.c` semantics: no BatchNorm, $B=2$, output $1470$, linear final layer, root-size box channels, IOU-rescored object confidence, and no-object loss on all non-responsible boxes.

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
