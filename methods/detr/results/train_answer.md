Object detection is fundamentally a set-prediction problem: for an image we want the unordered collection of objects, each represented by a class label and a bounding box, with no duplicates. The dominant detectors of the time, however, did not predict a set directly. They tiled the image with a dense field of reference candidates—anchors, region proposals, or center grids—and reduced detection to a large collection of independent per-candidate classification and regression problems. That reduction forces three hand-designed components that have nothing to do with the task itself: the candidate field with its scales, aspect ratios, and strides; an overlap-based assignment heuristic that labels many candidates positive for the same object; and non-maximum suppression, which prunes the duplicate predictions that the assignment necessarily creates. The model is therefore not end-to-end: the loss optimizes a surrogate per-reference task, while the desired clean set is manufactured by a separate, non-learned post-process.

The root cause is that training admits many predictions per object. If we could instead train with a one-to-one correspondence between predictions and ground-truth objects, duplicates would become a liability rather than an inevitability, and NMS would have no work left. The challenge is to make that correspondence learnable while remaining competitive on a hard benchmark like COCO. The solution is DETR, which stands for DEtection TRansformer.

DETR treats detection as direct end-to-end set prediction. It commits to a fixed number N of output slots, chosen larger than the maximum number of objects expected in an image, and adds a special “no object” class so that unused slots can remain silent. The output is always N predictions, but only the slots matched to real objects emit actual detections. Because a set has no natural ordering, the loss must be permutation-invariant: we cannot fix prediction i to ground-truth object i. Instead, for each image we solve a bipartite matching between the N predictions and the ground-truth set, padded to N with the no-object symbol, and supervise only the matched pairs. This matching is computed exactly with the Hungarian algorithm, using a cost that combines class probability and box similarity. The one-to-one nature of the matching means that if two slots both predict the same object, only one can be matched to it; the other is matched to no-object and is trained to be silent. De-duplication is therefore baked into the objective.

The architecture that makes this work is a Transformer encoder-decoder. A ResNet backbone extracts a compact feature map, which is projected to the transformer dimension and flattened into a sequence of tokens. A 2D sinusoidal positional encoding is added so that spatial location is preserved. The encoder self-attends over these tokens to build a global image representation. The decoder then produces the N output slots in parallel from N learned object queries—distinct learned embeddings that break the symmetry among slots. Decoder self-attention lets the slots communicate and divide up the objects, while cross-attention gathers the relevant image evidence for each slot. Parallel decoding is natural for a set, where order is meaningless, and is far faster than the autoregressive decoders used in earlier set-prediction attempts. Each decoder output passes through a shared head: a linear layer predicts class logits, including the no-object class, and a small MLP predicts absolute box coordinates in [0, 1] through a sigmoid.

The loss has two parts. The matching cost, used only to pick the assignment, keeps class and box terms on the same scale: it uses the predicted probability of the true class and L1 plus negative GIoU distance for the box. Once the Hungarian assignment is fixed, the training loss uses a proper negative log-likelihood for classification across all slots, with the no-object class down-weighted by roughly ten to prevent the model from collapsing to silence. The box loss combines L1 for coordinate fidelity and 1 minus GIoU for scale-invariant overlap; GIoU is essential because plain IoU has zero gradient when boxes do not overlap, which is common early in training. The box terms are normalized by the number of target objects in the batch. Because the decoder is six layers deep and only the final output is used at inference, auxiliary decoding losses are applied at every decoder layer with shared prediction heads; this deep supervision helps the model learn the correct cardinality of objects per image.

```python
import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h,
                        cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def box_iou(boxes1, boxes2):
    area1, area2 = box_area(boxes1), box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union, union


def generalized_box_iou(boxes1, boxes2):
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / area


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DETR(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples):
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        hs = self.transformer(self.input_proj(src), mask,
                              self.query_embed.weight, pos[-1])[0]
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b}
                                  for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
        return out


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                         box_cxcywh_to_xyxy(tgt_bbox))
        C = (self.cost_bbox * cost_bbox + self.cost_class * cost_class
             + self.cost_giou * cost_giou).view(bs, num_queries, -1).cpu()
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices]


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i)
                               for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels(self, outputs, targets, indices, num_boxes):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        return {'loss_ce': F.cross_entropy(src_logits.transpose(1, 2),
                                           target_classes, self.empty_weight)}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        return {'loss_bbox': loss_bbox.sum() / num_boxes,
                'loss_giou': loss_giou.sum() / num_boxes}

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float,
                                    device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes, min=1).item()
        losses = {}
        for loss in self.losses:
            losses.update(getattr(self, f'loss_{loss}')(outputs, targets, indices, num_boxes))
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = getattr(self, f'loss_{loss}')(aux_outputs, targets, indices, num_boxes)
                    losses.update({k + f'_{i}': v for k, v in l_dict.items()})
        return losses


class PostProcess(nn.Module):
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)
        boxes = box_cxcywh_to_xyxy(out_bbox)
        img_h, img_w = target_sizes.unbind(1)
        scale = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale[:, None, :]
        return [{'scores': s, 'labels': l, 'boxes': b}
                for s, l, b in zip(scores, labels, boxes)]
```

Training uses AdamW with weight decay around 1e-4 and gradient clipping around 0.1. The transformer is trained from scratch at a learning rate near 1e-4, while the ImageNet-pretrained backbone is fine-tuned more gently, around 1e-5, with its batch-normalization statistics frozen. The long schedule is important: bipartite matching plus a from-scratch transformer converges more slowly than standard detectors. At inference, each slot’s best real class is taken, its box is scaled back to image coordinates, and no non-maximum suppression is applied. The result is a detector that outputs the desired set directly, without anchors, assignment heuristics, or post-processing.
