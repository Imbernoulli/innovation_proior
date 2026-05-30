# DETR

## Problem

Detection is a set-prediction task — output each object once, as a (class, box) pair — but
conventional detectors recast it as dense per-candidate classification and regression over
anchors / proposals / center grids. That forces three hand-designed components: the candidate
field, an overlap-based (many-to-one) assignment heuristic, and non-maximum suppression (NMS)
to remove the duplicates the assignment creates. DETR removes all three and predicts the set
directly, end-to-end.

## Key idea

Two ingredients make direct set prediction work:

1. **A set-based global loss with bipartite matching.** Predict a fixed number N of slots
   (N ≫ typical object count). Pad ground truth to N with a "no object" symbol ∅. Find the
   one-to-one assignment (permutation σ) of predictions to ground-truth objects of minimum
   total cost via the Hungarian algorithm, then supervise the matched pairs. One-to-one
   matching makes duplicates a loss liability, so **no NMS is needed**.

2. **A Transformer encoder-decoder with parallel decoding.** A CNN backbone produces image
   features; the encoder self-attends over them (with 2D positional encoding) for global
   context; the decoder takes N learned **object queries** and, via self-attention among the
   queries and cross-attention into the encoder memory, emits N predictions **in parallel**.
   Self-attention lets the slots coordinate and avoid claiming the same object.

## Final objective

**Matching cost** (per ground-truth/prediction pair; probability, not negative log, to stay
commensurate with the box term):

  L_match(y_i, ŷ_σ(i)) = −1[c_i≠∅]·p̂_σ(i)(c_i) + 1[c_i≠∅]·L_box(b_i, b̂_σ(i)).

σ̂ = argmin_σ Σ_i L_match(y_i, ŷ_σ(i)), solved by the Hungarian algorithm.

**Hungarian (training) loss** over matched pairs (NLL for class; ∅ down-weighted ~10×):

  L = Σ_i [ −log p̂_σ̂(i)(c_i) + 1[c_i≠∅]·L_box(b_i, b̂_σ̂(i)) ].

**Box loss** (L1 for coordinate fidelity + GIoU for scale-invariant overlap with gradient on
disjoint boxes), normalized by #objects in the batch:

  L_box(b, b̂) = λ_iou·(1 − GIoU(b, b̂)) + λ_L1·‖b − b̂‖₁,   (λ_L1 = 5, λ_iou = 2)

  GIoU(A,B) = IoU(A,B) − |C \ (A∪B)| / |C|,   C = smallest box enclosing A and B.

In the matcher, the constant part of `1 − GIoU` is dropped, so the implemented assignment
cost uses `−GIoU`; the argmin is unchanged.

Boxes are predicted **absolutely** (cx, cy, w, h ∈ [0,1] via sigmoid), not as anchor deltas.
**Auxiliary losses**: a shared prediction head + full Hungarian loss after every decoder layer
(deep supervision; helps converge to the right object count).

## Architecture

```
image → ResNet backbone (frozen BN) → C×H×W features
      → 1×1 conv to d=256 → flatten to HW tokens (+ 2D sine positional encoding)
      → Transformer encoder (6 layers, self-attention)
      → Transformer decoder (6 layers): N=100 learned object queries,
        self-attention + cross-attention, decoded in parallel
      → per-slot shared head: 3-layer MLP → box (sigmoid); linear → class (incl. ∅)
```

Training: AdamW, weight decay 1e-4, gradient clip 0.1; transformer lr 1e-4, backbone lr 1e-5;
Xavier init; long schedule. The criterion returns named losses; the training loop applies the
weight dictionary, including `loss_ce: 1`, `loss_bbox: 5`, `loss_giou: 2`, and matching
auxiliary keys when auxiliary decoding losses are enabled.

## Working code

```python
import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops.boxes import box_area


# ----- box geometry + GIoU --------------------------------------------------
def box_cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


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
    area = wh[:, :, 0] * wh[:, :, 1]          # |C|, smallest enclosing box
    return iou - (area - union) / area        # GIoU


# ----- model ----------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

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
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)   # +1 for no-object
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)    # object queries
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples):
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask,
                              self.query_embed.weight, pos[-1])[0]   # [layers, b, N, d]
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


# ----- bipartite matching ---------------------------------------------------
class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2):
        super().__init__()
        self.cost_class, self.cost_bbox, self.cost_giou = cost_class, cost_bbox, cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        cost_class = -out_prob[:, tgt_ids]                    # -prob of true class
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)      # L1
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                         box_cxcywh_to_xyxy(tgt_bbox))
        C = (self.cost_bbox * cost_bbox + self.cost_class * cost_class
             + self.cost_giou * cost_giou).view(bs, num_queries, -1).cpu()
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices]


# ----- set criterion --------------------------------------------------------
class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef                           # down-weight no-object ~10x
        self.register_buffer('empty_weight', empty_weight)

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        losses = {'loss_ce': F.cross_entropy(src_logits.transpose(1, 2),
                                             target_classes, self.empty_weight)}
        if log and target_classes_o.numel() > 0:
            pred = src_logits[idx].argmax(-1)
            losses['class_error'] = 100 - 100 * (pred == target_classes_o).float().mean()
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        return {'cardinality_error': F.l1_loss(card_pred.float(), tgt_lengths.float())}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
        return {'loss_bbox': loss_bbox.sum() / num_boxes,
                'loss_giou': loss_giou.sum() / num_boxes}

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'cardinality': self.loss_cardinality,
        }
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def _normalized_num_boxes(self, outputs, targets):
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float,
                                    device=next(iter(outputs.values())).device)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_boxes)
            world_size = torch.distributed.get_world_size()
        else:
            world_size = 1
        return torch.clamp(num_boxes / world_size, min=1).item()

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        indices = self.matcher(outputs_without_aux, targets)
        num_boxes = self._normalized_num_boxes(outputs_without_aux, targets)
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {'log': False} if loss == 'labels' else {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    losses.update({k + f'_{i}': v for k, v in l_dict.items()})
        return losses


# ----- inference postprocessing ---------------------------------------------
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

Inference: softmax the class logits, drop the ∅ class, take each slot's best real class, and
scale its box back to image coordinates. There is no NMS.
