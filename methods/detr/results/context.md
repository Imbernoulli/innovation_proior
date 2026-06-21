## Research question

Object detection asks for a *set*: given an image, output the bounding boxes and category
labels of every object of interest. The output is genuinely a set — unordered, of variable
size, with no two elements naming the same object. The strong detectors of the time
reformulate detection as a large bag of *independent* per-location problems defined over a
dense field of reference candidates (region proposals, anchor boxes, or a grid of
object-center hypotheses): each candidate is classified and its box regressed on its own. This
reformulation brings with it a candidate field (its scales, aspect ratios, strides), a rule
that decides which candidate is "responsible" for which ground-truth object, and a
post-processing step that collapses many near-duplicate predictions back down to one box per
object.

The setting under study: how to train detection as direct set prediction — so that the
system's output *is* the loss objective — and have it remain competitive with the strongest
available detectors on a challenging benchmark. End-to-end set prediction had already been
applied to structured-output tasks such as machine translation and speech recognition.

## Background

**Detection as dense surrogate prediction.** The dominant paradigm makes predictions
*relative to initial guesses*. A dense set of reference boxes is laid over the image; for
each, the model predicts a class and a small offset that deforms the reference into a final
box. Two-stage detectors first generate data-dependent proposals and then refine them;
single-stage detectors regress directly from a fixed anchor grid or from candidate object
centers. The candidate field is large (tens of thousands of references) so that some
reference lands near every object. This paradigm uses three hand-designed pieces.

1. *The candidate field.* Anchor scales, aspect ratios, and strides are chosen by hand.
   A diagnostic study of the prevailing detectors showed that the gap between anchor-based
   and center-based detectors largely closes once the assignment of references to ground
   truth is controlled for.
2. *The assignment heuristic.* Training needs a target for every reference. The standard
   rule is based on overlap: a reference whose Intersection-over-Union (IoU) with a
   ground-truth box exceeds a threshold is a positive for that object, below another
   threshold a negative, in between ignored. Because the field is dense, *many* references
   pass the threshold for the same object — assignment is many-to-one by construction.
3. *Non-maximum suppression (NMS).* Many positives per object yield many overlapping
   predicted boxes per object at inference. They are pruned by a greedy procedure: sort by
   confidence, keep the top box, discard any box whose IoU with it exceeds a threshold,
   repeat. NMS runs outside the trained model with its own threshold.

In this paradigm the loss optimizes per-reference surrogate classification and regression,
and the final clean set of one box per object is produced by the separate NMS stage.

**Permutation-invariant set losses.** Outside detection, the canonical way to train a
fixed-size set predictor against ground truth is to make the loss invariant to the order of
the predictions. The standard device is a bipartite matching: pair each ground-truth element
with exactly one prediction by solving a minimum-cost assignment, then supervise each matched
pair. The Hungarian algorithm solves the assignment exactly in cubic time. A one-to-one
matching enforces permutation-invariance and gives each target a unique predictor. Early deep
set predictors used this with recurrent encoder-decoders that emit boxes one at a time,
demonstrated on small data.

**Attention and the Transformer.** A self-attention layer updates every element of a
sequence by aggregating, with learned softmax weights, information from every other element;
it explicitly models all pairwise interactions and is equivariant to input order up to an added
positional encoding. The encoder-decoder Transformer stacks such layers: an encoder of
self-attention over the inputs, a decoder that self-attends over its own outputs and
cross-attends to the encoder. Scaled dot-product attention computes, for query/key/value
projections of the inputs, softmax(QKᵀ/√d_k)·V; multi-head attention runs M such heads of
dimension d_k = d_model/M in parallel and concatenates them. The decoder is originally
autoregressive (one output token at a time); parallel (non-autoregressive) decoding —
emitting all outputs at once — has been developed for audio, translation, and masked language
modeling.

**Box overlap as a loss.** Coordinate regression with an L1 loss treats the four box numbers
independently and its magnitude scales with box size, so large boxes dominate even when their
*relative* error matches a small box's. The natural scale-invariant quantity is IoU itself,
but IoU is zero — and has zero gradient — whenever two boxes do not overlap. The generalized
IoU (GIoU) is defined for disjoint boxes too: with C the smallest axis-aligned box enclosing
both boxes A and B,

  GIoU(A,B) = IoU(A,B) − |C \ (A ∪ B)| / |C|,

which stays defined for disjoint boxes (it grows as the boxes are dragged toward overlap),
lies in [−1, 1], and reduces to IoU whenever the enclosing box is exactly the union,
including the containment case. It is scale-invariant, being a ratio of areas.

## Baselines

**Faster R-CNN (Ren et al., 2015), and its strengthened descendants.** A two-stage detector.
A Region Proposal Network slides over backbone features, and at each location, for each of
several anchors, predicts an objectness score and four box-regression offsets
(t_x, t_y, t_w, t_h) parametrizing a box *relative to that anchor*. High-scoring proposals
are pooled (RoIAlign) into a second-stage head that re-classifies and re-refines them. Targets
are assigned by IoU thresholds; the large positive/negative imbalance is handled by
*subsampling* a fixed ratio of negatives to positives in each minibatch. The pipeline uses
anchors, the IoU assignment rule, and NMS at both proposal and final stage. The line has been
strengthened over years with feature pyramids for multi-scale features, GIoU-augmented box
losses, and longer schedules.

**Single-stage anchor / center detectors.** Predict class and box offset densely over an
anchor grid (with a focal loss for the foreground/background imbalance) or over a grid
of candidate object centers, in one pass. They drop the proposal stage and keep the dense
candidate field, the overlap-based assignment, and NMS.

**Early set-based and learnable-NMS detectors.** Some detectors trained with a permutation-
invariant (bipartite-matching) loss but modeled the relations between predictions with
convolutional or fully-connected layers, alongside a hand-designed NMS. Learnable-NMS and
relation networks model pairwise relations between predictions with attention and can drop the
post-processing, using hand-crafted context features (e.g. proposal-box coordinates) to do so.

**Recurrent set predictors.** Encoder-decoders on CNN features that emit boxes/masks one at a
time with an RNN under a bipartite-matching loss. Autoregressive, evaluated on small datasets.

## Evaluation settings

The natural yardstick is COCO object detection: ~118k training images, ~5k validation,
80 object categories, many objects per image across a wide range of scales. The standard
metric is average precision (AP) averaged over IoU thresholds from 0.50 to 0.95, reported
overall and broken out by object size (AP_S / AP_M / AP_L for small / medium / large) and at
fixed thresholds (AP_50, AP_75). Detectors are also compared on inference cost (FLOPs,
frames per second) and parameter count. Standard practice freezes backbone batch-norm
statistics and initializes the backbone from ImageNet pretraining; the comparison is run
under matched data augmentation and training schedule. Panoptic segmentation (a unified
pixel-level recognition task combining things and stuff) is an adjacent benchmark with its
own PQ metric.

## Code framework

The available primitives are a CNN classification backbone usable as a feature extractor,
ordinary PyTorch module and loss abstractions, optimizers, and routine box utilities
(format conversion, IoU, coordinate scaling). The scaffold below wires these together and
leaves empty the slots that still have to be designed: how image features become a set of
detections, how predictions are assigned to ground-truth objects without assuming an order,
how the matched set is scored, and how normalized boxes are converted back to image
coordinates.

```python
import torch
from torch import nn
import torchvision

def box_cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)

def box_iou(a, b):
    # standard pairwise IoU + union area
    pass  # TODO existing helper

def generalized_box_iou(a, b):
    # TODO: box-overlap comparison to use when coordinates alone are not enough
    raise NotImplementedError


class SetDetector(nn.Module):
    """Image -> a fixed-size set of (class, box) predictions, in one pass."""
    def __init__(self, num_classes, num_slots):
        super().__init__()
        self.num_slots = num_slots
        self.backbone = nn.Sequential(
            *list(torchvision.models.resnet50(pretrained=True).children())[:-2]
        )
        # TODO: turn a CNN feature map into num_slots class logits and boxes
        self.set_predictor = None
        self.class_head = None
        self.box_head = None

    def forward(self, images):
        feats = self.backbone(images)
        # TODO: produce {"pred_logits": ..., "pred_boxes": ...}
        raise NotImplementedError


class AssignmentSolver(nn.Module):
    """Choose which predictions are responsible for which target objects."""
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        # TODO: return, for each image, matched prediction indices and target indices
        raise NotImplementedError


class SetCriterion(nn.Module):
    """Score a set of predictions against a set of ground-truth objects."""
    def __init__(self, num_classes, assignment, weight_dict=None):
        super().__init__()
        self.num_classes = num_classes
        self.assignment = assignment
        self.weight_dict = weight_dict or {}

    def loss_labels(self, outputs, targets, indices):
        # TODO: classification loss for matched objects and unused slots
        raise NotImplementedError

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        # TODO: box loss for the matched pairs
        raise NotImplementedError

    def forward(self, outputs, targets):
        indices = self.assignment(outputs, targets)
        # TODO: combine set assignment, class loss, and box loss
        raise NotImplementedError


class PostProcess(nn.Module):
    """Convert normalized model outputs to per-image detector outputs."""
    def forward(self, outputs, image_sizes):
        # TODO: convert normalized boxes and class logits to scores, labels, boxes
        raise NotImplementedError
```
