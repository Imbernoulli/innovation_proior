## Research question

Object detection on PASCAL VOC asks for more than assigning one image-level label. A detector must name every object of interest in an image and return a tight bounding box for each one. The immediate question is whether the new gains from large convolutional networks in whole-image classification can transfer to this harder setting, where localization and recognition have to happen together.

A classifier can ignore object position once it has enough evidence for a label, while a detector is judged by whether its box overlaps the right object by at least 0.5 intersection-over-union. The high-capacity CNN that changed ImageNet classification has tens of millions of parameters, but detection datasets with box labels are small.

Progress on PASCAL VOC had been steady but incremental for several years. Strong systems relied on hand-designed gradient-orientation features and gained through ensembles, context rescoring, and minor variants.

## Background

Hand-designed visual features dominated detection. SIFT and HOG summarize local gradient orientations in blocks, giving useful but shallow representations. Deformable models and spatial pyramids build on these features, but the underlying visual code is fixed by the designer rather than learned through multiple stages.

Large supervised CNNs reopened the feature question. A network trained on ImageNet classification, with five convolutional layers, high-capacity fully connected layers, rectified nonlinearities, dropout, and millions of labeled images, produced a step change in classification. What remains open is how to reuse that hierarchy when the target task has few box labels and requires precise location.

The localization choices are constrained. Directly regressing object boxes from the image is conceptually simple, and concurrent deep-network regression results exist for this setting. A dense sliding-window classifier is a familiar paradigm; a deep CNN's high-level units have large receptive fields and coarse effective stride. A third paradigm already exists: generate a modest set of category-independent candidate regions, then classify each region.

Category-independent region proposal methods make that third option plausible. Selective search begins with graph-based oversegmentation, greedily merges neighboring regions using color, texture, size, and fill compatibility, diversifies over color spaces and similarity choices, and emits bounding boxes from the resulting hierarchy. Its fast mode gives roughly a few thousand windows per image, far fewer than exhaustive sliding windows while retaining high recall.

## Baselines

Deformable part models are the strongest HOG-based reference point. A model has a coarse root filter, higher-resolution part filters, and quadratic deformation costs that let parts move around their anchors. The best part placements are latent, and training uses a latent SVM with hard-negative mining. DPM also refines boxes from inferred part geometry.

The most direct region-based comparison is a selective-search spatial-pyramid bag-of-visual-words detector. It uses the same kind of candidate boxes, then encodes each region with dense SIFT variants, multiple codebooks, a multi-level spatial pyramid, and a histogram-intersection-kernel SVM.

Other strong VOC systems add richer context, segmentation, or regionlets on top of hand-designed features. They establish the leaderboard level a learned hierarchy must beat without relying on an ensemble of unrelated cues. On the ImageNet detection side, a sliding-window CNN detector is a natural comparison because it uses CNN features with a different localization strategy.

## Evaluation settings

The central benchmark is PASCAL VOC object detection, especially VOC 2007 for validation and design ablations and VOC 2010-2012 for held-out test submissions. The metric is average precision per class and mean AP across classes. A detection is counted correct only when it has the right class and IoU at least 0.5 with an unmatched ground-truth box; duplicate detections become false positives.

ImageNet classification data is available as a large auxiliary source with image-level labels but no detection boxes. That makes it useful for learning visual features, not directly for supervised box localization. A separate 200-class ImageNet detection benchmark gives a broader-scale test once the design choices are fixed.

Useful diagnostics include error-type analysis that separates poor localization from confusion with similar classes, confusion with dissimilar classes, and background false positives. That diagnostic matters because a detector can have excellent recognition features and still lose AP through boxes that are close but not tight enough.

## Code framework

The scaffold already supplies the pieces one could reasonably have before the missing detector design: a category-independent proposal routine, a fixed-input CNN feature extractor, image crop/resize and mean-subtraction utilities, linear classifiers with hard-negative mining, ridge regression, IoU computations, and greedy non-maximum suppression.

```python
import numpy as np

def selective_search(image, mode="fast"):
    # Returns candidate boxes, independent of object category.
    ...
    return boxes

def cnn_forward(batch_227x227, layer="fc7"):
    # Forward pass through a pretrained fixed-input classification CNN.
    ...
    return features

def nms(boxes, scores, iou_thresh):
    # Greedy per-class non-maximum suppression.
    ...
    return keep_indices

def detect(image, cnn, ...):
    # TODO: build a detector that uses the available proposals, CNN,
    # classifiers, and box-refinement primitives.
    raise NotImplementedError
```
