## Research question

Object detection on a benchmark like PASCAL VOC asks for more than a single image-level
label: for each image, name every object of interest and draw a tight bounding box around it.
The precise question is whether the dramatic gains that large convolutional networks brought
to whole-image *classification* can be carried over to *detection*, which additionally
demands localization. Two obstacles stand in the way. First, localization: a classifier
ingests one whole image and emits one label, whereas a detector must decide *where* each of
possibly many objects sits. Second, data scarcity: a high-capacity CNN has tens of millions
of parameters and wants millions of labeled examples, but detection datasets with
box-level annotations are small (thousands of images), far too little to train such a network
from scratch without overfitting.

The pain point is concrete. Progress on VOC detection had been slow for several years; the
strongest systems were built on hand-designed gradient-orientation features (SIFT, HOG) and
their accuracy had plateaued, with the leaderboard advancing mainly through ensembling and
minor variants. Meanwhile, a large CNN had just produced a step-change in ImageNet
classification accuracy. A solution would have to bridge these two worlds: turn a
classification CNN into an object detector that localizes accurately, while learning its
high-capacity weights from the limited detection data available.

## Background

**Hand-designed features and their ceiling.** The dominant detection features were blockwise
histograms of gradient orientation — SIFT and HOG — representations one can loosely associate
with early visual cortex. They are shallow and fixed: no learned hierarchy of features above
the first stage. On VOC detection, systems built on them had stagnated. The prevailing
intuition was that richer, *learned*, multi-stage feature hierarchies should be more
informative for recognition than a single hand-designed stage, mirroring the several
processing stages believed to exist downstream in the primate visual pathway.

**The CNN resurgence.** Convolutional networks trained by backpropagation and stochastic
gradient descent (LeCun et al.) had existed since the late 1980s but fell out of favour in the
2000s. In 2012 a large CNN (Krizhevsky et al.), trained on 1.2 million labeled ImageNet
images with rectified-linear nonlinearities and dropout, produced a large jump in ILSVRC
classification accuracy. This reopened a sharp question, vigorously debated at the time: to
what extent do CNN classification results on ImageNet generalize to detection on VOC?

**Localization paradigms on the table.** Three broad strategies for *where* existed.
(1) *Regression* — directly regress object box coordinates from the image; concurrent work
(Szegedy et al.) reported this fared poorly in practice (about 30.5% mAP on VOC2007).
(2) *Sliding window* — slide a classifier densely over positions and scales. CNNs had been
used this way for constrained categories (faces, pedestrians), but only with very shallow
networks (two conv+pool layers) so that spatial resolution stayed high. A five-conv-layer
network has units with large receptive fields (about 195×195 pixels) and large effective
strides (about 32×32 pixels) in the input, which makes precise localization within the
sliding-window paradigm an open technical problem. (3) *Recognition using regions*
(Gu et al.) — generate a manageable set of candidate regions and classify each. This had
proven successful for both detection and semantic segmentation.

**Category-independent region proposals.** A line of work produced a few thousand
class-agnostic candidate windows likely to contain objects, decoupling *where might an object
be* from *what is it*. Selective search (Uijlings et al.) is the canonical example: it begins
with a fast graph-based over-segmentation (Felzenszwalb–Huttenlocher) into small initial
regions, then *greedily merges* the pair of neighbouring regions that are most similar —
combining color-histogram, texture-histogram, size, and fill/shape compatibility into the
similarity — recomputes similarities to the merged region, and repeats until the whole image
is one region. Every region created along the way (and the bounding boxes of diversified runs
over several color spaces and similarity measures) is emitted as a proposal. The result is a
small, data-driven, class-independent set with very high recall (around 99% at ~10k locations);
a "fast" mode yields roughly 2000 proposals per image.

**Transfer learning when labels are scarce.** The standard remedy for too-little task data was
*unsupervised* pre-training followed by supervised fine-tuning. Whether other forms of
pre-training help when a large auxiliary *labeled* dataset is available was, for this setting,
largely unexplored.

**Fixed-input CNNs and warping.** The classification CNN requires a fixed 227×227 input.
A region proposal is an arbitrary rectangle, so it must be transformed to that size. Options
include enclosing it in the tightest square and scaling isotropically (with or without the
surrounding context), or anisotropically scaling (warping) the rectangle to fill the input.
Each can also be padded with a border of surrounding image context; when the source rectangle
runs off the image, the missing pixels can be filled with the image mean.

## Baselines

**Deformable part models (DPM; Felzenszwalb et al.).** The strongest HOG-based detector. An
object is a star-structured model: a coarse *root* HOG filter (in the spirit of Dalal–Triggs)
plus several higher-resolution *part* filters, each attached by a spring-like deformation cost.
The detection score at a location is the root response plus, for each part, the best
part-filter response over nearby displacements minus a quadratic deformation penalty. Part
placements are latent and the whole model is trained with a latent SVM. A v5 release reaches
about 33–34% mAP on VOC. DPM also includes a bounding-box regression stage that predicts a
refined box from the inferred part locations. The gap it leaves: the features are hand-designed
HOG, shallow and fixed, and accuracy has plateaued.

**Spatial-pyramid bag-of-visual-words (UVA; Uijlings et al.).** The most directly comparable
prior system because it consumes the *same* selective-search proposals. Each region is encoded
by a four-level spatial pyramid populated with densely sampled SIFT, Extended OpponentSIFT, and
RGB-SIFT descriptors, each vector-quantized against 4000-word codebooks, producing a roughly
360k-dimensional vector, classified by a histogram-intersection-kernel SVM. It reaches about
35.1% mAP on VOC2010. The gap: an enormous, hand-engineered, encoding-heavy feature
(two orders of magnitude higher-dimensional than a CNN feature would be), and a nonlinear
kernel SVM — accurate relative to DPM but slow and still shallow.

**Regionlets and SegDPM.** Other strong VOC systems (Regionlets ~39.7%, SegDPM ~40.4% with
extra context/segmentation rescoring), all built on hand-designed features. They mark the
state of the leaderboard a richer-feature system would have to clear.

**Sliding-window CNN (OverFeat).** A CNN applied in a multiscale sliding-window fashion for
detection; the best result on the 200-class ILSVRC2013 detection set at the time (about 24.3%
mAP). The gap: it commits to dense sliding-window localization, which is awkward for a deep
network with large receptive fields and strides.

## Evaluation settings

The natural yardsticks are PASCAL VOC detection (VOC 2007 for ablating design choices, VOC
2010–2012 for held-out test through the evaluation server) with 20 object classes, and the
200-class ILSVRC2013 detection dataset. The metric is average precision (AP) per class and
mean AP (mAP) across classes, where a detection counts as correct if its intersection-over-
union (IoU) with a ground-truth box of the right class exceeds 0.5, with duplicate detections
of the same object counted as false positives. PASCAL best practice is to fix all
hyperparameters on VOC2007 and submit to the test server sparingly. ImageNet (ILSVRC2012
classification, 1.2M images, 1000 classes) is available as a large auxiliary dataset with
image-level labels only. A diagnostic tool (Hoiem et al.) categorizes detection errors
(localization vs. confusion vs. background) and is the natural instrument for understanding
failure modes.

## Code framework

The available primitives are: a region-proposal routine that returns a few thousand candidate
boxes per image; a pre-trained convolutional classification network usable as a fixed-input
feature extractor (forward pass to a chosen layer); standard image utilities (crop, resize,
mean subtraction); linear-SVM training with hard-negative mining; ridge regression in closed
form; and a greedy non-maximum-suppression routine. The scaffold wires these primitives
together and leaves the design of the detector itself unfilled.

```python
import numpy as np

def selective_search(image, mode="fast"):
    # existing region-proposal routine: returns ~2000 candidate boxes (x,y,w,h)
    ...
    return boxes

def cnn_forward(batch_227x227, layer="fc7"):
    # existing pre-trained classification CNN, mean-subtracted fixed-size input,
    # forward pass to a chosen layer; returns feature vectors
    ...
    return feats

def nms(boxes, scores, iou_thresh):
    # existing greedy non-maximum suppression
    ...
    return keep_indices


def detect(image, cnn, ...):
    # TODO: using the primitives above, build a detector that turns the
    #       classification CNN into one that localizes objects on this image.
    raise NotImplementedError
```
