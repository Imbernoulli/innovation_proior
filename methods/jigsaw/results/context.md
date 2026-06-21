# Context

## Research question

Supervised pretraining on ImageNet gives convolutional features that transfer to detection, segmentation, and other tasks — but it depends on a million hand-drawn class labels. The question is whether comparably transferable features can be learned from *unlabeled* images alone, by designing a pretext task whose supervision is generated for free from the image itself. The task should be solvable by understanding the *content and spatial structure* of objects, and it should work from *single still images*, not require video or extra sensors.

## Background

**Self-supervised learning.** A recent strategy that exploits labels available "for free" within the data. One distinguishes labels tied to a non-visual signal (egomotion, audio) from labels derived purely from the structure of the data (the spatial arrangement of pixels). The latter is appealing because it needs nothing but the image.

**Shortcuts (diagnostic facts about pretext tasks).** Several incidental cues let a network solve spatial-arrangement pretext tasks without learning semantic features:
- *Low-level statistics:* adjacent image regions share similar pixel mean and standard deviation, so a network can match neighbors by these statistics alone.
- *Edge continuity / texture across boundaries:* contiguous patches have continuous edges and matching texture at their shared border.
- *Chromatic aberration:* lenses produce a small relative spatial shift between color channels that grows from image center to border; a network can read off a patch's original position from this shift.
- *Absolute position:* if the task is set up so a patch's appearance maps to a fixed position, the network learns "this appearance lives here," and the features encode 2D location rather than object semantics.

**Transfer via pre-train + fine-tune.** The established way to use self-supervised features: pretrain on the pretext task, then copy the convolutional weights into a standard network and fine-tune on the target task. Yosinski et al. showed early conv layers are general-purpose while later layers are task-specific, so where one extracts features matters.

**Part-based models of objects.** Classic vision modeled objects as constellations of parts with an appearance term and a configuration (geometry) term. A task that requires reasoning about which part goes where would, in principle, learn both part appearance and part configuration.

## Baselines

**Doersch et al. 2015 (context prediction).** Single-image pretext: take a center tile on a 3×3 grid and a second tile in one of the 8 neighboring positions; train a network to classify the relative position of the second tile.

**Wang & Gupta 2015 (tracking).** Mine triplets by tracking patches across video frames; two patches of the same tracked object are pulled together, a third pushed apart, learning a patch-similarity metric.

**Agrawal et al. 2015 (egomotion).** A siamese network predicts the egomotion between two frames, supervised by odometry sensors.

**AlexNet (Krizhevsky et al. 2012).** The standard supervised CNN (conv1–conv5 plus fully connected layers, 61M parameters) and the architecture into which any pretext-learned convolutional weights would be transferred and against whose supervised features they would be measured.

## Evaluation settings

- **Pretraining data:** 1.3M ImageNet training images, used *without* labels; images resized to 256×256.
- **Transfer benchmarks (pretrain + fine-tune):** PASCAL VOC 2007 classification (Krähenbühl et al. framework), VOC 2007 object detection (Fast R-CNN), VOC 2012 semantic segmentation (FCN). Convolutional weights are copied into a standard AlexNet (stride 4 first layer); the new fully connected layers are randomly initialized.
- **ImageNet 2012 classification with layer-locking:** lock conv1…conv-k, randomly initialize and retrain the rest, to locate where features transition from general to task-specific; accuracy averaged over 10 crops.
- **Image retrieval:** nearest neighbors of pool5 features (VOC 2007 test boxes as queries, trainval boxes as gallery), ranked by inner product of normalized features.
- **Optimization:** SGD without batch normalization, batch size 256, base learning rate 0.01.
- **Baselines for comparison:** supervised AlexNet, Doersch et al., Wang & Gupta, Pathak et al. (context encoder), random initialization.

## Code framework

The primitives that already exist: the AlexNet convolutional stack and fully connected layers; SGD with a step learning-rate schedule; standard image cropping/resizing; the pre-train-then-fine-tune transfer recipe. The slots below are what a single-image spatial-arrangement pretext task would fill in.

```python
import numpy as np
import torch
import torch.nn as nn

# --- per-tile feature extractor: AlexNet conv stack (exists) ---
def alexnet_conv_stack():
    # conv1..conv5 with ReLU / maxpool / LRN, as in AlexNet
    ...

# --- turning one image into a training example ---
def make_pretext_example(image):
    # TODO: how is the free supervisory label generated from the image?
    #       (what is cropped, how is it perturbed, what is the target label?)
    pass

def build_label_space():
    # TODO: the set of possible labels for the pretext task -- and how it is chosen
    pass

# --- the network that consumes the pretext input and predicts its label ---
class PretextNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = alexnet_conv_stack()       # transferable part
        # TODO: how are the (possibly multiple) inputs processed, and what
        #       is the classifier head?

    def forward(self, x):
        # TODO: produce a distribution over the pretext label space
        pass

# --- training and transfer ---
def train_pretext(net, loader, opt):
    # standard classification training on the free labels
    ...

def transfer_to_alexnet(pretext_net):
    # copy conv weights into a standard AlexNet, randomly init the rest, fine-tune
    ...
```
