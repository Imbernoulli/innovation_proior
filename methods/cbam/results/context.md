# Context

## Research question

Improvements to convolutional networks have mostly come from three architectural levers: depth (more layers), width (more channels per layer), and cardinality (grouped/parallel branches). These all scale capacity, but they treat every channel and every spatial location of a feature map as equally worth computing on. A convolution blends information across channels and across space simultaneously, and not all of that information is useful for a given input — some channels respond to irrelevant patterns, some spatial regions are background clutter.

The question this raises: can a network learn, adaptively per input, *which* parts of a feature map to emphasize and which to suppress — and do so with a module so lightweight it can be dropped into *every* convolutional block of an existing architecture with negligible extra parameters and compute, trained end-to-end, without redesigning the backbone? A good solution would be a small, general "feature refinement" unit that takes an intermediate feature map and returns a re-weighted one of the same shape, improving accuracy as a plug-in.

## Background

The relevant state of the field is network-engineering along depth/width/cardinality, plus an emerging line of attention modules.

**Capacity-scaling backbones.** ResNet (He et al. 2016) stacks identity-shortcut residual blocks to reach great depth; WideResNet (Zagoruyko & Komodakis 2016) widens instead, showing a shallow-but-wide net can beat a very deep one; ResNeXt (Xie et al. 2016) and Xception (Chollet 2016) add cardinality via grouped convolutions; DenseNet (Huang et al. 2016) concatenates all previous features. All increase representation power, none introduce input-adaptive feature selection.

**Attention.** In human perception, attention selectively focuses on salient parts rather than processing a whole scene uniformly. Several works brought this into CNNs. Residual Attention Network (Wang et al. 2017) computes a full 3-D attention mask with an encoder-decoder branch and multiplies it into the features; effective but heavy, since it produces the whole H×W×C mask directly. Squeeze-and-Excitation (Hu et al. 2017) is the compact, influential predecessor: it models inter-channel relationships by *squeezing* each channel's spatial extent into a single number via global average pooling, passing the resulting C-vector through a small bottleneck MLP (reduce to C/r, ReLU, expand back to C), applying a sigmoid, and rescaling each channel — a pure channel-attention "excitation."

Two diagnostic facts about these existing systems frame the work:
- SE uses only global *average* pooling to summarize a channel. Average pooling gives a smooth, dense summary in which every spatial position contributes equally; it is a known property that this kind of summary smooths over strong localized responses.
- SE operates only on channels — it has no spatial component, so it re-weights which channels respond but not which locations within the map. Spatial focus is known to matter (e.g. SCA-CNN, Chen et al. 2016).
- Pooling *along the channel axis* (collapsing C into 1 or 2 maps) is known to highlight informative spatial regions effectively (Zagoruyko & Komodakis 2017, attention transfer). Each channel of a feature map can be viewed as a feature detector (Zeiler & Fergus 2014).
- Class Activation Mapping (Zhou et al. 2016) showed global average pooling helps localize the spatial extent of objects.

## Baselines

**Squeeze-and-Excitation (SE).** Channel attention only: M_c(F) = σ(W_1·ReLU(W_0·GAP(F))), with W_0 ∈ R^{C/r×C}, W_1 ∈ R^{C×C/r}, applied as a per-channel multiply. Lightweight, plug-in, and a strong accuracy boost. Gaps: (1) summarizes each channel with average pooling alone; (2) has no spatial component, so it cannot suppress background regions.

**Residual Attention Network.** Computes a 3-D soft attention mask via an encoder-decoder (bottom-up/top-down) branch and multiplies it into the trunk features, with a residual formulation for trainability. Gap: directly producing the full C×H×W mask costs substantial parameters and computation, so it is not a cheap drop-in for arbitrary backbones.

**Plain capacity scaling (depth/width/cardinality).** ResNet/WideResNet/ResNeXt. Gap: increases raw capacity but applies it uniformly across channels and locations; no input-adaptive emphasis or suppression of features.

## Evaluation settings

- **Image classification**: ImageNet-1K (Deng et al. 2009), 1.2M train / 50K val, 1000 classes; report top-1/top-5 error, single 224×224 center crop. Backbones: ResNet, WideResNet, ResNeXt, MobileNet, with the same data-augmentation as ResNet; SGD, learning rate 0.1 dropped every 30 epochs, 90 epochs.
- **Object detection**: MS COCO and PASCAL VOC 2007, attaching the module to detector backbones.
- **Visualization**: Grad-CAM (Selvaraju et al. 2017) to inspect where the network attends.
- **Ablation knobs**: reduction ratio r fixed to 16; the module's internal design choices (how statistics are pooled, how any sub-modules are arranged) and any conv kernel sizes are swept empirically.

## Code framework

The primitives exist: `nn.Conv2d`, `nn.Linear`, `nn.BatchNorm2d`, `nn.ReLU`, sigmoid, and spatial pooling. A "feature refinement" module is a unit that takes a feature map F ∈ R^{C×H×W} and returns a re-weighted map of the same shape, to be inserted inside a convolutional block (e.g. on the residual branch before the skip add). The internal mechanism — what statistics to gather, along which axes, and how to turn them into gates — is the open question.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    """Produce an attention map from F and multiply it back in.
    Which axis it acts on and how it pools/projects is the open design question."""
    def __init__(self, channels):
        super().__init__()
        # TODO: the pooling + projection that yields an attention gate
        pass

    def forward(self, x):
        # TODO: gate = sigmoid(some function of pooled stats of x); return x * gate
        return x

class FeatureRefinement(nn.Module):
    """Drop-in module: refine an intermediate feature map, same shape in and out."""
    def __init__(self, channels):
        super().__init__()
        # TODO: which attention sub-module(s), and in what arrangement
        pass

    def forward(self, x):
        # TODO
        return x

# Inserted inside a residual block, e.g.:
#   out = conv_layers(x)
#   out = FeatureRefinement(C)(out)   # refine before the residual add
#   out = out + shortcut(x)
```
