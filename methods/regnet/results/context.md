# Context

## Research question

How do you find a good convolutional network architecture in a way that actually teaches you something? Two paradigms dominate, and each leaves a gap. Manual design — the lineage from LeNet through AlexNet, VGG, and ResNet — produced not just strong networks but transferable *design principles*: convolution matters, depth matters, residual connections matter. But manual exploration gets unwieldy as the number of interacting choices (per-stage widths, depths, group widths, bottleneck ratios) explodes. Neural architecture search (NAS) automates the search inside a fixed space and finds excellent models, but its output is a single network *instance* tuned to one setting (one FLOP budget, one piece of hardware). That instance teaches you little: it does not tell you why it is good, and there is no obvious rule for adapting it to a different compute regime.

The precise goal: a paradigm that combines the interpretability of manual design with the efficiency of automated procedures — one that yields not a single network but a *family* of simple, regular networks parametrized by a few scalars, where (a) good models are densely concentrated so a tiny random search finds them, (b) the parametrization is interpretable enough to read off design principles, and (c) those principles generalize across compute regimes, schedule lengths, and block types. A solution would have to provide a small, low-dimensional space that is provably better-populated than the unconstrained one, plus the empirical methodology to demonstrate it.

## Background

The field state rests on a sequence of manually designed ConvNets and a methodology for comparing not models but model *populations*.

On the architecture side: VGG (Simonyan & Zisserman 2015) established stacking small 3×3 convs and doubling channel width when spatial resolution halves. ResNet (He et al. 2016) made deep networks trainable with residual connections F(x)+x and introduced the bottleneck block (1×1 reduce → 3×3 → 1×1 expand). ResNeXt (Xie et al. 2017) replaced the bottleneck's 3×3 with a grouped convolution, parametrized by a group width, trading a "cardinality" dimension against width. MobileNetV2 (Sandler et al. 2018) proposed the inverted residual with a linear bottleneck — expand wide in the middle, contract at the ends — and Xception (Chollet 2017) the extreme of grouping, depthwise convolution. EfficientNet (Tan & Le 2019), found by NAS, scales depth, width, and input resolution jointly with a compound coefficient. Squeeze-and-Excitation (Hu et al. 2018) adds lightweight channel attention: global-pool the feature map, pass through a small two-layer bottleneck MLP with a sigmoid, and rescale each channel.

On the methodology side, the load-bearing prior is Radosavovic et al. (2019), "On Network Design Spaces." Its key idea: a *design space* is a (possibly infinite) population of architectures; rather than comparing the single best model found in two spaces, sample n models from each, train them, and compare the resulting *error distributions*. The summary statistic is the error empirical distribution function (EDF), F(e) = (1/n) Σ_i 1[e_i < e], the fraction of sampled models with error below e; a design space whose EDF is shifted toward lower error is the better space. To read which value of a structural parameter (depth, a width) the best models prefer, an empirical bootstrap over (parameter, error) pairs gives a confidence interval for the best value. This distribution-level view is what makes it possible to talk about a *space* being good rather than a model.

Diagnostic facts about existing systems that frame the work:
- Network *structure* — the per-stage allocation of blocks and channels — determines how compute, parameters, and memory are distributed through the graph, and is the primary driver of accuracy/efficiency for a fixed block type.
- Population statistics are cheap to gather: training 100 models at 400 MFLOPs for 10 epochs costs roughly the same total compute as training one ResNet-50 at 4 GFLOPs for 100 epochs, so distribution-level conclusions are affordable.
- Beyond FLOPs and parameters, *activations* (the summed sizes of all conv output tensors) can correlate more strongly with wall-clock runtime on memory-bound accelerators (GPUs, TPUs) than FLOPs do.
- The complexity of common operators is known in closed form: a 1×1 conv on width w, resolution r costs w²r² flops and w² params; a 3×3 group conv with group width g costs 3²·w·g·r² flops; a 3×3 depthwise conv costs 3²·w·r²; all have wr² activations.

## Baselines

The natural comparison points are the standard manually designed families and the NAS-found state of the art.

**ResNet / ResNeXt.** A stem, four stages of residual bottleneck blocks at decreasing resolution, then a pooled FC head. The bottleneck block is 1×1 (reduce by a ratio) → 3×3 (group) conv → 1×1 (expand), BatchNorm and ReLU after each conv, residual add. Standard practice: double width across stages, use deeper models for higher compute budgets, use a bottleneck ratio > 1. Gap: the per-stage allocation of depth and width is set by hand or convention; there is no principle that says how many blocks each stage should get, and the conventions (deeper-for-bigger, bottleneck > 1) have never been tested at the population level.

**MobileNetV2 inverted bottleneck / depthwise.** Block goes narrow → wide → narrow with a depthwise spatial conv in the wide middle and a linear (activation-free) projection. Effective for mobile efficiency. Gap: whether the inverted bottleneck and depthwise convolution actually help *as design choices* (versus plain grouped bottlenecks) for a population of structure-optimized networks is untested.

**EfficientNet.** A NAS-found base network scaled by a compound rule over depth, width, and resolution. Strong accuracy/FLOPs, but trained with heavy training-time enhancements (advanced augmentation, longer schedules, etc.) that confound architecture comparisons; it is slow on GPUs partly due to high activation counts. Gap: as a single searched instance plus a scaling rule, it provides no interpretable principle and its gains are entangled with its training recipe.

## Evaluation settings

- **Dataset/metric**: ImageNet-1K (Deng et al. 2009), top-1 error on the validation set; generalization checked on ImageNetV2 (Recht et al. 2019).
- **Population-design regime**: sample n=500 models in a 360–400 MFLOP band, train each 10 epochs. SGD with momentum 0.9, batch size 128 on 1 GPU, half-period cosine schedule, initial learning rate 0.05, weight decay 5×10⁻⁵, input resolution 224. Ten epochs suffice for robust *population* statistics.
- **Analysis regime**: fewer models (100) trained longer (25 epochs, lr 0.1) to resolve finer per-parameter trends.
- **Complexity yardsticks**: FLOPs (multiply-adds; MF = 10⁶, GF = 10⁹), parameters, activations, and measured inference time (e.g. 64 images on a V100 GPU).
- **Comparison protocol**: per FLOP regime, pick the best model from a small random sample of the space and retrain a few times at 100 epochs for a robust error estimate, under a single controlled training setup with no training-time enhancements.

## Code framework

The primitives exist: PyTorch convs (with `groups` for grouped/depthwise), BatchNorm, ReLU, adaptive average pool, linear layers. A network is built from a stem, a list of stages (each a sequence of residual bottleneck blocks at one resolution/width), and a pooled classifier head — all driven by *per-stage* lists of widths, depths, strides, bottleneck ratios, and group widths. The open question is how those per-stage lists should be set.

```python
import numpy as np
import torch.nn as nn

def conv2d(w_in, w_out, k, stride=1, groups=1): ...   # padding to keep size
def norm2d(w): ...                                     # BatchNorm2d
def activation(): ...                                  # ReLU

class SE(nn.Module):
    """Optional channel-attention sub-block placed inside a bottleneck."""
    def __init__(self, w_in, w_se):
        super().__init__()
        # TODO: squeeze (global pool) -> reduce -> act -> expand -> gate

class BottleneckTransform(nn.Module):
    """1x1 -> 3x3 group conv [+ optional channel attention] -> 1x1."""
    def __init__(self, w_in, w_out, stride, params):
        super().__init__()
        # bottleneck width, group count, optional attention reduction
        # TODO: build a, 3x3 group conv b, optional attention, c

class ResBottleneckBlock(nn.Module):
    """x + f(x), with a stride/width-change projection on the shortcut."""
    def __init__(self, w_in, w_out, stride, params):
        super().__init__()
        ...

class AnyStage(nn.Module):
    def __init__(self, w_in, w_out, stride, d, block, params):
        super().__init__()
        # d identical blocks; first carries the stride

class AnyNet(nn.Module):
    """Builds stem -> stages -> head from explicit PER-STAGE lists."""
    def __init__(self, stem_w, widths, depths, strides, bot_muls, group_ws,
                 head_w, num_classes, se_r=0):
        super().__init__()
        # stem; one AnyStage per (d, w, s, b, g); pooled FC head

def generate_structure(*args):
    """Produce the per-stage (widths, depths) lists AnyNet consumes."""
    pass  # TODO
```
