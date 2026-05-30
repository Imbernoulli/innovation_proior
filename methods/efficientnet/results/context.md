## Research question

Image classifiers based on convolutional networks keep getting more accurate by getting bigger.
The 2012 winner (AlexNet) had a few tens of millions of parameters; by 2017 the strongest single
model (SENet) reached 82.7% ImageNet top-1 with ~145M parameters; by 2018 a giant model trained
with pipeline parallelism across many accelerators (GPipe) reached 84.3% with ~557M parameters at
480×480 input. The pattern is clear: when a larger compute budget is available, people "scale up"
an existing network for better accuracy.

What is *not* clear is how to scale. A convolutional network has three obvious knobs that all cost
compute and all buy accuracy: how deep it is (number of layers), how wide it is (number of channels
per layer), and how large the input image is (resolution). Common practice fixes a network at one
budget and then, given more resources, enlarges *one* of these dimensions — usually depth, sometimes
width, increasingly resolution. Scaling two or three together is possible but is done by hand, which
is tedious and tends to land on configurations that are good neither for accuracy nor for FLOPS.

The precise problem: given a baseline network and a target resource budget (FLOPS and memory), find
a *principled, reusable* rule for distributing that budget across depth, width, and resolution that
beats single-dimension scaling — and whose cost of being derived does not blow up as the target
budget grows. A good solution must (a) say concretely how many extra layers, channels, and pixels to
add for a given budget, (b) make the budget knob mean something consistent, and (c) be cheap enough
to apply to a whole family of model sizes.

## Background

A convolutional network can be written as a composition of layers grouped into stages, where every
layer in a stage shares the same operator and only the first layer of a stage changes the spatial
size: a network is a list of stages, each stage i defined by an operator F_i repeated L_i times,
operating on a tensor of shape ⟨H_i, W_i, C_i⟩. As one moves through the network the spatial size
H_i×W_i shrinks (pooling/strided convs) while the channel count C_i grows. The three scalable
quantities are exactly the length L_i (depth), the channel count C_i (width), and the input spatial
size H_i, W_i (resolution).

Three pieces of accumulated wisdom describe what each knob buys, and where each saturates:

- **Depth.** Stacking more layers is the most common way to scale (it is how an 18-layer residual
  net becomes a 200-layer one). Deeper networks capture richer, more hierarchical features and tend
  to transfer well. But depth is hard to train — gradients vanish through long stacks — and although
  skip connections and batch normalization largely fix trainability, the *accuracy* return on extra
  depth diminishes sharply: a ~1000-layer residual net is no more accurate than a ~100-layer one.

- **Width.** Increasing the number of channels is the usual lever for small models (the "depth
  multiplier" of mobile networks scales channels). Wider layers capture finer-grained features and
  are easier to optimize than very deep ones. But a network that is very wide and shallow struggles
  to form high-level abstractions, and accuracy saturates quickly as width alone is pushed up.

- **Resolution.** Feeding larger images lets the network see finer patterns. Input size has crept up
  over the years — from 224×224 in early networks to 299×299, 331×331, and 480×480 in the largest
  recent models, with 600×600 common in detection. Again, accuracy rises with resolution but the
  gain tapers off at very high resolutions.

The diagnostic that frames everything: scaling up *any single* one of these dimensions improves
accuracy, but for each of them the curve flattens — accuracy gain diminishes as the model gets
bigger, with each single-dimension curve plateauing around the low 80s in top-1. So single-dimension
scaling has a built-in ceiling.

A second, sharper diagnostic concerns the *interaction* between dimensions. If width is scaled while
depth and resolution are held at their baseline values, accuracy saturates early. But if the same
width scaling is applied on top of a network that is *already* deeper and run at *higher* resolution,
width scaling keeps paying off and reaches markedly higher accuracy at the *same* FLOPS. In other
words the three dimensions are not independent: a higher-resolution image contains more pixels per
object, which calls for a larger receptive field (more depth) and more channels (more width) to use
those pixels. Prior theoretical and empirical work had already noted that depth and width are coupled
and both matter for expressive power; the open piece was a quantitative relationship spanning all
three of depth, width, and resolution at once.

Two cheap, near-free architectural primitives underpin efficient modern networks and are part of the
landscape:

- **Depthwise separable convolution.** A standard k×k convolution from M input channels to N output
  channels over a D_F×D_F map costs about k²·M·N·D_F². Factor it into a *depthwise* convolution (one
  k×k filter per input channel, no channel mixing: k²·M·D_F²) followed by a *pointwise* 1×1
  convolution that mixes channels (M·N·D_F²). The total is k²·M·D_F² + M·N·D_F², which is the
  original cost times (1/N + 1/k²) — about an 8–9× reduction for 3×3 kernels — at a small accuracy
  cost. This is the workhorse of mobile-size networks.

- **Channel attention (squeeze-and-excitation).** A feature map's channels are not equally useful for
  a given input. A squeeze-and-excitation unit collapses each channel's spatial map to a single
  number by global average pooling (squeeze), passes that channel descriptor through a small
  two-layer bottleneck (a fully connected layer that reduces the channel count by a ratio, then one
  that restores it) ending in a sigmoid, and uses the result as a per-channel multiplicative gate
  (excite) on the original feature map. It re-weights channels by global context for negligible
  compute.

## Baselines

The prior methods a new scaling rule would be measured against, or would build on:

- **Residual networks (He et al., 2016).** The canonical depth-scalable family. A block computes
  x + F(x) so that identity is the default and gradients reach early layers undiminished, making very
  deep networks trainable. Residual nets are scaled by adding layers (e.g. from 18 to 200). They are
  the de-facto depth-scaling baseline. Limitation here: depth-only scaling, and its accuracy return
  saturates (a thousand-layer variant ≈ a hundred-layer one).

- **Wide residual networks (Zagoruyko & Komodakis, 2016).** Argue that one can trade depth for width:
  a shallower-but-much-wider residual net matches or beats a very deep thin one and trains faster.
  The canonical width-scaling baseline. Limitation: width-only; very wide shallow nets miss
  high-level features, and accuracy saturates with width.

- **Depthwise separable / mobile networks v1 (Howard et al., 2017; Xception, Chollet, 2017).** Build
  whole networks out of depthwise separable convolutions, with a global "width multiplier" and a
  "resolution multiplier" to trade accuracy for size. The standard way to make small efficient
  networks scalable. Limitation: scaling is per-single-dimension multipliers, hand-set; no rule for
  jointly balancing depth, width, resolution.

- **Inverted-residual mobile networks v2 (Sandler et al., 2018).** Refine the mobile block into an
  *inverted residual with a linear bottleneck*. Each block keeps a thin channel representation at its
  endpoints, *expands* it with a 1×1 convolution (typically 6×) into a high-dimensional space, applies
  a 3×3 depthwise convolution there, then *projects* back down with a 1×1 convolution that is left
  **linear** — no activation — because a ReLU applied in the narrow bottleneck destroys information
  that cannot be recovered. The residual skip connects the thin bottleneck endpoints, which is cheap
  to add and keeps the large expanded tensor from ever having to be carried across blocks (a memory
  win). This block is the standard efficient building unit. Limitation: still scaled by a single
  width/resolution multiplier.

- **Squeeze-and-excitation networks (Hu et al., 2018).** Insert the channel-attention unit above into
  a strong backbone. Reached 82.7% ImageNet top-1, the best single model of its time, but at ~145M
  parameters. Limitation: extremely large; great accuracy per nothing-much-compute of the SE unit
  itself, but the host network is heavy.

- **Platform-aware architecture search (Tan et al., 2018).** Instead of hand-designing the mobile
  block, search for it. A reinforcement-learning controller searches a factorized hierarchical space
  (each stage may pick its own kernel size, expansion ratio, whether to include squeeze-and-excitation,
  whether to add a skip) under a multi-objective reward that multiplies accuracy by a soft penalty on
  measured latency, ACC(m)·(LAT(m)/T)^w. Produces strong mobile networks built from inverted-residual
  blocks. This search machinery is available to *design a baseline* rather than to scale one;
  measured latency can be swapped for FLOPS when no specific device is targeted. Limitation: search is
  affordable at mobile size but its cost explodes for large models, so it cannot itself answer the
  scaling question.

- **Pipeline-parallel giant model (Huang et al., 2018).** Pushes accuracy by brute-force scaling — a
  much larger network at 480×480 resolution, trainable only by partitioning across accelerators with
  a specialized pipeline-parallelism library. Reaches 84.3% top-1 with ~557M parameters. This is the
  state of the art that any efficient scaling method must compete with on accuracy while using far
  fewer resources. Limitation: enormous and unwieldy; scaling done ad hoc (mostly resolution + size).

## Evaluation settings

- **Primary benchmark.** ImageNet (ILSVRC-2012): ~1.28M training images, 50k validation images,
  1000 classes. Report single-crop, single-model top-1 and top-5 accuracy. A common protocol holds
  out a small subset of the training set (e.g. ~25k images, a "minival") for early stopping and then
  evaluates the early-stopped checkpoint on the original validation set. Test-set numbers, when
  reported, come from the held-out 100k test images submitted to the evaluation server.

- **Transfer benchmarks.** Take an ImageNet-pretrained network and fine-tune on a suite of smaller
  classification datasets: CIFAR-10 and CIFAR-100, Birdsnap, Stanford Cars, Flowers, FGVC Aircraft,
  Oxford-IIIT Pets, Food-101. These span coarse and fine-grained recognition with train sizes from a
  couple thousand to tens of thousands of images.

- **Yardsticks.** Accuracy is plotted against model size (number of parameters) and against compute
  (FLOPS); inference latency is measured on a real CPU (batch size 1, single core) to confirm the
  FLOPS picture matches wall-clock speed.

- **Training recipe available at the time.** RMSProp optimizer (decay and momentum ~0.9), batch-norm
  momentum 0.99, weight decay 1e-5, a learning rate that warms to ~0.25 and decays geometrically;
  the smooth activation x·σ(x); AutoAugment data augmentation; stochastic depth (drop whole residual
  branches with some survival probability); dropout before the classifier, increased for larger
  models.

## Code framework

Cheap convolution primitives, stochastic-depth regularization, a generic residual building-block
slot, a stage-structured body, a classifier head, and a training loop are enough to hold the work.
The open pieces are the block internals, the baseline stage table, and the budget-to-shape rule that
returns concrete depth, width, resolution, and regularization settings.

```python
import torch
import math
import torch.nn as nn
import torch.nn.functional as F


BN_MOMENTUM = 0.01  # PyTorch value corresponding to TensorFlow batch-norm momentum 0.99
BN_EPS = 1e-3


class Conv2dSamePadding(nn.Conv2d):
    """Convolution primitive with TensorFlow-style SAME padding."""
    def forward(self, x):
        ih, iw = x.shape[-2:]
        kh, kw = self.weight.shape[-2:]
        sh, sw = self.stride
        dh, dw = self.dilation
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * sh + (kh - 1) * dh + 1 - ih, 0)
        pad_w = max((ow - 1) * sw + (kw - 1) * dw + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2,
                          pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def conv_bn_act(in_ch, out_ch, kernel_size, stride, groups=1, act=True):
    """Standard conv -> batchnorm -> (optional) activation primitive.
       groups == in_ch == out_ch gives a depthwise convolution."""
    layers = [
        Conv2dSamePadding(in_ch, out_ch, kernel_size, stride,
                          groups=groups, bias=False),
        nn.BatchNorm2d(out_ch, momentum=BN_MOMENTUM, eps=BN_EPS),
    ]
    if act:
        layers.append(nn.SiLU())  # x * sigmoid(x); the smooth activation in use
    return nn.Sequential(*layers)


def drop_connect(x, drop_rate, training):
    if not training or not drop_rate:
        return x
    keep = 1.0 - drop_rate
    mask = keep + torch.rand([x.shape[0], 1, 1, 1],
                             dtype=x.dtype, device=x.device)
    mask = torch.floor(mask)
    return x / keep * mask


class Block(nn.Module):
    """The per-stage building block. Its internal structure is exactly what
       this design has to decide; for now it is an empty slot."""
    def __init__(self, in_ch, out_ch, kernel_size, stride, **kwargs):
        super().__init__()
        # TODO: the building block we will design (how to spend channels and
        #       spatial filtering cheaply, what to gate, where to skip).
        raise NotImplementedError

    def forward(self, x, drop_rate=None):
        # TODO
        raise NotImplementedError


# TODO: the baseline body. A list of stages (operator, #channels, #layers,
#       kernel, stride, input resolution) that fixes the per-layer operator and
#       leaves only L_i, C_i, H_i, W_i to be scaled. To be decided.
BASELINE_STAGES = None


def scale_width(channels, width_coeff):
    # TODO: map a baseline channel count to a scaled one for a given budget.
    raise NotImplementedError


def scale_depth(num_layers, depth_coeff):
    # TODO: map a baseline layer-repeat count to a scaled one for a given budget.
    raise NotImplementedError


def resolve_budget(budget_multiplier):
    """Turn an available compute multiplier into width, depth, resolution,
       dropout, and drop-connect settings."""
    # TODO: the budget-to-shape rule.
    raise NotImplementedError


class ConvNet(nn.Module):
    """Generic stage-structured classifier: stem -> stages of Blocks -> head."""
    def __init__(self, stages, width_coeff, depth_coeff, dropout,
                 drop_connect_rate, num_classes=1000):
        super().__init__()
        self.drop_connect_rate = drop_connect_rate

        # stem
        stem_ch = scale_width(32, width_coeff)
        self.stem = conv_bn_act(3, stem_ch, kernel_size=3, stride=2)

        # body: build each stage from the (to-be-designed) Block, with the
        # channel counts and layer-repeats adjusted for the budget.
        self.blocks = nn.ModuleList()
        in_ch = stem_ch
        # TODO: iterate BASELINE_STAGES, apply scale_width / scale_depth, and
        #       append Blocks while updating in_ch. Depends on the block and table.

        # head + classifier
        head_ch = scale_width(1280, width_coeff)
        self.head = conv_bn_act(in_ch, head_ch, kernel_size=1, stride=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(head_ch, num_classes)

    def forward(self, x):
        x = self.stem(x)
        for idx, b in enumerate(self.blocks):
            drop_rate = self.drop_connect_rate * float(idx) / len(self.blocks)
            x = b(x, drop_rate)
        x = self.head(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.dropout(x))


# Training loop already exists in pre-method form.
def train_step(model, images, labels, optimizer):
    optimizer.zero_grad()
    logits = model(images)              # images at the chosen input resolution
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    return loss.item()
```
