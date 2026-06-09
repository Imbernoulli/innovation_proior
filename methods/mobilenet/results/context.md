# Context: efficient convolutional networks for on-device vision

## Research question

Convolutional networks have become the default tool for image recognition, and the dominant way to push accuracy higher has been to make them deeper and wider. That trend produces excellent benchmark numbers but expensive models: the largest classification networks of the time require on the order of tens of billions of multiply-adds and tens to hundreds of millions of parameters per forward pass. Many real deployments — phones, embedded cameras, robots, self-driving cars, augmented-reality headsets — must run recognition in real time on a device with a tight compute, memory, and power budget, often with no server round-trip.

The question is therefore: **can we design a convolutional architecture that is small and, more importantly, *fast* (low latency) on commodity mobile hardware, while giving up only a small amount of accuracy — and can we do it in a way that lets a developer dial the model down to whatever resource budget they actually have?** Two things make this hard. First, parameter count and *actual inference cost* are not the same thing; a network can be small in bytes yet slow, or fast yet large. Second, the cost of a standard convolution is multiplicative in several independent factors (kernel size, input channels, output channels, spatial resolution), so shrinking the network naively along any one axis sacrifices a lot of representational power.

## Background

A standard convolutional layer maps an input feature map of size `D_F × D_F × M` (spatial size `D_F`, `M` input channels) to an output of size `D_F × D_F × N` (with stride 1 and same padding), using a kernel of shape `D_K × D_K × M × N`. Its output is

```
G[k,l,n] = Σ_{i,j,m} K[i,j,m,n] · F[k+i-1, l+j-1, m].
```

The number of multiply-adds is

```
D_K · D_K · M · N · D_F · D_F.
```

The structurally important fact is that this cost is *multiplicative* in four independent quantities: the spatial kernel size `D_K²`, the number of input channels `M`, the number of output channels `N`, and the spatial map size `D_F²`. Every factor scales the others, so there is no single term to shrink in isolation.

Several lines of work in the surrounding literature attack pieces of this cost. *Factorized convolutions* (in the Inception line, Szegedy et al.) replace an `n × n` spatial convolution with a `n × 1` followed by a `1 × n` convolution, and lean heavily on `1 × 1` convolutions as cheap channel-mixing / bottleneck layers. *Depthwise separable convolution* originates in Sifre's rigid-motion scattering work (2014) and was used in the first layers of Inception models to cut early-layer compute. *Flattened networks* (Jin et al., 2014) take factorization to the extreme, building a network out of fully separable rank-1 (1D) filters, demonstrating that very aggressive factorization can still learn. *Factorized networks* (Wang et al., 2016) introduce a similar factorized convolution together with topological connections. Concurrently, the *Xception* architecture (Chollet, 2016) scales depthwise separable filters up across an entire large network. A separate diagnostic fact about hardware also matters: a `1 × 1` convolution is exactly a dense matrix multiply (GEMM) over the reshaped activations and needs no `im2col` memory reordering, whereas a general `k × k` convolution is typically mapped to GEMM only after an `im2col` copy (as in Caffe); unstructured-sparse matrix operations are usually *not* faster than dense ones until very high sparsity. So where the compute sits — in dense `1 × 1` convs versus general convs — directly affects realized latency, not just the theoretical multiply-add count.

Batch normalization (Ioffe & Szegedy, 2015) — normalizing each channel's pre-activations using minibatch statistics, with learned scale and shift — was by then standard between convolution and nonlinearity, and made deep convolutional stacks trainable and well-conditioned. ReLU was the default nonlinearity.

## Baselines

- **VGG (Simonyan & Zisserman, 2014).** A deep stack of `3 × 3` standard convolutions. Very accurate but enormous: on the order of 15 billion multiply-adds and ~138M parameters at `224 × 224`. Core idea: depth from small kernels. Gap: far too heavy for a phone, dominated by full convolutions whose cost multiplies kernel size by both channel counts.

- **GoogLeNet / Inception (Szegedy et al.).** Inception modules combine parallel branches of `1 × 1`, `3 × 3`, `5 × 5` convolutions and pooling, with `1 × 1` bottlenecks to control channel count, and later factorized spatial convolutions (`n × 1` then `1 × n`). More efficient than VGG (~1.5B mult-adds, ~6.8M params) but the module topology is intricate and hand-tuned, and the design knobs are not a single clean dial for trading accuracy against latency.

- **SqueezeNet (Iandola et al., 2016).** "Fire" modules squeeze channels with `1 × 1` convolutions before expanding, reaching AlexNet-level accuracy with ~50× fewer parameters and under 1MB. Core idea: aggressive `1 × 1` bottlenecking for tiny *parameter count*. Gap: optimizes model size, not latency — its multiply-add count remains high, so it is small on disk but not necessarily fast.

- **AlexNet (Krizhevsky et al., 2012).** The network that started the deep-CNN era; large fully-connected layers dominate its ~60M parameters. A natural low-accuracy reference point for "small/fast" comparisons.

- **Compression of pretrained networks** — pruning + quantization + Huffman coding (Han et al.), hashing (Chen et al.), product quantization (Wu et al.), low-rank/tensor factorization of trained filters (Jaderberg et al.; Lebedev et al.), and knowledge distillation (Hinton et al., teaching a small student from a large teacher). Core idea: start from a trained large model and make it smaller/faster after the fact. Gap relative to the goal here: these post-hoc methods do not, by themselves, give a developer a clean family of architectures parameterized to hit an arbitrary compute budget *from scratch*; distillation in particular is complementary rather than competing.

## Evaluation settings

- **ImageNet (ILSVRC) classification** at `224 × 224` input — 1000 classes, top-1 accuracy the headline metric. Also reduced input resolutions (`192`, `160`, `128`) as a compute knob.
- **Cost metrics:** multiply-adds (Mult-Adds) per forward pass and parameter count, reported in millions, alongside accuracy — because the whole point is the accuracy-vs-cost trade-off, not accuracy alone.
- **Transfer / downstream tasks** used as additional yardsticks for a backbone: fine-grained classification (Stanford Dogs), large-scale geo-localization (PlaNet-style cell classification), face-attribute classification (with distillation), face embeddings (FaceNet-style triplet embeddings, via distillation), and object detection on COCO under Faster-RCNN and SSD frameworks (mAP at IoU 0.50:0.05:0.95).
- **Training stack:** TensorFlow with RMSProp and asynchronous gradient descent, in the style used for Inception V3.

## Code framework

The pieces below already exist before the architecture is designed: a conv-BN-ReLU primitive, a generic network skeleton that stacks blocks then classifies, and a standard training loop. The one empty slot is the block design — the layer pattern that the whole network is built from.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Known primitive: convolution + batch-norm + ReLU.
def conv_bn_relu(in_ch, out_ch, kernel, stride, padding, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride, padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class Block(nn.Module):
    """The repeating building block of the network — to be designed."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        # TODO: what convolutional structure goes here?
        pass

    def forward(self, x):
        # TODO
        pass


class Net(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # A first standard conv stem, then a stack of Blocks, then classify.
        self.stem = conv_bn_relu(3, 32, kernel=3, stride=2, padding=1)
        self.blocks = self._make_blocks(in_ch=32)   # TODO: channel/stride schedule
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def _make_blocks(self, in_ch):
        # TODO: build the body from a list of (out_ch, stride) using Block
        pass

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def train_step(model, images, labels, optimizer):
    # Standard supervised classification step (RMSProp upstream).
    logits = model(images)
    loss = F.cross_entropy(logits, labels)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return loss.item()
```
