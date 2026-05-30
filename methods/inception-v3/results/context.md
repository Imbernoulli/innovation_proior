# Context

The aim is to scale up a deep convolutional image-classification network so that added computation buys as much accuracy as possible — keeping the parameter count and FLOPs low enough for mobile-vision and big-data settings — using the architectures, normalization, optimizers, and framework code available in late 2015. The target is the ILSVRC-2012 1000-class benchmark.

## Research question

Since 2014, very deep convolutional networks have become the mainstream of computer vision, and gains in ImageNet classification accuracy reliably transfer to detection, segmentation, pose, video, and other tasks — so an architectural improvement to the classification backbone improves nearly everything downstream. That makes scaling up the backbone the highest-leverage move available.

But scaling up naively is wasteful. There are two contrasting reference architectures. One is built from a single uniform primitive (stacked 3×3 convolutions): conceptually simple, but it evaluates at a high computational cost and uses about three times the parameters of the 2012 AlexNet. The other is a heterogeneous, parallel-branch design that achieves comparable accuracy at roughly 5 million parameters — a 12× reduction relative to AlexNet's 60 million — and much lower compute, which is what makes it usable in memory- and compute-limited settings. The catch is that the efficient architecture is *complex*, and its complexity makes it hard to modify: if you scale it up naively — e.g. double every filter-bank width — you incur a **4× increase in both compute and parameters**, which is prohibitive when the accuracy gain is modest. Worse, the original description of that architecture did not spell out *why* its many design decisions were made, so adapting it to a new budget without destroying its efficiency is guesswork.

The precise question, then: what *general principles* govern scaling a convolutional network so that every added multiply-add is spent efficiently? And given such principles, how should the expensive pieces — large-filter convolutions, grid-size reductions, the classifier head, the training objective — be re-engineered so a deeper, more accurate network costs only modestly more than the efficient baseline rather than multiplicatively more?

## Background

By late 2015 the field state is: depth and width both help, gains transfer across tasks, and the practical bottleneck has shifted from "can we train deep nets at all" to "can we get the accuracy of a huge net at a cost we can afford." Several load-bearing facts and tools are in hand.

- **Two stacked 3×3 convolutions have the receptive field of one 5×5**, with fewer parameters and an extra nonlinearity in between. A 5×5 convolution is disproportionately expensive: with the same number of filters it costs 25/9 ≈ 2.78× a 3×3. This is the seed of "factorize big filters into small ones."
- **Generous 1×1 dimension reduction** — reducing channel count with a cheap 1×1 convolution *before* an expensive spatial convolution — is what makes the efficient parallel-branch architecture affordable. The working hypothesis behind it: in a vision network, nearby activations are highly correlated, so reducing dimensionality before spatial aggregation loses little information.
- **Batch Normalization** (Ioffe & Szegedy 2015): normalize each layer's pre-activations over the mini-batch to zero mean / unit variance with a learnable scale and shift. It permits much higher learning rates, speeds convergence, and acts as a mild regularizer. A BN-equipped version of the efficient architecture is the immediate predecessor being improved upon.
- **Auxiliary classifiers**: small classifier heads attached to intermediate layers, originally introduced to push gradient into lower layers and combat vanishing gradients in very deep nets (and argued by Lee et al. 2014 to promote more stable learning). A diagnostic observation on *existing* nets sets up a re-interpretation: a network trained with and without the auxiliary head shows *virtually identical* training progression until late in training, when the auxiliary version pulls slightly ahead; and removing the *lower* of two auxiliary heads has no adverse effect on final quality. So the original "evolves low-level features" rationale is suspect — the head behaves more like a regularizer.
- **Optimizers and stabilizers**: momentum SGD (Sutskever et al. 2013); RMSProp (per-parameter adaptive step sizes); gradient clipping (Pascanu et al. 2012) to stabilize training; and exponential moving averages of parameters for evaluation.

Four design principles, distilled from large-scale experimentation, frame the rest (they are speculative guidelines, but grave deviations from them tended to degrade networks):
1. **Avoid representational bottlenecks, especially early.** The representation size should decrease gently from input to output; extreme compression at any cut loses information no later layer can recover.
2. **Higher-dimensional representations are easier to process locally.** More activations per tile → more disentangled features → faster training.
3. **Spatial aggregation can be done over lower-dimensional embeddings with little loss of representational power.** Reduce dimension *before* a spatial (e.g. 3×3) convolution; adjacent units are strongly correlated, so the reduction loses little and even speeds learning.
4. **Balance width and depth.** For a fixed compute budget, increase per-stage filter count and network depth in parallel.

A diagnostic finding on resolution that informs the input design: at *constant* compute (achieved by reducing the stride of the first layers or removing the first pool for smaller inputs), receptive fields of 79×79, 151×151, and 299×299 reach top-1 accuracies of 75.2%, 76.4%, and 76.6% respectively — i.e. lower-resolution inputs nearly match high-resolution ones when the compute is held fixed.

## Baselines

**The uniform-3×3 deep net (VGGNet, Simonyan & Zisserman 2014).** Depth from one primitive: stacks of 3×3 convolutions, doubling filters when halving the spatial map, ending in large fully-connected layers. *Core math:* two stacked 3×3s = one 5×5 receptive field, three = 7×7, with fewer parameters and added nonlinearity. *Gap:* architectural simplicity at a high compute cost and ~3× AlexNet's parameter count — most of the parameters in the fully-connected head; no mechanism for spending compute efficiently.

**The efficient parallel-branch net (GoogLeNet / Inception, Szegedy et al. 2014/2015).** Replace a uniform stack with **Inception modules**: parallel branches (1×1, 3×3, 5×5, pooling) whose outputs are concatenated, with 1×1 convolutions as cheap dimension-reduction bottlenecks so a 22-layer net stays at ~5M parameters. Includes auxiliary classifiers. *Core idea:* heterogeneous multi-scale processing per module + aggressive 1×1 reduction. *Gap:* highly complex and hard to modify; naive scaling (doubling filter widths) costs 4× compute/params; the design rationale was never documented, so efficiency is fragile under modification. This is the architecture being scaled up.

**Batch-normalized Inception (Ioffe & Szegedy 2015).** The efficient architecture with BN after convolutions. *Role:* the immediate predecessor whose accuracy and compute set the bar; the new network aims for substantially better accuracy at only a modest (~2.5×) compute increase over it.

**Denser, higher-performing successors (He et al. 2015, PReLU/“delving deep”).** Higher accuracy from denser/heavier networks. *Gap:* much higher computational cost and parameter count — the accuracy-per-FLOP trade is what an efficient scaled-up net would attack.

**Hard one-hot cross-entropy training (standard practice).** Train the classifier to maximize the log-probability of the single ground-truth label, `q(k) = δ_{k,y}`. *Gap:* the maximum is approached only as the ground-truth logit grows unboundedly larger than the rest, which (a) encourages overfitting — full probability mass on the training label does not guarantee generalization — and (b) drives the largest logit far above the others, which with the bounded gradient `∂ℓ/∂z_k = p(k) − q(k) ∈ [−1,1]` reduces the model's ability to adapt: the model becomes over-confident.

## Evaluation settings

- **ILSVRC-2012 classification, 1000 classes.** ~1.28M training images. Metrics: top-1 and top-5 error. Both single-frame (single-crop) and multi-crop / multi-model-ensemble evaluation protocols are standard.
- **Cost metrics.** Computational cost in multiply-adds (multiplications) per inference, and parameter count — these are first-class because the whole exercise is accuracy-per-compute. (A fully-convolutional net has one multiply per weight per activation, so reducing compute also reduces parameters.)
- **Controlled comparisons that exist at the time.** Holding compute roughly constant while varying a single architectural axis — input resolution (above), or linear vs. ReLU activation in a factorized layer, or one grid-reduction scheme vs. another — to isolate the effect of that axis.
- **Input pipeline.** Fixed 299×299×3 RGB input for the main network (with the option of reduced stride / removed first pool for lower-resolution inputs at matched compute).
- **Training infrastructure.** Distributed synchronous/asynchronous SGD across tens of GPU replicas with small per-replica batches; evaluation on a parameter EMA.

## Code framework

The available code is a deep-CNN image-classification harness with batch-normalized convolution as the base primitive. The libraries supply convolution, batch normalization, ReLU, pooling, concatenation, an SGD/RMSProp optimizer with exponential LR decay and gradient clipping, a softmax cross-entropy loss, and an optional auxiliary-head hook. The scaffold is the harness with empty slots for the repeated multi-branch module, the grid-size-reduction module, the classifier objective, and the auxiliary head.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBN(nn.Module):
    # the base primitive: conv (no bias) -> BN -> ReLU
    def __init__(self, in_ch, out_ch, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_ch, eps=0.001)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class MultiBranchModule(nn.Module):
    # TODO: the repeated parallel-branch module we'll design — several branches
    # (cheap 1x1 reductions feeding spatial convolutions, plus a pooling branch),
    # concatenated on the channel axis. Exactly how the spatial convolutions are
    # structured (filter shapes, how many layers per branch) is the open slot.
    def __init__(self, in_ch):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class GridReduction(nn.Module):
    # TODO: the module that halves the spatial grid while increasing channels,
    # without creating a representational bottleneck. Structure is open.
    def __init__(self, in_ch):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class AuxHead(nn.Module):
    # TODO: an intermediate classifier head (pool -> 1x1 -> ... -> logits).
    def __init__(self, in_ch, num_classes):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class Net(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True):
        super().__init__()
        # stem: a stack of small convolutions + pooling, open in detail
        self.stem = nn.Sequential()        # TODO
        self.body = nn.Sequential()        # TODO: stages of the modules above,
                                           # with grid reductions between them
        self.aux = AuxHead(768, num_classes) if aux_logits else None
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.body(self.stem(x))
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)


def classification_loss(logits, target, num_classes):
    # TODO: the training objective for the classifier (the standard choice is
    # softmax cross-entropy against the one-hot target; whether to modify the
    # target distribution is open).
    raise NotImplementedError


# --- training harness ---
model = Net(num_classes=1000, aux_logits=True)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.045, alpha=0.9, eps=1.0)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.94)  # per 2 epochs
# gradient clipping at 2.0; evaluation on an EMA of parameters
```
