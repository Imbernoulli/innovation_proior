## Research question

Depth has become the single most reliable lever for image-recognition accuracy. Deep
convolutional networks integrate low/mid/high-level features end-to-end, and the "level" of a
feature is enriched by the number of stacked layers, so the leading ImageNet entries keep
winning by getting deeper — from eight layers, to sixteen-nineteen, to roughly twenty-two. The
obvious next move is to keep stacking. The question is whether learning a better network is as
easy as adding more layers.

It is not. Once two known obstacles are removed — exploding/vanishing gradients, handled by
variance-preserving initialization and by normalization layers inside the net — networks of a
few tens of layers *do* start to converge. But a second, separate obstacle then surfaces, and
it is the one that matters here: as a plain stack of layers gets deeper, accuracy first
saturates and then *degrades*, and the degradation is visible already on the **training** set,
not only at test time. A 56-layer plain net has higher training error than a 20-layer one
built the same way; the same reversal shows up going from an 18-layer to a 34-layer plain net.

This is strange, because a deeper model contains its shallower counterpart as a special case.
Take a trained shallow net, copy its layers into the bottom of a deeper net, and make every
extra layer compute the identity; the deeper net then computes exactly the shallow net's
function and must have *no higher* training error. A solution that is at least as good
provably exists in the deeper model's parameter space. Yet the optimizer, given the deeper
architecture and trained from scratch, lands somewhere worse. The capacity is there; the
solver cannot reach it. The precise goal is to remove that gap — to make a network of, say, a
hundred-plus layers at least as easy to optimize as a shallow one, so that the accuracy gains
everyone expects from depth actually materialize instead of reversing.

## Background

The field state is "depth wins, if you can train it." A few load-bearing pieces are in place.

**Depth by stacking small filters.** The prevailing recipe (Simonyan & Zisserman 2014) fixes a
simple, uniform design and increases depth: almost all convolutions are 3×3 (the smallest
receptive field that still distinguishes left/right and up/down), stride 1 with 1-pixel padding
so spatial resolution is preserved, and periodic 2×2 max-pooling. Two rules keep the per-layer
compute balanced as the net deepens: for a given output-map size all layers carry the same
number of filters, and whenever a pooling stage halves the spatial size the filter count is
doubled, so the time complexity per layer stays roughly constant down the stack. Pushing this
to sixteen-nineteen weight layers gave a clear accuracy jump — strong evidence that depth, not
width or filter size, is the active ingredient. The same design also carries a heavy
three-layer fully-connected head (two 4096-wide layers) that dominates the parameter count.

**Normalized initialization and in-network normalization.** Variance-preserving initialization
for rectifier nets (He et al. 2015) keeps the signal variance roughly constant across layers at
the start of training, so forward activations neither blow up nor collapse through a deep stack.
Batch Normalization (Ioffe & Szegedy 2015) goes further and normalizes each layer's
pre-activations *during* training: for a feature over a mini-batch it computes
`x̂ = (x − μ_B) / √(σ_B² + ε)` and then restores representational freedom with a learned scale
and shift `y = γ·x̂ + β`, using running statistics at test time. BN lets nets train with much
higher learning rates, makes them far less sensitive to initialization, and regularizes enough
to often drop dropout. Together these mean a deep *plain* net no longer fails at the starting
line — it converges, which makes it possible to isolate what is left.
With BN in place, the forward activations of a deep plain net keep non-zero variance and the
back-propagated gradients keep healthy norms — both can be checked directly — so whatever makes
the 56-layer plain net train worse than the 20-layer one is **not** a vanishing/exploding
gradient. The deep plain net still reaches competitive accuracy eventually, so the solver is
not broken; it appears to converge to the right region very slowly. The degradation is an
optimization-conditioning phenomenon, observed and isolated, waiting to be explained.

**Parameter-free heads.** Global average pooling (Lin et al. 2013) replaces the fat FC head by
taking the spatial mean of each final feature map and feeding the resulting vector straight to
the classifier. It has no parameters to overfit, acts as a structural regularizer, and sums out
spatial position, making the prediction robust to small translations.

**Reformulation/preconditioning as a known trick (outside deep learning).** In several fields a
problem is solved far faster by re-expressing it in terms of *residuals*. Image
representations like VLAD and the Fisher Vector encode vectors by their residual with respect to
a dictionary, and for vector quantization encoding the residual is more effective than encoding
the original vector. For solving PDEs, multigrid methods and hierarchical-basis preconditioning
work on the residual solution between a coarse and a fine scale, and these residual-aware
solvers converge much faster than solvers that ignore the residual structure. The standing
lesson is that a good reformulation or preconditioning can make a hard optimization easy without
changing what is ultimately representable.

**Shortcut connections as an old idea.** Connections that skip one or more layers have a long
history: an early MLP practice added a direct linear path from input to output; auxiliary
classifiers wired to intermediate layers were used to fight vanishing gradients; "inception"
modules mix a shortcut branch with deeper branches; and centering schemes route responses,
gradients, and errors through shortcut paths. Skipping layers is, in itself, well-trodden
ground.

## Baselines

**Plain very-deep ConvNet (VGG-style; Simonyan & Zisserman 2014).** Stack 3×3 conv + ReLU,
periodic max-pool with channel doubling, a 3-FC head, softmax. Core idea: depth as the lever,
small filters to make depth affordable. Limitation: past a couple dozen layers it exhibits the
degradation problem — a deeper plain stack has *higher training error* than a shallower one
built identically, even with BN and good init keeping its gradients healthy. The deeper model
can represent everything the shallower one can and more, but the solver does not find a solution
as good as the shallower net's, and the fix is not "train longer" (degradation persists with
3× the iterations). It also pays for a heavy FC head.

**Highway networks (Srivastava, Greff & Schmidhuber 2015).** The contemporary attempt to make
very deep nets trainable by letting information skip layers through a *learned gate*, inspired by
LSTM gating. A layer computes
`y = H(x, W_H)·T(x, W_T) + x·C(x, W_C)`,
where `H` is the usual nonlinear transform, `T` is a learned, data-dependent "transform gate"
(typically `T = σ(W_T x + b_T)`), and `C` is a "carry gate" (often `C = 1 − T`). When `T → 0`
the layer carries its input through unchanged; when `T → 1` it behaves like a plain layer. With
this gating, nets of many hundreds of layers can be optimized by plain SGD. Limitations: the
gates carry their own parameters and add computation; because `C` is learned and data-dependent
the carry path can *close* (`C → 0`), at which point the layer reverts to an ordinary
transform and the skip is lost exactly where deep nets seem to need it; and despite training to
extreme depth, this line did not demonstrate accuracy *gains* from going past ~100 layers — it
showed depth could be optimized, not that depth then paid off.

## Evaluation settings

The natural yardsticks already in use:

- **ImageNet 2012 classification** (ILSVRC): 1000 classes, 1.28M training images, 50k
  validation, 100k test (test-server scored). Metrics: top-1 and top-5 error. Standard
  training augmentation: shorter side rescaled into `[256, 480]` for scale jitter, 224×224
  random crops with horizontal flip, per-pixel mean subtraction, and color augmentation.
  Evaluation by 10-crop testing, or fully-convolutional scoring averaged over several scales.
- **CIFAR-10**: 50k train / 10k test, 10 classes, 32×32 images, per-pixel mean subtracted.
  Simple augmentation: 4-pixel reflect-style padding then a random 32×32 crop or its flip;
  single-view 32×32 testing. Used here as the small-scale setting to probe the *behavior* of
  extreme depth, not to chase state of the art. Metric:
  test error (%).
- **Diagnostic protocol for the degradation phenomenon**: build matched plain nets at two
  depths (e.g. 20 vs 56 on CIFAR, 18 vs 34 on ImageNet) under identical width/optimizer/init,
  and compare their *training* and validation error curves through training. Also instrument BN
  layer outputs (response standard deviations) and gradient norms to test whether signals or
  gradients are vanishing.
- Optimizer protocol: SGD with momentum 0.9, weight decay 1e-4, mini-batch 256 (ImageNet) /
  128 (CIFAR), learning rate starting at 0.1 and divided by 10 on plateau, weights from the
  rectifier-aware initialization, BN after every convolution, no dropout.

## Code framework

Any candidate design plugs into the deep-conv-net training machinery that already exists: a stem that
maps the image into a feature map, a body that is a stack of repeated building blocks organized
into stages (each stage runs at one spatial resolution and one channel width, downsampling and
doubling channels between stages by the VGG complexity rule), a parameter-free pooling head, and
a linear classifier — trained by mini-batch SGD with momentum and weight decay, with batch
normalization after each convolution. Everything in that harness is settled. What is *not*
settled is the internal wiring of one repeated building block: how a few convolution-plus-norm
layers should be arranged so that stacking very many of them stays easy to optimize. That is the
one empty slot.

```python
import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with 1-pixel padding (the VGG primitive), no bias (BN follows)."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Block(nn.Module):
    """One repeated building block of the deep stack. It takes a feature map of
    `inplanes` channels and returns `planes` channels, optionally downsampling by
    `stride`. The convolution + BatchNorm + ReLU primitives below already exist;
    how to wire a few of them together so that a very deep stack of these blocks
    stays optimizable is exactly what is to be designed."""

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        # Available primitives: conv3x3, nn.BatchNorm2d, nn.ReLU.
        # TODO: fill in the building-block wiring.
        pass

    def forward(self, x):
        # TODO: produce the block output from x using the primitives above.
        pass


class DeepConvNet(nn.Module):
    """The fixed harness: VGG-style stem, four stages of stacked Blocks with
    channel doubling / spatial halving between stages, a parameter-free global
    average pooling head, and a linear classifier. Only `Block` is open."""

    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_stage(block, 64, layers[0])
        self.layer2 = self._make_stage(block, 128, layers[1], stride=2)
        self.layer3 = self._make_stage(block, 256, layers[2], stride=2)
        self.layer4 = self._make_stage(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))      # parameter-free GAP head
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():                         # rectifier-aware init
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_stage(self, block, planes, n_blocks, stride=1):
        blocks = [block(self.inplanes, planes, stride)]
        self.inplanes = planes
        for _ in range(1, n_blocks):
            blocks.append(block(self.inplanes, planes))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)


# existing mini-batch SGD training loop the network plugs into
def train(model, loss_fn, data_loader, optimizer):
    for images, labels in data_loader:
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)     # cross-entropy + softmax
        loss.backward()
        optimizer.step()                   # SGD, momentum 0.9, weight decay 1e-4
```

The stem, the staged body, the GAP head, the classifier, the init, and the SGD loop are all
fixed. The single open piece is `Block` — how to arrange a few conv-BN-ReLU layers so that a
stack of very many blocks remains as easy to optimize as a shallow one.
