## Research question

Every state-of-the-art deep network is a tall stack of residual blocks with a normalization
layer (batch normalization for vision, layer normalization for sequence models) wired into
each block. Normalization is universally credited with four things at once: it stabilizes
training, it lets you use a large learning rate, it accelerates convergence, and it improves
generalization. The trouble is that nobody can say *which* of those four benefits is actually
intrinsic to normalization and which are incidental — and that ignorance has a cost. The
normalization op forces a batch dependence into the forward pass, couples examples within a
minibatch, behaves differently at train and test time, adds bookkeeping for running
statistics, and quietly entangles itself with weight decay and the learning-rate schedule in
ways that are hard to reason about. The precise question is whether a deep residual network
can be trained reliably *without any normalization layer* and still match a normalized net on
all four counts simultaneously: trained at the *same* (maximal) learning rate, converging at
the *same* speed, and generalizing *at least as well* — for networks that may be hundreds or
thousands of layers deep. If the answer is yes, then the only thing that has to be replaced is
whatever normalization was secretly doing at the *start* of training; if it is no, the field
needs to understand exactly what about normalization is irreplaceable. Either way the first
job is to pin down, mathematically, why a plain residual network without normalization fails
to train at all at a useful learning rate.

## Background

**Residual networks and why they go deep.** Since He et al. (2016), the dominant vision (and
later sequence) architecture is a deep stack of residual blocks. A plain (normalization-free)
residual network with blocks `F_1, ..., F_L` and input `x_0` computes

```
x_l = x_0 + sum_{i=0}^{l-1} F_i(x_i),
```

so each block adds a learned correction to a running sum that always carries the identity
forward. The skip connection is what makes optimization of very deep nets tractable: gradients
have a short path back to every layer.

**Positively homogeneous functions.** A function `f` is positively homogeneous of degree one
(p.h.) if `f(alpha * x) = alpha * f(x)` for every `alpha > 0`. This is not an exotic property:
a linear or convolutional layer with no bias is p.h., as is ReLU, max-pooling, average
pooling, addition, concatenation, and dropout; and a composition of p.h. functions is again
p.h. So a residual network built from bias-free conv/linear layers and ReLUs — exactly what
you get when you strip out the normalization layers and zero the biases — is, as a function of
its inputs and as a function of any homogeneous subset of its weights, positively homogeneous
of degree one. The cross-entropy classification loss is `l(z, y) = -y^T (z - logsumexp(z))`
for one-hot label `y` and logits `z`.

**Standard initialization and the variance story.** Glorot & Bengio (2010) and He et al.
(2015) set per-layer weight variances so that the variance of the activations is preserved
from layer to layer. For ReLU networks, He initialization draws weights from
`N(0, 2 / fan)` (so the per-coordinate standard deviation is `sqrt(2 / fan)`), with biases at
zero; this is designed so that a single layer maps unit-variance input to unit-variance
output. These analyses were built for plain feed-forward nets, and they say nothing about what
the *additive skip connection* does to the variance budget.

**The diagnostic that matters: variance explodes with depth in a residual net.** Take the
residual recursion `x_{l+1} = x_l + F_l(x_l)` and assume, as at initialization, that the
branch is zero-mean given its input, `E[F_l(x_l) | x_l] = 0`. By the law of total variance,

```
Var[x_{l+1}] = E[Var[F_l(x_l) | x_l]] + Var[x_l].
```

The skip connection therefore *forces* the variance to grow with depth: it can never shrink,
because a nonnegative term is added at every block. Worse, with a standard variance-preserving
initialization each branch is tuned so that its output variance roughly equals its input
variance, `Var[F_l(x_l) | x_l] ~ Var[x_l]`, which makes `Var[x_{l+1}] ~ 2 * Var[x_l]` — the
activation variance roughly *doubles* every block and so grows like `2^l`, exponentially in
depth. This is a known phenomenon for residual nets (Hanin & Rolnick 2018 on activation scale;
Balduzzi et al. 2017 on gradient statistics; Yang & Schoenholz 2017). It is a pre-method fact
about the world: it follows from the architecture and the standard init, before any new method
is proposed.

**Shattered gradients (Balduzzi et al. 2017).** In plain deep ReLU networks the correlation
between gradients at nearby inputs decays exponentially with depth, so the gradient field
comes to resemble white noise; skip connections slow this decay. This established that the
*joint behavior* of gradients across depth — not just their average magnitude — is what breaks
in deep nets, and that residual connections matter for it.

**What normalization is believed to do, and the open question about why.** Batch
normalization (Ioffe & Szegedy 2015) divides activations by the batch standard deviation and
subtracts the batch mean, then restores representation power with a learnable per-channel scale
`gamma` and shift `beta`. It was originally justified as reducing "internal covariate shift";
Santurkar et al. (2018) argued instead that its real effect is to *smooth the loss surface*.
There is no consensus. Crucially for what follows, a normalized layer's output is *invariant
to rescaling the weights that feed it*, `y(alpha * w) = y(w)`, because the division by the
activation standard deviation cancels any scalar on the weights.

**Weight decay, scale invariance, and the effective learning rate (van Laarhoven 2017).** That
scale invariance has a consequence everyone uses but few state: under normalization, an `L2`
penalty on the weights has *no* regularizing effect on the function — it only shrinks the
weight norm `||w||`. And because the output is invariant to weight scale, the gradient of a
normalized layer is (to leading order) orthogonal to its own weight vector. For a gradient
step `w <- w - eta * grad`, the *effective* step size measured on the unit-normalized weight
is `eta / ||w||^2`. So as the learning rate is annealed under weight decay, `||w||` shrinks,
which *raises* the effective learning rate `eta / ||w||^2` — normalization hands you an
automatic effective-LR schedule for free. Any normalization-free scheme that wants to match a
normalized net's training dynamics has to reproduce this weight-norm behavior, or it will need
a different, hand-tuned learning-rate schedule.

**Prior attempts at normalization-free residual training, and where each stalls.**
- *Scaled recurrence* (Balduzzi et al. 2017; Gehring et al. 2017): replace the residual
  recursion with `x_l = sqrt(1/2) * (x_{l-1} + F_l(x_{l-1}))`. This keeps the activation
  variance at `O(1)`, but unrolling the recurrence shows the contribution of each residual
  branch is multiplied by a power of `sqrt(1/2)` that decays geometrically toward the input,
  so the residual branches are suppressed exponentially with depth — which undercuts the
  reason for having them.
- *Data-dependent rescaling, LSUV* (Mishkin & Matas 2015): orthogonal-initialize each layer,
  then rescale it, using one forward pass on a real minibatch, so each layer's output has unit
  variance — in effect mimicking what batch normalization would do on the first batch. It
  controls the *scale* of activations and gradients at initialization, with a calibration pass
  over data.
- *Near-zero residual init* (Srivastava et al. 2015; a 2016 linearized residual-net analysis;
  Goyal et al. 2017; Kingma & Dhariwal 2018): initialize each residual branch at (or near) the zero function so
  the network starts close to identity. Goyal et al. (2017) implement this inside
  normalization by setting the *last* batch-norm `gamma` of each residual block to zero, so the
  branch starts as identity. These works observe empirically that small/zero residual init
  helps optimization. The linearized analysis proves, in the regime around zero init, that the
  critical points are global minima.

Each of these limits the magnitude of activations or gradients at initialization, or starts
the branches small; yet each still fails to train stably at the maximal learning rate for very
deep residual networks, suppresses the deep branches, or requires a data-dependent calibration
pass.

## Baselines

A normalization-free initialization would be measured against, and reacts to, these:

- **He / Kaiming initialization (He et al. 2015, arXiv:1502.01852).** Draw conv/linear weights
  from `N(0, 2 / fan)` for ReLU networks, biases at zero. *Limitation:* derived for plain
  feed-forward nets, it ignores the additive skip connection, so applied to a residual net it
  produces the `2^l` variance explosion above; the resulting logit blowup at initialization is
  what makes a plain residual net untrainable at a useful learning rate.

- **Xavier / Glorot initialization (Glorot & Bengio 2010).** Same variance-preserving idea
  with a fan-in/fan-out average, for symmetric activations. *Limitation:* same blind spot to
  the residual sum; a Xavier-initialized deep residual net without normalization cannot be
  trained at the maximal learning rate, and even at a small fraction of it converges only very
  slowly.

- **Batch normalization (Ioffe & Szegedy 2015).** The default. Normalize activations to
  zero-mean unit-variance per channel over the batch, then affine-transform with learnable
  `gamma, beta`. Stabilizes training, enables high learning rates, accelerates convergence,
  improves generalization. *Limitation (the thing under examination):* it introduces a batch
  dependence and a train/test discrepancy, costs `O(channels)` extra parameters and running
  statistics per layer, and — most importantly here — its benefits are bundled together so
  that it is unknown which are intrinsic; it is the incumbent whose necessity is in doubt.

- **Layer normalization (Ba et al. 2016).** The sequence-model analogue, normalizing over the
  feature dimension of a single example. *Limitation:* same questions, and it is reported to be
  *critical* for stable training of deep sequence models — removing it is said to make training
  unstable, a claim a normalization-free method would have to overturn.

- **sqrt(1/2) scaled recurrence (Balduzzi et al. 2017; Gehring et al. 2017).** *Limitation:*
  geometric suppression of deep branches; fails to train at the normalization learning rate
  beyond ~100 layers.

- **LSUV (Mishkin & Matas 2015).** *Limitation:* data-dependent (needs a calibration batch);
  controls only per-layer activation/gradient scale, so it too fails to train stably at the
  maximal learning rate for very deep residual nets.

## Evaluation settings

The natural yardsticks already in use, at the time, for "can you train a deep residual net":

- **Depth stress test:** Wide-ResNet (width 1) on CIFAR-10, trained at the default batch-norm
  learning rate `0.1`, weight decay `5e-4`, at increasingly large depths;
  measure how far into training one can get (e.g. accuracy after the first epoch) as a function
  of depth. The metric is whether training even proceeds at the maximal learning rate as depth
  grows.
- **Image classification:** ResNet / Wide-ResNet on CIFAR-10 and SVHN; ResNet-50/101 on
  ImageNet. Default SGD with momentum `0.9`, weight decay, step or cosine learning-rate
  schedule, standard augmentation (random crop with padding, horizontal flip). Metric: test
  error.
- **Machine translation:** the Transformer (Vaswani et al. 2017) on IWSLT German-English and
  WMT English-German, following standard scaled-training recipes; metric BLEU.
- Protocol: hold the architecture, optimizer, schedule, data pipeline, and loss fixed; vary
  only the normalization / initialization, so that any difference is attributable to it.

## Code framework

A standard residual-network training harness already exists; the open slot is how a
normalization-free residual network should be parameterized and initialized. The substrate is
generic: a data pipeline, an SGD-with-momentum optimizer, a cross-entropy loss, residual blocks
with additive skips, a stack of those blocks into a network, and a training loop. The scaffold
below keeps the residual architecture and training loop ordinary, with one neutral placeholder
for the branch design and one neutral placeholder for initialization.

```python
import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ResidualBlock(nn.Module):
    """A residual block with no normalization layer and an additive skip."""

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.downsample = downsample
        # TODO: the branch structure to use inside the residual path

    def forward(self, x):
        # TODO: the residual branch computation
        pass


class ResNet(nn.Module):
    """A stack of residual blocks, no normalization anywhere."""

    def __init__(self, block, layers, num_classes=10):
        super().__init__()
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        initialize(self)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.AvgPool2d(1, stride=stride)
        seq = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        for _ in range(1, blocks):
            seq.append(block(planes, planes))
        return nn.Sequential(*seq)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


def initialize(model):
    # TODO: choose initial parameter values for this normalization-free net
    pass


# existing training loop the network plugs into
def train(model, loss_fn, data_loader, optimizer):
    for inputs, targets in data_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), targets)
        loss.backward()
        optimizer.step()
```

The harness supplies the data, optimizer, loss, residual-network shell, and training loop. The
empty branch and initialization slots are the only places left open.
