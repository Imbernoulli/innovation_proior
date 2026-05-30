# Context

## Research question

Network depth is one of the strongest levers on the expressiveness of a convolutional network — both in theory and in the empirical record (AlexNet's 5 conv layers, VGG's 19, GoogLeNet's 22, ResNet's 152). But pushing depth up brings three coupled problems that make very deep networks hard to use:

- **Vanishing gradients.** As error is backpropagated through many layers, repeated multiplication by small weights shrinks the gradient signal, so the earliest layers barely learn.
- **Diminishing feature reuse (forward information loss).** The forward analog: features of the input, or of early layers, are progressively "washed out" by repeated multiplication with (randomly initialized) weight matrices, so later layers struggle to recover meaningful directions.
- **Long training time.** Forward and backward passes scale linearly with depth; a 152-layer ResNet takes weeks to converge on ImageNet.

This sets up a dilemma. Short networks flow information well forward and backward and train quickly, but lack the capacity for complex vision concepts. Deep networks have the capacity but are slow and difficult to train. The precise question: is there a training procedure that gives the optimization behavior of a *short* network — fast, with strong gradient flow — while still producing a *deep*, high-capacity model to deploy? A solution would have to reconcile "short during training" with "deep at test time" within a single set of shared weights, and ideally do so cheaply, with minimal change to an existing deep architecture.

## Background

The relevant prior work is the lineage of methods for training very deep nets and for stochastic regularization.

**Better initialization and Batch Normalization.** Early efforts attacked vanishing gradients with greedy layer-wise pretraining and careful initialization (Glorot & Bengio 2010). Batch Normalization (Ioffe & Szegedy 2015) standardizes the mean and variance of hidden activations per mini-batch, which reduces vanishing gradients and has a regularizing effect; it is the default in modern deep nets.

**Skip connections.** Highway Networks (Srivastava et al. 2015) added parameterized gated skip connections — "information highways" whose learned gates control how much signal bypasses a block, possibly crossing several layers. ResNets (He et al. 2016) simplified this by making the skip an identity. A residual block computes H_ℓ = ReLU(f_ℓ(H_{ℓ-1}) + id(H_{ℓ-1})), where f_ℓ is a small stack (e.g. Conv-BN-ReLU-Conv-BN) and id is identity (or a linear projection when dimensions change). The identity path lets gradients and features pass between layers unimpeded, which is what makes hundred-plus-layer training feasible. A motivating diagnostic that produced ResNets: plain deep networks exhibit *higher training error* as depth grows — a degradation that more parameters should not cause, attributed to optimization difficulty when signals propagate through many layers.

**Stochastic regularization.** Dropout (Srivastava et al. 2014) multiplies each hidden activation by an independent Bernoulli variable during training, reducing co-adaptation of units and acting like training an ensemble of exponentially many thinned subnetworks; at test time the activations are scaled by the keep probability. Follow-ups include DropConnect (Wan et al. 2013, which drops weights instead of activations), Maxout, and DropIn.

Two diagnostic facts about existing systems that frame the work:
- Dropout's regularizing benefit is known to weaken when combined with Batch Normalization; experiments on 110-layer ResNets with BN show essentially no improvement from Dropout at various rates. So a different stochastic mechanism is needed for deep BN-ResNets.
- He et al.'s record-setting results came partly from an *ensemble* of ResNets of varying depth — suggesting depth diversity itself is valuable.

A structural fact about the architecture that the method will exploit: in these ResNets the input to a residual block is always non-negative, because it is the output of a preceding ReLU (or, for the first block, of an initial Conv-BN-ReLU stem). A ReLU applied to a non-negative input is the identity.

## Baselines

**Constant-depth ResNet.** The network as built: L residual blocks, each H_ℓ = ReLU(f_ℓ(H_{ℓ-1}) + id(H_{ℓ-1})), all blocks active every iteration. Strong, but its training cost scales with the full depth L, and very deep variants (1000+ layers) suffer vanishing gradients / diminishing feature reuse and overfit. Gap: no mechanism to make the *effective* training depth smaller than the deployed depth.

**Dropout on a deep ResNet.** Add Dropout inside or around the blocks for regularization. Gap: makes the network *thinner* (drops individual activations) but not *shorter* — it does nothing about the long gradient/forward chains of a deep net — and its benefit largely vanishes in the presence of Batch Normalization.

**Highway Networks.** Gated skip connections that can route around layers. Gap: the gates are learned and parameterized (extra parameters, always on at test in a fixed learned way); they do not stochastically and freely shorten the network during training, and ResNet already showed the simpler identity skip trains better.

## Evaluation settings

- **Datasets/metric**: image classification test error on CIFAR-10, CIFAR-100, SVHN, and ImageNet.
- **Architectures**: ResNets following He et al. — the basic two-conv block (Conv-BN-ReLU-Conv-BN) on CIFAR/SVHN, the bottleneck block on ImageNet. A 110-layer CIFAR ResNet corresponds to L = 54 residual blocks; the method is also pushed beyond 1000 layers on CIFAR.
- **Training**: SGD with momentum, standard ResNet learning-rate schedules and data augmentation; Batch Normalization throughout. Reported alongside wall-clock training time, since reducing it is an explicit goal.
- **Comparison axis**: test error *and* training time, against the constant-depth ResNet of the same architecture.

## Code framework

The primitive that exists is a ResNet built from residual blocks, each with a transformation branch f_ℓ and an identity/projection skip, composed into a deep stack with a classifier head. The block already has the identity path the method needs; the open question is a per-block training/test rule and a per-block scalar that governs it.

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """H = ReLU(f(H_prev) + skip(H_prev)).
    f = Conv-BN-ReLU-Conv-BN; skip = identity (or projection/pool when shape changes).
    The block carries a scalar hyper-parameter whose role is the open design question."""
    def __init__(self, in_ch, out_ch, stride=1, block_hparam=None):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = self._make_skip(in_ch, out_ch, stride)
        self.relu = nn.ReLU(inplace=True)
        self.block_hparam = block_hparam   # TODO: what is this per-block scalar, and how is it used?

    def _make_skip(self, in_ch, out_ch, stride):
        # identity, with avg-pool downsample and zero-pad channels when widening
        ...

    def forward(self, x):
        # TODO: training-time and test-time behavior of the residual branch
        return self.relu(self.f(x) + self.skip(x))

class ResNet(nn.Module):
    def __init__(self, block_counts, hparams, num_classes=10):
        super().__init__()
        # stem (Conv-BN-ReLU) -> L residual blocks -> global pool -> linear head
        # TODO: assign each block its scalar hyper-parameter
        ...
    def forward(self, x):
        ...
```
