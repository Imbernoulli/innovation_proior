# Context

## Research question

Network depth is one of the strongest levers on the expressiveness of a convolutional network — both in theory and in the empirical record (AlexNet's 5 conv layers, VGG's 19, GoogLeNet's 22, ResNet's 152). Pushing depth increases capacity and performance, but forward and backward passes scale linearly with depth, and training very deep networks is computationally expensive. The question is how to train very deep residual networks more effectively, in both optimization quality and wall-clock time.

## Background

The relevant prior work is the lineage of methods for training very deep nets and for stochastic regularization.

**Better initialization and Batch Normalization.** Early efforts attacked vanishing gradients with greedy layer-wise pretraining and careful initialization (Glorot & Bengio 2010). Batch Normalization (Ioffe & Szegedy 2015) standardizes the mean and variance of hidden activations per mini-batch, which reduces vanishing gradients and has a regularizing effect; it is the default in modern deep nets.

**Skip connections.** Highway Networks (Srivastava et al. 2015) added parameterized gated skip connections — "information highways" whose learned gates control how much signal bypasses a block, possibly crossing several layers. ResNets (He et al. 2015/2016) simplified this by making the skip parameter-free whenever shapes match. A residual block computes H_ℓ = ReLU(f_ℓ(H_{ℓ-1}) + s_ℓ(H_{ℓ-1})), where f_ℓ is a small stack (e.g. Conv-BN-ReLU-Conv-BN) and s_ℓ is identity in same-shape blocks, average-pool plus zero-padding in CIFAR transition blocks, or a projection shortcut in ImageNet-style bottleneck transitions. The shortcut path lets gradients and features pass between layers with little obstruction, which is what makes hundred-plus-layer training feasible. A motivating diagnostic that produced ResNets: plain deep networks exhibit *higher training error* as depth grows — a degradation that more parameters should not cause, attributed to optimization difficulty when signals propagate through many layers.

**Stochastic regularization.** Dropout (Srivastava et al. 2014) multiplies each hidden activation by an independent Bernoulli variable during training, reducing co-adaptation of units and acting like training an ensemble of exponentially many thinned subnetworks; in the original test-time rule, the retained outgoing weights are scaled by the keep probability, equivalently matching the expected contribution of each unit. Follow-ups include DropConnect (Wan et al. 2013, which drops weights instead of activations), Maxout, and DropIn.

He et al.'s record-setting results on ImageNet came partly from an ensemble of ResNets of varying depth.

A structural fact about the CIFAR/SVHN architecture: the input to a residual block is always non-negative, because it is the output of a preceding ReLU (or, for the first block, of an initial Conv-BN-ReLU stem). Same-shape identity shortcuts and average-pool/zero-pad transition shortcuts preserve non-negativity; learned projection shortcuts need to be treated as the shortcut branch, not as a literal identity.

## Baselines

**Constant-depth ResNet.** The network as built: L residual blocks, each H_ℓ = ReLU(f_ℓ(H_{ℓ-1}) + s_ℓ(H_{ℓ-1})), all blocks active every iteration. Strong baseline for image classification, with training cost scaling with the full depth L.

**Dropout on a deep ResNet.** Dropout applied inside or around the blocks for regularization. It multiplies individual activations by Bernoulli masks, leaving the overall network depth unchanged.

**Highway Networks.** Gated skip connections that can route signal around layers. The gates are learned and parameterized, adding extra parameters that are fixed in a learned way at test time; ResNets showed that the simpler identity skip is competitive.

## Evaluation settings

- **Datasets/metric**: image classification test error on CIFAR-10, CIFAR-100, SVHN, and ImageNet.
- **Architectures**: ResNets following He et al. — the basic two-conv block (Conv-BN-ReLU-Conv-BN) on CIFAR/SVHN, the bottleneck block on ImageNet. A 110-layer CIFAR ResNet corresponds to L = 54 residual blocks; the method is also pushed beyond 1000 layers on CIFAR.
- **Training**: SGD with momentum, standard ResNet learning-rate schedules and data augmentation; Batch Normalization throughout. Reported alongside wall-clock training time, since reducing it is an explicit goal.
- **Comparison axis**: test error *and* training time, against the constant-depth ResNet of the same architecture.

## Code framework

The primitive that exists is a ResNet built from residual blocks, each with a transformation branch f_ℓ and a shape-aware shortcut s_ℓ, composed into a deep stack with a classifier head. The block exposes an unused per-block hyper-parameter slot whose role and use are the open design question.

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """H = ReLU(f(H_prev) + skip(H_prev)).
    f = Conv-BN-ReLU-Conv-BN; skip = identity, pool+pad, or projection.
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
        self.block_hparam = block_hparam   # TODO: what is this per-block hyper-parameter, and how is it used?

    def _make_skip(self, in_ch, out_ch, stride):
        # identity, with avg-pool downsample and zero-pad channels when widening
        ...

    def forward(self, x):
        # TODO: how the residual branch behaves (see open design question)
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
