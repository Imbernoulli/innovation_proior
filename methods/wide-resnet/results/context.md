# Context

The task is to redesign a residual image classifier under the knowledge available in mid-2016: residual shortcuts, pre-activation blocks, batch normalization, dropout, CIFAR/SVHN/ImageNet practice, and GPU training constraints are available, but the residual block's capacity allocation and regularization are still open.

## Pressure On Depth

Residual networks have changed the practical depth limit. A block computes an additive shortcut form, usually written as `x_{l+1} = x_l + F(x_l, W_l)`, so the model can learn a residual correction while the shortcut carries the signal forward. This solves much of the degradation problem that made plain very deep nets hard to optimize, and it enables networks with hundreds or even more than a thousand layers.

That success creates a new problem. The best residual results keep spending depth for small accuracy gains, so training time grows sharply. The shortcut also creates a structural ambiguity: because the identity path can carry forward activations and carry backward gradients, a residual block's weights are not forced to become essential. The useful work may concentrate in only some blocks while many others pass along small corrections. Stochastic-depth training makes this concern concrete by randomly bypassing whole residual blocks during training and still improving very deep nets.

## Fixed Machinery

The starting architecture is the pre-activation residual template. Instead of the original `conv -> BN -> ReLU` ordering, the residual branch uses `BN -> ReLU -> conv`, which preserves a cleaner identity path after addition and is the better available residual-unit design for very deep CIFAR-style networks.

The fixed classifier scaffold has one `3x3` stem convolution, then three residual groups at decreasing spatial resolutions, then final batch normalization, ReLU, global average pooling, and a fully connected classifier. Downsampling happens at the first block of the second and third groups. Shortcut connections are identities when shape permits and projections when resolution or channel count changes.

The code base has ordinary convolution, batch normalization, ReLU, dropout, cross-entropy, SGD with momentum/Nesterov, weight decay, and step learning-rate decay. It can instantiate a residual block repeatedly, but the block's internal convolution pattern, feature-plane count, and regularization location are the variables to decide.

## Open Design Axes

There are three direct ways to increase a residual block's representational power. One can add more convolutions inside a block, increase the number of feature planes in the convolutions, or increase the spatial kernel size. The kernel-size option is constrained by the VGG/Inception evidence that stacked small filters are effective, so filters larger than `3x3` are not the interesting first move.

Two knobs remain. Let `l` denote the number of convolutions inside a block, and let `k` denote a multiplier on the feature-plane counts in the three residual groups. The basic pre-activation residual block has `l = 2` and two `3x3` convolutions. The open question is how to allocate a fixed training budget between more blocks, more convolutions per block, and more channels per convolution.

Regularization is also unresolved. Batch normalization regularizes, but its effect is entangled with data augmentation. Dropout is available, but placing dropout on the identity shortcut is dangerous because that shortcut is the mechanism that makes residual optimization work. If dropout is used, its position must not corrupt the skip path.

## Baselines To Beat

The thin deep residual baseline treats depth as the main scaling axis and uses narrow blocks to keep the parameter count manageable. The extreme CIFAR versions reach hundreds or a thousand layers, with strong accuracy but high training cost.

The pre-activation residual baseline improves the residual unit ordering and gives the strongest thin deep reference point. It is the right baseline for any CIFAR/SVHN residual-block redesign.

Highway networks provide the gated-shortcut predecessor. They show that very deep networks can be trained when information has carry paths, but the gates add learned control and are not the same as parameter-free residual identities.

Stochastic depth is not merely a competitor; it is a diagnostic. If randomly removing entire residual blocks during training can improve a very deep residual net, then depth alone is not being fully used.

Classic wider convolutional families such as VGG and Inception are relevant counterexamples to the thin-depth prior. They do not settle the residual design, but they show that substantial channel capacity was already a viable image-recognition strategy before the residual era.

## Evaluation Scaffold

The main small-image settings are CIFAR-10 and CIFAR-100, each with 50,000 training images and 10,000 test images at `32x32`, using horizontal flips and random crops from 4-pixel reflected padding. Mean/std normalization is needed for direct comparison with residual baselines; ZCA whitening appears in some experiments.

SVHN supplies the low-augmentation stress test: roughly 600,000 digit images, no augmentation, and only scaling to `[0,1]`. If batch normalization overfits without augmentation, the train-loss and test-error curves should reveal it.

ImageNet checks whether the conclusions transfer to larger images and bottleneck-style residual practice, using top-1/top-5 error and wall-clock training speed as practical constraints.

The scaffold to fill is:

```python
class ResidualBlock(nn.Module):
    # Known: pre-activation residual branch, identity/projection shortcut,
    # residual addition. Open: convolution list, channel multiplier, and
    # any regularizer placement.
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class Net(nn.Module):
    # Fixed stem, three residual groups, final BN/ReLU, global average pool, FC.
    # The block, per-group feature counts, and depth/width tradeoff are open.
    pass
```
