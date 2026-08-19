## Research question

Redesign a residual image classifier's building block. Residual shortcuts, pre-activation ordering,
batch normalization, dropout, and CIFAR/SVHN training practice are all available, but the residual
block's capacity allocation — how many convolutions per block, what kernel-size pattern, how many
feature planes — and its regularization are still open. The only design freedom is the block
(`ResidualBlock`) and the per-group feature-plane counts of a fixed three-stage classifier scaffold;
everything else (SGD training loop, augmentation, dataset splits) is fixed.

## Prior art / Background / Baselines

- **Residual networks (He et al. 2015).** A block computes `x_{l+1} = x_l + F(x_l, W_l)`, an additive
  identity shortcut around a residual branch. This solves much of the degradation problem that made
  very deep plain nets hard to optimize and enables networks with hundreds to over a thousand layers.
  A "bottleneck" block variant (`1x1 -> 3x3 -> 1x1`) exists specifically to keep each block cheap so
  more of them can be stacked; it is a depth-maximizing device, not examined here.
- **Identity-mappings pre-activation residual unit.** Reordering the block from `conv -> BN -> ReLU`
  to `BN -> ReLU -> conv` trains faster and gets better results, and is the strongest available thin
  deep residual reference: on CIFAR-10/CIFAR-100 test error (%, mean/std normalization, batch size
  128), depth-110 (1.7M params) gets 6.37/-, depth-164 (1.7M) gets 5.46/24.33, depth-1001 (10.2M) gets
  4.92/22.71 (a parenthetical 4.64% exists for depth-1001 but was measured at batch size 64, not
  directly comparable to the batch-128 protocol used everywhere else). The same source also reports
  that on that architecture, dropout at ratio 0.5 applied directly on the identity shortcut fails to
  converge to a good solution (>20% CIFAR-10 test error, against a 6.61% ResNet-110 baseline under a
  slightly different training protocol used for that ablation) — multiplicative manipulations on a
  shortcut (scaling, gating, a 1x1 conv standing in for identity, dropout) are documented to hamper
  signal propagation because the shortcut is the most direct path through the stack.
- **Original (non-pre-activation) residual networks.** depth-110 (1.7M): 6.43/25.16 CIFAR-10/100;
  depth-1202 (10.2M): 7.93/27.82 — accuracy degrades past roughly a thousand layers even without an
  optimization failure, i.e. depth alone stops paying off well before it stops training.
  Highway networks are the gated-shortcut predecessor: very deep nets trainable via learned gates
  rather than a parameter-free identity.
- **Stochastic depth.** Randomly bypasses whole residual blocks during training and remains
  competitive with deterministically-deep residual nets: depth-110 (1.7M): 5.23/24.58; depth-1202
  (10.2M): 4.91/-. The fact that whole blocks can be skipped at random without destroying accuracy is
  evidence that a residual chain does not require every block to be individually essential — the
  identity shortcut means gradient can flow around a block's own weights, so nothing forces every
  block to learn something useful.
- **Self-account: batch norm vs. dropout as regularizers.** Prior to this work, on a CIFAR-10 VGG
  network built specifically to test whether batch normalization and dropout compete or complement
  each other, stacking the two together reached 92.44% accuracy; removing either one dropped it to
  91.4%. This is a measured result, not a claim.
- **Wide non-residual families.** VGG and Inception show that substantial channel width, not just
  depth, is a viable image-recognition strategy — and residual networks with a comparable number of
  parameters and depth to a VGG-style network are constructible (would just require k in the low
  8-10 range on the group-width convention below), a live comparison point.

## Fixed machinery

The starting architecture is the pre-activation residual template. The classifier scaffold has one
`3x3` stem convolution (16 channels, fixed), then three residual groups at decreasing spatial
resolution, then final batch normalization, ReLU, global average pooling, and a fully connected
classifier. Downsampling happens at the first block of the second and third groups. Shortcut
connections are identities when input/output shape matches and `1x1`, stride-matched projections
when it does not. The code base has ordinary convolution, batch normalization, ReLU, dropout,
cross-entropy, SGD with Nesterov momentum, weight decay, and step learning-rate decay, and can
instantiate a residual block repeatedly — the block's internal convolution pattern, feature-plane
count, and regularizer placement are the variables to decide.

## Open design axes

There are three direct ways to increase a residual block's representational power: add more
convolutions inside a block, increase the number of feature planes, or increase spatial kernel size.
The kernel-size option is constrained by the VGG/Inception evidence that stacked small filters are
effective, so filters larger than `3x3` are not the interesting first move. Two knobs remain: let `l`
denote the number of convolutions inside a block and `M` denote the specific list of kernel sizes used
(so the basic block is `M=(3,3)`, i.e. `l=2`); let `k` denote a widening multiplier on the feature-plane
counts of the three residual groups (`k=1` recovers the thin baseline). The question is how to
allocate a fixed training budget between more blocks, more convolutions per block, and more channels
per convolution — and, separately, where regularization should be added once channel count grows.

Regularization is unresolved. Batch normalization regularizes, but its effect is entangled with data
augmentation and, per the self-account above, does not fully substitute for dropout on at least one
prior architecture. Dropout is available; its placement relative to the residual branch and the
identity shortcut, and whether it helps or hurts a wide residual block, is open — and the identity-
mappings ablation above is a specific warning against putting it on the shortcut itself.

## Evaluation scaffold

The main small-image settings are CIFAR-10 and CIFAR-100, each 50,000 training / 10,000 test images
at `32x32`, with horizontal flips and random crops from 4-pixel reflected padding as the only
augmentation. Two CIFAR preprocessing conventions are both in active use in the surrounding
literature: ZCA whitening (following the maxout-network convention) and simple mean/std
normalization. The thin-residual-network baselines quoted above (original ResNet, pre-act-ResNet,
stochastic depth) all report their numbers under mean/std normalization; ZCA whitening is a separate,
not-yet-cross-validated-against-baselines convention. SVHN supplies the low-augmentation stress test:
roughly 600,000 digit images, no augmentation, only scaling to `[0,1]`. Training protocol is fixed
throughout: SGD with Nesterov momentum, cross-entropy loss, initial learning rate 0.1 (CIFAR) / 0.01
(SVHN), momentum 0.9, dampening 0, weight decay 5e-4, batch size 128; CIFAR trains 200 epochs with
the learning rate multiplied by 0.2 at epochs 60/120/160, SVHN trains 160 epochs with the learning
rate multiplied by 0.1 at epochs 80/120. Test error (%) is the reported metric throughout, generally
as the median over 5 runs where multiple runs are available.

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
