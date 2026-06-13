## Research question

A modern convolutional network inserts a normalization layer after (almost) every convolution: it
re-centers and re-scales the intermediate features so that very deep stacks train at all, and it speeds
and stabilizes optimization at the high learning rate the schedule uses. The single thing being designed
here is **that layer** — a drop-in replacement for `nn.BatchNorm2d` that takes a feature map `[B, C, H, W]`
and returns one of the same shape, standardizing it by some mean and variance and applying a learnable
affine. Everything else about the system is fixed: the backbones, the optimizer, the data pipeline, the
schedule, the loss. So whatever accuracy difference appears between rungs is attributable purely to the
normalization layer, and I cannot lean on re-tuning the learning rate or the schedule to rescue a clever
choice of statistics.

## Prior art before the first rung (the normalization lineage)

The layer the first rung reacts to is itself the resolution of a short line of normalization designs.
These are the baselines that precede the ladder; the editable contract below is the slot all of them fill.

- **Batch normalization (Ioffe & Szegedy 2015).** For each channel `c`, pool the mean and variance over
  the batch and the spatial axes — the set `{k : k_C = c}`, spanning `(N, H, W)` — standardize, then a
  per-channel affine `γ x̂ + β`. It eases optimization, lets very deep nets train, and the noise from
  random batch sampling acts as a mild regularizer. Gap: the statistic is estimated *over the batch*, so
  its quality degrades as the per-device batch shrinks (noisier, more biased mean/variance), and the layer
  must keep frozen running statistics for inference, so the function it computes at test time is not the
  one it computed during training — a train/test discrepancy that also breaks under distribution shift.
- **Instance normalization (Ulyanov, Vedaldi & Lempitsky 2016).** For each sample *and each channel*,
  pool over `(H, W)` only — `{k : k_N = i_N, k_C = i_C}`. Batch-independent, identical at train and test,
  and strong for style transfer because it discards per-image contrast/style. Gap for recognition: a
  channel is standardized using only its own spatial map, blind to every other channel, so it cannot use
  the dependence *between* channels at all, and it throws away the per-image global scale that often
  carries class-discriminative signal.
- **Layer normalization (Ba, Kiros & Hinton 2016).** For each sample, pool over *all* of `(C, H, W)` —
  `{k : k_N = i_N}`. Batch-independent, the standard in transformers. Gap for conv features: it forces
  every channel of a sample to share one mean and variance, i.e. it assumes the channels are
  interchangeable, but conv channels are different filters (edges, colors, textures) whose response
  distributions genuinely differ, so a single shared statistic over all of them is too coarse.

Each of these is a different choice of *which feature positions share one mean/variance*; the per-channel
affine and the standardization are identical across them. The ladder is a walk through that design space.

## The fixed substrate

The backbones, the training loop, and the data are frozen and must not be touched. Three evaluation
configurations: **ResNet-56 on CIFAR-100**, **ResNet-110 on CIFAR-100**, and **MobileNetV2 on
FashionMNIST** (grayscale repeated to 3 channels, resized to 32×32). The CIFAR ResNets are the
He-2016 CIFAR variant — a 3×3 stem into three stages of `BasicBlock`s at 16/32/64 channels, global
average pool, linear head — with a `CustomNorm` after every conv (stem, both convs in each block, and the
1×1 projection shortcut). MobileNetV2 is the width-1.0 CIFAR-adapted variant with a `CustomNorm` after
every conv inside each inverted-residual block. The optimizer is SGD (`lr=0.1`, `momentum=0.9`,
`weight_decay=5e-4`) under cosine annealing over 200 epochs, batch size 128, with `RandomCrop(32, pad=4)`
+ `RandomHorizontalFlip` augmentation and cross-entropy loss. Weight init is fixed: Kaiming on convs;
any 1-D normalization weight initialized to 1 and bias to 0. Because the batch is a comfortable 128 per
device, the small-batch failure of the batch statistic is *not* the pressure here — the pressure is which
statistics best fit these specific conv features under this specific schedule.

## The editable interface

Exactly one region is editable — the `CustomNorm` class in `pytorch-vision/custom_norm.py` (lines 31–45).
It must be a drop-in replacement for `nn.BatchNorm2d`: constructed as `CustomNorm(num_features)` where
`num_features` is the channel count `C`, taking `[B, C, H, W]` to `[B, C, H, W]`, numerically stable in
both train and eval. Inside it I may compute statistics over any combination of the batch, channel, and
spatial axes, add learnable affine parameters, group channels, mix several normalizations, or make the
normalization input-dependent — as long as the constructor signature and the tensor shape are preserved
and nothing else (backbones, activations, datasets, loss, optimizer) changes.

The starting point is the scaffold default: a plain `nn.BatchNorm2d`. Each rung replaces exactly this
class body and nothing else.

```python
# EDITABLE region of pytorch-vision/custom_norm.py (lines 31-45) -- default fill
class CustomNorm(nn.Module):
    """Custom normalization layer. Drop-in replacement for BatchNorm2d.

    Args:
        num_features: number of channels C
    Input: [B, C, H, W]
    Output: [B, C, H, W]
    """

    def __init__(self, num_features):
        super().__init__()
        self.norm = nn.BatchNorm2d(num_features)

    def forward(self, x):
        return self.norm(x)
```

## Evaluation settings

Three (architecture, dataset) configurations, each run on the single configured seed {42}, reporting
**best test accuracy (%, higher is better)** reached at any point during the 200-epoch run:
`test_acc_resnet56-cifar100`, `test_acc_resnet110-cifar100`, and `test_acc_mobilenetv2-fmnist`. The two
CIFAR-100 ResNets are the harder, primary targets (100 classes, deep stacks); the FashionMNIST run is
near its accuracy ceiling and moves little. The normalization module must preserve tensor shape, accept
the expected channel count, and remain numerically stable in train and eval.
