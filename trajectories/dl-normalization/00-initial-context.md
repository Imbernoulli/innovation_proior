## Research question

Current convolutional networks insert a normalization layer after almost every convolution. The task is to design **that layer** — a drop-in replacement for `nn.BatchNorm2d` that takes a feature map `[B, C, H, W]` and returns one of the same shape, standardizing it by some mean and variance and applying a learnable affine. Everything else in the system is fixed: backbones, optimizer, data pipeline, schedule, and loss. Any accuracy difference between rungs is therefore attributable purely to the normalization layer.

## Prior art / Background / Baselines

- **Batch normalization (Ioffe & Szegedy 2015).** For each channel, it pools the mean and variance over the batch and spatial axes and applies a per-channel affine `γ x̂ + β`. Gap: the statistic is estimated over the batch, so quality degrades as the per-device batch shrinks and the layer must keep frozen running statistics for inference, creating a train-test discrepancy.

- **Instance normalization (Ulyanov, Vedaldi & Lempitsky 2016).** For each sample and each channel, it pools over the spatial axes and applies a per-channel affine. Gap: on standard recognition tasks it underperforms batch normalization, because standardizing each channel map independently discards the per-image global scale that carries useful signal for classification.

- **Layer normalization (Ba, Kiros & Hinton 2016).** For each sample, it pools over all channels and spatial positions and applies a per-channel affine. Gap: on convolutional features it lags batch normalization, because forcing every channel to share a single mean and variance ignores that different channels often have very different response distributions.

## Fixed substrate / Code framework

The backbones, training loop, and data are frozen. Three evaluations: **ResNet-56 on CIFAR-100**, **ResNet-110 on CIFAR-100**, and **MobileNetV2 on FashionMNIST** (grayscale repeated to 3 channels, resized to 32×32). The CIFAR ResNets use the He-2016 variant with a `CustomNorm` after every conv. MobileNetV2 uses the width-1.0 CIFAR-adapted variant with a `CustomNorm` after every conv in each inverted-residual block. The optimizer is SGD (`lr=0.1`, `momentum=0.9`, `weight_decay=5e-4`) with cosine annealing over 200 epochs, batch size 128, `RandomCrop(32, pad=4)` + `RandomHorizontalFlip`, and cross-entropy loss. Weight init is fixed: Kaiming on convs; normalization weights to 1 and biases to 0. Because the batch is 128 per device, the small-batch failure of the batch statistic is not the dominant pressure here.

## Editable interface

Exactly one region is editable — the `CustomNorm` class in `pytorch-vision/custom_norm.py` (lines 31–45). It must be a drop-in replacement for `nn.BatchNorm2d`: constructed as `CustomNorm(num_features)` where `num_features` is the channel count `C`, taking `[B, C, H, W]` to `[B, C, H, W]`, numerically stable in both train and eval. Inside it you may compute statistics over any combination of the batch, channel, and spatial axes, add learnable affine parameters, mix several normalizations, or make the normalization input-dependent — as long as the constructor signature and tensor shape are preserved and nothing else changes.

The starting point is the default scaffold below: a plain `nn.BatchNorm2d`. Each rung replaces exactly this class body and nothing else.

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

Three (architecture, dataset) configurations, each run on seed {42}, reporting **best test accuracy (%, higher is better)** reached during the 200-epoch run: `test_acc_resnet56-cifar100`, `test_acc_resnet110-cifar100`, and `test_acc_mobilenetv2-fmnist`. The two CIFAR-100 ResNets are the primary targets; the FashionMNIST run is near its ceiling and moves little. The module must preserve tensor shape, accept the expected channel count, and remain numerically stable in train and eval.
