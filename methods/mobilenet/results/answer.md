# MobileNet, distilled

## Problem

Run accurate image recognition in real time on mobile/embedded hardware under tight compute, memory, and power budgets — and give a developer a simple way to trade accuracy for latency/size to hit any target budget. The obstacle is that a standard convolution's cost is multiplicative in four factors (kernel size, input channels, output channels, spatial resolution), so naive shrinking is costly.

## Key idea

Replace each standard convolution with a **depthwise separable convolution**, which factors the convolution's two entangled jobs — spatial filtering and channel mixing — into two cheaper steps:

1. **Depthwise convolution** — one `D_K × D_K` spatial filter per input channel, no cross-channel mixing. Cost `D_K · D_K · M · D_F · D_F`.
2. **Pointwise convolution** — a `1 × 1` convolution mixing the `M` channels into `N`. Cost `M · N · D_F · D_F`.

Each followed by BatchNorm + ReLU. Where a standard conv costs `D_K · D_K · M · N · D_F · D_F` (the channel-mixing factor `N` and the kernel factor `D_K²` *multiply*), the separable version costs the *sum* `D_K² · M · D_F² + M · N · D_F²`. The reduction ratio is

```
(D_K²·M·D_F² + M·N·D_F²) / (D_K²·M·N·D_F²) = 1/N + 1/D_K².
```

For `3 × 3` kernels (`D_K² = 9`) and the usual `N ≥ 64`, this is ≈ `1/9`: **8–9× less computation** at a small accuracy cost. Almost all remaining compute sits in the `1 × 1` convolutions, which are dense matrix multiplies (GEMM, no `im2col`), so the savings translate to real latency.

## Architecture

- First layer: a **full** `3 × 3` conv, `3 → 32` channels, stride 2 (the input has only 3 channels — nothing to separate yet).
- Body: 13 depthwise-separable blocks. Channel schedule `32 → 64 → 128 → 128 → 256 → 256 → 512 → (×6 at 512) → 1024 → 1024`. Downsampling is folded into stride-2 depthwise convs: `224 → 112 → 56 → 28 → 14 → 7`.
- Every conv (full, depthwise, pointwise) is followed by BatchNorm + ReLU, except the final fully-connected layer.
- Global `7 × 7` average pool → FC `1024 → 1000` → softmax. 28 layers counting depthwise and pointwise separately.

Two global hyper-parameters dial the budget:

- **Width multiplier `α ∈ (0, 1]`** (typical `1, 0.75, 0.5, 0.25`): scale every channel count `M → αM`, `N → αN`. The dominant pointwise term scales as `α²`, so compute and parameters drop ≈ `α²`. Defines a new, thinner model trained from scratch.
- **Resolution multiplier `ρ ∈ (0, 1]`**, set implicitly by input resolution (`224, 192, 160, 128`): scales every spatial map, so compute drops ≈ `ρ²`. Parameters are unaffected (they don't depend on feature-map size).

Combined per-layer cost: `D_K² · αM · (ρD_F)² + αM · αN · (ρD_F)²`.

Training: RMSProp with asynchronous gradient descent; *less* augmentation/regularization than large models (small models overfit less) — no auxiliary heads, no label smoothing, reduced crop distortion, and little/no weight decay on the depthwise filters (they hold very few parameters).

## Code

```python
import torch
import torch.nn as nn


def conv_bn_relu(in_ch, out_ch, kernel, stride, padding, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride, padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        # Depthwise: one 3x3 filter per channel (groups == in_ch). Stride downsamples here.
        self.depthwise = conv_bn_relu(in_ch, in_ch, 3, stride, 1, groups=in_ch)
        # Pointwise: 1x1 channel mixing; dense GEMM, carries ~all the compute.
        self.pointwise = conv_bn_relu(in_ch, out_ch, 1, 1, 0)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class MobileNet(nn.Module):
    # (output_channels, stride) per separable block.
    cfg = [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1),
           (512, 2), (512, 1), (512, 1), (512, 1), (512, 1), (512, 1),
           (1024, 2), (1024, 1)]

    def __init__(self, num_classes=1000, width_mult=1.0):
        super().__init__()

        def c(ch):
            return max(8, int(ch * width_mult))   # width multiplier alpha

        in_ch = c(32)
        self.stem = conv_bn_relu(3, in_ch, 3, stride=2, padding=1)  # full conv stem

        blocks = []
        for out_ch, stride in self.cfg:
            out_ch = c(out_ch)
            blocks.append(DepthwiseSeparableBlock(in_ch, out_ch, stride))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# Resolution multiplier rho is just the input image size: 224 / 192 / 160 / 128.
# e.g. MobileNet(width_mult=0.5) on 160x160 inputs == "0.5 MobileNet-160".
```

The reference TensorFlow-Slim implementation expresses the depthwise step as `separable_conv2d(..., filters=None, depth_multiplier=1)` and the pointwise step as `conv2d(..., [1, 1])`, with the width multiplier applied as `depth = max(int(d · α), 8)` per layer; the PyTorch form above uses a grouped convolution (`groups = in_channels`) for the depthwise step, which is the same operation.
