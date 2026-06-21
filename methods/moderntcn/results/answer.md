# ModernTCN, distilled

ModernTCN is a pure-convolution backbone for general time series analysis that revives convolution for
the five TS tasks by importing the modern-ConvNet recipe (large depthwise kernels + pointwise channel
mixing, à la ConvNeXt) and adapting it to the two channel-like axes of a multivariate series. One
backbone serves forecasting, imputation, anomaly detection, and classification by swapping only the head.
Introduced as "ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis" (Donghao
& Xue, ICLR 2024 Spotlight, https://openreview.net/forum?id=vpJMJerXHU; official code
https://github.com/luodhhh/ModernTCN).

## Problem it solves

Convolution had been written off for time series on the charge that 1-D temporal convs have too small an
effective receptive field (ERF) to model long-range dependence, ceding the benchmarks to
attention/MLP models. ModernTCN shows the charge was about *small* kernels and *old* conv blocks, not
convolution itself: a modernized conv design achieves a large ERF and matches/beats TimesNet and the
Transformer/MLP models at convolutional cost — and, unlike channel-independent forecasters, models
cross-variable dependence explicitly.

## Key idea

Separate three kinds of mixing into three operators, never conflated:

- **Time** — a **large depthwise** 1-D conv. Depthwise (one kernel per channel) makes a size-31/51 kernel
  affordable, giving each channel a large ERF in one layer. Trained as a **large + small re-param**
  (`ReparamLargeKernelConv`: large branch and small branch, each conv+BN, summed) so the wide kernel
  optimizes well and the fine local detail has a clean path; the two branches fuse to **one** equivalent
  large kernel at inference (`merge_kernel`, verified numerically exact).
- **Feature** — **ConvFFN1**: a 1×1 grouped conv (`groups = nvars`) mixing the `D` features within each
  variable (the inverted-bottleneck FFN, `D → r·D → D`, weights shared across variables).
- **Variable** — **ConvFFN2**: a 1×1 grouped conv (`groups = dmodel`) mixing across the `M` variables per
  feature — the explicit cross-variable operator channel-independent models drop.

A block is DWConv → BatchNorm → ConvFFN1 → ConvFFN2 → residual, on a 4-D tensor `[B, M, D, N]`. The stem
patch-embeds each variable independently (shared weights), stages with strided-conv downsampling compound
the ERF, and a flatten-and-project head reads off the task output. BatchNorm (not LayerNorm) because TS
outliers corrupt per-token statistics.

## Why it works

The large depthwise kernel + downsampling answers the ERF objection directly and cheaply; the re-param
small branch makes the wide kernel trainable from scratch; the two distinct grouped FFNs let cross-feature
and cross-variable mixing be controlled independently, and the cross-variable ConvFFN2 is exactly what
helps on multivariate problems whose signal is cross-channel covariance, where channel-independent and
head-only-mixing models stall. Shallow stacks and shared/grouped weights guard against overfitting on the
modest benchmark sizes.

## Hyperparameters (UEA classification defaults)

`patch_size=8`, `patch_stride=4`, `downsample_ratio=2`, `ffn_ratio=1`, `num_blocks=[1,1]` (2 stages),
`large_size=[31,29]`, `small_size=[5,5]`, `dims=[64,64]`, `dropout=0.1`; depthwise groups `=nvars·dmodel`,
ConvFFN1 `groups=nvars`, ConvFFN2 `groups=dmodel`; head `Linear(nvars · head_nf, num_class)`. Optimizer
per the Time-Series-Library classification protocol (RAdam, CrossEntropy, accuracy on UEA).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(in_ch, out_ch, kernel_size, stride, padding, groups, dilation=1, bias=False):
    seq = nn.Sequential()
    seq.add_module('conv', nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride,
                                     padding=padding, dilation=dilation, groups=groups, bias=bias))
    seq.add_module('bn', nn.BatchNorm1d(out_ch))
    return seq


def fuse_bn(conv, bn):
    kernel = conv.weight
    std = (bn.running_var + bn.eps).sqrt()
    t = (bn.weight / std).reshape(-1, 1, 1)
    return kernel * t, bn.bias - bn.running_mean * bn.weight / std


class ReparamLargeKernelConv(nn.Module):
    """Depthwise large-kernel temporal conv: large+small branches (conv+BN), summed; fuse at inference."""

    def __init__(self, channels, large_size, small_size, groups):
        super().__init__()
        self.large_size, self.small_size = large_size, small_size
        self.large = conv_bn(channels, channels, large_size, 1, large_size // 2, groups, bias=False)
        self.small = conv_bn(channels, channels, small_size, 1, small_size // 2, groups, bias=False)

    def forward(self, x):
        if hasattr(self, 'lkb_reparam'):                # merged single kernel (inference)
            return self.lkb_reparam(x)
        return self.large(x) + self.small(x)            # two branches (training)

    def merge_kernel(self):
        eq_k, eq_b = fuse_bn(self.large.conv, self.large.bn)
        sk, sb = fuse_bn(self.small.conv, self.small.bn)
        pad = (self.large_size - self.small_size) // 2
        eq_k, eq_b = eq_k + F.pad(sk, [pad, pad]), eq_b + sb
        merged = nn.Conv1d(self.large.conv.in_channels, self.large.conv.out_channels, self.large_size,
                           padding=self.large_size // 2, groups=self.large.conv.groups, bias=True)
        merged.weight.data, merged.bias.data = eq_k, eq_b
        self.lkb_reparam = merged
        self.__delattr__('large'); self.__delattr__('small')


class Block(nn.Module):
    def __init__(self, large_size, small_size, dmodel, dff, nvars, drop=0.1):
        super().__init__()
        self.dw = ReparamLargeKernelConv(nvars * dmodel, large_size, small_size, groups=nvars * dmodel)
        self.norm = nn.BatchNorm1d(dmodel)
        self.ffn1pw1 = nn.Conv1d(nvars * dmodel, nvars * dff, 1, groups=nvars)
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(nvars * dff, nvars * dmodel, 1, groups=nvars)
        self.ffn1drop1, self.ffn1drop2 = nn.Dropout(drop), nn.Dropout(drop)
        self.ffn2pw1 = nn.Conv1d(nvars * dmodel, nvars * dff, 1, groups=dmodel)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(nvars * dff, nvars * dmodel, 1, groups=dmodel)
        self.ffn2drop1, self.ffn2drop2 = nn.Dropout(drop), nn.Dropout(drop)

    def forward(self, x):                                # x: [B, M, D, N]
        inp = x
        B, M, D, N = x.shape
        x = self.dw(x.reshape(B, M * D, N))
        x = self.norm(x.reshape(B, M, D, N).reshape(B * M, D, N))
        x = x.reshape(B, M, D, N).reshape(B, M * D, N)
        x = self.ffn1drop2(self.ffn1pw2(self.ffn1act(self.ffn1drop1(self.ffn1pw1(x)))))
        x = x.reshape(B, M, D, N).permute(0, 2, 1, 3).reshape(B, D * M, N)
        x = self.ffn2drop2(self.ffn2pw2(self.ffn2act(self.ffn2drop1(self.ffn2pw1(x)))))
        x = x.reshape(B, D, M, N).permute(0, 2, 1, 3)
        return inp + x


class Stage(nn.Module):
    def __init__(self, ffn_ratio, num_blocks, large_size, small_size, dmodel, nvars, drop=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([Block(large_size, small_size, dmodel, dmodel * ffn_ratio,
                                           nvars, drop) for _ in range(num_blocks)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.nvars = configs.enc_in
        self.num_class = configs.num_class
        self.patch_size = getattr(configs, 'patch_size', 8)
        self.patch_stride = getattr(configs, 'patch_stride', 4)
        self.downsample_ratio = getattr(configs, 'downsample_ratio', 2)
        ffn_ratio = getattr(configs, 'ffn_ratio', 1)
        num_blocks = getattr(configs, 'num_blocks', [1, 1])
        large_size = getattr(configs, 'large_size', [31, 29])
        small_size = getattr(configs, 'small_size', [5, 5])
        dims = getattr(configs, 'dims', [64, 64])
        drop = getattr(configs, 'dropout', 0.1)
        self.num_stage = len(num_blocks)

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(
            nn.Conv1d(1, dims[0], self.patch_size, stride=self.patch_stride),
            nn.BatchNorm1d(dims[0])))
        for i in range(self.num_stage - 1):
            self.downsample_layers.append(nn.Sequential(
                nn.BatchNorm1d(dims[i]),
                nn.Conv1d(dims[i], dims[i + 1], self.downsample_ratio, stride=self.downsample_ratio)))

        self.stages = nn.ModuleList([
            Stage(ffn_ratio, num_blocks[s], large_size[s], small_size[s], dims[s], self.nvars, drop)
            for s in range(self.num_stage)])

        patch_num = self.seq_len // self.patch_stride
        d_model = dims[self.num_stage - 1]
        fold = self.downsample_ratio ** (self.num_stage - 1)
        self.head_nf = d_model * (patch_num // fold if patch_num % fold == 0 else patch_num // fold + 1)
        self.class_dropout = nn.Dropout(getattr(configs, 'class_dropout', 0.0))
        self.head_class = nn.Linear(self.nvars * self.head_nf, self.num_class)

    def forward_feature(self, x):                        # x: [B, M, L]
        B, M, L = x.shape
        x = x.unsqueeze(-2)
        for i in range(self.num_stage):
            B, M, D, N = x.shape
            x = x.reshape(B * M, D, N)
            if i == 0 and self.patch_size != self.patch_stride:
                pad = self.patch_size - self.patch_stride
                x = torch.cat([x, x[:, :, -1:].repeat(1, 1, pad)], dim=-1)
            elif i > 0 and N % self.downsample_ratio != 0:
                pad = self.downsample_ratio - (N % self.downsample_ratio)
                x = torch.cat([x, x[:, :, -pad:]], dim=-1)
            x = self.downsample_layers[i](x)
            _, D_, N_ = x.shape
            x = self.stages[i](x.reshape(B, M, D_, N_))
        return x

    def classification(self, x_enc, x_mark_enc):
        x = x_enc.permute(0, 2, 1)
        x = self.forward_feature(x)
        x = self.class_dropout(F.gelu(x))
        x = x.reshape(x.shape[0], -1)
        return self.head_class(x)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        return None
```
