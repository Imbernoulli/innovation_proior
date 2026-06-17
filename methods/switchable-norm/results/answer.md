# Switchable Normalization (SN), distilled

Switchable Normalization is a normalization layer that, instead of committing to one
normalizer, computes instance-, layer-, and batch-statistics and normalizes with a *learned
convex combination* of them. Each layer carries six scalar control parameters; two softmaxes
turn them into importance weights — three for the means, three for the variances — and those
weights are trained end-to-end by backprop alongside the network. So each normalization layer
learns, from data, how much to lean on each kind of statistics, per layer, jointly with the
filters.

## Problem it solves

Deep networks put a normalization layer after every convolution, but the field's practice is to
pick one normalizer (batch / instance / layer) by hand and stamp it into every layer. The best
choice depends on the task, the architecture, and — sharply — the batch size, and it plausibly
differs across layers of one network. A single global hand-made choice is cumbersome and
suboptimal; in particular batch-statistic normalization degrades as the per-GPU sample count
shrinks and fails to converge in the reported one-sample-per-GPU setting.

## Key idea

Write every normalizer in one common form and note they differ *only* in which pixels they pool
statistics over. For input `h ∈ R^{N×C×H×W}`:

- Instance (IN): pool over `(H,W)` per sample & channel — `μ_in, σ²_in ∈ R^{N×C}`.
- Layer (LN): pool over `(C,H,W)` per sample — `μ_ln, σ²_ln ∈ R^{N}`.
- Batch (BN): pool over `(N,H,W)` per channel — `μ_bn, σ²_bn ∈ R^{C}`.

SN normalizes with a weighted blend of all three:

```
ĥ_{ncij} = γ · (h_{ncij} − Σ_k w_k μ_k) / sqrt(Σ_k w'_k σ²_k + ε) + β,     k ∈ {in, ln, bn}
w_k  = softmax(λ_k)   over {in, ln, bn}          (weights for the means)
w'_k = softmax(λ'_k)  over {in, ln, bn}          (weights for the variances)
```

So `Σ_k w_k = 1`, `Σ_k w'_k = 1`, all weights in `[0,1]`. Six control parameters
`λ_in,λ_ln,λ_bn,λ'_in,λ'_ln,λ'_bn` per layer, shared across channels: only `6` parameters on top
of the `2C` affine.

Why each choice:
- **Softmax of free `λ`** gives a differentiable convex combination on the simplex (weights ≥ 0,
  sum to 1) with no projection step — so the blend stays a genuine averaged statistic (denominator
  is a convex combination of variances, hence positive and bounded) and `λ` trains by ordinary
  backprop.
- **Separate weights for means and variances** because a minibatch variance estimate is noisier
  than a minibatch mean estimate (and the variance sits under the square root); the layer can
  down-weight a noisy variance without discarding a usable mean.
- **Channel-shared weights** keep the parameter count negligible (per-channel/group is a noted
  extension).

## Computation reuse — keeps cost at O(NCHW)

The three statistics are nested averages of the per-instance ones, so compute `μ_in, σ²_in` once
and derive the rest by the **law of total variance** (total variance = mean within-group
variance + variance of group means):

```
μ_in = (1/HW) Σ_ij h_{ncij}              σ²_in = (1/HW) Σ_ij (h_{ncij} − μ_in)²
μ_ln = (1/C) Σ_c μ_in                    σ²_ln = (1/C) Σ_c (σ²_in + μ_in²) − μ_ln²
μ_bn = (1/N) Σ_n μ_in                    σ²_bn = (1/N) Σ_n (σ²_in + μ_in²) − μ_bn²
```

`σ²_in + μ_in²` is the per-channel second raw moment; averaging it and subtracting the squared
overall mean recovers the variance over the bigger pooling set. Total cost `O(NCHW)`, the same as
a single normalizer.

## Why it works (weight-space / geometric reading)

As a simplifying weight-space comparison, reparameterize a filter as direction ×
length (weight normalization, `v·(wᵢᵀx)/‖wᵢ‖₂`). For a zero-mean unit-variance patch `x`, the
three normalizers are written as:

```
ĥ_in = γ (wᵢᵀx)/‖wᵢ‖₂ + β                                       (free length)
ĥ_bn = γ (wᵢᵀx)/‖wᵢ‖₂ + β,  s.t. γ ≤ v                          (length regularized → shorter filter)
ĥ_ln = γ (wᵢᵀx)/(‖wᵢ‖₂ + Σ_{j≠i}‖wⱼ‖₂) + β                      (least constrained → can take γ > v)
```

BN's `γ ≤ v` arises because treating the random batch statistics in the expected loss yields
population normalization plus an adaptive `ζ(h)γ²` regularizer — generalization at the price of
batch noise. Combining,

```
ĥ_sn = γ (wᵢᵀx)/(‖wᵢ‖₂ + w_ln Σ_{j≠i}‖wⱼ‖₂) + β,   s.t. w_bn γ ≤ v
```

`w_ln` slides toward LN's learning ability; `w_bn` sets the strength of the length
regularization. SN is a continuous dial between **learning ability** and **generalization**.
This explains small-batch behavior: when the batch is small BN's `γ²` regularization is driven
by noise, so gradient descent lowers `w_bn` and raises `w_ln`. At `N=1`, the training statistic
for BN is the same spatial statistic as IN, so the batch branch can be removed and the layer
uses the IN/LN switch.

## Joint single-set training (not bilevel)

`λ` and the filters are optimized together by backprop on the *same* training set — unlike
architecture search, which uses a train/val split to stop the control parameters overfitting.
Safe here because the switched normalizers do not differ in capacity (same affine, same stats)
and two of them *add* regularization, so choosing normalizers improves generalization rather
than memorizing.

## Backprop (for frameworks without autodiff)

Standard normalization backward plus softmax-Jacobian gradients into the control parameters.
With ĥ = γh̃ + β, h̃ = (h−μ)/sqrt(σ²+ε):

```
∂L/∂h̃ = (∂L/∂ĥ)·γ
∂L/∂σ² = −1/(2(σ²+ε)) Σ_ij (∂L/∂h̃)·h̃
∂L/∂μ  = −1/sqrt(σ²+ε) Σ_ij (∂L/∂h̃)
∂L/∂h  = (∂L/∂h̃)/sqrt(σ²+ε)
       + [2w_in(h−μ_in)/(HW) ∂L/∂σ² + 2w_ln(h−μ_ln)/(CHW) Σ_c(∂L/∂σ²)_c + 2w_bn(h−μ_bn)/(NHW) Σ_n(∂L/∂σ²)_n]
       + [w_in/(HW) ∂L/∂μ + w_ln/(CHW) Σ_c(∂L/∂μ)_c + w_bn/(NHW) Σ_n(∂L/∂μ)_n]
∂L/∂γ  = Σ_nij (∂L/∂ĥ)·h̃ ,   ∂L/∂β = Σ_nij ∂L/∂ĥ
∂L/∂λ_in = w_in(1−w_in) Σ_nc[(∂L/∂μ)μ_in]
         − w_in w_ln  Σ_nc[(∂L/∂μ)μ_ln]
         − w_in w_bn  Σ_nc[(∂L/∂μ)μ_bn]                      (cyclic for λ_ln, λ_bn)
∂L/∂λ'_in = w'_in(1−w'_in) Σ_nc[(∂L/∂σ²)σ²_in]
          − w'_in w'_ln  Σ_nc[(∂L/∂σ²)σ²_ln]
          − w'_in w'_bn  Σ_nc[(∂L/∂σ²)σ²_bn]                 (cyclic for λ'_ln, λ'_bn)
```

The gradients are exactly the softmax Jacobian `∂w_k/∂λ_z = w_k(δ_kz − w_z)` chained through
`μ = Σ w_k μ_k` and `σ² = Σ w'_k σ²_k`. Tying the mean and variance weights would be more
compact; with tied weights the mean and variance terms add into a single combined gradient.

## Inference

IN and LN are recomputed per sample at test. For BN, freeze a population estimate after training.
Two options: a moving average accumulated during training, or **batch average** — freeze the
network and SN weights, push some training minibatches, and average the per-minibatch BN
means/variances. Batch average uses the final settled network for every sample, so it is less
biased and converges faster and more stably; a small number of samples suffices. Default `ε=1e-5`.

## Working code

This block mirrors the released `switchablenorms/Switchable-Normalization` PyTorch implementation
for `SwitchNorm2d`, including the `using_bn` switch, `last_gamma`, `momentum=0.9`, running
buffers, and the reuse path from instance statistics. The shorter scaffold implementation below
instead spells out the statistics directly and therefore sets `unbiased=False` for the
population-variance denominator `1/|I|`.

```python
import torch
import torch.nn as nn


class SwitchNorm2d(nn.Module):
    """Switchable Normalization for [N, C, H, W] feature maps. Drop-in replacement for
    BatchNorm2d. Each layer learns, via softmax importance weights, how much to lean on
    instance (per-sample, per-channel), layer (per-sample, all-channel), and batch
    (per-channel, across-batch) statistics. weight=gamma, bias=beta."""

    def __init__(self, num_features, eps=1e-5, momentum=0.9,
                 using_moving_average=True, using_bn=True, last_gamma=False):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))   # gamma
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))    # beta
        n = 3 if using_bn else 2
        self.mean_weight = nn.Parameter(torch.ones(n))    # control params for the means
        self.var_weight = nn.Parameter(torch.ones(n))     # control params for the variances
        if using_bn:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))
        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        self.weight.data.fill_(0 if self.last_gamma else 1)
        self.bias.data.zero_()

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(x.dim()))
        N, C, H, W = x.size()
        x = x.view(N, C, -1)

        # instance statistics (one pass), then derive layer/batch by law of total variance
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)
        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2                       # per-channel 2nd raw moment
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum).add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum).add_((1 - self.momentum) * var_bn.data)
                else:                                       # batch-average accumulation
                    self.running_mean.add_(mean_bn.data)
                    # Divide accumulated buffers by the number of minibatches before eval.
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = self.running_mean
                var_bn = self.running_var

        mean_w = torch.softmax(self.mean_weight, 0)         # -> simplex, differentiable
        var_w = torch.softmax(self.var_weight, 0)

        if self.using_bn:
            mean = mean_w[0] * mean_in + mean_w[1] * mean_ln + mean_w[2] * mean_bn
            var = var_w[0] * var_in + var_w[1] * var_ln + var_w[2] * var_bn
        else:
            mean = mean_w[0] * mean_in + mean_w[1] * mean_ln
            var = var_w[0] * var_in + var_w[1] * var_ln

        x = (x - mean) / (var + self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias
```

Minimal drop-in form (direct statistics, no running stats — recompute all three from `x`
each forward, statistics matched to the `[B,C,H,W]` axes):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomNorm(nn.Module):
    """Switchable Normalization, drop-in for BatchNorm2d. CustomNorm(num_features)."""

    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.ones(num_features))    # gamma
        self.bias = nn.Parameter(torch.zeros(num_features))     # beta
        self.mean_weight = nn.Parameter(torch.ones(3))          # lambda  (in, ln, bn)
        self.var_weight = nn.Parameter(torch.ones(3))           # lambda' (in, ln, bn)

    def forward(self, x):                                       # x: [B, C, H, W]
        mean_w = F.softmax(self.mean_weight, dim=0)
        var_w = F.softmax(self.var_weight, dim=0)
        mean_in = x.mean(dim=(2, 3), keepdim=True)              # over (H, W)
        var_in = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        mean_ln = x.mean(dim=(1, 2, 3), keepdim=True)           # over (C, H, W)
        var_ln = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
        mean_bn = x.mean(dim=(0, 2, 3), keepdim=True)           # over (B, H, W)
        var_bn = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
        mean = mean_w[0] * mean_in + mean_w[1] * mean_ln + mean_w[2] * mean_bn
        var = var_w[0] * var_in + var_w[1] * var_ln + var_w[2] * var_bn
        x_norm = (x - mean) / (var + self.eps).sqrt()
        return x_norm * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
```

## Variants

- **Sparse SN:** apply `argmax` to each layer's control parameters after training so each layer
  keeps a single normalizer.
- **Group SN:** split channels into groups and give each group its own softmax — more switch
  capacity at the cost of more control parameters.
