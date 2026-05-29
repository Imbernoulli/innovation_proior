# Real NVP — method summary

## Problem

Learn an exact probabilistic model p(x) of high-dimensional continuous data (natural
images) that simultaneously supports (1) exact log-likelihood evaluation, (2) exact and
*efficient, parallel* sampling, (3) exact and efficient inference of latents, and (4) a
usable, semantically organized latent space. Existing families each give only a subset:
undirected models need MCMC/AIS (no exact likelihood or sampling); VAEs optimize a bound
with approximate inference and a fixed L2 reconstruction cost (blurry samples);
autoregressive models are exact but sample sequentially with no latent space; GANs give
sharp samples but no likelihood and no encoder.

## Key idea

Model the data as a **bijection** f: x → z (with g = f⁻¹) into a simple latent prior, and
train by exact maximum likelihood via the change-of-variables formula:

  log p_X(x) = log p_Z(f(x)) + log |det( ∂f(x)/∂xᵀ )|.

The only obstacle is the Jacobian determinant (O(D³) in general). Real NVP — *real-valued
non-volume-preserving* transformations — makes it cheap by building f from **affine
coupling layers** whose Jacobian is triangular by construction.

**Affine coupling layer.** Split x into two parts; copy one part, affinely transform the
other conditioned on the copied part:

  y_{1:d}   = x_{1:d}
  y_{d+1:D} = x_{d+1:D} ⊙ exp(s(x_{1:d})) + t(x_{1:d}),

with s, t: R^d → R^{D−d} arbitrary networks. The Jacobian is lower-triangular,

  ∂y/∂xᵀ = [[ I_d, 0 ], [ ∂y_{d+1:D}/∂x_{1:d}ᵀ, diag(exp(s(x_{1:d}))) ]],

so its log-determinant is simply

  log|det| = Σ_j s(x_{1:d})_j

— independent of the Jacobians of s and t, which may therefore be arbitrarily complex. The
inverse needs no inverse of s or t:

  x_{1:d}   = y_{1:d},   x_{d+1:D} = ( y_{d+1:D} − t(y_{1:d}) ) ⊙ exp( −s(y_{1:d}) ).

Forward and inverse cost the same → sampling is as fast as inference. Because det = exp(Σ s)
≠ 1 in general, each layer is non-volume-preserving (the key advance over additive,
volume-preserving coupling) and can reallocate probability mass.

**Composition.** Stacking is free: determinants multiply (log-dets add) and inverses
reverse, (f_b∘f_a)⁻¹ = f_a⁻¹∘f_b⁻¹. Alternate the partition so every coordinate gets
transformed.

**Masking for images.** Implement the partition with a binary mask b:

  y = b ⊙ x + (1 − b) ⊙ ( x ⊙ exp(s(b⊙x)) + t(b⊙x) ),

using two masks: a **spatial checkerboard** (b=1 where i+j is odd) and a **channel-wise**
mask (b=1 on the first half of channels). s, t are rectified residual conv nets. For
numerical stability s = (learned per-channel scale)·tanh(raw output), wrapped in weight
norm, so the raw scale is bounded while the learned range grows under explicit control.

**Multi-scale architecture.** A **squeeze** reshapes s×s×c → (s/2)×(s/2)×4c (absolute
determinant 1), trading space for channels and enabling channel-wise masking. Per scale: 3
checkerboard couplings → squeeze → 3 channel-wise couplings; then **factor out** half the
dimensions as finished latents and recurse on the rest:

  h⁽⁰⁾ = x;  (z⁽ⁱ⁺¹⁾, h⁽ⁱ⁺¹⁾) = f⁽ⁱ⁺¹⁾(h⁽ⁱ⁾);  z⁽ᴸ⁾ = f⁽ᴸ⁾(h⁽ᴸ⁻¹⁾);  z = (z⁽¹⁾,…,z⁽ᴸ⁾).

This cuts compute/memory/parameters, distributes the loss (deep-supervision effect), and
builds a coarse-to-fine latent hierarchy. Hidden width in s, t doubles after each squeeze.

**Batch norm in the flow.** Batch norm is a per-dimension affine map, so it is just another
bijector: the forward normalization x ↦ (x−μ̃)/√(σ̃²+ε) has log-det −½ Σ_i
log(σ̃_i²+ε), while the inverse x ↦ x√(σ̃²+ε)+μ̃ has log-det +½ Σ_i log(σ̃_i²+ε).
A running-average variant (μ̃_{t+1}=ρμ̃_t+(1−ρ)μ̂_t, likewise σ̃²; backprop only through
current-batch stats) makes it robust at small batch size. This stabilizes the exp(s)
objective and lets the coupling stack go deep.

**Preprocessing and prior.** For normalized input x ∈ [0,1], dequantize the underlying
integer level as r = (255x+u)/256, u~U[0,1], then constrain and logit-transform
v = α+(1−2α)r with α=.05, y=logit(v). The preprocessing log-det is
softplus(y)+softplus(−y)+log(1−2α) per dimension. Prior p_Z is an isotropic unit
Gaussian. The likelihood used for discrete data is log p_Z(z) + sldj − D·log k with
k=256, so bits/dim is −[log p_Z(z)+sldj−D·log k]/(D·log 2). Optimize with Adam.

## Working code

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WNConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=True):
        super().__init__()
        self.conv = nn.utils.weight_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias))

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.in_norm = nn.BatchNorm2d(channels)
        self.in_conv = WNConv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.out_norm = nn.BatchNorm2d(channels)
        self.out_conv = WNConv2d(channels, channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        skip = x
        x = self.in_conv(F.relu(self.in_norm(x)))
        x = self.out_conv(F.relu(self.out_norm(x)))
        return x + skip


class ResNet(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, num_blocks,
                 kernel_size=3, padding=1, double_after_norm=False):
        super().__init__()
        self.in_norm = nn.BatchNorm2d(in_ch)
        self.double_after_norm = double_after_norm
        self.in_conv = WNConv2d(2 * in_ch, mid_ch, kernel_size, padding, bias=True)
        self.in_skip = WNConv2d(mid_ch, mid_ch, kernel_size=1, padding=0, bias=True)
        self.blocks = nn.ModuleList([ResidualBlock(mid_ch) for _ in range(num_blocks)])
        self.skips = nn.ModuleList([
            WNConv2d(mid_ch, mid_ch, kernel_size=1, padding=0, bias=True)
            for _ in range(num_blocks)
        ])
        self.out_norm = nn.BatchNorm2d(mid_ch)
        self.out_conv = WNConv2d(mid_ch, out_ch, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x = self.in_norm(x)
        if self.double_after_norm:
            x = 2. * x
        x = self.in_conv(F.relu(torch.cat((x, -x), dim=1)))
        x_skip = self.in_skip(x)
        for block, skip in zip(self.blocks, self.skips):
            x = block(x)
            x_skip = x_skip + skip(x)
        return self.out_conv(F.relu(self.out_norm(x_skip)))


def squeeze_2x2(x, reverse=False, alt_order=False):
    if reverse:
        b, c4, h, w = x.size()
        c = c4 // 4
        if alt_order:
            x = x.view(b, 4, c, h, w).permute(0, 2, 1, 3, 4)
            x = x[:, :, [0, 2, 3, 1]].contiguous().view(b, c4, h, w)
        return x.view(b, c, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).contiguous().view(
            b, c, 2 * h, 2 * w)

    b, c, h, w = x.size()
    x = x.view(b, c, h // 2, 2, w // 2, 2)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, 4 * c, h // 2, w // 2)
    if alt_order:
        x = x.view(b, c, 4, h // 2, w // 2)
        x = x[:, :, [0, 3, 1, 2]].permute(0, 2, 1, 3, 4).contiguous().view(
            b, 4 * c, h // 2, w // 2)
    return x


def checkerboard_mask(h, w, reverse=False, device=None):
    cb = [[((i % 2) + j) % 2 for j in range(w)] for i in range(h)]
    mask = torch.tensor(cb, dtype=torch.float32, device=device)
    if reverse:
        mask = 1 - mask
    return mask.view(1, 1, h, w)


class Rescale(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels, 1, 1))

    def forward(self, x):
        return self.weight * x


class FlowBatchNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-4, rho=0.99):
        super().__init__()
        self.eps = eps
        self.rho = rho
        self.register_buffer("running_mean", torch.zeros(1, num_channels, 1, 1))
        self.register_buffer("running_var", torch.ones(1, num_channels, 1, 1))

    def _stats(self, x):
        if self.training:
            batch_mean = x.mean(dim=(0, 2, 3), keepdim=True)
            batch_var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
            mean = self.rho * self.running_mean + (1. - self.rho) * batch_mean
            var = self.rho * self.running_var + (1. - self.rho) * batch_var
            with torch.no_grad():
                self.running_mean.copy_(mean.detach())
                self.running_var.copy_(var.detach())
            return mean, var
        return self.running_mean, self.running_var

    def forward(self, x, sldj, reverse=False):
        _, _, h, w = x.size()
        mean, var = (self.running_mean, self.running_var) if reverse else self._stats(x)
        log_var = torch.log(var + self.eps)
        if reverse:
            x = x * torch.exp(0.5 * log_var) + mean
            if sldj is not None:
                sldj = sldj + 0.5 * h * w * log_var.view(-1).sum()
            return x, sldj
        x = (x - mean) * torch.exp(-0.5 * log_var)
        if sldj is not None:
            sldj = sldj - 0.5 * h * w * log_var.view(-1).sum()
        return x, sldj


class CouplingLayer(nn.Module):
    """Affine coupling: y_change = x_change * exp(s) + t."""
    def __init__(self, in_channels, mid_channels, num_blocks, channel_wise, reverse_mask):
        super().__init__()
        self.channel_wise = channel_wise
        self.reverse_mask = reverse_mask
        cond_in = in_channels // 2 if channel_wise else in_channels
        self.st_net = ResNet(cond_in, mid_channels, 2 * cond_in, num_blocks,
                             double_after_norm=not channel_wise)
        self.rescale = nn.utils.weight_norm(Rescale(cond_in))
        self.flow_bn = FlowBatchNorm(in_channels)

    def forward(self, x, sldj, reverse=False):
        if reverse:
            x, sldj = self.flow_bn(x, sldj, reverse=True)

        if not self.channel_wise:                              # spatial checkerboard
            b = checkerboard_mask(x.size(2), x.size(3), self.reverse_mask, x.device)
            s, t = self.st_net(x * b).chunk(2, dim=1)
            s = self.rescale(torch.tanh(s))                    # keep exp(s) numerically sane
            s, t = s * (1 - b), t * (1 - b)
            if reverse:
                x = (x - t) * s.mul(-1).exp()
                if sldj is not None:
                    sldj = sldj - s.view(s.size(0), -1).sum(-1)
            else:
                x = x * s.exp() + t
                sldj = sldj + s.view(s.size(0), -1).sum(-1)
        else:                                                  # channel-wise split
            if self.reverse_mask:
                x_id, x_change = x.chunk(2, dim=1)
            else:
                x_change, x_id = x.chunk(2, dim=1)
            s, t = self.st_net(x_id).chunk(2, dim=1)
            s = self.rescale(torch.tanh(s))
            if reverse:
                x_change = (x_change - t) * s.mul(-1).exp()
                if sldj is not None:
                    sldj = sldj - s.view(s.size(0), -1).sum(-1)
            else:
                x_change = x_change * s.exp() + t
                sldj = sldj + s.view(s.size(0), -1).sum(-1)
            x = (torch.cat((x_id, x_change), dim=1) if self.reverse_mask
                 else torch.cat((x_change, x_id), dim=1))
        if not reverse:
            x, sldj = self.flow_bn(x, sldj, reverse=False)
        return x, sldj


class _RealNVP(nn.Module):
    """One scale: 3 checkerboard couplings -> squeeze -> 3 channel couplings ->
    squeeze+split -> recurse on half the dims."""
    def __init__(self, scale_idx, num_scales, in_channels, mid_channels, num_blocks):
        super().__init__()
        self.is_last = scale_idx == num_scales - 1
        self.in_couplings = nn.ModuleList([
            CouplingLayer(in_channels, mid_channels, num_blocks, False, False),
            CouplingLayer(in_channels, mid_channels, num_blocks, False, True),
            CouplingLayer(in_channels, mid_channels, num_blocks, False, False),
        ])
        if self.is_last:
            self.in_couplings.append(
                CouplingLayer(in_channels, mid_channels, num_blocks, False, True))
        else:
            self.out_couplings = nn.ModuleList([
                CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, True, False),
                CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, True, True),
                CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, True, False),
            ])
            self.next = _RealNVP(scale_idx + 1, num_scales,
                                 2 * in_channels, 2 * mid_channels, num_blocks)

    def forward(self, x, sldj, reverse=False):
        if not reverse:
            for c in self.in_couplings:
                x, sldj = c(x, sldj, reverse)
            if not self.is_last:
                x = squeeze_2x2(x)
                for c in self.out_couplings:
                    x, sldj = c(x, sldj, reverse)
                x = squeeze_2x2(x, reverse=True)
                x = squeeze_2x2(x, alt_order=True)
                x, x_split = x.chunk(2, dim=1)
                x, sldj = self.next(x, sldj, reverse)
                x = torch.cat((x, x_split), dim=1)
                x = squeeze_2x2(x, reverse=True, alt_order=True)
        else:
            if not self.is_last:
                x = squeeze_2x2(x, alt_order=True)
                x, x_split = x.chunk(2, dim=1)
                x, sldj = self.next(x, sldj, reverse)
                x = torch.cat((x, x_split), dim=1)
                x = squeeze_2x2(x, reverse=True, alt_order=True)
                x = squeeze_2x2(x)
                for c in reversed(self.out_couplings):
                    x, sldj = c(x, sldj, reverse)
                x = squeeze_2x2(x, reverse=True)
            for c in reversed(self.in_couplings):
                x, sldj = c(x, sldj, reverse)
        return x, sldj


class RealNVP(nn.Module):
    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8):
        super().__init__()
        self.register_buffer('data_constraint', torch.tensor([0.9]))
        self.flows = _RealNVP(0, num_scales, in_channels, mid_channels, num_blocks)

    def _preprocess(self, x):
        y = (x * 255. + torch.rand_like(x)) / 256.            # dequantize
        y = (2 * y - 1) * self.data_constraint
        y = (y + 1) / 2
        y = y.log() - (1. - y).log()                          # logit
        ldj = F.softplus(y) + F.softplus(-y) + self.data_constraint.log()
        return y, ldj.view(ldj.size(0), -1).sum(-1)

    def _postprocess(self, y):
        x = y.sigmoid()
        x = (2. * x - 1.) / self.data_constraint
        return ((x + 1.) / 2.).clamp(0., 1.)

    def forward(self, x, reverse=False):
        sldj = None
        if not reverse:
            x, sldj = self._preprocess(x)
        return self.flows(x, sldj, reverse)

    def sample(self, z):
        y, _ = self.forward(z, reverse=True)
        return self._postprocess(y)


class RealNVPLoss(nn.Module):
    """Change-of-variables NLL with an isotropic unit-Gaussian prior."""
    def __init__(self, k=256):
        super().__init__()
        self.k = k

    def log_likelihood(self, z, sldj):
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        prior_ll = (prior_ll.view(z.size(0), -1).sum(-1)
                    - np.log(self.k) * np.prod(z.size()[1:]))
        return prior_ll + sldj

    def forward(self, z, sldj):
        return -self.log_likelihood(z, sldj).mean()

    def bits_per_dim(self, z, sldj):
        dims = np.prod(z.size()[1:])
        return (-self.log_likelihood(z, sldj) / (dims * np.log(2))).mean()
```

Training is a standard loop: `z, sldj = model(x); loss = RealNVPLoss()(z, sldj);
loss.backward()` with Adam. For reporting, `RealNVPLoss().bits_per_dim(z, sldj)` applies
the same −D·log k discrete correction and divides by D·log 2. Sampling draws `z ~ N(0, I)`
and calls `model.sample(z)`. Inference is the same forward pass that yields `z`.
