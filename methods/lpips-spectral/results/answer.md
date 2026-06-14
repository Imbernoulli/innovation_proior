# LPIPS (+ spectral), distilled

LPIPS — Learned Perceptual Image Patch Similarity — is a perceptual distance between images:
push both images through a frozen ImageNet-pretrained backbone, unit-normalize each layer's
activations in the channel dimension, take the squared feature difference, weight it by a
learned non-negative per-channel vector (a no-bias `1x1` conv), average over space, and sum
over layers. The weights are calibrated to human two-alternative-forced-choice judgments.
Because the distance is differentiable, it doubles as a perceptual training loss. The
`spectral` extension stacks LPIPS with complementary spatial- and frequency-domain terms — a
finite-difference gradient `L1`, a multi-scale `L1`, and an FFT-amplitude `L1` — so the loss
supervises perceptual structure, edges, coarse layout, and the high-frequency spectrum at once.

## Problem it solves

Per-pixel `L2`/PSNR is not perceptual: it assumes pixel independence and spatial alignment,
so it is blind to structure and to small geometric ambiguity (blur is a tiny `L2` change, a
large perceptual one), and as a training loss it drives a predictor to the blurry mean of
plausible outputs. Hand-designed metrics (SSIM/FSIM) are shallow, fixed, and assume
registration, so they fail on geometric distortion. Feature-space "perceptual losses"
(Gatys 2015; Johnson 2016) work much better but are uncalibrated. The goal: a distance that
tracks human similarity across a wide distortion space (including CNN artifacts), and that
can be used as a differentiable perceptual training loss for image generation.

## Key idea

Measure distance in the activations of a generally-trained deep network, but make it
*perceptual by calibration*:

- **Channel unit-normalization** before comparing — divide each layer's activation by its
  per-location channel norm, so widely varying channel magnitudes do not dominate; what
  decides a channel's influence is a learned weight, not its raw scale.
- **Learned non-negative per-channel weights `w_l`** (a `1x1` conv applied to squared feature
  differences) — calibrate which feature channels are perceptual. `w_l = 1` reduces the metric
  to (averaged) cosine distance in feature space, i.e. the off-the-shelf perceptual loss;
  calibration improves on it. Non-negativity is required: a larger feature difference must
  never reduce the distance (project `w_l >= 0` each step).
- **BCE through a learned comparator**, not a fixed-margin rank loss — fit `w_l` to noisy,
  pairwise 2AFC judgments. Only a thin layer on top of a frozen representation is fit, so the
  metric borrows competence from a representation that learned about the world (it
  generalizes) rather than overfitting a free-form fit to the seen distortions.

The result tracks humans across architectures and even self-supervised training — perceptual
similarity is an emergent property of having learned useful structure — and beats `L2`, SSIM,
FSIM by large margins.

## Distance (LPIPS metric)

With `L` chosen layers, channel-normalized activations `yhat^l`, learned non-negative
per-channel weights `w_l`:

```
d(x, x0) = sum_l (1 / (H_l W_l)) * sum_{h,w} || w_l ⊙ ( yhat^l_{hw} - yhat0^l_{hw} ) ||_2^2
```

In the canonical implementation this is realized as: standardize the input (optionally map
`[0,1]`→`[-1,1]`, then a fixed per-channel affine), channel-normalize each layer's features,
square their differences, apply a no-bias `1x1` conv (the non-negative `w_l`), spatially
average, and sum over layers.

Calibration loss on triplets `(x, x0, x1, h)`, `h in {0,1}` the human choice, distances
`d0 = d(x,x0)`, `d1 = d(x,x1)`, learned comparator `G` (conceptually two small FC-ReLU layers
into a sigmoid; the canonical code feeds `G` the five relational features
`d0, d1, d0-d1, d0/(d1+eps), d1/(d0+eps)` through `1x1` convs):

```
L = - h log G(d0, d1) - (1 - h) log( 1 - G(d0, d1) ) ,   project w_l >= 0 each step.
```

## Spectral training-loss stack

Used as a training loss on a produced image vs target. In a velocity-based generator on the
linear path `z_t=(1-t)x+t*eps`, `v=eps-x`, the predicted velocity implies a denoised image
`x_hat = x_t - t*v_pred`; its error is exactly `x_hat - x = t*(v_target - v_pred)`, so the
image-space gradient back to `v_pred` is scaled by `t`. LPIPS is the primary perceptual
signal; complementary terms cover where it is blind:

- **Finite-difference gradient `L1`** (Gradient Difference Loss, Mathieu 2016, `alpha=1`):
  matches horizontal/vertical neighbor differences, forcing sharp edges where a feature loss
  tolerates soft ones.
- **Multi-scale `L1`** (Mathieu 2016 / LapSRN, Lai 2017): downsample-and-compare, supervising
  coarse structure the fine terms miss.
- **FFT-amplitude `L1`** (Fourier-space loss, Fuoli 2021): the high-frequency detail spatial
  losses diffusely underproduce is a concentrated amplitude deficit in the spectrum. Use
  `rfft2` (half spectrum suffices by Hermitian symmetry `F{x}_{u,v}=conj F{x}_{-u,-v}`),
  compare *amplitude* (energy) not phase (energy is the failure mode; amplitude is shift-
  tolerant). The clean amplitude loss is `L_F = (2/UV) sum_{u<U/2,v} | |Yhat|_{u,v}-|Y|_{u,v} |`;
  as an auxiliary term, a plain mean over the `rfft2` coefficients carries the same gradient
  direction with a fixed scale folded into the weight:
  `L_spec = mean( | rfft2(x_hat).abs() - rfft2(x).abs() | )`.

Per-sample objective, with velocity-MSE kept as the unscaled correctness anchor:

```
loss = || v_pred - v_target ||^2
     + perceptual_w * ( 0.5 * LPIPS(x_hat, x)     # primary perceptual: largest
                      + 0.3 * L_grad(x_hat, x)     # edges
                      + 0.2 * L_multi(x_hat, x)    # coarse structure
                      + 0.2 * L_spec(x_hat, x) )   # frequency spectrum
perceptual_w = (1 - t)^2 * 1[t > 0.1]             # concentrate on low-noise; gate small-t
x_hat        = x_t - t * v_pred  (clamped to [-1,1] before the perceptual terms)
```

The `0.5 > 0.3 > 0.2 = 0.2` ordering ranks centrality to perceptual quality (feature-space,
then edges, then the structural/spectral helpers); the four are the stack's own knobs (not
from the prior work), kept below the MSE anchor so they never pull the prediction off the
velocity target. The `(1-t)^2` schedule weights perceptual terms toward clean-ish samples
where `x_hat` is a faithful image; the `t > 0.1` gate drops the small-`t` region where the
auxiliary's velocity leverage (scaled by `t`) has shrunk to nearly nothing.

## Working code

LPIPS metric (faithful to the canonical implementation — channel-normalize, `1x1`-conv
learned weights on squared differences, spatial average, sum over layers):

```python
import torch
import torch.nn as nn


def normalize_tensor(x, eps=1e-10):
    norm = torch.sqrt((x ** 2).sum(dim=1, keepdim=True))   # channel-dim unit norm
    return x / (norm + eps)


def spatial_average(x):
    return x.mean([2, 3], keepdim=True)


class NetLinLayer(nn.Module):
    """Learned per-channel weight w_l as a 1x1 conv (no bias) = weighted channel sum."""
    def __init__(self, chn_in, use_dropout=False):
        super().__init__()
        layers = ([nn.Dropout()] if use_dropout else [])
        layers += [nn.Conv2d(chn_in, 1, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)


class LPIPS(nn.Module):
    def __init__(self, backbone, chns):
        super().__init__()
        self.net = backbone                          # frozen pretrained feature extractor
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.L = len(chns)
        self.lins = nn.ModuleList([NetLinLayer(c) for c in chns])

    def forward(self, in0, in1):                      # inputs in [-1, 1]
        outs0, outs1 = self.net(in0), self.net(in1)
        val = 0
        for kk in range(self.L):
            f0 = normalize_tensor(outs0[kk])
            f1 = normalize_tensor(outs1[kk])
            diff = (f0 - f1) ** 2                     # squared feature difference
            val = val + spatial_average(self.lins[kk].model(diff))
        return val                                    # d(in0, in1) = sum over layers


# Calibration: fit only w_l (and comparator G) to human 2AFC judgments
class Dist2Logit(nn.Module):
    """G over the five relational distance features (canonical Dist2LogitLayer)."""
    def __init__(self, chn_mid=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(5, chn_mid, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(chn_mid, chn_mid, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(chn_mid, 1, 1), nn.Sigmoid(),
        )

    def forward(self, d0, d1, eps=0.1):
        return self.model(torch.cat(
            (d0, d1, d0 - d1, d0 / (d1 + eps), d1 / (d0 + eps)), dim=1))


def calibrate(metric, comparator, triplet_loader, opt):
    bce = nn.BCELoss()
    for x, x0, x1, h in triplet_loader:
        d0, d1 = metric(x, x0), metric(x, x1)
        loss = bce(comparator(d0, d1), h)            # -h log G - (1-h) log(1-G)
        opt.zero_grad(); loss.backward(); opt.step()
        for lin in metric.lins:                       # project w_l >= 0
            for p in lin.parameters():
                p.data.clamp_(min=0)
```

Spectral training-loss stack (filling the loss slot of a velocity-based generator trainer,
using a calibrated `lpips_fn` and finite-difference / multiscale helpers):

```python
import torch


def lpips_spectral_auxiliary(v_pred, v_target, x, x_t, t, lpips_fn,
                             compute_gradient_loss, compute_multiscale_loss):
    B = v_pred.shape[0]
    t_img = t.reshape(B, *([1] * (v_pred.ndim - 1)))           # broadcast t over image dims
    t_flat = t_img.flatten()

    x_hat = x_t - t_img * v_pred                               # implied denoised image
    mask = t_flat > 0.1
    weight = ((1.0 - t_flat) ** 2) * mask.float()

    zeros = torch.zeros(B, device=v_pred.device, dtype=v_pred.dtype)
    loss_lpips, loss_grad = zeros.clone(), zeros.clone()
    loss_multi, loss_spec = zeros.clone(), zeros.clone()

    if mask.any():
        xh = x_hat[mask].clamp(-1, 1).float()
        xc = x[mask].clamp(-1, 1).float()
        loss_lpips[mask] = lpips_fn(xh, xc).view(-1).float()
        loss_grad[mask]  = compute_gradient_loss(xh, xc).view(-1).float()
        loss_multi[mask] = compute_multiscale_loss(xh, xc).view(-1).float()
        fh = torch.fft.rfft2(xh, dim=(-2, -1)).abs()           # half spectrum (Hermitian)
        fc = torch.fft.rfft2(xc, dim=(-2, -1)).abs()
        loss_spec[mask] = (fh - fc).abs().mean(dim=(1, 2, 3)).float()

    return weight * (0.5 * loss_lpips + 0.3 * loss_grad
                     + 0.2 * loss_multi + 0.2 * loss_spec)


def total_loss(v_pred, v_target, x, x_t, t, lpips_fn,
               compute_gradient_loss, compute_multiscale_loss):
    base = ((v_pred - v_target) ** 2).flatten(1).mean(1)       # unscaled velocity anchor
    aux = lpips_spectral_auxiliary(v_pred, v_target, x, x_t, t, lpips_fn,
                                   compute_gradient_loss, compute_multiscale_loss)
    return (base + aux).mean()
```
