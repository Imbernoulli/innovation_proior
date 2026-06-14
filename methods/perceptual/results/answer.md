# LPIPS (Learned Perceptual Image Patch Similarity), distilled

LPIPS measures the perceptual distance between two image patches as a channel-normalized,
per-channel-calibrated Euclidean distance in the feature space of a fixed, pretrained
convolutional network. It rests on the finding that internal activations of networks trained to
predict the structure of natural images (classification, but also self-supervised and even
unsupervised objectives) can behave like a perceptual space, and that a small linear recalibration
on top can ground that space in human judgments without relearning the whole representation.

## Problem it solves

A distance `d(x, x0)` on image patches whose ordering of pairs agrees with human perception.
Per-pixel `l2`/PSNR assume pixel independence and are fooled by blur and spatial shifts;
hand-designed metrics (SSIM/MS-SSIM/FSIM) use shallow local statistics, degrade under geometric
distortion, and are not grounded in human data. Human similarity depends on high-order structure, is
context-dependent, and may not be a true metric, so fitting a function directly to judgments is
fragile — LPIPS instead starts from a representation where plain Euclidean distance is already nearly
perceptual and makes only a small learned correction.

## Key idea

Compare patches in deep feature space, but (1) **unit-normalize across channels** at each spatial
location so the comparison is of feature *direction* (cosine), not raw activation energy; (2) give
each channel a **learned non-negative** coefficient, fit to human 2AFC judgments; (3) **spatially
average** within a layer and **sum across layers** (low layers carry appearance, deep layers
semantics). With all coefficients set to 1 this reduces exactly to summed cosine distance, so the
learned metric is a strict generalization of the unlearned one.

## Final form

For a frozen network `F` with `L` chosen layers, let `y_hat^l, y0_hat^l` be the channel-unit-normalized
activations at layer `l` (shape `H_l x W_l x C_l`):

```
d(x, x0) = sum_l  (1 / (H_l W_l))  sum_{h,w}  || w^l ⊙ ( y_hat^l_{hw} - y0_hat^l_{hw} ) ||_2^2,
           y_hat^l_{hw} = y^l_{hw} / (|| y^l_{hw} ||_2 + eps),   w^l in R^{C_l},  w^l >= 0.
```

The implementation first forms the squared-difference map, then applies a `1x1` convolution (1 output
channel, no bias). Its non-negative coefficients `a_c` are the channel weights on squared
disagreements, corresponding to `a_c = (w_c)^2` in the norm expression above; with all `a_c = 1`, the
distance is the summed cosine baseline.

## Calibration (training the linear weights, the "lin" variant)

Keep `F` frozen; learn only the per-channel linear coefficients (e.g. ~1472 params for VGG; AlexNet
conv1-5 = 64+192+384+256+256 = 1152). A small predictor `G` maps a distance pair `(d0, d1)` to a
judgment probability `h_hat in (0,1)`: input `[d0, d1, d0-d1, d0/(d1+eps), d1/(d0+eps)]` (eps=0.1) →
two 32-unit FC-LeakyReLU(0.2) layers → 1-unit FC → sigmoid. Fit the channel coefficients and `G` end to end by binary
cross-entropy on 2AFC triplets `(x, x0, x1, h)`, `h in {0,1}` (split judges → target 0.5):

```
L(x, x0, x1, h) = - h log G(d0, d1) - (1 - h) log( 1 - G(d0, d1) ),   d0 = d(x,x0), d1 = d(x,x1).
```

After every optimizer step, project the `1x1`-conv coefficients to be non-negative: a larger
disagreement in any feature must never make two patches *closer*. Optimizer: Adam, lr `1e-4`,
beta1 `0.5`, 5 epochs at `1e-4` then 5 epochs linear decay, batch 50. Use the learned `G` rather than a
fixed-margin ranking loss because one uniform margin cannot fit triplets with very different degrees
of ambiguity. Variants: **lin** (learn only the linear channel coefficients on a frozen trunk),
**tune** (fine-tune all of `F`), **scratch** (random init, train all). Available
trunks: VGG `[64,128,256,512,512]`, AlexNet `[64,192,384,256,256]`, SqueezeNet
`[64,128,256,384,384,512,512]`.

## Why each choice

- **Feature space, not pixels** — pixels assume independence (blur/shift fool `l2`); deep features
  encode high-order structure and behave perceptually as an emergent property.
- **Channel unit-normalization** — raw activations vary wildly in magnitude across channels; without
  it, a few loud channels dominate and the metric measures gain, not pattern. Normalized ⇒ cosine
  (`||u-v||^2 = 2 - 2<u,v>` for unit vectors).
- **Learned non-negative channel coefficients** — not all channels matter equally to perception;
  cosine weights all equally. Non-negativity is structural: more disagreement must not reduce
  distance.
- **Frozen trunk, linear calibration** — the perceptual structure is the emergent property of the
  pretrained features; a light recalibration preserves it and transfers, where heavier fitting risks
  specializing to the calibration distortions.
- **Learned `G` over fixed-margin ranking** — relative closeness varies across triplets; a learned
  soft boundary fits this, a uniform margin fights it.
- **Sum over layers, spatial average within** — low/high layers carry appearance/semantics; spatial
  averaging converts each layer's local disagreement map into a patch-level scalar on a comparable
  scale across feature-map sizes.
- **Fixed input scaling** (ImageNet stats) — keeps patches in the regime the trunk was trained on.

## Working code

```python
import torch
import torch.nn as nn


def normalize_tensor(feat, eps=1e-10):
    # unit-normalize across channels at each location -> cosine comparison of feature direction
    norm = torch.sqrt(torch.sum(feat ** 2, dim=1, keepdim=True))
    return feat / (norm + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)


class ScalingLayer(nn.Module):
    # fixed ImageNet-style input normalization used before the feature trunk
    def __init__(self):
        super().__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    # per-channel learned coefficient as a 1x1 conv -> 1 channel, no bias
    def __init__(self, chn_in, use_dropout=True):
        super().__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers += [nn.Conv2d(chn_in, 1, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class LPIPS(nn.Module):
    def __init__(self, trunk, chns, lpips=True, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.net = trunk                      # frozen pretrained backbone; forward -> list of L feats
        self.chns = chns
        self.L = len(chns)
        self.lpips = lpips                    # True: learned coefficients; False: cosine baseline
        if lpips:
            self.lins = nn.ModuleList([NetLinLayer(c, use_dropout=use_dropout) for c in chns])

    def forward(self, in0, in1, normalize=False):
        if normalize:                         # flip if inputs are in [0,1] -> [-1,1]
            in0, in1 = 2 * in0 - 1, 2 * in1 - 1
        out0 = self.net(self.scaling_layer(in0))
        out1 = self.net(self.scaling_layer(in1))
        val = 0
        for kk in range(self.L):
            f0, f1 = normalize_tensor(out0[kk]), normalize_tensor(out1[kk])
            diff = (f0 - f1) ** 2
            if self.lpips:
                res = spatial_average(self.lins[kk](diff), keepdim=True)
            else:
                res = spatial_average(diff.sum(dim=1, keepdim=True), keepdim=True)
            val = val + res
        return val


class Dist2LogitLayer(nn.Module):
    # G: distance pair -> judgment probability, via [d0, d1, d0-d1, d0/d1, d1/d0]
    def __init__(self, chn_mid=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(5, chn_mid, 1, bias=True), nn.LeakyReLU(0.2, True),
            nn.Conv2d(chn_mid, chn_mid, 1, bias=True), nn.LeakyReLU(0.2, True),
            nn.Conv2d(chn_mid, 1, 1, bias=True), nn.Sigmoid())

    def forward(self, d0, d1, eps=0.1):
        return self.model(torch.cat((d0, d1, d0 - d1, d0 / (d1 + eps), d1 / (d0 + eps)), dim=1))


class BCERankingLoss(nn.Module):
    def __init__(self, chn_mid=32):
        super().__init__()
        self.net = Dist2LogitLayer(chn_mid=chn_mid)
        self.loss = nn.BCELoss()

    def forward(self, d0, d1, judge):         # judge in [-1,1]; per in [0,1]
        per = (judge + 1.) / 2.
        self.logit = self.net(d0, d1)
        return self.loss(self.logit, per)


def clamp_weights(net):                       # project learned 1x1-conv coefficients >= 0
    for m in net.modules():
        if hasattr(m, 'weight') and getattr(m, 'kernel_size', None) == (1, 1):
            m.weight.data = torch.clamp(m.weight.data, min=0)
```
