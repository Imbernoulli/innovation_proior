# LPIPS, distilled

LPIPS (Learned Perceptual Image Patch Similarity) is a perceptual distance between two images:
run both through a frozen, ImageNet-pretrained CNN, channel-normalize the features at several
layers, take the squared difference, scale it per channel by a small learned non-negative weight
vector calibrated on human similarity judgments, average over space, and sum over layers. It rests
on the idea that an internal feature space trained for visual prediction is a better starting point
than pixels or a from-scratch triplet classifier, so the human data is used for a small calibration
rather than to learn the whole representation.

## Problem it solves

Per-pixel `ℓ2`/PSNR is blind to structure (blur is cheap, an imperceptible shift is expensive);
hand-built metrics (SSIM/MS-SSIM/FSIM) are shallow fixed formulas that break under spatial
ambiguity and cannot be adapted to human data; directly fitting a similarity predictor to human
pairwise preferences risks learning triplet-specific quirks because the judgments are
context-dependent and pairwise. LPIPS provides a cheap, differentiable distance designed to order
image pairs by human similarity judgments across distortions and real algorithm outputs, usable
both as an evaluation metric and as a training loss.

## Key idea

Borrow the emergent perceptual structure of a pretrained classifier instead of learning it:

- Extract feature stacks from `L` layers of a frozen network `F`.
- **Channel-wise unit-normalize** each per-location feature vector,
  `ŷ^l_{hw} = y^l_{hw} / (||y^l_{hw}||₂ + ε)` — so loud channels can't dominate and per-layer
  distances are comparable. With unit weights this reduces to cosine distance because
  `||a - b||² = 2 - 2 cos(a, b)` for unit vectors.
- Add the **minimal learnable calibration**: a non-negative per-channel weight vector
  `w_l ∈ ℝ^{C_l}` (e.g. 1152 scalars for AlexNet) — too few to overfit the labels, enough to
  reweight feature channels while leaving the trunk fixed. `w_l = 1` recovers cosine distance.

## Final form

The distance:

```
d(x, x0) = Σ_l (1 / (H_l W_l)) Σ_{h,w} || w_l ⊙ ( ŷ^l_{hw} − ŷ0^l_{hw} ) ||²₂
```

Implemented per layer in the package as: channel L2-normalize features → squared difference →
dropout plus a no-bias `1×1` conv whose learned coefficients are clamped nonnegative → spatial
average → sum over layers. The equation writes the conceptual channel scale inside the norm; the
package stores the equivalent nonnegative coefficients after the square has been absorbed into the
linear weights on squared channel differences.

Calibration ("lin" config: freeze `F`, learn only the linear calibration): a small head `G` maps a
distance pair to a preference probability `ĥ = G(d0, d1) ∈ (0, 1)`, where `h = 0` means `x0` is
chosen and `h = 1` means `x1` is chosen, trained with binary cross-entropy against `h`:

```
L = − h log G(d0, d1) − (1 − h) log(1 − G(d0, d1))
```

A *learned* `G` (rather than a fixed-margin ranking loss) supplies a pair-dependent margin.
Non-negativity of the stored `1×1` coefficients is required because they multiply squared feature
differences, and is enforced by clamping them to `≥ 0` after each Adam step (`lr = 1e-4`,
`β1 = 0.5`; five epochs at the initial rate, five epochs with linear decay, batch size 50).

## Design choices and why

- **Frozen trunk + linear calibration ("lin") is the default**, with fine-tuning ("tune") and
  training from scratch ("scratch") kept as ablations: the general representation is the asset, so
  the safest calibrated metric updates only the small linear heads and the preference head.
- **Channel normalization** before comparison; `ε = 1e-10` floors the norm.
- **Squared L2, spatial average, layer sum**: aggregates multi-scale evidence (low-level early
  layers, semantic late layers) into a patch-level scalar.
- **Input handling as a loss**: inputs are expected on `[-1, 1]` (or converted from `[0, 1]` with
  `normalize=True`) and then passed through a fixed per-channel affine standardization
  (shift `(-.030, -.088, -.188)`, scale `(.458, .448, .450)`) before the frozen trunk.

## Working code

The metric (faithful to the canonical `lpips` package), runnable as either a metric or a
differentiable loss:

```python
import inspect
import os
import torch
import torch.nn as nn
from torchvision import models as tv


def normalize_tensor(x, eps=1e-10):
    norm = torch.sqrt((x ** 2).sum(dim=1, keepdim=True))   # channel-wise L2 per location
    return x / (norm + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)


class ScalingLayer(nn.Module):
    """Fixed per-channel standardization the frozen backbone expects."""
    def __init__(self):
        super().__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, x):
        return (x - self.shift) / self.scale


class AlexBackbone(nn.Module):
    """Pretrained AlexNet, frozen, sliced into 5 relu feature stages."""
    def __init__(self, requires_grad=False):
        super().__init__()
        feats = tv.alexnet(pretrained=True).features
        self.slices = nn.ModuleList()
        for a, b in [(0, 2), (2, 5), (5, 8), (8, 10), (10, 12)]:
            s = nn.Sequential()
            for x in range(a, b):
                s.add_module(str(x), feats[x])
            self.slices.append(s)
        if not requires_grad:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x):
        out, h = [], x
        for s in self.slices:
            h = s(h)
            out.append(h)
        return out


class NetLinLayer(nn.Module):
    """Learned nonnegative per-channel coefficients as a 1x1 conv (no bias) + optional dropout."""
    def __init__(self, chn_in, use_dropout=False):
        super().__init__()
        layers = [nn.Dropout()] if use_dropout else []
        layers += [nn.Conv2d(chn_in, 1, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class LPIPS(nn.Module):
    """Conceptual form: sum_l mean_hw ||w_l * (yhat0^l - yhat1^l)||_2^2.
    The package stores the equivalent nonnegative coefficients on squared differences in 1x1 convs."""
    def __init__(self, net='alex', pretrained=True, version='0.1', use_dropout=True, model_path=None):
        super().__init__()
        assert net == 'alex'                                # vgg/squeeze: swap slices + chns
        self.version = version
        self.scaling_layer = ScalingLayer()
        self.net = AlexBackbone(requires_grad=False)        # frozen ('lin' configuration)
        self.chns = [64, 192, 384, 256, 256]
        self.L = len(self.chns)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout)
        self.lins = nn.ModuleList([self.lin0, self.lin1, self.lin2, self.lin3, self.lin4])
        if pretrained:
            if model_path is None:
                model_path = os.path.abspath(os.path.join(
                    os.path.dirname(inspect.getfile(self.__init__)), 'weights/v%s/%s.pth' % (version, net)))
            self.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)

    def forward(self, in0, in1, normalize=False):
        if normalize:                                       # inputs in [0,1] -> [-1,1]
            in0, in1 = 2 * in0 - 1, 2 * in1 - 1
        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1)) if self.version == '0.1' else (in0, in1)
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        val = 0
        for l in range(self.L):
            f0, f1 = normalize_tensor(outs0[l]), normalize_tensor(outs1[l])
            diff = (f0 - f1) ** 2
            val = val + spatial_average(self.lins[l](diff))
        return val
```

Calibration on human triplets (`G` head + BCE + non-negativity projection):

```python
class Dist2LogitLayer(nn.Module):
    """Small G mapping a distance pair to P(h=1), the second distorted patch being preferred."""
    def __init__(self, chn_mid=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(5, chn_mid, 1, bias=True), nn.LeakyReLU(0.2, True),
            nn.Conv2d(chn_mid, chn_mid, 1, bias=True), nn.LeakyReLU(0.2, True),
            nn.Conv2d(chn_mid, 1, 1, bias=True), nn.Sigmoid())

    def forward(self, d0, d1, eps=0.1):
        return self.model(torch.cat((d0, d1, d0 - d1, d0 / (d1 + eps), d1 / (d0 + eps)), dim=1))


def train_on_human_judgments(metric, G, triplet_loader, nepoch=5, nepoch_decay=5, lr=1e-4):
    bce = nn.BCELoss()
    params = list(metric.lins.parameters()) + list(G.parameters())   # frozen trunk: coeffs + G only
    opt = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.999))
    for epoch in range(1, nepoch + nepoch_decay + 1):
        for x, x0, x1, h in triplet_loader:
            d0, d1 = metric(x, x0), metric(x, x1)
            loss = bce(G(d0, d1), h.view(d0.shape))                  # h=0 for x0, h=1 for x1
            opt.zero_grad(); loss.backward(); opt.step()
            for lin in metric.lins:                                  # project learned coeffs >= 0
                for m in lin.model:
                    if isinstance(m, nn.Conv2d):
                        m.weight.data.clamp_(min=0)
        if epoch > nepoch:                                           # linear decay, as in the training script
            new_lr = lr * max(0, 1 - (epoch - nepoch) / float(nepoch_decay))
            for group in opt.param_groups:
                group['lr'] = new_lr
```

Used as an auxiliary perceptual loss:

```python
lpips = LPIPS(net='alex', pretrained=True).eval()
# x_hat and x_target are image tensors on [-1, 1]; use normalize=True for [0, 1]
loss = loss_primary + lam * lpips(x_hat, x_target)   # differentiable; backprops into x_hat
```
