# U-Net: a symmetric encoder–decoder for dense biomedical segmentation

## Problem

Produce a per-pixel label map for a microscopy image (membrane/cell/background), trained from only a few dozen annotated images, with crisp boundaries and the ability to separate touching objects of the same class. Classification CNNs give one label per image and destroy localization via pooling; patch-based pixel classifiers are slow/redundant and trade context against localization through patch size.

## Key idea

A **U-shaped fully convolutional network**: a contracting path (encoder) that builds context while halving resolution and doubling channels, and a symmetric expanding path (decoder) that upsamples back to resolution. **Skip connections** copy high-resolution feature maps from each encoder level and **concatenate** them with the upsampled decoder features at the matching level; convolutions after each concatenation *learn* to fuse fine localization ("where") with coarse context ("what"). The decoder is wide (many channels), so context propagates all the way up to high resolution. Three further pieces make it work in the few-image biomedical regime: **valid (unpadded) convolutions** with center-cropped skips, enabling a seamless **overlap-tile** strategy (with mirror-extrapolation at image borders); a **weighted soft-max cross-entropy** whose weight map spikes in the thin gaps between touching cells; and heavy **elastic-deformation data augmentation**.

## Architecture

- Contracting path: 4 stages of `[3×3 conv (valid) → ReLU] ×2` then `2×2 max-pool /2`; channels 64→128→256→512, bottom block 1024.
- Expanding path: 4 stages of `2×2 up-conv (×2 size, ÷2 channels) → concat cropped skip → [3×3 conv → ReLU] ×2`; channels 512→256→128→64.
- Head: `1×1 conv` mapping the 64-d per-pixel vector to K class scores. No fully connected layers; 23 conv layers total.
- Sizes (valid convs shrink by 2 px/conv): 572×572 input → 388×388 output. Choose input tiles so every 2×2 pool acts on an even-sized map.

## Objective

Pixel-wise soft-max `p_k(x) = exp(a_k(x)) / Σ_{k'} exp(a_{k'}(x))`, weighted cross-entropy energy

  E = Σ_{x∈Ω} w(x) · log p_{ℓ(x)}(x)   (maximize; equivalently minimize the weighted NLL)

with weight map

  w(x) = w_c(x) + w_0 · exp( −(d_1(x) + d_2(x))² / (2σ²) ),   w_0 = 10, σ ≈ 5 px,

where w_c balances class frequencies, and d_1, d_2 are distances to the borders of the nearest and second-nearest cell. The Gaussian term is large only where a pixel is squeezed between two cells (both distances small), forcing the network to learn the separating ridges.

## Training

- SGD, **batch size 1** (favor large input tiles over batch size), **momentum 0.99** (stable gradient from many single-image steps).
- He initialization: Gaussian weights with standard deviation √(2/N), N = fan-in (e.g. 3×3 conv over 64 channels → N = 576).
- Augmentation: **random elastic deformation** (Gaussian displacements, σ ≈ 10 px, on a coarse grid, bicubically interpolated), plus shift/rotation/gray-value jitter; dropout at the end of the contracting path for further implicit augmentation.
- Inference on large images: overlap-tile (feed a larger input tile per output tile; tiles abut seamlessly), mirror-extrapolate the missing context at image borders.

## Code

This implementation uses valid convolutions, center-cropped skip features, a symmetric decoder, weighted per-pixel cross-entropy, and one-image SGD updates.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def init_he(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        nn.init.normal_(m.weight, mean=0.0, std=(2.0 / n) ** 0.5)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class DoubleConv(nn.Module):
    """[3x3 valid conv -> ReLU] x2. Valid: output loses 2 px per conv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """2x2 max-pool then double-conv; channels double per step."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x):
        return self.conv(self.pool(x))


def center_crop(skip, target_hw):
    _, _, h, w = skip.shape
    th, tw = target_hw
    dh, dw = (h - th) // 2, (w - tw) // 2
    return skip[:, :, dh:dh + th, dw:dw + tw]


class Up(nn.Module):
    """2x2 up-conv (x2 size, /2 channels) -> concat cropped skip -> double-conv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)   # in_ch = up'd (in_ch//2) + skip (in_ch//2)
    def forward(self, x, skip):
        x = self.up(x)
        skip = center_crop(skip, x.shape[2:])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, n_classes=2):
        super().__init__()
        self.inc   = DoubleConv(in_ch, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1   = Up(1024, 512)
        self.up2   = Up(512, 256)
        self.up3   = Up(256, 128)
        self.up4   = Up(128, 64)
        self.head  = nn.Conv2d(64, n_classes, kernel_size=1)
        self.apply(init_he)

    def forward(self, x):
        c1 = self.inc(x)
        c2 = self.down1(c1)
        c3 = self.down2(c2)
        c4 = self.down3(c3)
        b  = self.down4(c4)
        x  = self.up1(b,  c4)
        x  = self.up2(x,  c3)
        x  = self.up3(x,  c2)
        x  = self.up4(x,  c1)
        return self.head(x)


def weighted_softmax_ce(logits, target, weight_map):
    """Loss = sum_x w(x) * (-log p_l(x)); labels/weights cropped to the valid interior."""
    th = (target.shape[-2] - logits.shape[-2]) // 2
    tw = (target.shape[-1] - logits.shape[-1]) // 2
    target = target[:, th:th + logits.shape[-2], tw:tw + logits.shape[-1]]
    weight_map = weight_map[:, th:th + logits.shape[-2], tw:tw + logits.shape[-1]]
    logp = F.log_softmax(logits, dim=1)
    nll = F.nll_loss(logp, target, reduction='none')
    return (weight_map * nll).mean()


def make_weight_map(instance_label, w0=10.0, sigma=5.0):
    """w(x) = w_c(x) + w0*exp(-(d1+d2)^2 / (2 sigma^2)).
    Positive IDs are individual cells; binary masks are split into components."""
    import numpy as np
    from scipy.ndimage import distance_transform_edt, label as cc_label

    instance_label = np.asarray(instance_label)
    semantic = (instance_label > 0).astype(np.int64)
    wc = np.ones_like(instance_label, dtype=np.float32)
    classes, counts = np.unique(semantic, return_counts=True)
    freq = {c: cnt / semantic.size for c, cnt in zip(classes, counts)}
    for c in classes:
        wc[semantic == c] = 1.0 / (len(classes) * freq[c])

    cell_ids = [i for i in np.unique(instance_label) if i != 0]
    if len(cell_ids) > 1:
        cell_masks = [instance_label == cell_id for cell_id in cell_ids]
    else:
        components, n = cc_label(instance_label > 0)
        cell_masks = [components == i for i in range(1, n + 1)]

    if len(cell_masks) >= 2:
        dists = np.stack([distance_transform_edt(~cell_mask)
                          for cell_mask in cell_masks], axis=0)
        dists.sort(axis=0)
        d1, d2 = dists[0], dists[1]
        w = wc + w0 * np.exp(-((d1 + d2) ** 2) / (2 * sigma ** 2))
        w[semantic > 0] = wc[semantic > 0]      # boundary term targets background gaps
    else:
        w = wc
    return torch.from_numpy(w.astype(np.float32))


def elastic_deform(image, label, alpha=10.0, grid=3):
    """Gaussian displacements (std=alpha px) on a coarse grid, bicubic to per-pixel."""
    import numpy as np
    from scipy.ndimage import map_coordinates, zoom

    h, w = image.shape
    dx = zoom(np.random.randn(grid, grid) * alpha, (h / grid, w / grid), order=3)
    dy = zoom(np.random.randn(grid, grid) * alpha, (h / grid, w / grid), order=3)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    coords = [(yy + dy).ravel(), (xx + dx).ravel()]
    img = map_coordinates(image, coords, order=3).reshape(h, w)
    lab = map_coordinates(label, coords, order=0).reshape(h, w)
    return img, lab


def train(net, images, labels, steps=1000, lr=0.01):
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.99)
    net.train()
    for step in range(steps):
        i = step % len(images)
        img, lab = elastic_deform(images[i], labels[i])
        w = make_weight_map(lab)
        x = torch.from_numpy(img).float()[None, None]
        t = torch.from_numpy((lab > 0).astype("int64"))[None]
        logits = net(x)
        loss = weighted_softmax_ce(logits, t, w[None])
        opt.zero_grad(); loss.backward(); opt.step()
```
