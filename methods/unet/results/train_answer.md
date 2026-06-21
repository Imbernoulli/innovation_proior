The task is to assign a class label to every pixel of a microscopy image — a full segmentation map at the input resolution, not one image-level label — and to do it from only a few dozen annotated images, with boundaries crisp enough that two objects of the same class that physically touch are still separated into distinct instances. This is the everyday biomedical regime: tens of training images rather than the million-image datasets that made deep classification work, and cells packed membrane-to-membrane, where a correct segmentation has to trace the thin ridge of background separating adjacent cells or a whole cluster collapses into one blob and instance counting fails. The machinery I have on hand is a classification CNN — conv, ReLU, max-pool, repeated until the spatial map collapses and fully connected layers crush it to a vector — and that machinery is built to do the opposite of what I need. Pooling with stride is the operation that builds context: it enlarges the receptive field and grants tolerance to small shifts, which is exactly why classification works. But the same pooling step that grants context destroys localization, because once a 2×2 region is reduced to one value I no longer know which of those four positions the activation came from. Context and localization pull against each other through the pooling stride, and that single tension is what any dense-prediction method has to resolve.

The prior per-pixel approaches each hit one side of that tension. The sliding-window patch classifier of Ciresan et al. (2012), which won the EM challenge, labels a pixel by running a classification CNN on a square patch centered on it; it genuinely localizes, and the number of extractable patches dwarfs the number of images, which helps when images are scarce. But it runs the full network once per pixel — hundreds of thousands of forward passes for a 512×512 image — and neighboring patches overlap in all but one column, so it recomputes essentially the same convolutions over and over; it is grotesquely redundant and slow. Worse, its patch size is a single dial trading context against localization: a large patch admits more pooling and more context but coarser localization, a small patch localizes sharply but sees almost nothing. There is no setting that wins both. The fully convolutional networks of Long et al. (2014) cure the redundancy: reinterpret the fully connected head as 1×1 convolutions and the whole net becomes translation-equivariant end to end, mapping a full image to a grid of class scores in a single forward pass. But that grid is coarse — downsampled by the pooling stride, e.g. 32× — and upsampling it back is blurry, because the deepest layer is precisely the one that threw the fine spatial detail away. Their skip connections that sum coarse and fine score maps help, but the recovered boundaries stay imprecise, and the method was validated on large natural-image datasets, not the few-image biomedical regime with thin separations between touching objects.

I propose U-Net: a U-shaped fully convolutional network, a contracting path that builds context mirrored by a symmetric expanding path that rebuilds resolution, with skip connections across the middle that carry the localization information forward. The reasoning that fixes its shape is this. Making the net fully convolutional already kills the redundancy wall, so I keep that. The remaining wall is that upsampling the deep map is blurry — but the localization information did not vanish from the network, only from the deep layers; the shallow layers, before all the pooling, are still at high resolution and fire on edges and fine texture, they simply lack the context to know what those edges mean. So I have two complementary signals at different depths: deep features that know "what" but are coarse, shallow features that know "where" but are dumb. The fix is to route the high-resolution shallow features forward and combine them with the upsampled deep features so the blurry "what" gets re-sharpened by the crisp "where". Now the decisive design choice: *how* to combine them. The obvious move — predict scores on both paths and sum them — commits the network to a fixed linear mixture it can only rescale, and it keeps the expanding path thin, carrying only a few score channels so context never really reaches high resolution. I do the opposite on both counts. I concatenate the upsampled deep features with the shallow features into one fat stack and put real $3\times3$ convolutions after the concatenation, so the network *learns* per location how a fine edge should modulate a coarse decision rather than accepting a fixed blend. And I make the expanding path wide, symmetric to the contracting path, so context is carried richly all the way up rather than trickling through a handful of scores. Drawn out, it is a U: down on the left building context, up on the right rebuilding resolution, rungs across the middle carrying localization.

Concretely the contracting path is four stages of two $3\times3$ convolutions each with a ReLU, then a $2\times2$ max-pool with stride 2, doubling the channel count at every pool — $64 \to 128 \to 256 \to 512$, with $1024$ at the bottom of the U. Doubling channels as the spatial area shrinks by $4\times$ preserves representational capacity so the network does not bottleneck as resolution drops. The expanding path mirrors it: at each step a $2\times2$ up-convolution (a backwards-strided, learnable transposed convolution, which can be initialized to bilinear) doubles the spatial size and halves the channels, then the cropped contracting-level feature map is concatenated, then two $3\times3$ convolutions with ReLU; up, concatenate, convolve, four times, back to $64$ channels at full resolution. A final $1\times1$ convolution maps each $64$-dimensional per-pixel vector to $K$ class scores. There is no fully connected layer anywhere, so the network is size-agnostic; it is $23$ convolutional layers in total.

A choice that looks minor but is load-bearing is padding. A $3\times3$ convolution can pad its borders to keep the size constant, but padding fabricates values (zeros) at the edge, so output pixels near the border are computed partly from invented data — and at the seams between tiles that produces visible artifacts and the tiles do not stitch. I use valid (unpadded) convolutions only, so every output pixel is computed exclusively from real input pixels for which the full receptive field exists. The price is that the map shrinks by a one-pixel ring per convolution, two pixels per pair, so the output is an interior of the input. That is why the skips must be center-cropped: the contracting-path map at a given level underwent fewer valid-conv shrinkages than the expanding-path map it joins, so it is larger, and I crop its outer ring to match before concatenating. The arithmetic closes cleanly — a $572\times572$ input yields a $388\times388$ output ($572 \to 570 \to 568$, pool to $284$, and so on down to $32\to30\to28$ at the bottom, then up through $56,104,200,392$ to $388$) — and the input tile size is not free: every $2\times2$ pool must act on an even-sized map ($568,280,136,64$ are all even) or the halving is ambiguous and the two sides of the U fail to line up. The shrinking output is not a defect but the enabling feature of an overlap-tile strategy: to predict an output tile I feed a larger input tile containing it plus the context ring the valid convs need, and because every output pixel sees only real input, adjacent output tiles abut with no seam. Memory then limits the tile size, not the image size. At the outer frame of the whole image, where the needed context lies outside the image, I refuse zero-padding and instead mirror the border strip outward — a reasonable prior for tissue, since the statistics just past an edge look like those just inside it.

The harder, domain-specific problem is touching cells of the same class, and the generic dense-prediction story does nothing for it. If I predict cell-versus-background per pixel, two cells that touch are predicted as one connected blob; I need the network to carve a thin separating line of background in the crevice between them, but that crevice is only a few pixels wide and under an ordinary pixel-wise loss it is a rounding error against millions of easy interior pixels. So I make the loss notice. Over the final map I take a pixel-wise soft-max,
$$p_k(x) = \frac{\exp(a_k(x))}{\sum_{k'=1}^{K} \exp(a_{k'}(x))},$$
the soft approximation to the max, and form a weighted cross-entropy energy
$$E = \sum_{x \in \Omega} w(x)\,\log p_{\ell(x)}(x),$$
maximized — equivalently I minimize the weighted negative log-likelihood. The whole game is in $w(x)$, which I build from two terms. The first, $w_c(x)$, balances class frequencies by up-weighting the rarer class, so that background does not swamp membrane; necessary but not sufficient, because $w_c$ treats every background pixel alike. The second term must spike precisely in the thin gap and nowhere else. The signature of a crevice pixel is that it is close to the border of one cell *and* close to the border of a second cell, so for each background pixel I compute $d_1(x)$, the distance to the nearest cell border, and $d_2(x)$, the distance to the second-nearest. In open background both are large; deep inside or far from a second cell, $d_2$ is large; only in the squeeze between two cells are *both* small. A Gaussian in $d_1 + d_2$ captures exactly that:
$$w(x) = w_c(x) + w_0 \cdot \exp\!\left(-\frac{(d_1(x) + d_2(x))^2}{2\sigma^2}\right),$$
with $w_0 = 10$ so the separating ridges get a heavy emphasis and $\sigma \approx 5$ pixels so the boost is concentrated in a band the width of the separation I am teaching. When a pixel is wedged between two cells $d_1 + d_2 \approx 0$, the exponential is $\approx 1$, and it earns an extra $w_0$; as it moves away the Gaussian collapses and the bonus vanishes. The weight map is precomputed per image so it costs nothing at train time.

The last problem is the one I started with: thirty images, on which a network this deep would memorize in its sleep. I cannot get more annotations, so I manufacture variety by synthesizing exactly the variations the task should be invariant to — shift, rotation, gray-value changes, and above all *deformation*, because tissue is soft and the same structure appears in a continuum of deformed shapes, the dominant real variation here and one I can simulate. The key augmentation is random elastic deformation: lay random Gaussian displacement vectors (about $10$ pixels of standard deviation) on a coarse $3\times3$ grid, interpolate them bicubically to a smooth per-pixel warp, and apply it to both image and label, so each step shows a freshly, plausibly deformed example without a single new annotation; a dropout layer at the end of the contracting path adds further implicit augmentation. Two training mechanics fall out of the architecture. I want large input tiles so each pass covers a lot of image, which with limited memory forces the batch down to one image; a single-image gradient is noisy, so I crank momentum to $0.99$, letting the running average pool information across many recent single-image steps — high momentum substitutes for the batch I gave up. And in a net this deep with two paths meeting at every concatenation, careless initialization makes some branches saturate while others go dead, so I draw initial weights from a Gaussian with standard deviation $\sqrt{2/N}$, $N$ the fan-in (for a $3\times3$ conv over $64$ channels, $N = 9\cdot64 = 576$), which keeps each feature map at roughly unit variance through the depth.

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
