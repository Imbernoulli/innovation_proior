OK, let me think this through from scratch. I have a microscope image of tissue — neuronal membranes in an electron-microscopy stack, or cells in phase-contrast — and I need to produce a label for every single pixel: membrane or not, cell or background. Not one label for the image. A whole map, the same size as the input. And the brutal constraint is that I have maybe thirty annotated images. Thirty. The networks that just took over vision — the 8-layer ImageNet net, the very deep VGG nets — were trained on a million labeled images. I have thirty. So whatever I build has to be wildly more data-efficient, and it has to emit "where", not just "what".

Start with what a classification net actually does, because that's the machinery I have lying around. Conv, ReLU, max-pool, conv, ReLU, max-pool… the spatial map shrinks and shrinks, the channel count grows, and at the end a couple of fully connected layers crush the whole thing into one vector of class scores. And here's the thing I keep coming back to: that spatial collapse is not an accident, it's the *point*. Pooling throws away spatial resolution on purpose. It buys translation tolerance and a big receptive field — by the deep layers a single unit "sees" a large chunk of the image, which is exactly the context you need to say "this is a cat". For classification, forgetting *where* the cat is inside the image is a feature, not a bug.

But for me "where" is the entire job. So I'm immediately staring at a tension, and I want to name it precisely because it's going to drive everything. The operation that gives me context — pooling, which enlarges the receptive field — is the same operation that destroys localization, because after I pool a 2×2 region down to one value I no longer know which of those four positions the activation came from. Context and localization are pulling against each other through the pooling stride. Any honest dense-prediction method has to confront this.

How did people get per-pixel output before? The approach that won the EM challenge in 2012, Ciresan and colleagues: to label a pixel, cut out a square patch *centered on that pixel*, run it through a normal classification CNN, and take the network's output as the label of the center pixel. Then slide that window over every pixel. It's clever — it turns a classifier into a localizer, because each forward pass is "decided" at one specific pixel, and as a bonus the number of training patches you can extract is far larger than the number of images, which helps a lot when images are scarce. Both of those are real virtues and I want to keep them in mind.

But look at the cost. To label one 512×512 image I run the full network 262,144 times, once per pixel. And almost every one of those runs is redundant: the patch for pixel (i, j) and the patch for pixel (i, j+1) overlap in all but one column, so I'm recomputing essentially the same convolutions over and over. It's grotesquely wasteful and slow. That's the first wall. And there's a second, subtler wall baked into the patch size. A big patch means I can afford more pooling layers, so more context — but more pooling means coarser localization of that center decision. A small patch localizes sharply but sees almost no context. The patch size is a single dial that trades context against localization, and there is no setting that gives me both. So Ciresan's method has the two diseases written right into its structure: redundant compute, and a context/localization knob I can't win.

Can I kill the redundancy? Here's an observation that, once you see it, is hard to un-see. If I take a classification CNN and feed it an image *larger* than its nominal input, the convolutions and poolings just keep working — they're translation-equivariant — and the only thing standing in the way is the fully connected head, which demands a fixed input size. But a fully connected layer over an H×W feature map is just a convolution with an H×W kernel. So reinterpret the FC layers as 1×1 convolutions, and suddenly the whole network is convolutional from end to end. Feed it the *whole* image once, and out comes a spatial grid of class scores — one forward pass, every pixel's patch effectively evaluated simultaneously with all the shared computation actually shared. That's the fully-convolutional idea, and it annihilates the redundancy wall completely. No more per-pixel re-runs.

So why don't I just stop here? Because of the *other* wall, which the fully-convolutional reinterpretation does nothing for. The output grid is coarse. If the network pooled five times with stride 2, the output is downsampled by 32; my dense map is 32× too small and 32× too blurry. I've made it fast but not precise. I still haven't beaten localization-versus-context — I've just moved it.

I need to get from the coarse, high-context, deep feature map back up to full resolution. The first move is obvious: upsample. And I don't want a fixed interpolation if I can avoid it; let the network learn the upsampling. A convolution with stride 2 downsamples; running that operation backwards — a "backwards-strided" or transposed convolution, an up-convolution — upsamples by 2 with learnable weights (you can initialize it to bilinear so it starts sane). Stack a few of those and I can blow the coarse map back up to input size.

Try that and stare at the result. It's blurry. Of course it is — I'm trying to reconstruct fine spatial detail from a map that *threw the fine spatial detail away* at every pooling step. The deep feature map knows *what* is there with great confidence and almost nothing about precisely *where* the boundary sits, because boundaries are exactly the high-frequency information pooling discarded. Upsampling can't invent information that was destroyed. Wall again, and it's the same wall wearing a different coat: the deepest layer is all context and no localization.

So where did the localization information go? It didn't vanish from the network — it vanished from the *deep* layers. The shallow layers, before all the pooling, still have it: they're at high resolution and they fire on edges and fine texture. They just don't have the context to know what those edges *mean*. So I have two complementary signals living at different depths: deep layers that know "what" but are coarse, shallow layers that know "where" but are dumb. The fix writes itself — route the high-resolution shallow features forward and combine them with the upsampled deep features, so the upsampled-but-blurry "what" gets re-sharpened by the crisp "where". Skip connections from the contracting side to the upsampling side.

That's the structure I'll build on. One contracting path of conv-and-pool that builds context, one expanding path that upsamples back to resolution, and links across the middle that hand the high-frequency detail from each contracting level to the matching expanding level so the boundaries come back sharp. Good localization and good context, at last, in a single pass.

Now I have to make a real design decision about *how* the two signals are combined, and the obvious thing and the right thing differ. The obvious thing is to predict class scores on the deep path, predict class scores from the shallow features, and *add* the two score maps — fuse by summation. That works, but it's committing to a fixed linear combination of the two predictions; the network can only scale them, it can't really *reason* about how a fine edge should modulate a coarse decision. And worse, it keeps the upsampling path thin — if all I'm carrying up is a handful of class-score channels, I'm not propagating much context to high resolution at all. I keep thinking about that: context lives in *many channels* of rich features, and if my whole expanding path is just a few score maps being nudged by skips, the context never really makes it up to where the resolution is.

So flip both of those. First: instead of summing two score maps, *concatenate* the upsampled deep features with the shallow features into one fat stack of channels, and then put real 3×3 convolutions after the concatenation. Now the network doesn't have to accept a fixed mixture — it can *learn* how to assemble the fine detail and the coarse context together, per location. Second: make the expanding path *wide*. Give the upsampling layers lots of feature channels, symmetric to the contracting path, so context is carried richly all the way up to full resolution rather than trickling up through a thin trickle of scores. When I draw it, the expanding side ends up mirroring the contracting side almost exactly, and the picture is a U: down on the left building context, up on the right rebuilding resolution, rungs across the middle carrying localization. The channels should grow as I go down and shrink symmetrically as I go up.

Let me pin the contracting path concretely. Two 3×3 convolutions, each with a ReLU, then a 2×2 max-pool with stride 2 to halve the resolution — and at every pooling step I *double* the number of feature channels. Why double? Because as the spatial map shrinks by 4× in area, I want to preserve representational capacity; doubling channels is the standard way to keep the network from bottlenecking as resolution drops. So 64 channels at the top, then 128, 256, 512, and 1024 at the bottom of the U.

The expanding path is the mirror. At each step: a 2×2 up-convolution that doubles the spatial size and *halves* the channel count (1024→512, say); then concatenate the cropped feature map from the matching contracting level; then two 3×3 convolutions with ReLU. Up, concatenate, convolve; up, concatenate, convolve — four times — until I'm back at the top resolution with 64 channels. And at the very end, a 1×1 convolution maps each 64-component per-pixel feature vector to K class scores. No fully connected layer anywhere; the whole thing stays convolutional, so it's size-agnostic.

Now, a detail I almost glossed but it matters a great deal: padding. When I do a 3×3 convolution, do I pad the borders to keep the size, or do I take only the "valid" part and let the map shrink by a one-pixel ring? Padding is convenient — sizes stay constant, no cropping needed. But padding means inventing values at the border (zeros, usually), and the output pixels near the edge are then computed partly from fabricated data. For most of the image that's a small sin, but I want a method that segments *arbitrarily large* images by tiling, and at tile seams a padded border would produce visible artifacts and the tiles wouldn't stitch cleanly. So I make a different choice: **valid convolutions only, no padding.** Every output pixel is then computed exclusively from real input pixels for which the full receptive field is available. The price is that the map shrinks: a 3×3 valid conv removes a one-pixel ring all around, so −2 pixels per convolution, −4 pixels per pair of convolutions before each pool. The output of the whole U is smaller than the input by a fixed border.

That shrinkage has an immediate consequence for the skip connections, and it's why I said "cropped" a moment ago. The contracting-path feature map at a given level was computed with fewer valid-conv shrinkages than the expanding-path map I'm about to concatenate it with, so it's *larger*. To concatenate, I center-crop the contracting feature map down to the expanding map's size. The crop just discards the outer ring that the expanding side no longer has support for. Fine.

Let me actually run the arithmetic to convince myself it's consistent, because with valid convs the sizes have to line up exactly at the bottom and through every pool. Take an input tile of 572×572. Two valid 3×3 convs: 572 → 570 → 568. Pool: 284. Two convs: 284 → 282 → 280. Pool: 140. Two convs: 140 → 138 → 136. Pool: 68. Two convs: 68 → 66 → 64. Pool: 32. Two convs at the bottom: 32 → 30 → 28. Now up: up-conv doubles 28 → 56, concatenate the contracting map from the 64-level, center-cropped from 64 to 56, two convs 56 → 54 → 52. Up 52 → 104, crop the 136-map to 104, convs 104 → 102 → 100. Up 100 → 200, crop the 280-map to 200, convs 200 → 198 → 196. Up 196 → 392, crop the 568-map to 392, convs 392 → 390 → 388. Output: 388×388. So a 572×572 input yields a 388×388 segmentation. The numbers close cleanly — and notice I had to pick the input size carefully: there's a constraint that every 2×2 max-pool must act on an even-sized map (568, 280, 136, 64 are all even), otherwise the halving is ambiguous and the up-side won't line up with the down-side for cropping. So the input tile size isn't free; I choose it so all the pooling sizes come out even.

Now — output smaller than input. At first that feels like a defect. But it's exactly what makes the tiling work, and that's the next thing I need, because a single 512×512 image already strains GPU memory and I want to segment images far larger than fit at once. Here's the strategy. To produce an output tile, I feed the network a *larger* input tile that contains that output region plus the surrounding ring of context the valid convolutions need. Because every output pixel sees only real input and the network is fully convolutional, the output tiles **abut seamlessly** — predict the next tile by sliding the input window over, and the two output tiles join with no seam, no border artifact. The whole image gets covered by overlapping input tiles producing non-overlapping output tiles. Memory now limits the *tile* size, not the *image* size.

There's one snag at the outer edge of the whole image. Output pixels right at the image border need input context that lies *outside* the image — there isn't any. I refuse to pad with zeros (same objection as before: fabricated data, artifacts). What's the least-wrong way to extrapolate? Mirror the image across its own edge. Reflect the border strip outward to synthesize the missing context ring. Mirroring is a reasonable prior for tissue — the statistics just past the edge look like the statistics just inside it — and it lets the valid-conv network produce honest predictions all the way to the image border. So: overlap tiles for the interior, mirror-extrapolation for the outer frame.

Architecture settled. Now the harder, domain-specific problem, and it's the one that the generic dense-prediction story doesn't touch at all: **touching cells.** In the cell datasets the objects are the same class and they're packed against each other, membrane to membrane. If I just predict "cell vs background" per pixel, two cells that touch get predicted as one connected blob, and downstream instance counting is ruined. I need the network to carve a thin separating line of "background" exactly in the crevice between two adjacent cells. The trouble is that crevice is only a few pixels wide. Under an ordinary pixel-wise loss it's negligible — a rounding error against the millions of easy interior pixels — and the network will happily merge the cells because the loss barely notices.

So I have to *make* the loss notice. Let me set up the loss carefully. Over the final feature map I take a pixel-wise soft-max: for class k at pixel x,

  p_k(x) = exp(a_k(x)) / ( Σ_{k'=1}^{K} exp(a_{k'}(x)) ),

where a_k(x) is the activation in channel k at position x. This is the usual soft approximation to the max — p_k(x) ≈ 1 for the dominant channel, ≈ 0 otherwise. Then cross-entropy at each pixel penalizes the deviation of p at the *true* label ℓ(x) from 1. Ordinarily I'd sum −log p_{ℓ(x)}(x) over all pixels with equal weight. Instead I attach a per-pixel weight w(x) and form the energy

  E = Σ_{x∈Ω} w(x) · log p_{ℓ(x)}(x)

(maximized — equivalently I minimize −E, the weighted negative-log-likelihood). The whole game is now in choosing w(x).

Two things need fixing through w(x), and I'll build it from two terms. First, plain class imbalance: background pixels vastly outnumber membrane/separation pixels, so without correction the net just predicts the majority class. Add a class-balancing weight w_c(x) that up-weights the rarer class so each class contributes comparably to the loss. That's standard and necessary but not sufficient — w_c treats all background pixels alike, and the *separating* background between two touching cells is what I specifically care about.

So I need a second term that spikes precisely in those thin gaps and nowhere else. What characterizes a pixel sitting in the crevice between two cells? It is close to the border of *one* cell and *also* close to the border of *another* cell. So compute, for each background pixel, d_1(x) = the distance to the border of the nearest cell, and d_2(x) = the distance to the border of the *second*-nearest cell. In the middle of a wide background region, d_1 and d_2 are both large. Deep inside or far from a second cell, d_2 is large. But right in the squeeze between two cells, *both* d_1 and d_2 are small — that's the signature I want. So make the extra weight large exactly when d_1 + d_2 is small and decay it away as the sum grows. A Gaussian in (d_1 + d_2) does that cleanly:

  w(x) = w_c(x) + w_0 · exp( − (d_1(x) + d_2(x))² / (2σ²) ).

When a pixel is wedged between two cells, d_1 + d_2 ≈ 0, the exponential ≈ 1, and that pixel gets an extra w_0 of weight; as the pixel moves away from the crevice, d_1 + d_2 grows, the Gaussian collapses, and the bonus vanishes. I'll set w_0 = 10 so the separating ridges get a heavy emphasis, and σ ≈ 5 pixels so the boost is concentrated in a band a few pixels wide — exactly the width of the separation I'm trying to teach. The separating borders themselves I extract up front with morphological operations on the instance ground truth, then precompute the whole weight map per image so it costs nothing at train time. Now the loss genuinely cares about the few pixels that decide whether two cells are one object or two.

That's the architecture and the loss. The last problem is the one I started with and never actually solved: thirty images. A network this deep will memorize thirty images in its sleep. I can't get more annotations, so I have to manufacture variety, and the right way to manufacture variety is to ask: what does the network legitimately need to be *invariant* to, and synthesize exactly those variations? For microscopy: shift and rotation (a cell is a cell at any position or angle), gray-value changes (illumination varies), and — the big one — *deformation*. Tissue is soft; it stretches and squashes; the same biological structure appears in a continuum of deformed shapes. That's the dominant real variation in this domain, and crucially I can *simulate* it. So the key augmentation is **random elastic deformation**: lay down random displacement vectors on a coarse 3×3 grid, drawn from a Gaussian with about 10 pixels of standard deviation, then interpolate per-pixel displacements with bicubic interpolation to get a smooth warp, and apply that warp to both image and label. Each training step the network sees a freshly, plausibly deformed version, and it learns deformation invariance without ever being shown a real second example. For thirty images, this is the thing that makes training possible at all. (A dropout layer at the end of the contracting path adds a bit more implicit augmentation on top.)

A couple of training mechanics fall out of the architecture. I deliberately want *large* input tiles so each forward pass covers a lot of image and the GPU memory is well used — but large tiles plus limited memory means I can't also have a large batch. So push it to the extreme: batch size of one image. A single-image gradient is noisy, though. I recover a stable estimate by cranking the momentum way up, to 0.99, so that the running average of gradients effectively pools information from a large number of recently seen single-image steps — high momentum substitutes for the batch I gave up. And initialization, in a net this deep with two paths meeting at every concatenation: if I'm careless, some branches will saturate and others will go dead. For alternating conv/ReLU layers the fix is to draw initial weights from a Gaussian with standard deviation √(2/N), N the number of incoming connections to a unit, which keeps each feature map at roughly unit variance through the depth — for a 3×3 conv reading 64 channels, N = 9·64 = 576, so std = √(2/576).

Let me write it as code, building it the way I reasoned it: a double-conv block (the [3×3 valid conv → ReLU] twice that repeats everywhere), a downsampling step that pools then double-convs, an upsampling step that up-convolves then center-crops the skip and concatenates then double-convs, a 1×1 head, the weighted soft-max cross-entropy, the weight-map builder, the elastic deformation, and a one-image-at-a-time training loop with momentum 0.99.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def init_he(m):
    # std = sqrt(2/N), N = fan-in, so each feature map keeps ~unit variance
    # through the deep alternating conv/ReLU stack (else paths explode or die).
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        nn.init.normal_(m.weight, mean=0.0, std=(2.0 / n) ** 0.5)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class DoubleConv(nn.Module):
    """The repeating unit: two 3x3 VALID convs, each + ReLU.
    Valid (no padding) => every output pixel has full real-input support,
    and the map shrinks by 2 px per conv (4 px per block)."""
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
    """Contracting step: 2x2 max-pool (halve resolution), then double-conv.
    Channels double at each step (capacity preserved as area shrinks)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x):
        return self.conv(self.pool(x))


def center_crop(skip, target_hw):
    # contracting map is larger (fewer valid-conv shrinks); crop its outer ring
    # to match the expanding map so they can be concatenated.
    _, _, h, w = skip.shape
    th, tw = target_hw
    dh, dw = (h - th) // 2, (w - tw) // 2
    return skip[:, :, dh:dh + th, dw:dw + tw]


class Up(nn.Module):
    """Expanding step: 2x2 up-conv (double resolution, halve channels),
    concatenate the cropped contracting-path skip (restores 'where'),
    then double-conv (LEARNS how to fuse fine detail with coarse context)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)   # in_ch = up'd (in_ch//2) + skip (in_ch//2)
    def forward(self, x, skip):
        x = self.up(x)
        skip = center_crop(skip, x.shape[2:])
        x = torch.cat([skip, x], dim=1)          # concatenate, not sum
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, n_classes=2):
        super().__init__()
        self.inc   = DoubleConv(in_ch, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)             # bottom of the U
        self.up1   = Up(1024, 512)
        self.up2   = Up(512, 256)
        self.up3   = Up(256, 128)
        self.up4   = Up(128, 64)
        self.head  = nn.Conv2d(64, n_classes, kernel_size=1)   # per-pixel -> K scores
        self.apply(init_he)

    def forward(self, x):
        c1 = self.inc(x)                          # high-res, fine detail ("where")
        c2 = self.down1(c1)
        c3 = self.down2(c2)
        c4 = self.down3(c3)
        b  = self.down4(c4)                       # low-res, rich context ("what")
        x  = self.up1(b,  c4)                     # rungs of the U carry detail across
        x  = self.up2(x,  c3)
        x  = self.up3(x,  c2)
        x  = self.up4(x,  c1)
        return self.head(x)                       # logits, smaller than input


def weighted_softmax_ce(logits, target, weight_map):
    # Loss = sum_x w(x) * (-log p_{l(x)}(x)); labels cover the valid interior.
    th = (target.shape[-2] - logits.shape[-2]) // 2   # labels cover the valid interior
    tw = (target.shape[-1] - logits.shape[-1]) // 2
    target = target[:, th:th + logits.shape[-2], tw:tw + logits.shape[-1]]
    weight_map = weight_map[:, th:th + logits.shape[-2], tw:tw + logits.shape[-1]]
    logp = F.log_softmax(logits, dim=1)
    nll = F.nll_loss(logp, target, reduction='none')  # -log p_{l(x)}(x) per pixel
    return (weight_map * nll).mean()


def make_weight_map(instance_label, w0=10.0, sigma=5.0):
    """w(x) = w_c(x) + w0*exp(-(d1+d2)^2 / (2 sigma^2)).
    instance_label is 0 for background and positive IDs for individual cells.
    For background pixels, d1,d2 are distances to the nearest / 2nd-nearest
    cell border; the Gaussian spikes in the gap between touching cells."""
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
        dists = np.stack([
            distance_transform_edt(~cell_mask)
            for cell_mask in cell_masks
        ], axis=0)
        dists.sort(axis=0)
        d1, d2 = dists[0], dists[1]
        w = wc + w0 * np.exp(-((d1 + d2) ** 2) / (2 * sigma ** 2))
        w[semantic > 0] = wc[semantic > 0]         # boundary term applies to background gaps
    else:
        w = wc
    return torch.from_numpy(w.astype(np.float32))


def elastic_deform(image, label, alpha=10.0, grid=3):
    """Smooth random warp: Gaussian displacements (std=alpha px) on a coarse grid,
    bicubically interpolated to per-pixel. Teaches deformation invariance --
    the key to training from ~30 images."""
    import numpy as np
    from scipy.ndimage import map_coordinates, zoom

    h, w = image.shape
    dx = zoom(np.random.randn(grid, grid) * alpha, (h / grid, w / grid), order=3)
    dy = zoom(np.random.randn(grid, grid) * alpha, (h / grid, w / grid), order=3)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    coords = [(yy + dy).ravel(), (xx + dx).ravel()]
    img = map_coordinates(image, coords, order=3).reshape(h, w)        # bicubic
    lab = map_coordinates(label, coords, order=0).reshape(h, w)        # nearest for labels
    return img, lab


def train(net, images, labels, steps=1000, lr=0.01):
    # batch = 1 (favor large tiles over batch size); momentum 0.99 recovers a
    # stable gradient from many single-image steps.
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

Tracing the whole chain back: I need per-pixel labels from very few images, and pooling forces context and localization to fight. Making the net fully convolutional kills the per-patch redundancy and gives me a coarse map in one pass; upsampling that map is blurry because the deep layer threw away the localization; so I route the high-resolution detail forward through skip connections and let convolutions after a *concatenation* learn to fuse fine "where" with coarse "what", in a wide, symmetric U. Valid convolutions keep every prediction honest and, by making the output an interior of the input, enable seamless overlap-tiling with mirror-extrapolation at the image border. A weighted soft-max cross-entropy, with a Gaussian boost in the thin gap between touching cells, forces the network to draw the separating ridges that plain loss would ignore. And aggressive elastic-deformation augmentation, plus single-image batches with momentum 0.99 and √(2/N) initialization, is what lets a network this deep actually train from thirty images.
