Let me start from what actually hurts. I want a label — or more generally an output value — at every single pixel of an image, and I want the output map to come out the same spatial size as the input, with the value sitting on the correct pixel. Segmentation is the case in front of me: cell vs. membrane, traced to the pixel, with the thin boundaries intact. The trouble is that the two halves of this task pull in opposite directions. To know *what* is at a pixel I have to look at a wide neighborhood — whether a pixel is membrane or interior is only decidable once I've integrated a lot of surrounding structure, which means a deep stack with a large receptive field. The cheap way to grow the receptive field is to pool, to throw spatial samples away so each later unit sees more of the image per pixel of feature map. But to say *where* — to put the label on exactly this pixel and not the one next to it, to keep a one-pixel-wide boundary one pixel wide — I need spatial resolution, and pooling is precisely the thing that destroyed it. So the deeper I go to settle the *what*, the more the *where* dissolves. A plain classification convnet run as-is takes this to the limit: it pools the whole image down to a coarse grid, and answers "what is in this image" beautifully while answering "what is at this pixel" not at all. That is the knot I have to cut.

What do people do today? The sliding window. To label a pixel, crop a patch around it and run a classifier on that patch; do that for every pixel and assemble the map. Ciresan and colleagues did exactly this for neuronal membranes and won the EM challenge with it, so it works. And it has one property I genuinely want to keep in the back of my mind: the number of training *patches* is gigantic even when the number of training *images* is tiny, which is the regime I'm stuck in — tens of annotated images, not thousands. But the patch approach has two problems I can't live with. It's brutally slow and redundant: I rerun the whole network from scratch for each pixel, and adjacent patches overlap almost entirely, so I'm recomputing nearly identical features millions of times. And worse, the patch size is a single knob that trades the two things I just said are in tension — make the patch bigger and I get more context but more pooling and so coarser localization; make it smaller and I localize better but starve the network of context. The patch *forces* me to choose, and I refuse to choose. I want both at once.

So forget per-patch. The first real move I can borrow is the fully-convolutional reframing — Long, Shelhamer and Darrell's idea. A classification net ends in fully-connected layers, but a fully-connected layer applied to a fixed-size feature map is just a convolution with a kernel the size of that map; rewrite it as a 1×1 convolution and suddenly the whole network is convolutional, accepts an image of any size, and produces a *spatial* output in one pass. No more per-patch redundancy — the overlapping computation is shared automatically by the convolution. That fixes the speed problem cleanly. But it doesn't fix the resolution problem; it relocates it. The output of that fully-convolutional net is still at the network's final stride — a coarse grid, say one prediction per 32 input pixels, because all the pooling is still in there building the receptive field. I've made it fast, I haven't made it sharp.

How do I get the resolution back? The output is small, the input is big, so I have to *upsample* inside the network, and I want that upsampling learned end-to-end, not a fixed post-hoc interpolation. Here the deconvolution idea is the right primitive: upsampling by a factor f is, in a precise sense, a convolution with a fractional input stride 1/f, which I can implement as a "backwards" strided convolution — run the forward and backward passes of an ordinary stride-f convolution in reverse. Initialize it to bilinear interpolation if I like, but let the weights train. So now I can build a second half of the network that climbs back up in resolution by stacking these learnable upsampling layers, and learn the whole thing through the pixelwise loss. Good — that's the skeleton of an answer: contract down to build context, then expand back up to recover resolution.

But I have to ask the question that the bare upsample-and-be-done version dodges: when I upsample the coarse, deep feature map back to full resolution, *where does the sharp localization come from*? It can't come from the coarse map — that map has a 32-pixel stride; the fine spatial detail was literally pooled out of it several layers ago. Upsampling a blurry thing gives me a bigger blurry thing. The detail I want still exists, but it exists *only* in the early, high-resolution feature maps, the ones from before the pooling threw resolution away. Those early maps know *where* the edges are; they just don't yet know *what* anything is, because they're shallow and have a tiny receptive field. The deep coarse map is the opposite: it knows *what*, blurrily *where*. So the two halves of the answer to my *what*/*where* tension are sitting in two different places in the network — the *where* in the early high-res layers, the *what* in the late low-res ones — and the entire game is to bring them back together.

FCN does bring them together, and watching exactly *how* it does it tells me what to fix. It takes a finer-stride pooling layer, slaps a 1×1 convolution on it to produce a few extra per-pixel class scores, upsamples the coarse prediction by 2×, and *adds* the two score maps; do it twice, at two finer layers, and the output gets sharper. They sum rather than take a max because max fusion made training unstable through gradient switching. Fine — but stare at what's actually being fused. It's *class-score maps*: thin, low-dimensional things, one channel per class, produced by collapsing a rich high-resolution feature map down through a 1×1 conv *before* fusion. So almost all of the content of those early high-resolution features — dozens or hundreds of channels of texture and edge and local-appearance information — is discarded at the 1×1 conv, and only a handful of class scores survive to be added in. And it happens at only two or three points in the network, and the recombination is a fixed addition, not something the network gets to learn how to do. And the climb back up is shallow: essentially upsample-and-add, with barely any learned processing on the way to full resolution. No wonder the detail is limited.

So don't summarize the early high-resolution maps into class scores before fusing — *route the full feature maps themselves* from each level of the contracting path across to the matching level of the expanding path, while they're still rich. And don't *add* them to the upsampled coarse features — **concatenate** them. Why concatenate rather than add? Because addition forces the two tensors into the same channel space and pre-commits how they combine — it bakes in "these correspond, sum them" before the network has any say. Concatenation stacks the *where*-channels (fine, high-res, from the encoder) right next to the *what*-channels (coarse context, just upsampled) and hands the whole pile to a subsequent convolution, which is then free to *learn* how to mix localization with context — to learn, channel by channel, position by position, how to assemble a sharp output from blurry-but-knowledgeable plus sharp-but-ignorant. FCN already discovered that fusion is delicate — summing was the only thing that trained, max broke it — which to me is a hint that the fusion shouldn't be a fixed arithmetic op at all; it should be learnable, and concatenation-then-convolution is exactly the way to make it learnable.

And I shouldn't do this at just two or three layers. The detail lost to pooling is lost at *every* pooling step, a little more each time, so the cure has to be applied at *every* resolution: at each step of the climb back up, upsample, then concatenate the same-resolution feature map saved from the way down, then convolve. That makes the expanding path a true mirror of the contracting path — same number of stages, each stage matched to a stage on the other side by a skip connection. The topology stops being a line, or a line with two side-channels; it becomes a U: down the left arm, across the bottom, up the right arm, with a horizontal skip bridging each pair of levels at equal resolution.

Now there's a subtler design choice I almost glossed: how *wide* should the expanding path be — how many feature channels on the way up? FCN's decoder is thin (it's pushing around class scores). But think about what the upward path has to accomplish. It isn't just interpolating; it has to *propagate the context* — the *what* it learned at the bottom — all the way up to full resolution, fusing it with localization at every level. If the up path is narrow, it simply can't carry enough context channels up to high resolution; the context gets bottlenecked and the high-res output ends up locally sharp but globally confused. So I want a *large* number of feature channels in the expanding path too — comparable to the contracting path. That symmetry isn't decoration; it's what lets context reach high resolution. The U has two fat arms, not one fat and one thin.

Let me pin down the rest of the contracting path concretely, because the choices there are mostly inherited from what already works but each has a reason. Each stage is two 3×3 convolutions, each followed by a ReLU, then a 2×2 max-pool with stride 2 to halve the resolution. Why two 3×3 convs instead of one bigger filter? Two stacked 3×3s have the receptive field of a single 5×5 but with fewer parameters and an extra nonlinearity in between — the lesson from very deep small-filter nets — so I get more expressiveness per stage at lower cost. Why max-pool with stride 2? It's the cheap receptive-field-doubling primitive; I'll revisit it later for the variant where I'd rather learn the downsampling. And crucially: at each downsampling step I *double the number of feature channels*. The spatial grid shrinks by 4× (half in each dimension) at each pool, and if I held the channel count fixed the representational capacity would collapse stage by stage. Doubling the channels roughly compensates — it keeps the total amount of representation per stage in the same ballpark while the network trades spatial extent for semantic abstraction. Down the left arm, then: resolution halves, channels double, four or five times, until I'm at a small grid with many channels — that's the bottom of the U, maximum context, minimum resolution.

The climb back up mirrors it exactly. Each expansive step: upsample the feature map; halve the channels with a 2×2 "up-convolution"; concatenate the cropped same-resolution map from the contracting path; then two 3×3 convs, each with a ReLU, to mix the concatenated stack. Halving channels on the way up mirrors the doubling on the way down, so by the time I'm back at full resolution I'm back to the base channel count, and a final 1×1 convolution maps that last feature vector at each pixel to the desired number of output classes. And note what's *not* anywhere in this network: no fully-connected layers at all. A fully-connected layer would pin the input to a fixed size and re-introduce exactly the rigidity the fully-convolutional reframing bought me out of. Keeping it all-convolutional keeps the arbitrary-input-size property — important because I want to feed it big tiles.

That word "cropped" I slipped in needs justifying, because it's a real design decision tied to a concrete operational goal, not an accident. If I use *unpadded* — "valid" — convolutions, every 3×3 conv eats a one-pixel border, so the feature map shrinks by two pixels per conv, and the output ends up smaller than the input by a fixed, predictable border. That's why the encoder map has to be center-cropped before I concatenate it onto the (smaller) decoder map — they're at the same resolution but the encoder one is a bit bigger because it's seen fewer border-eating convs at that point. Why would I *want* valid convolutions and this whole cropping headache, instead of just padding every conv to keep sizes equal? Because of the tiling. These microscopy images are huge — far bigger than fits in GPU memory at once — so I have to process them in tiles. If I only ever output pixels for which the *full* receptive-field context genuinely existed in the input (which is exactly what valid convolutions guarantee — no output pixel ever depended on invented, padded, edge values), then I can tile the image into overlapping windows, predict the valid interior of each, and stitch them seamlessly with no boundary artifacts; for the true image border I mirror-pad the input. That "overlap-tile" trick is what lets the network run on arbitrarily large images under a fixed memory budget, and it only works cleanly if the output is the honest, context-complete valid region. So valid-conv-plus-crop is the price for seamless tiling, and it's worth paying for segmentation of giant images. (I'll flag for later that a task with a *fixed* small input and a hard same-size-output requirement would instead pad every conv to keep sizes equal and skip the cropping entirely — there the tiling rationale doesn't apply.)

Now the few-data problem, which is the other half of why the sliding window was attractive (all those patches). I have maybe thirty annotated images. A network this size will memorize thirty images in its sleep. The patch trick manufactured data by chopping; I'll manufacture data by *deforming*. The dominant real variation in tissue is elastic deformation — cells and membranes bend and stretch — so if I apply random smooth elastic warps to the training images and their masks, I teach the network deformation invariance directly, from few images, without ever having to collect the warped versions. Concretely: sample random displacement vectors on a coarse grid from a Gaussian, interpolate them smoothly per pixel, and warp. Add the usual shift/rotation/gray-value jitter, and put a dropout at the deep end of the contracting path for a bit of implicit augmentation on top. This is the piece that makes thirty images enough.

There's a second data-specific difficulty: separating *touching* objects of the same class. Two cells pressed against each other are both "cell," and the thin gap between them is "background," but it's a tiny number of pixels and a plain loss will happily smear across it and merge the two cells. So I weight the loss. The objective is a pixelwise softmax fed into a cross-entropy that, at each pixel, penalizes the deviation of the true-label probability p_{ℓ(x)}(x) from 1, but with a per-pixel weight w(x) — written as the weighted log-likelihood energy

  p_k(x) = exp(a_k(x)) / Σ_{k'} exp(a_{k'}(x)),
  E = Σ_x w(x) · log p_{ℓ(x)}(x),

which training drives upward (equivalently it minimizes −E), where ℓ(x) is the true label and a_k(x) the activation for class k at pixel x. The weight map is precomputed per ground-truth mask as

  w(x) = w_c(x) + w_0 · exp( −(d_1(x) + d_2(x))² / (2σ²) ),

with w_c balancing class frequencies, d_1 and d_2 the distances to the nearest and second-nearest cell borders, and I'll use w_0 = 10, σ ≈ 5 pixels. Read the exponential term: it's large exactly where d_1 + d_2 is *small*, i.e. in the narrow corridor between two nearby cells, and decays away from it. So it dumps a big loss weight precisely on the thin separating membranes, forcing the network to learn to keep touching cells apart. It's a targeted fix for the failure mode the data exhibits, not a generic reweighting.

And because I'm going to run this deep network with many paths through it, initialization isn't optional. With alternating conv and ReLU layers, if the weights are off-scale the variance of activations explodes or collapses as it propagates, and with several paths through the U some paths can end up dominating while others never contribute. Draw the initial weights from a Gaussian with standard deviation √(2/N), N the number of incoming connections to a unit — for a 3×3 conv over 64 input channels that's N = 9·64 = 576 — and each feature map starts at roughly unit variance, which is the condition for the whole thing to train at all. Training itself: ordinary SGD, but I'll favor large input tiles over a large batch (to use the memory the tiles need), pushing the batch down to a single image, and compensate with a high momentum, 0.99, so that a large number of recently-seen single-image gradients accumulate into each update and the effective batch is recovered in time rather than in space.

So I now have the whole shape. Down the left arm: [3×3 conv, ReLU] ×2, max-pool /2, double channels — repeat. Across the bottom: the deepest, widest, coarsest features. Up the right arm: upsample, halve channels, concatenate the cropped same-level encoder features, [3×3 conv, ReLU] ×2 — repeat. A final 1×1 conv to the outputs. Skip connections at every level carrying the full high-resolution features across and concatenating them so a conv can learn to fuse *where* with *what*. That U is the architecture.

Let me make sure I can actually instantiate it as code, and here I want to land it in the form the field actually uses now, because the same encoder-decoder-with-concatenating-skips is the workhorse far beyond segmentation. Take the case where the input is a fixed small image and I have to predict a same-size continuous field — a denoiser that, given a noisy image and a scalar noise level, predicts the noise. Two things change from the segmentation original, and both follow from the new setting. First, the output must be *exactly* the same spatial size as the input — no shrinking border allowed — so I drop valid-conv-plus-crop and instead **pad every 3×3 convolution by 1** so sizes are preserved; the tiling rationale that justified valid convs simply doesn't apply when the input is a fixed 32×32. The skip connection then concatenates encoder and decoder maps that are already exactly aligned, no cropping. Second, there's an auxiliary scalar — the noise level, an integer timestep — that the whole network must be conditioned on, with the *same* weights handling every level. The clean way is to embed that scalar and inject it into every block: map the timestep through a fixed sinusoidal embedding (the same family of multi-frequency sin/cos features used to encode position in sequence models — a scalar t becomes a vector whose coordinates are sin and cos of t at geometrically-spaced frequencies, so nearby t's get nearby embeddings and the network can read off the level at any resolution), then through a small MLP, and *add* the resulting vector into each convolutional block. One network, all noise levels, conditioned per block.

While I'm modernizing the block, two more swaps that the setting motivates. I'll make each stage's conv block a *residual* block — the two convs compute a correction that's added to a shortcut of the input — because the denoiser is deep and residual connections make a deep stack trainable (the identity path keeps gradients alive). And I'll normalize inside each block, but not with batch normalization: the effective batch here can be tiny and batch statistics would be noise. **Group normalization** computes its statistics over groups of channels within a single example, independent of the batch, so it's stable at batch size one — exactly the regime I argued I'd be training in. Pair it with a smooth activation (SiLU) instead of bare ReLU. For downsampling, rather than max-pool I'll use a strided convolution, which *learns* its downsampling filter instead of fixing it to a max — the same upgrade FCN made on the upsampling side, now applied on the way down; and for upsampling, nearest-neighbor interpolation followed by a conv, which avoids the checkerboard artifacts that bare transposed convolutions are prone to. The U-topology, the per-level concatenating skips, the channel doubling/halving — all unchanged. Only the block internals and the conditioning are new, and each change traces to a property of the new task.

One genuinely open question in this denoiser setting is whether convolution at the per-resolution stages is enough, or whether the network also needs an explicit all-to-all interaction — a global mixing layer where every spatial position attends to every other — to coordinate long-range structure. Such a global mixer is expensive: its cost grows with the square of the number of spatial positions, so at a 32×32 grid (over a thousand positions) it is prohibitive, while at the 4×4 bottom of the U (sixteen positions) it is nearly free. The U already gives each output pixel a large receptive field through pooling, so in principle convolution alone could propagate global structure. The lean hypothesis to test is exactly this: strip the global mixer out of every per-resolution stage — every down stage and every up stage purely convolutional — and keep a single such mixing layer only at the very bottom of the U, where the grid is smallest and the cost is negligible. That isolates the question to "does per-resolution convolution suffice," while not paying for global mixing anywhere it would actually hurt throughput.

Here is the architecture filled into the dense-prediction harness, in the residual-conv denoiser form, every piece tied back to a step in the reasoning:

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim, max_period=10000, downscale_freq_shift=1.0):
    # scalar noise level t -> multi-frequency sin/cos vector, so the same net
    # reads off the level at every resolution (sequence-model positional sinusoid).
    half = dim // 2
    exponent = -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32)
    exponent = exponent / (half - downscale_freq_shift)
    freqs = torch.exp(exponent)
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)   # [B, dim]


class ResBlock(nn.Module):
    # two 3x3 convs as a residual correction (deep denoiser stays trainable);
    # GroupNorm (batch-independent, stable at batch size 1) + SiLU;
    # the timestep embedding is added in after the first conv -> per-block conditioning.
    def __init__(self, in_ch, out_ch, temb_ch, groups=32, eps=1e-6, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch, eps=eps)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)      # padded: keep size
        self.temb_proj = nn.Linear(temb_ch, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch, eps=eps)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb_proj(F.silu(temb))[:, :, None, None]    # inject noise level
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)                                   # residual


class Downsample(nn.Module):
    # learned downsampling: strided 3x3 conv (replaces fixed max-pool).
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=0)    # asymmetric pad below

    def forward(self, x):
        return self.conv(F.pad(x, (0, 1, 0, 1)))                 # keep even halving


class Upsample(nn.Module):
    # nearest-neighbor upsample then conv (avoids transposed-conv checkerboard).
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class AttnBlock(nn.Module):
    # the lone global mixer: every position attends to every other. Only affordable
    # at the smallest grid, so it lives at the bottleneck and nowhere per-resolution.
    def __init__(self, ch, groups=32, eps=1e-6, heads=1):
        super().__init__()
        self.norm = nn.GroupNorm(groups, ch, eps=eps)
        self.attn = nn.MultiheadAttention(ch, heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W).transpose(1, 2)     # [B, HW, C]
        h, _ = self.attn(h, h, h)
        return x + h.transpose(1, 2).reshape(B, C, H, W)          # residual


class ConvUNet(nn.Module):
    """Pure-convolutional U-Net denoiser: contracting path of residual conv
    blocks with channel-doubling + learned downsampling, a bottleneck, and a
    symmetric expanding path that at EVERY level concatenates the same-resolution
    encoder features (the 'where') onto the upsampled decoder features (the
    'what') and lets the next conv learn to fuse them. No per-resolution global
    mixing -- the hypothesis that per-resolution convolution suffices; a single
    global mixer survives only at the bottleneck, where the grid is smallest."""

    def __init__(self, in_channels=3, out_channels=3,
                 channels=(128, 256, 256, 256), layers_per_block=2,
                 groups=32, eps=1e-6, dropout=0.0):
        super().__init__()
        temb_ch = channels[0] * 4
        self.temb_dim = channels[0]
        self.time_mlp = nn.Sequential(                           # embed -> MLP -> temb
            nn.Linear(channels[0], temb_ch), nn.SiLU(), nn.Linear(temb_ch, temb_ch))

        self.conv_in = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        # contracting path: resnets (saving each output to skip across), then downsample
        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        ch = channels[0]
        skip_chs = [ch]
        for i, out_ch in enumerate(channels):
            blocks = nn.ModuleList()
            for _ in range(layers_per_block):
                blocks.append(ResBlock(ch, out_ch, temb_ch, groups, eps, dropout))
                ch = out_ch
                skip_chs.append(ch)                              # 'where' features kept
            self.down_blocks.append(blocks)
            if i < len(channels) - 1:                           # double channels as we shrink
                self.downsamplers.append(Downsample(ch))
                skip_chs.append(ch)
            else:
                self.downsamplers.append(None)

        # bottleneck: deepest, coarsest, widest features (max context); the lone
        # global mixer sits here between two residual blocks (cheap at this grid)
        self.mid1 = ResBlock(ch, ch, temb_ch, groups, eps, dropout)
        self.mid_attn = AttnBlock(ch, groups, eps)
        self.mid2 = ResBlock(ch, ch, temb_ch, groups, eps, dropout)

        # expanding path: upsample, CONCAT same-level encoder skip, resnets to fuse
        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        for i, out_ch in enumerate(reversed(channels)):
            blocks = nn.ModuleList()
            for _ in range(layers_per_block + 1):               # +1: one block per skip
                blocks.append(ResBlock(ch + skip_chs.pop(), out_ch, temb_ch, groups, eps, dropout))
                ch = out_ch
            self.up_blocks.append(blocks)
            self.upsamplers.append(Upsample(ch) if i < len(channels) - 1 else None)

        self.norm_out = nn.GroupNorm(groups, ch, eps=eps)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, x, timestep):
        temb = self.time_mlp(timestep_embedding(timestep, self.temb_dim))

        h = self.conv_in(x)
        skips = [h]
        for blocks, down in zip(self.down_blocks, self.downsamplers):
            for block in blocks:
                h = block(h, temb)
                skips.append(h)                                  # save 'where' at this level
            if down is not None:
                h = down(h)
                skips.append(h)

        h = self.mid2(self.mid_attn(self.mid1(h, temb)), temb)   # bottleneck (one global mixer)

        for blocks, up in zip(self.up_blocks, self.upsamplers):
            for block in blocks:
                h = torch.cat([h, skips.pop()], dim=1)           # concat, not add: learnable fuse
                h = block(h, temb)
            if up is not None:
                h = up(h)

        return self.conv_out(F.silu(self.norm_out(h)))           # same-size output field
```

Let me trace the causal chain back. I started with the irreducible tension of dense prediction: pooling buys the receptive field that decides *what* is at a pixel but destroys the resolution that decides *where*. The sliding window dodged this only by paying in speed and by forcing a single patch-size knob to trade context against localization, so I dropped it. The fully-convolutional reframing killed the speed and per-patch redundancy in one move by sharing computation, but left the output coarse. Learnable in-network upsampling let me climb back to full resolution, but upsampling a deep coarse map can't recover detail that was pooled away — the detail survives only in the early high-resolution maps. FCN reinjected those early maps, but only at two or three points and only as thin class scores combined by a fixed addition, so almost none of their content reached the output and the upward path stayed shallow. Routing the *full* high-resolution feature maps across from every encoder level and **concatenating** them onto the decoder — letting a learned convolution fuse *where* with *what* rather than committing to a fixed sum — fixed all three at once, and making the expanding path as deep and as wide as the contracting path gave it the capacity to carry context up to full resolution. That symmetry is the U. The contracting details (two stacked 3×3 convs per stage for cheap receptive field and extra nonlinearity; channel-doubling to hold capacity against the shrinking grid; no fully-connected layers to keep arbitrary input size) and the data-regime details (elastic-deformation augmentation and a border-weighted loss to learn from tens of images and to split touching cells; √(2/N) initialization and high momentum at batch size one) make it trainable in the few-data setting. Casting the same architecture as a same-size image-to-field predictor swaps valid-conv-plus-crop for padded convs (no shrinking border, no tiling rationale), and adds per-block conditioning on a scalar level through a sinusoidal-then-MLP embedding, residual blocks and batch-independent group normalization for a deep small-batch network, and learned strided-conv downsampling with checkerboard-free nearest-neighbor upsampling — while keeping the per-level concatenating skips untouched, and leaving every per-resolution stage purely convolutional — a single global mixer retained only at the cheap, smallest-grid bottleneck — to test whether per-resolution convolution alone suffices.
