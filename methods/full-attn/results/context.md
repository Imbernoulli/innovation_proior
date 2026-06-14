# Context: long-range structure in convolutional image generators (circa 2017-2018)

## Research question

A convolutional generator (or, equivalently, a convolutional denoiser inside a generative model)
turns a low-resolution feature map into a high-resolution image by stacking layers whose every
output value is a function of a small spatial neighborhood of its input. The problem this leaves
open is **coordinating image content across long spatial distances**. Convolution has a fixed,
local receptive field: a single conv layer can only relate a pixel to its immediate neighbors, so
two far-apart regions of an image become statistically coupled only after the signal has passed
through enough layers for their receptive fields to overlap. That makes long-range dependencies
hard to represent (a small stack of layers simply cannot reach across the image), hard to optimize
(many layers must be tuned in concert to express one global relationship), and brittle (such
multi-layer couplings tend to fail on inputs unlike the training set). The visible consequence is
that convolutional generators reproduce *local* statistics — texture, color, edges — convincingly
while getting *global* geometry wrong: a generated animal can have correct fur everywhere yet
mutually inconsistent or duplicated limbs; a background can be locally plausible but globally
discontinuous. The precise goal is a same-shape feature-map component that preserves the
convolutional backbone's local path, adds a content-dependent route for long-distance coordination,
stays computationally viable at image feature-map sizes, and starts from a behavior that does not
destabilize a network whose local modeling already works.

## Background

The dominant image-generation backbone is the deep convolutional network. Convolutional GANs
(Radford et al. 2016; Karras et al. 2018) and the convolutional denoising networks used inside
score- and diffusion-based generative models are built almost entirely from convolutional residual
blocks, often with normalization and a per-step (timestep or noise-level) embedding folded in. The
governing fact about these backbones is the **locality of convolution**: a kernel of size 3×3
mixes only a 3×3 neighborhood, so the receptive field grows only linearly with depth, and any
relationship between two positions more than a few pixels apart must be carried through many
intermediate layers. Capturing long-range structure is understood to be central to perception, and
two families of tools were on the table for it.

The first is the **classical non-local idea** from image processing: non-local means and bilateral
filtering compute the value at a pixel as a weighted average of *all* other pixels, weighted by
feature similarity rather than spatial proximity. This is a fixed (non-learned) operation, but it
establishes the template — *response at a location = similarity-weighted sum over all locations* —
that captures distant dependencies in one step.

The second is **attention from sequence modeling**. An attention layer computes the response at one
position as a weighted average of value vectors at all positions, where the weights come from a
learned, content-dependent compatibility between a query at that position and keys at the others.
The relevant pieces of background here are precise:

- **Scaled dot-product attention** (Vaswani et al. 2017). With queries `Q`, keys `K`, values `V`,
  the layer computes `softmax(QKᵀ/√d_k) V`. The softmax turns learned similarities into
  nonnegative weights that sum to one, so each output is a convex combination of value vectors —
  a content-addressed average. The `1/√d_k` factor is load-bearing: if the components of a query
  and a key are independent with mean 0 and variance 1, their dot product `q·k = Σ_{i=1}^{d_k} q_i
  k_i` has mean 0 and variance `d_k`, so for large channel dimension the logits grow like `√d_k`,
  the softmax saturates, and its gradients vanish; dividing by `√d_k` restores unit-variance logits.
- **Multi-head attention** (Vaswani et al. 2017). Rather than one attention over the full channel
  dimension, project to `h` lower-dimensional subspaces (`d_k = d_v = d_model/h`), attend in each
  independently, then concatenate and project. Because each head is `1/h` as wide, the total cost is
  comparable to one full-width head, while different heads can specialize to different relations.
  Each attention sublayer is wrapped in a residual connection and normalization.
- **Self-attention** (a.k.a. intra-attention; Cheng et al. 2016; Parikh et al. 2016): the special
  case where queries, keys, and values all come from the *same* set of positions, so the layer
  relates a sequence to itself.

These pieces were developed for sequences. The motivating observations that frame the problem are
qualitative and pre-method: that convolution's receptive field is local and therefore long-range
coupling is slow to form with depth; that classical non-local filtering already captures distant
dependencies by similarity-weighted averaging; and that a similarity-weighted average over all
positions costs `O(N²)` in the number of feature locations `N = H·W`, which is cheap at a small
feature map and expensive at a large one. The cost of `N²` pairwise interactions per layer is the
hard constraint that any application of attention to a feature map has to live within.

## Baselines

These are the prior building blocks a long-range mechanism for image generators would be measured
against and built from.

**Stacked convolutional / residual backbone (Radford et al. 2016; He et al. 2016 for the residual
block).** The default. A residual block computes `x + F(x)` where `F` is a short stack of
convolutions (with normalization and nonlinearity), so signals and gradients flow through the
identity path and depth becomes safe. Long-range dependencies are reached purely by stacking: after
enough downsampling/conv layers the receptive field eventually spans the image. **Limitation:**
the coupling between two distant positions is mediated by a long chain of local operations, so it
is expensive in depth to represent, hard to optimize jointly, and statistically fragile; enlarging
the kernels to shorten the chain forfeits the efficiency that makes local convolution attractive in
the first place.

**Non-local operation / non-local block (Wang et al. 2018).** A generic operation that computes the
response at position `i` as a normalized similarity-weighted sum over all positions `j`:

```
y_i = (1 / C(x)) · Σ_j  f(x_i, x_j) · g(x_j),
```

where `f` is a pairwise affinity and `g` a per-position (unary) embedding, both implemented as 1×1
convolutions. Two instantiations matter. The *embedded-Gaussian* form `f(x_i,x_j) =
exp(θ(x_i)ᵀφ(x_j))` with `C(x) = Σ_j f` makes `y_i` exactly a **softmax over `j`** of learned
similarities — i.e. the sequence self-attention layer is recovered as a special case. The
*dot-product* form `f(x_i,x_j) = θ(x_i)ᵀφ(x_j)` normalizes by the number of positions `C(x) = N`
instead. The operation is wrapped into a **non-local block**

```
z_i = W_z y_i + x_i,
```

a residual around the non-local op, with `W_z` a 1×1 conv **initialized to zero** so that at the
start of training the block is exactly the identity and can be dropped into an already-trained
network without breaking it. For efficiency the embedding channels are bottlenecked to half the
input channels, and the keys/values can be spatially subsampled (e.g. by max-pooling) to cut the
pairwise cost. **Limitation / open question:** this block was demonstrated for *recognition* —
image classification, object detection, and video — where it improves accuracy. Whether a
similarity-weighted, all-positions average helps a *generative* convolutional network coordinate
the long-range structure it gets wrong, and how to insert it so it does not destabilize generator
training, is not addressed.

**Larger convolution kernels / deeper stacks.** The brute-force alternative for more reach.
**Limitation:** capacity grows but at the cost of the computational and statistical efficiency of
local convolution; a bigger kernel still imposes a fixed-shape neighborhood rather than a
content-dependent one, and added depth makes the optimization of long-range coordination harder,
not easier.

## Evaluation settings

The natural yardsticks for an image-generation backbone are sample-quality metrics computed against
a reference dataset, plus the generator's own training objective.

- **Datasets.** For the small-image regime, CIFAR-10 (32×32, 50,000 training images), used
  unconditionally; for the large-scale conditional regime, ImageNet (LSVRC2012) at resolutions such
  as 128×128.
- **Fréchet Inception Distance (FID).** Embed real and generated images with a fixed Inception
  network and compute the Fréchet (Wasserstein-2) distance between the two Gaussian-fit feature
  distributions; lower is better. It is the standard distributional measure of sample quality and
  diversity.
- **Inception Score (IS)** — the KL between the per-sample conditional class distribution and the
  marginal under an Inception classifier; higher is better — used as a complementary metric where
  applicable.
- **Protocol.** Fix the data pipeline, the training objective, the optimizer and its schedule, and
  the sampler; vary only the architecture; generate a fixed number of samples (e.g. tens of
  thousands) and score them against the reference set. For a diffusion denoiser the generative
  objective is a fixed per-step regression target and the sampler is a fixed deterministic or
  stochastic update; for a GAN the objective is an adversarial (e.g. hinge) loss with stabilizers
  such as spectral normalization and separate generator/discriminator learning rates.

## Code framework

The scaffold is a standard convolutional UNet-style denoiser/generator harness. The data pipeline,
the optimizer, the loss target, the per-step (timestep) embedding, the convolutional residual block,
and the up/down-sampling are already fixed. The harness exposes one generic same-shape feature-map
hook inside each scale; the hook is empty.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Existing convolutional residual block: norm -> conv -> add timestep embedding ->
    norm -> conv, wrapped in a residual. Mixes information locally only."""

    def __init__(self, in_ch, out_ch, temb_ch, groups=32):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb_proj = nn.Linear(temb_ch, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return self.skip(x) + h


class FeatureMapBlock(nn.Module):
    """Unresolved per-resolution operator. It takes a feature map [B, C, H, W]
    plus an optional timestep embedding and returns a feature map of the same shape.
    The convolutional path already mixes information locally; this is the empty slot."""

    def __init__(self, channels, groups=32):
        super().__init__()
        # TODO: the block we will design here.
        pass

    def forward(self, x, temb=None):
        # TODO: process the feature map and return a tensor of the same shape as x.
        raise NotImplementedError


class UNetDenoiser(nn.Module):
    """Existing UNet harness: conv-in, a stack of ResBlock plus one same-shape
    feature-map hook at each scale with downsampling between levels, a bottleneck,
    then the mirrored upsampling path, then conv-out. Only `FeatureMapBlock` is unspecified."""

    def __init__(self, in_ch=3, base_ch=128, ch_mult=(1, 2, 2, 2),
                 num_res_blocks=2, temb_ch=512):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        # ... timestep embedding MLP, down/up sampling, ResBlocks and FeatureMapBlocks
        #     wired per resolution level, conv_out -> in_ch (omitted; standard plumbing) ...

    def forward(self, x, timestep):
        # embed timestep -> temb; run conv_in; at each resolution apply ResBlock then
        # FeatureMapBlock; downsample; bottleneck; upsample with skip connections;
        # conv_out. Returns a same-shape noise/score prediction.
        raise NotImplementedError
```

The harness supplies a feature map at each resolution; `FeatureMapBlock` is the only open slot.
