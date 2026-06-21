## Research question

Diffusion models are the leading generative models of images, and to date they are built on a convolutional
U-Net backbone. In language and in visual recognition, the transformer operates as a single standard
architecture across tasks, and it carries a property of interest to image generation — *predictable scaling*:
making a transformer deeper/wider or feeding it more tokens lowers loss and improves downstream quality, with
smooth, measurable trends in compute.

The goal here is a diffusion backbone built from a standard transformer — patchify the (latent) image into
tokens and run an off-the-shelf Vision-Transformer stack — so that the transformer's scaling behaviour applies
to generative modelling. A diffusion denoiser is not an unconditional function of the image: at every step it
must be told the diffusion timestep `t` (where in the noising schedule we are), and for class-conditional
generation it must also be told the class label `c`. A vanilla transformer block — pre-norm, multi-head
self-attention, residual, pre-norm, MLP, residual — processes a single token sequence and takes `(t, c)`
as a global, per-example side signal. The question is *how to condition a transformer diffusion block on
`(t, c)`*.

## Background

**Diffusion denoisers and what they consume.** A Gaussian diffusion model defines a forward process
`q(x_t|x_0) = N(x_t; sqrt(alpha_bar_t) x_0, (1-alpha_bar_t) I)` and trains a network to invert it. With the
noise-prediction reparameterization the training loss is a simple MSE, `||eps_theta(x_t, t) - eps||^2`, but the
network is always a function of the noisy input *and the timestep* `t`; for conditional generation it is also a
function of the class label `c`. Timestep and class are turned into vectors first: `t` via a sinusoidal
frequency embedding followed by a small MLP, `c` via a learned per-class embedding table; both produced at the
backbone's hidden width so they can be combined and broadcast.

**Normalization with a learned affine.** Modern deep nets normalize activations and then apply a learned
per-channel affine. Layer normalization (Ba, Kiros & Hinton 2016) normalizes each token across its feature
dimension to zero mean / unit variance, `x_hat = (x - mu)/sigma`, then applies `gamma * x_hat + beta` with
learned vectors `gamma, beta`; it is the normalization used inside transformers because it is independent of
batch and sequence statistics. `gamma` and `beta` are per-channel knobs that rescale and re-bias every feature,
and one line of work makes them *functions of some conditioning input* rather than learned constants, so a
network is steered per example through normalization.

**Feature-wise modulation as a conditioning primitive.** Perez et al. (FiLM, AAAI 2018) define a general
layer: Feature-wise Linear Modulation applies, per feature channel, an affine transform whose coefficients are
regressed from a conditioning input,
`FiLM(F_{i,c} | gamma_{i,c}, beta_{i,c}) = gamma_{i,c} * F_{i,c} + beta_{i,c}`, with `gamma = f(x_i)` and
`beta = h(x_i)` produced (in practice) by a single small network that emits the `(gamma, beta)` vector. It is a
*conditional, per-channel affine* — agnostic to spatial location — and unifies a family of
conditional-normalization methods under one operation. In generative models the same primitive appears as
adaptive instance normalization: `AdaIN(x_i, y) = y_{s,i} * (x_i - mu(x_i))/sigma(x_i) + y_{b,i}` (Huang &
Belongie 2017; StyleGAN, Karras et al. 2019), where a "style" vector supplies the post-normalization scale and
bias.

**Adaptive normalization in diffusion.** Dhariwal & Nichol (2021), improving diffusion U-Nets, ablated how to
inject conditioning into each residual block and settled on adaptive group normalization:
`AdaGN(h, y) = y_s * GroupNorm(h) + y_b`, where `h` are the block's intermediate activations and
`y = [y_s, y_b]` is a linear projection of the (combined) timestep-and-class embedding. They reported that this
adaptive-normalization injection beat the older "add the embedding then group-norm" recipe. This is defined
inside a convolutional U-Net block.

**Initializing residual blocks as the identity.** Goyal et al. (2017), training ImageNet in one hour at very
large batch, initialize the *last* batch-norm scale `gamma` of each residual block to 0: with `gamma = 0` the
block's residual branch outputs zero, so the forward/backward signal initially flows only through the identity
skip connection, and "this initialization improves all models but is particularly helpful for large minibatch
training." Diffusion U-Nets adopt a sibling trick, zero-initializing the final convolution of each residual
block before the skip add. A deep residual network is thus made to start as a stack of identities, growing its
nonlinearity from there.

**The transformer / ViT substrate.** The Vision Transformer (Dosovitskiy et al. 2020) showed a standard
transformer (Vaswani et al. 2017) operating on a sequence of linearly-embedded image patches matches or beats
convolutional nets at scale. A ViT block is pre-norm: `x = x + MHSA(LN(x))`, then `x = x + MLP(LN(x))`, with
the MLP typically `4x` wide and a GELU nonlinearity, and sinusoidal/learned positional embeddings added to the
patch tokens. The transformer scaling laws are a property of this standard design.

**The surrounding pipeline.** Latent diffusion (Rombach et al. 2021) trains the diffusion model in the latent
space of a frozen VAE encoder, cutting compute by an order of magnitude. Classifier-free guidance (Ho &
Salimans 2022) trains the conditional model with the label randomly dropped to a learned null embedding, then
at sampling time extrapolates between conditional and unconditional predictions to trade diversity for
fidelity. These frame the training/sampling setup.

## Baselines

**In-context conditioning (append conditioning tokens).** Embed `t` and `c` as two extra tokens and
prepend/append them to the image-token sequence, exactly like a ViT `cls` token, run unmodified transformer
blocks, and drop the extra tokens at the end. This makes zero change to the block and adds negligible Gflops;
the conditioning acts on the image tokens through self-attention.

**Cross-attention.** Borrowed from the encoder-decoder transformer (Vaswani et al. 2017) and from latent
diffusion's text conditioning: keep the image tokens as queries and add, after the self-attention sublayer, a
multi-head cross-attention layer whose keys and values come from the conditioning embeddings of `t` and `c`.
Each image token attends to the conditioning. This is the most expressive option and natural when the
conditioning is itself a sequence (e.g. text); it adds an attention sublayer to every block.

**Adaptive normalization of the block (the convolutional-U-Net form).** As above (AdaGN / FiLM / AdaIN):
replace a normalization layer's learned affine with scale and shift regressed from the conditioning embedding,
`y_s * Norm(h) + y_b`. Condition by modulating normalization, channel-wise, with one function applied
identically across spatial positions. It is the cheapest of the three (one small projection, no extra
attention) and matches the inductive bias of a global signal (the same modulation everywhere). It is defined
inside a convolutional residual block (group norm, conv branches).

## Evaluation settings

The natural yardsticks already in use for class-conditional image generation at the time:

- **Datasets.** Class-conditional ImageNet at `256x256` and `512x512` for the large-scale regime; for a small,
  fast class-conditional setting, CIFAR-10 (`32x32`, 10 classes). Diffusion is run in a VAE latent space (e.g.
  a downsample-factor-8 encoder maps a `256x256x3` image to a `32x32x4` latent) or directly in pixel space for
  small images.
- **Backbone configs.** Standard transformer size tiers that jointly scale depth `N`, hidden width `d`, and
  head count (Small / Base / Large / XL), and a patch-size knob `p` (e.g. 2, 4, 8) that trades token count `T =
  (I/p)^2` against compute. For the small CIFAR setting, a compact convolutional U-Net at a few channel widths
  (e.g. ~9M / ~36M / ~140M parameters) is the analogous yardstick.
- **Training.** AdamW, a constant learning rate around `1e-4` to `2e-4`, batch size in the hundreds, an
  exponential moving average of weights (decay ~0.9999) used for all reported evaluation, horizontal-flip
  augmentation only.
- **Diffusion + sampling.** A linear noise schedule with `t_max = 1000` steps from `1e-4` to `2e-2`; sampling
  via a DDPM or DDIM sampler with a reduced number of steps (e.g. 50–250), optionally with classifier-free
  guidance.
- **Metric.** Fréchet Inception Distance (FID), the standard generative-image metric, computed between
  generated samples and the dataset's reference statistics (lower is better); FID is sensitive to
  implementation details, so a fixed evaluation suite (e.g. clean-fid against the train set) is used for
  comparability. Inception Score, sFID, and Precision/Recall serve as secondary metrics. Forward-pass Gflops
  is tracked as the complexity axis.

## Code framework

The available substrate is a standard patch-transformer diffusion harness: patchify the latent into tokens, add
positional embeddings, embed the timestep and class label into vectors, run a stack of transformer blocks, and
decode the tokens back to a noise prediction. The side-information path is the open slot.

```python
import torch
import torch.nn as nn


class TimestepEmbedder(nn.Module):
    """Existing: scalar diffusion timestep -> vector (sinusoidal features + MLP)."""
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(256, hidden_size), nn.SiLU(),
                                 nn.Linear(hidden_size, hidden_size))
    def forward(self, t):
        return self.mlp(sinusoidal_features(t, 256))   # sinusoidal_features: existing utility


class LabelEmbedder(nn.Module):
    """Existing: class label -> vector (learned table; +1 row for the null/dropped label)."""
    def __init__(self, num_classes, hidden_size, dropout_prob=0.1):
        super().__init__()
        self.table = nn.Embedding(num_classes + 1, hidden_size)
    def forward(self, y, train):
        # (label dropout for classifier-free guidance handled here)
        return self.table(y)


def build_side_information(t_emb, c_emb):
    """TODO: define the per-example side-information representation."""
    raise NotImplementedError


class BackboneBlock(nn.Module):
    """A standard pre-norm ViT block (LN -> MHSA -> residual; LN -> MLP -> residual),
    with a single empty slot for side information.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = MultiHeadSelfAttention(hidden_size, num_heads)     # existing
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = MLP(hidden_size, int(hidden_size * mlp_ratio))      # existing, GELU
        # TODO: side-information slot

    def forward(self, x, side):
        # TODO: fill the side-information slot.
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class FinalLayer(nn.Module):
    """Existing: final norm + linear decode of each token into patch pixels (predicted noise)."""
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
    def forward(self, x, side):
        # TODO: fill the side-information slot, if the head needs it.
        return self.linear(self.norm(x))


class PatchTransformerDiffusion(nn.Module):
    """Existing harness: patchify -> +pos_embed -> N blocks -> decode. The side-information
    path through the generic side-information slot is what gets designed."""
    def __init__(self, input_size, patch_size, in_channels, hidden_size, depth,
                 num_heads, num_classes, out_channels):
        super().__init__()
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)  # existing
        self.pos_embed = nn.Parameter(sincos_pos_embed(hidden_size, input_size, patch_size),
                                      requires_grad=False)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)
        self.blocks = nn.ModuleList(
            [BackboneBlock(hidden_size, num_heads) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, out_channels)

    def forward(self, x, t, y):
        x = self.x_embedder(x) + self.pos_embed
        side = build_side_information(self.t_embedder(t), self.y_embedder(y, self.training))
        for block in self.blocks:
            x = block(x, side)
        x = self.final_layer(x, side)
        return unpatchify(x)
```

The empty slot is the side-information path shared by the block stack and the final head.
