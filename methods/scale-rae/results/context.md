## Research question

Latent diffusion has become the backbone of high-quality image synthesis: instead of denoising in
pixel space, one trains a compact latent with a variational autoencoder (VAE) and runs the diffusion
model there. The latent is small (a few dozen channels) and is optimized purely for pixel
reconstruction, so it is not *semantic*: a model that must both *understand* an image and *generate*
one needs two different visual spaces — a high-dimensional semantic encoder for perception and a
low-dimensional VAE latent for generation.

A tempting alternative is to diffuse directly in the *semantic representation space* of a powerful
pretrained vision encoder — the same high-dimensional tokens used for perception. If a generator could
produce those tokens and a lightweight decoder could render them to pixels, a single space would serve
both seeing and generating. The Representation Autoencoders (RAE) line gives this route a
controlled-setting proof of concept: on class-conditional ImageNet, diffusion in a frozen
representation space (encoder frozen, only a decoder trained) converges faster and reconstructs better
than VAE-based diffusion.

The open question is: **can representation-space diffusion be carried from that controlled setting to
large-scale, freeform text-to-image generation?** That regime brings open-ended prompts, enormous
visual diversity (including hard cases like rendered text/typography), billion-parameter models, and an
LLM as the conditioning source.

## Background

**Latent diffusion and the VAE latent.** A VAE (Kingma & Welling, 2014) is trained to encode an image
into a low-dimensional latent and decode it back. Latent diffusion (Rombach et al., 2022) runs the
generative model in that latent rather than in pixels, cutting compute dramatically. Subsequent T2I
systems (SDXL, SD3, FLUX) improved the VAE with larger/higher-quality training, but the latent stays
low-dimensional (channel count typically below ~64) and is shaped only by reconstruction. A separate
line pushes compression further with discrete codes (VQ-VAE), at further cost to fidelity.

**Representation encoders.** In parallel, self-supervised learning (DINO, MAE, MoCo, SimCLR),
language supervision (CLIP, SigLIP), and their combinations (SigLIP-2) produce high-dimensional,
semantically structured patch features that transfer broadly across perception tasks. These are the
features unified models already use for *understanding*. A representative encoder, SigLIP-2 So400M at
patch size 14 and 224x224 input, emits a 16x16 grid (256 tokens) of dimension 1152.

**Why high-dimensional latents were considered hard to generate in.** Folklore held such features were
too "abstract" for generative modeling, or outright intractable. There is a concrete reason. A
diffusion model injects Gaussian noise across the entire latent during training; this spreads the data
distribution's support over the whole ambient space, turning what might be a thin data manifold into a
full-rank one. The model therefore needs capacity that scales with the *full* dimensionality, not the
intrinsic dimension. Made precise in the controlled study that introduced representation-space
diffusion: for a denoiser whose width `d` is smaller than the latent token dimension `n`, the training
loss is bounded below by the sum of the tail eigenvalues of the covariance of `(eps - x)`,
`L >= sum_{i=d+1}^{n} lambda_i`. Empirically a narrow denoiser simply fails to converge on these
latents, and convergence appears only once the denoiser's width meets or exceeds the token dimension.

**Noise schedules depend on dimension.** Rectified-flow systems (SD3) observed that *resolution*
changes the effective signal-to-noise ratio: recovering a roughly constant signal from `n` pixels has
uncertainty scaling like `sigma(t,n) = (t/(1-t)) * sqrt(1/n)`, so a higher-resolution image at the
same nominal timestep is effectively *less* corrupted and needs the schedule pushed toward more noise.
SD3 implement this by shifting timesteps, `t_m = alpha t_n / (1 + (alpha - 1) t_n)` with
`alpha = sqrt(m/n)`. The controlled representation-space study generalized the same correction from
spatial resolution to the *effective data dimension* `m = N x d` (tokens times channels), using a base
dimension `n = 4096`; without the shift, generation quality on ImageNet collapsed (gFID ~23 vs ~5).

**Design choices the controlled ImageNet RAE recipe added.** Three further ingredients were introduced
there to make representation-space diffusion work at ImageNet scale:
- A *wide, shallow denoising head*. Because ImageNet-scale denoisers were often narrower (~1024) than
  the latent dimension (e.g., 1152), and widening the whole backbone costs quadratically, a cheap fix
  appends a shallow but wide head on top of a normal-width backbone — the backbone produces a
  conditioning signal and the wide head produces the velocity — buying the required width locally.
- *Noise-augmented decoding*: train the decoder not only on true encoder tokens `z` but on perturbed
  `z + n`, `n ~ N(0, sigma^2 I)` with `sigma ~ |N(0, tau^2)|`, so the decoder tolerates the slightly
  off-manifold tokens that a diffusion sampler produces at inference.
- A decoder trained with the usual reconstruction stack: L1 + perceptual (LPIPS) + adversarial losses,
  encoder kept frozen.

**Measured behavior of decoders.** Several measurements frame the landscape:
- *Decoder data composition vs. scale.* Training a frozen-encoder decoder on data beyond ImageNet
  yields only marginal gains on ImageNet reconstruction, moderate gains on diverse web imagery (e.g.,
  YFCC), but essentially no improvement on rendered-text reconstruction until *text-specific* data is
  added — at which point text reconstruction improves sharply (rFID on a text-rendering set dropping
  from ~2.64 with ImageNet-only data to ~1.62 once text data is included).
- Modern T2I transformers at billions of parameters carry hidden widths of 2048 and up.
- Noise-augmented decoding uses a `tau` parameter that must be kept moderate (~0.2) to maintain
  decoder training stability.

**The conditioning interface.** MetaQuery shows that a text LLM can condition a generator through a
set of *learnable query tokens* placed at the image-generation slots in the sequence: the LLM processes
text plus queries, and the query-token hidden states (after a small MLP connector) become the diffusion
model's conditioning, trained with ordinary image-caption pairs and the standard diffusion loss.

## Baselines

**VAE-based latent diffusion (the incumbent T2I pipeline).** Encode with a strong VAE (e.g., the FLUX
VAE), run a transformer/U-Net diffusion model in the compressed latent, decode to pixels. Core math:
VAE gives latent `z = E(x)` with low channel count; the diffusion model is trained on `z` and decoded
by `D`.

**Standard transformer denoiser (DiT).** Patchify the latent into tokens, add positional embeddings,
process with transformer blocks conditioned on timestep (+ class/text) through adaptive LayerNorm —
the conditioning vector predicts per-block (shift, scale, gate) modulations applied around attention
and the MLP, with the gates zero-initialized. LightningDiT modernizes this backbone with RMSNorm,
SwiGLU feed-forward layers, QK-normalized attention, rotary position embeddings on the vision grid,
and Gaussian Fourier timestep embeddings.

**Rectified-flow / flow-matching diffusion.** Define a straight path `x_t = (1 - t) x + t eps` from
data `x` at `t=0` to noise `eps ~ N(0, I)` at `t=1`; the time-derivative is the constant velocity
`v = eps - x`; train a network `v_theta(x_t, t)` with plain MSE to that target; sample by integrating
the ODE backward from noise with Euler steps `x_{t-} = x_t + (sigma_{t-} - sigma_t) v_theta`. This is
the modern default (SD3, FLUX).

**Compressed-representation unified models.** Several unified systems use continuous representation
features but heavily downsample them for generation, or use compressed embeddings for both paths, or
pair a semantic encoder for understanding with a different encoder for generation that relies on a
strong separate diffusion decoder to return to pixels.

## Evaluation settings

- **Reconstruction quality (decoders).** rFID-50k computed on reconstructed images across three
  representative domains: ImageNet-1k (object-centric), a diverse web set (YFCC), and a held-out
  text-rendering/typography set. (Lower is better.)
- **Text-to-image generation.** GenEval (compositional prompt-following score) and DPG-Bench
  (dense-prompt graph score). Generation uses a fixed-step Euler sampler (50 steps).
- **Multimodal understanding.** Standard VL benchmarks: MME, TextVQA, AI2D, SeedBench, MMMU,
  MMMU-Pro.
- **Protocol.** Two-stage training (large-scale pretraining from scratch, then finetuning on a small
  high-quality set), with matched configurations so that the only variable under study is the latent
  space and its decoder. Models span LLM backbones {1.5B, 7B} and denoiser sizes {0.5B ... 9.8B}.

## Code framework

A practical scaffold needs a frozen image encoder, a text model, a trainable pixel decoder, a
transformer denoiser conditioned through adaptive normalization, a straight-path flow objective, and
optimizer groups that can treat pretrained and from-scratch parameters differently.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- frozen image encoder -------------------------------------------------
class FrozenImageEncoder(nn.Module):
    """Pretrained encoder, weights frozen. Maps an image to N tokens of dim D."""
    def __init__(self, model_name):
        super().__init__()
        self.backbone = load_pretrained(model_name)   # provided externally
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):                              # x: (B, 3, H, W)
        return self.backbone.patch_tokens(x)          # (B, N, D)


# ---- pixel decoder --------------------------------------------------------
class PixelDecoder(nn.Module):
    """ViT decoder: tokens -> pixels. Trained; encoder stays frozen."""
    def __init__(self, num_tokens, token_dim, patch_size, depth, width, heads):
        super().__init__()
        # TODO: token embed, transformer blocks, per-patch pixel head
        pass

    def unpatchify(self, patches):
        # TODO: (B, N, p*p*3) -> (B, 3, H, W)
        pass

    def forward(self, tokens):                         # (B, N, D) -> (B, 3, H, W)
        # TODO
        pass


def reconstruction_loss(x, x_hat):
    # TODO: pixel + perceptual + adversarial terms
    pass


# ---- conditioning interface -----------------------------------------------
class ConditioningTokens(nn.Module):
    """Learnable tokens inserted into image-generation slots."""
    def __init__(self, num_tokens, llm_dim):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(num_tokens, llm_dim) / llm_dim ** 0.5)

    def expand(self, batch):
        return self.tokens.unsqueeze(0).expand(batch, -1, -1)


class Connector(nn.Module):
    """Maps LLM hidden states into the denoiser's conditioning space."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # TODO
        pass

    def forward(self, h):
        # TODO
        pass


# ---- generic transformer denoiser -----------------------------------------
def modulate(x, shift, scale):
    # TODO: broadcast shift/scale to token grids
    pass


def gate(x, value):
    # TODO: broadcast gates to token grids
    pass


class DenoiserBlock(nn.Module):
    """Transformer block conditioned via adaptive LayerNorm."""
    def __init__(self, width, heads, cond_dim):
        super().__init__()
        # TODO: norm + attention + MLP + adaLN modulation
        pass

    def forward(self, x, c):
        # TODO
        pass


class Denoiser(nn.Module):
    """Predicts a flow velocity on a square token grid."""
    def __init__(self, num_tokens, token_dim, width, depth, heads, cond_dim):
        super().__init__()
        # TODO: reshape tokens to grid, embed, add position, run blocks, unpatchify
        pass

    def forward(self, x_t_grid, t, cond):              # -> velocity, same shape as x_t_grid
        # TODO
        pass


# ---- flow-matching objective ----------------------------------------------
def tokens_to_grid(x):
    # TODO: (B, N, D) -> (B, D, H, H)
    pass


def grid_to_tokens(x):
    # TODO: (B, D, H, H) -> (B, N, D)
    pass


class FlowMatching:
    """Straight-path flow matching: x_t = (1-t) x + t eps, target velocity eps - x."""
    def __init__(self, num_tokens, token_dim, base_dim=4096):
        # TODO: store the effective dimension ratio and sampler grid
        pass

    def sample_timestep(self, batch, device, dtype):
        # TODO: timestep distribution (and any dimension-dependent adjustment)
        pass

    def training_loss(self, denoiser, x, cond):
        x_grid = tokens_to_grid(x)
        eps = torch.randn_like(x_grid)
        t = self.sample_timestep(x_grid.shape[0], x_grid.device, x_grid.dtype)
        # TODO: build x_t = (1-t)x + t eps on the grid
        x_t = None
        target = eps - x_grid
        pred = denoiser(x_t, t, cond)
        return ((pred - target) ** 2).mean()

    @torch.no_grad()
    def euler_sample(self, denoiser, cond, shape, num_steps, device):
        # TODO: integrate the ODE from noise (t=1) to data (t=0)
        pass


# ---- training loop --------------------------------------------------------
class GenerationModel(nn.Module):
    def __init__(self, llm, encoder, decoder, connector, denoiser, flow, num_queries):
        super().__init__()
        self.llm = llm
        self.encoder = encoder
        self.decoder = decoder
        self.connector = connector
        self.denoiser = denoiser
        self.flow = flow
        self.cond_tokens = ConditioningTokens(num_queries, llm.hidden_size)

    def forward(self, prompt_ids, image, image_token_indices, answer_image_mask):
        # TODO: encode target image, insert conditioning tokens, gather their hidden states,
        # project to denoiser conditioning, and apply the flow loss only to answer images
        pass


def build_optimizer(model, lr, diffusion_lr, weight_decay):
    # TODO: parameter groups for pretrained text parameters and from-scratch diffusion head
    pass


def train_step(batch, model, optimizer):
    loss = model(batch["prompt_ids"], batch["image"],
                 batch["image_token_indices"], batch["answer_image_mask"])
    # TODO: backward + optimizer step
    return loss
```
