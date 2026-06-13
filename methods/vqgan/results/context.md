# Context

## Research question

How can we synthesize high-resolution images — up to the megapixel range, both unconditionally and under rich conditioning (a class label, a semantic segmentation map, a depth map, a low-resolution image) — with a model that captures the *global composition* of a scene, so that the result is not just locally plausible texture but globally coherent structure?

The obstacle is a mismatch between the two model families that are good at the two halves of this problem. The architecture that is unmatched at modeling long-range, global dependencies is the transformer: it lets every element attend to every other element, with no built-in assumption that interactions are local. But that generality is exactly what makes it expensive — attention computes inner products between *all* pairs of inputs, so its cost grows quadratically with the sequence length. For an image the sequence is the set of pixels, and the number of pixels itself grows quadratically with the side length, so the cost of a pixel-level transformer explodes with resolution. In practice this caps pixel-space transformers at tiny images (around 64x64). Convolutional networks are the opposite: their local kernels make cost grow only linearly with the number of pixels, so they scale, but their hard locality-and-translation-invariance bias makes them weak at the holistic, long-range reasoning that global image composition demands, and the proliferation of specialized convolutional layers built to inject global or task-specific structure suggests the bias is often too restrictive.

A satisfactory solution would therefore have to reconcile these tensions: the expressivity of a global model against the cost that expressivity incurs at high resolution, the perceptual fidelity of the final image, and applicability across conditioning signals without bespoke per-task architectures.

## Background

**Transformers and the quadratic-attention wall.** A transformer layer maps each input to a query, key, and value, and forms its output by attention,

`Attn(Q,K,V) = softmax(QKᵀ / √d_k) V`,

where the `QKᵀ` term scores every pair of positions. For a sequence of length `n` this is an `n×n` matrix, so both compute and memory scale as `O(n²)`. For autoregressive use the upper-triangular entries are masked to `−∞` so position `i` only attends to `≤ i`, and a final linear layer predicts the next token's logits. The ability to attend to *all* positions is precisely what lets transformers learn long-range structure — and precisely why they become infeasible on long sequences. On images the sequence length is the pixel count, which scales as the square of the resolution, so the wall arrives early. Efforts to push it back either restrict the attention to local neighborhoods (Parmar et al., 2018, Image Transformer; Weissenborn et al., 2020), which reintroduces a locality assumption and limits expressivity, or keep the full receptive field but reduce cost only from `n²` to `n√n` (Child et al., 2019, Sparse Transformers; Ho et al., 2019, Axial Attention), which still leaves resolutions beyond 64 pixels prohibitive.

**Convolutional autoregressive image models.** Convolutional architectures exploit the 2D locality of images: a `k×k` kernel restricts interactions to a local neighborhood, so applying it costs linearly in the number of pixels and quadratically only in the (small, fixed) kernel size. PixelRNN/PixelCNN (van den Oord et al., 2016) and PixelSNAIL (Chen et al., 2018) use masked convolutions to model `p(x) = Πᵢ p(xᵢ | x_{<i})` autoregressively over the pixel grid. They are tractable and scale with resolution, but the locality bias makes long-range structure hard to capture, and at low resolution transformers have been shown to outperform them.

**The two-stage idea.** Rather than model the data distribution directly in pixel space, first learn a compact encoding of the data and then learn a probabilistic model of *that* encoding. Dai & Wipf (2019) gave theoretical and empirical evidence that learning a representation with one model and its distribution with a second helps; similar gains were reported with normalizing flows over autoencoder latents (Esser et al., 2020; Rombach et al., 2020) and with GANs trained on autoencoder representations (Liu et al., 2019) or on low-resolution wavelet coefficients (Han et al., 2020). These works establish that decoupling representation learning from distribution modeling is a viable and often beneficial frame for generative modeling.

**Discrete representation learning (VQVAE).** van den Oord et al. (2017) introduced the Vector-Quantized Variational Autoencoder. An encoder `E` maps an image to a spatial grid of vectors `ẑ = E(x) ∈ ℝ^{h×w×n_z}`; each spatial vector is quantized to its nearest entry in a learned codebook `Z = {z_k}_{k=1}^{K} ⊂ ℝ^{n_z}`,

`z_q = q(ẑ) := (argmin_{z_k ∈ Z} ‖ẑ_{ij} − z_k‖) ∈ ℝ^{h×w×n_z}`,

and a decoder `G` reconstructs `x̂ = G(z_q)`. The argmin has zero gradient almost everywhere, so the straight-through estimator (Bengio et al., 2013) copies the decoder's gradient at `z_q` straight back to `E(x)`. Because the straight-through copy routes the reconstruction gradient *around* the codebook, the codebook needs its own signal; the loss is

`L_VQ = ‖x − x̂‖² + ‖sg[E(x)] − z_q‖² + β‖z_q − sg[E(x)]‖²`,

with `sg[·]` the stop-gradient. The middle term (the codebook loss) pulls each chosen prototype toward the encoder output that selected it — online k-means dictionary learning — and the last term (the commitment loss, weight `β`) pulls the encoder output toward its chosen prototype so the encoder commits to codes rather than letting its outputs drift. An equivalent view of the quantized grid is a sequence of `h·w` integer indices into the codebook. The original work modeled the distribution over those indices with a convolutional autoregressive prior (a PixelCNN), and a hierarchical extension (Razavi et al., 2019, VQVAE-2) used a hierarchy of such codes. The limitation that matters for high-resolution global modeling: these priors are still *convolutional* density estimators, so they inherit the difficulty of capturing long-range interactions, and the reconstruction objective is a pixel-space `L₂` loss.

**Per-pixel reconstruction losses and the conditional mean.** A per-pixel `L₂` (or `L₁`) reconstruction loss penalizes the average pixel error and is minimized, under any residual uncertainty, by predicting the conditional mean — which smears high-frequency texture into blur. The harder the spatial reduction from image to code grid, the more residual uncertainty there is per code, and the more pronounced this effect.

**Perceptual and adversarial training signals.** Beyond pixel losses, two families of learned similarity signals were established in the literature. *Perceptual losses* compare deep features of a fixed pretrained network rather than raw pixels: the Learned Perceptual Image Patch Similarity (LPIPS; Zhang et al., 2018) measures distance in VGG feature space and tracks human judgments of similarity, rewarding texture and structure that pixel losses ignore (Johnson et al., 2016; Dosovitskiy & Brox, 2016). *Adversarial losses* use a discriminator trained to tell real images from generated ones as a learned training signal to the generator. Larsen et al. (2015) combined a VAE with a GAN discriminator; for image-to-image translation, the patch-based discriminator of pix2pix (Isola et al., 2017) classifies each `N×N` patch as real or fake rather than scoring the whole image, which concentrates the signal on local texture realism, uses fewer parameters, and applies to arbitrary image sizes. Mentzer et al. (2020) used such ingredients for high-fidelity compression.

**GAN training objectives.** A generative adversarial network (Goodfellow et al., 2014) plays a min–max game `min_G max_D E_x[log D(x)] + E_z[log(1 − D(G(z)))]`. In practice the saturating `log(1−D)` term is often replaced by more stable surrogates; the hinge formulation trains `D` to push real logits above `+1` and fake logits below `−1` (`E[relu(1−D(x))] + E[relu(1+D(x̂))]`) and trains the generator to maximize `D(x̂)`. GAN training is notoriously unstable, especially early when the generator output is meaningless, motivating warm-up schedules and careful balancing of the adversarial term against any reconstruction term.

**Generative pretraining on pixels.** Chen et al. (2020, ImageGPT) trained transformers autoregressively directly on pixel sequences to probe representation learning. Because pixel sequences are so long, inputs were capped (a shallow vector-quantizer encodes images up to 192×192, and a deliberately *shallow*, small-receptive-field encoder is used to keep the discrete representation as pixel-like as possible). This is the opposite of what high-resolution global modeling wants: a powerful first stage that captures as much context as possible.

## Baselines

**VQVAE + convolutional autoregressive prior (van den Oord et al., 2017; Razavi et al., 2019).** Two-stage: a vector-quantized autoencoder trained with `L₂` reconstruction plus codebook and commitment losses, then a PixelCNN/PixelSNAIL prior over the code indices. Core idea and math are the `L_VQ` and quantization equations above. Gap: the `L₂` reconstruction yields blurry codebooks, which forces a low compression rate (short downsampling) to keep reconstructions acceptable — leaving long code grids; and the convolutional prior struggles with the long-range structure of those grids. Both halves push against high-resolution global synthesis.

**Pixel-space autoregressive transformers (Parmar et al., 2018; Child et al., 2019; Ho et al., 2019; Chen et al., 2020).** Model `p(x) = Πᵢ p(xᵢ | x_{<i})` over pixels with (sparse/axial/local) attention. Core idea: full or restricted self-attention over the pixel grid. Gap: `O(n²)` attention over a pixel sequence whose length is quadratic in resolution; sparse/axial variants only reduce to `n√n`; local variants reintroduce a locality bias. None scale past roughly 64–192 pixels.

**Pixel-space convolutional autoregressive models (PixelCNN, PixelSNAIL).** Masked-convolution AR density estimators, `p(x)=Πᵢ p(xᵢ|x_{<i})`. Core idea: causal masked convolutions over the grid. Gap: cheap and scalable but weak at long-range dependencies; outperformed by transformers at low resolution.

**GANs for image synthesis (Goodfellow et al., 2014; and conditional/patch variants, Isola et al., 2017).** Adversarial generators produce sharp images and dominate sample-quality benchmarks, including class-conditional synthesis. Core idea: the min–max game above; for conditioning, condition both `G` and `D` on side information. Gap as a *baseline to beat for general controllable synthesis*: GANs do not provide an explicit likelihood, can be unstable, and typically need task-specific architectures and losses for each conditioning modality rather than one general mechanism.

## Evaluation settings

The natural unconditional and conditional image-synthesis testbeds existing at the time: ImageNet (Deng et al., 2009) and a restricted animal-class subset of it; LSUN Churches & Towers (Yu et al., 2015); CelebA-HQ (Karras et al., 2018) and FFHQ (Karras et al., 2019) for faces; ADE20K (Zhou et al., 2016) and COCO-Stuff (Caesar et al., 2018) for semantic-layout-conditioned synthesis; DeepFashion (Liu et al., 2016) for pose-guided synthesis; and depth/edge maps (e.g. via MiDaS, Ranftl et al., 2020, and DeepLab, Chen et al., 2017) as structure conditioning. Conditioning modalities studied: semantic segmentation maps, depth maps, edge maps, low-resolution images (for super-resolution), poses, and class labels. Standard metrics: negative log-likelihood / bits-per-dim for the autoregressive code model; Fréchet Inception Distance and Inception Score for sample quality (computed with a standard toolkit such as torch-fidelity); LPIPS for reconstruction fidelity of the first stage. For comparing a transformer prior against a convolutional one, negative log-likelihood at matched parameter counts, matched wall-clock, and matched step counts.

## Code framework

Pre-method primitives that already exist: a convolutional encoder/decoder (a ResNet-style stack with downsampling/upsampling, group normalization, and a non-local/self-attention block usable at low resolution), the optimizer (Adam), automatic differentiation with a stop-gradient primitive `sg[·]`, a perceptual-distance module (LPIPS over a fixed pretrained VGG), a patch-based convolutional classifier (the discriminator), and a generic causal-decoder sequence model with token + positional embeddings, multi-head masked self-attention, an MLP block, and a cross-entropy next-token head. The slots the method will fill are left as stubs.

```python
import torch, torch.nn as nn, torch.nn.functional as F

# ---- existing convolutional building blocks ----
class ResnetBlock(nn.Module): ...      # GroupNorm + Swish + Conv, residual
class Downsample(nn.Module): ...        # stride-2 conv
class Upsample(nn.Module): ...          # nearest + conv
class AttnBlock(nn.Module): ...         # non-local self-attention over a feature map (used at low res)

class Encoder(nn.Module):
    # x (B,3,H,W) -> features (B,n_z,h,w), h=H/2^m
    ...
class Decoder(nn.Module):
    # features (B,n_z,h,w) -> image (B,3,H,W)
    def __init__(self, ...):
        ...
        self.conv_out = nn.Conv2d(...)  # final layer producing the output image

# ---- existing perceptual / adversarial primitives ----
class LPIPS(nn.Module):                 # fixed VGG feature distance (perceptual loss)
    def forward(self, x, x_hat): ...
class PatchDiscriminator(nn.Module):    # classifies N x N patches real/fake
    def forward(self, img): ...

# ---- the discrete bottleneck: TODO (the method designs this) ----
class Quantizer(nn.Module):
    def forward(self, z):
        # map encoder output to nearest codebook entry; return z_q, codebook/commitment loss, indices
        # TODO
        pass

# ---- the first-stage objective: TODO (the method designs this) ----
class ReconstructionObjective(nn.Module):
    def forward(self, codebook_loss, x, x_hat, optimizer_idx, global_step, last_layer):
        # define the first-stage training objective
        # TODO
        pass

# ---- first-stage model ----
class AutoEncoderModel(nn.Module):
    def __init__(self, ...):
        self.encoder = Encoder(...)
        self.decoder = Decoder(...)
        self.quantize = Quantizer(...)
        self.loss = ReconstructionObjective(...)
    def encode(self, x): ...   # -> z_q, loss, indices
    def decode(self, z_q): ... # -> x_hat
    def training_step(self, batch, optimizer_idx):
        # TODO: define the first-stage update(s)
        pass

# ---- second-stage prior over discrete codes ----
class CausalSequenceModel(nn.Module):   # generic GPT-style decoder
    def forward(self, idx, targets=None):
        # token+pos embed -> masked self-attention blocks -> logits; cross-entropy if targets given
        ...

class CodePrior(nn.Module):
    def __init__(self, first_stage, transformer):
        self.first_stage = first_stage
        self.transformer = transformer
    def image_to_sequence(self, x):
        # encode, quantize, read off indices, choose an ordering
        # TODO
        pass
    def training_step(self, batch):
        # maximize log-likelihood of the index sequence (optionally conditioned)
        # TODO
        pass
    @torch.no_grad()
    def sample(self, steps, condition=None, temperature=1.0, top_k=None):
        # autoregressively sample indices, map back to codes, decode to an image
        # TODO
        pass
```
