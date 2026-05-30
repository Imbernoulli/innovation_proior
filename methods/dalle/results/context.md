# Context

## Research question

Can text-to-image generation be made high-fidelity and *flexible* — controllable by arbitrary natural language, and even capable of rudimentary image-to-image translation — simply by scaling a single, general autoregressive model on a very large dataset, rather than by hand-engineering task-specific architectures?

For years, text-to-image systems were evaluated on small datasets (MS-COCO, CUB-200) and improved through bespoke machinery: multi-scale generators, attention modules, auxiliary text-image matching losses, extra conditioning signals. Despite real progress, samples still suffered object distortion, illogical placement, and unnatural blending of foreground and background. The open question is whether *dataset size and model size* are the binding constraint — i.e. whether a single large generative model, trained on hundreds of millions of image-text pairs, would subsume the specialized tricks and yield a flexible, language-controllable image generator with capabilities (like image-to-image translation) emerging rather than engineered.

## Background

**Scaling autoregressive transformers.** When compute, model size, and data are scaled carefully, autoregressive transformers (Vaswani et al., 2017) reach impressive quality across modalities: text (Radford et al., 2019), images modeled as pixel sequences (Chen et al., 2020), audio (Dhariwal et al., 2020). This suggests modeling text and image jointly as a single autoregressive *token* stream.

**Why not model pixels directly.** Using pixels as the image tokens is infeasible at high resolution: a 256×256×3 image is ~200k values, far too long a sequence for a transformer's quadratic attention. Worse, likelihood objectives over pixels tend to prioritize short-range dependencies (Salimans et al., 2017), so most of the modeling capacity is spent on high-frequency texture rather than the low-frequency structure that makes objects recognizable. A useful approach must first *compress* the image into a short sequence of tokens that preserve recognizable structure.

**Two-stage discrete-token compression.** Vector-quantized autoencoders (van den Oord et al., 2017; Razavi et al., 2019) established a two-stage recipe: first learn an autoencoder that maps an image to a small grid of *discrete* codes (and back), then fit a powerful autoregressive model over those codes. This decouples a fast feed-forward compressor from a slow expressive prior, and shrinks the sequence the prior must model by orders of magnitude. The discreteness makes the code grid a categorical sequence — exactly what an autoregressive transformer over a shared text+image vocabulary needs.

**Variational autoencoders and the ELBO.** The autoencoder side is naturally framed as a VAE (Kingma & Welling, 2013; Rezende et al., 2014): an encoder `q_φ(z|x)`, a decoder `p_θ(x|z)`, a prior, trained on the evidence lower bound `E_{q_φ(z|x)}[log p_θ(x|z)] − D_KL(q_φ(z|x) ‖ p(z))`. Upweighting the KL term by `β > 1` (Higgins et al., 2016) is a known knob on the latent channel.

**The discrete-latent gradient obstacle.** When `q_φ(z|x)` is *categorical* (picking one of K codes), the reparameterization gradient is unavailable. Two known workarounds: an online cluster-assignment / straight-through estimator (Bengio et al., 2013), as used by the vector-quantized autoencoders; and the *Gumbel-Softmax / Concrete* relaxation (Jang et al., 2016; Maddison et al., 2016), which replaces the hard categorical sample with a continuous, temperature-controlled, *reparameterizable* approximation that hardens into a true categorical as the temperature `τ → 0`.

**Pixel likelihood and bounded support.** The common `ℓ1` / `ℓ2` reconstruction objectives correspond to Laplace / Gaussian likelihoods for the pixels — but pixel values lie in a bounded interval while these distributions have support on the whole real line, so they waste likelihood mass outside the admissible pixel range.

**Sparse transformers and reranking.** The decoder-only sparse transformer (Child et al., 2019) uses factorized (row/column-style) attention masks to make attention over long 2D-structured sequences tractable. A pretrained contrastive image-text model (Radford et al., 2021) can score how well an image matches a caption, enabling sample reranking — a language-guided search over many candidates.

## Baselines

**GAN-based text-to-image (Reed et al., 2016; StackGAN, Zhang et al., 2017/2018; AttnGAN, Xu et al., 2018; DM-GAN, Zhu et al., 2019; DF-GAN, Tao et al., 2020).** Conditional GANs, often multi-scale, with attention and auxiliary text-image matching losses. Strengths: sharp images, the strongest prior FID/IS on MS-COCO (DF-GAN). Gap: trained on small datasets, task-specific architectures, and samples still show object distortion, illogical placement, and unnatural foreground/background blending; limited flexibility and no emergent cross-task ability.

**DRAW with caption conditioning (Mansimov et al., 2015).** A recurrent variational autoencoder extended to condition on captions; first to generate novel scenes from text. Gap: low fidelity relative to later adversarial approaches.

**Energy-based / pretrained-model optimization (Nguyen et al., 2017; Cho et al., 2020).** Optimize inputs to pretrained discriminative or cross-modal models for conditional generation. Gap: rely on specific pretrained components and optimization procedures rather than a single trained generator.

**Vector-quantized two-stage models (van den Oord et al., 2017; Razavi et al., 2019).** The compress-to-discrete-codes-then-fit-a-prior recipe this work builds on, using straight-through / online cluster assignment for the discrete bottleneck and a PixelCNN-family prior over the codes. Gap (as a *text-to-image* baseline): the prior is class-conditional or unconditional, not driven by free-form text, and the codes are modeled by convolutional autoregressive priors rather than a single text+image transformer.

## Evaluation settings

- **Datasets.** A collected dataset of 250 million internet image-text pairs (incorporating Conceptual Captions, Wikipedia text-image pairs, and a filtered subset of YFCC100M; aspect ratios restricted to [1/2, 2]) for the large model; Conceptual Captions (3.3M pairs) for smaller preliminary models. Zero-shot evaluation on MS-COCO and CUB-200 (never trained on their captions).
- **Metrics.** Fréchet Inception Distance and Inception Score on MS-COCO (zero-shot); human evaluation (which of two samples is more realistic / better matches a caption); qualitative inspection of compositionality, text rendering, and image-to-image translation. Data-overlap controls account for the MS-COCO validation images that leak through YFCC100M.
- **Protocol.** Stage 1: train the discrete autoencoder on images alone, by gradient descent on the relaxed ELBO, with annealing schedules for the relaxation temperature and step size and a ramp on the KL weight. Stage 2: freeze the autoencoder; BPE-encode captions (≤256 tokens, vocab 16384), encode images to a 32×32 grid of tokens (vocab 8192), concatenate, and train an autoregressive transformer with cross-entropy. Generate by sampling token streams from the transformer and reranking the decoded images with a contrastive model.

## Code framework

Pre-existing primitives: PyTorch `nn.Module`, convolutional ResNet blocks, `MaxPool2d` / nearest-neighbor upsampling, the Adam/AdamW optimizer, a BPE tokenizer, a decoder-only transformer with attention masks, and `F.cross_entropy`. A convolutional encoder/decoder and an autoregressive transformer over a token vocabulary already exist as components. What does not yet exist is how to bridge them: how the encoder's *categorical* output is turned into a differentiable training signal, the pixel likelihood the decoder is scored under, and how text and image tokens are joined into one modeled stream.

```python
import torch
from torch import nn
import torch.nn.functional as F

class ImageEncoder(nn.Module):
    # conv ResNet that maps a 256x256x3 image to a 32x32 grid of logits over K codebook entries
    def __init__(self, n_hid=256, vocab_size=8192):
        super().__init__()
        # TODO: 7x7 input conv, groups of bottleneck resblocks with downsampling,
        #       1x1 output conv producing [B, vocab_size, 32, 32] logits
        pass
    def forward(self, x):
        pass  # returns logits [B, K, 32, 32]

class ImageDecoder(nn.Module):
    # conv ResNet that maps a 32x32 grid of code vectors back to image-space statistics
    def __init__(self, n_hid=256, vocab_size=8192):
        super().__init__()
        # TODO: 1x1 input conv, resblocks with upsampling, 1x1 output conv
        pass
    def forward(self, z):
        pass  # returns per-pixel reconstruction statistics

def categorical_to_differentiable(logits, tau):
    # TODO: turn the encoder's categorical distribution into a differentiable,
    #       sampleable representation we can backprop the ELBO through.
    pass

def reconstruction_log_prob(x, decoder_out):
    # TODO: the pixel likelihood the decoder is scored under (must respect bounded pixel range)
    pass

class TokenTransformer(nn.Module):
    # decoder-only autoregressive transformer over a shared text+image token stream
    def __init__(self, n_layers=64):
        super().__init__()
        # TODO: token + positional (and image row/col) embeddings, sparse-attention blocks, output head
        pass
    def forward(self, tokens):
        pass  # returns next-token logits

# stage 1: train the autoencoder on images alone (relaxed ELBO)
def train_stage1(encoder, decoder, img, tau, beta, opt):
    logits = encoder(img)
    z = categorical_to_differentiable(logits, tau)       # TODO
    out = decoder(z)
    recon = -reconstruction_log_prob(img, out)           # TODO
    kl = None                                            # TODO: KL of categorical q vs uniform prior
    loss = recon + beta * kl                             # TODO assemble
    opt.zero_grad(); loss.backward(); opt.step()

# stage 2: freeze encoder/decoder, model concatenated text+image tokens autoregressively
def train_stage2(encoder, transformer, text_tokens, img, opt):
    image_tokens = encoder(img).argmax(dim=1).flatten(1) # TODO: how tokens are extracted for the prior
    stream = torch.cat([text_tokens, image_tokens], dim=1)
    logits = transformer(stream[:, :-1])
    loss = F.cross_entropy(logits.transpose(1, 2), stream[:, 1:])   # TODO: text/image weighting
    opt.zero_grad(); loss.backward(); opt.step()
```
