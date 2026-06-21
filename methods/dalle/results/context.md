# Context

## Research question

How can a single, general generative model produce images controllable by arbitrary natural language, when trained on a very large corpus of image-text pairs rather than on a hand-engineered, task-specific architecture tuned on a small dataset?

Text-to-image systems are commonly evaluated on small datasets (MS-COCO, CUB-200) and built from task-specific machinery: multi-scale generators, attention modules, auxiliary text-image matching losses, extra conditioning signals. The question here is what happens when *dataset size and model size* are scaled instead — whether a single large generative model, trained on hundreds of millions of image-text pairs, can serve as a language-controllable image generator.

## Background

**Scaling autoregressive transformers.** When compute, model size, and data are scaled carefully, autoregressive transformers (Vaswani et al., 2017) reach impressive quality across modalities: text (Radford et al., 2019), images modeled as pixel sequences (Chen et al., 2020), audio (Dhariwal et al., 2020). This suggests modeling text and image jointly as a single autoregressive *token* stream.

**Pixels as image tokens.** A 256×256×3 image is ~200k values, a sequence whose length interacts with the quadratic cost of transformer attention. Likelihood objectives over pixels tend to prioritize short-range dependencies (Salimans et al., 2017), placing modeling capacity on high-frequency texture relative to low-frequency structure.

**Two-stage discrete-token compression.** Vector-quantized autoencoders (van den Oord et al., 2017; Razavi et al., 2019) established a two-stage recipe: first learn an autoencoder that maps an image to a small grid of *discrete* codes (and back), then fit a powerful autoregressive model over those codes. This decouples a fast feed-forward compressor from a slow expressive prior, and shrinks the sequence the prior must model by orders of magnitude. In the established uses, the prior over the codes is class-conditional or unconditional and the codes are modeled by convolutional autoregressive priors.

**Variational autoencoders and the ELBO.** The autoencoder side is naturally framed as a VAE (Kingma & Welling, 2013; Rezende et al., 2014): an encoder `q_φ(z|x)`, a decoder `p_θ(x|z)`, a prior, trained on the evidence lower bound `E_{q_φ(z|x)}[log p_θ(x|z)] − D_KL(q_φ(z|x) ‖ p(z))`. Upweighting the KL term by `β > 1` (Higgins et al., 2016) is a known knob on the latent channel.

**Gradients through a discrete latent.** When `q_φ(z|x)` is *categorical* (picking one of K codes), the reparameterization gradient is unavailable. Two known approaches: an online cluster-assignment / straight-through estimator (Bengio et al., 2013), as used by the vector-quantized autoencoders; and the *Gumbel-Softmax / Concrete* relaxation (Jang et al., 2016; Maddison et al., 2016), which replaces the hard categorical sample with a continuous, temperature-controlled, *reparameterizable* approximation that hardens into a true categorical as the temperature `τ → 0`.

**Pixel likelihood and support.** The common `ℓ1` / `ℓ2` reconstruction objectives correspond to Laplace / Gaussian likelihoods for the pixels, which have support on the whole real line; pixel values lie in a bounded interval.

**Sparse transformers and reranking.** The decoder-only sparse transformer (Child et al., 2019) uses factorized (row/column-style) attention masks to make attention over long 2D-structured sequences tractable. A pretrained contrastive image-text model (Radford et al., 2021) can score how well an image matches a caption, enabling sample reranking — a language-guided search over many candidates.

## Baselines

**GAN-based text-to-image (Reed et al., 2016; StackGAN, Zhang et al., 2017/2018; AttnGAN, Xu et al., 2018; DM-GAN, Zhu et al., 2019; DF-GAN, Tao et al., 2020).** Conditional GANs, often multi-scale, with attention and auxiliary text-image matching losses; trained on small datasets with task-specific architectures. DF-GAN holds the strongest prior FID/IS on MS-COCO.

**DRAW with caption conditioning (Mansimov et al., 2015).** A recurrent variational autoencoder extended to condition on captions; first to generate novel scenes from text.

**Energy-based / pretrained-model optimization (Nguyen et al., 2017; Cho et al., 2020).** Optimize inputs to pretrained discriminative or cross-modal models for conditional generation.

**Vector-quantized two-stage models (van den Oord et al., 2017; Razavi et al., 2019).** The compress-to-discrete-codes-then-fit-a-prior recipe this work builds on, using straight-through / online cluster assignment for the discrete bottleneck and a PixelCNN-family prior over the codes; the prior is class-conditional or unconditional and the codes are modeled by convolutional autoregressive priors.

## Evaluation settings

- **Datasets.** A collected dataset of 250 million internet image-text pairs (incorporating Conceptual Captions, Wikipedia text-image pairs, and a filtered subset of YFCC100M; aspect ratios restricted to [1/2, 2]) for the large model; Conceptual Captions (3.3M pairs) for smaller preliminary models. Zero-shot evaluation on MS-COCO and CUB-200 (never trained on their captions).
- **Metrics.** Fréchet Inception Distance and Inception Score on MS-COCO (zero-shot); human evaluation (which of two samples is more realistic / better matches a caption); qualitative inspection of compositionality, text rendering, and image-to-image translation. Data-overlap controls account for the MS-COCO validation images that leak through YFCC100M.
- **Protocol.** The generative model is trained on the image-text pairs and then evaluated zero-shot; captions are BPE-encoded (≤256 tokens, vocab 16384). At generation time a caption is supplied and images are produced for scoring against the metrics above.

## Code framework

Pre-existing primitives: PyTorch `nn.Module`, convolutional ResNet blocks, `MaxPool2d` / nearest-neighbor upsampling, the Adam/AdamW optimizer, a BPE tokenizer, a decoder-only transformer with attention masks, and `F.cross_entropy`. A convolutional encoder/decoder and an autoregressive transformer over a token vocabulary already exist as components. What does not yet exist is how they fit together into a working training procedure.

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
    # TODO
    pass

def reconstruction_log_prob(x, decoder_out):
    # TODO
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
    kl = None                                            # TODO
    loss = recon + beta * kl                             # TODO assemble
    opt.zero_grad(); loss.backward(); opt.step()

# stage 2: freeze encoder/decoder, model concatenated text+image tokens autoregressively
def train_stage2(encoder, transformer, text_tokens, img, opt):
    image_tokens = encoder(img).argmax(dim=1).flatten(1) # TODO: how tokens are extracted for the prior
    stream = torch.cat([text_tokens, image_tokens], dim=1)
    logits = transformer(stream[:, :-1])
    loss = F.cross_entropy(logits.transpose(1, 2), stream[:, 1:])   # TODO
    opt.zero_grad(); loss.backward(); opt.step()
```
