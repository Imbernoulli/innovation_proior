## Research question

Vision Transformers treat an image as a sequence of flattened patches and run a standard Transformer encoder over them. They reach excellent accuracy, but only when trained on very large *labeled* corpora — on ImageNet-scale data alone, trained from random initialization, they underperform convolutional networks and are unstable. The architecture has weak built-in image priors (no locality or translation-equivariance baked in the way a convolution has), so it needs either a lot of labeled data or a lot of inductive bias to do well.

The question is therefore: **can we pretrain a vision Transformer on large quantities of *unlabeled* images, with a self-supervised objective, so that the resulting encoder transfers strongly (and converges fast and stably) when fine-tuned on downstream tasks such as image classification and semantic segmentation?** A good solution would (a) need no human annotations during pretraining, (b) produce an encoder whose representations carry semantic structure, and (c) be cheap and stable enough to scale to large models.

In natural language, one self-supervised recipe — masked-token denoising — dominates: corrupt part of the input, predict the missing part, transfer the encoder. The crux of the question is whether an equally simple denoising objective exists for images, given that images are continuous pixel arrays rather than sequences of discrete symbols.

## Background

**Transformers and the patch interface.** A Transformer encoder operates on a sequence of vectors with self-attention and feed-forward sublayers. To apply it to an image of shape `H×W×C`, the image is reshaped into `N = HW/P²` non-overlapping patches of size `P×P`, each flattened and linearly projected to a `D`-dimensional embedding; learnable position embeddings are added and a special pooling token is prepended. The encoder outputs one contextualized vector per patch. With `224×224` images and `16×16` patches this gives a `14×14` grid, i.e. `N = 196` patches. This patch-embedding interface is the standard way a Transformer "sees" an image. Such models are data-hungry: empirically they need far more training data than CNNs before they generalize, which is what motivates pretraining on unlabeled data.

**Masked-token denoising in NLP.** The prevailing self-supervised recipe for Transformers in language masks out a fraction (~15%) of the *discrete* tokens in a sequence, replaces them with a special `[MASK]` symbol, runs the bidirectional encoder over the corrupted sequence, and trains the model to predict each masked token. Because the vocabulary is a fixed, finite set of discrete symbols (words or sub-word units), prediction is a clean classification problem: a softmax over the vocabulary at each masked position, trained with cross-entropy. This works precisely *because* language already comes pre-tokenized into a well-defined discrete vocabulary; the model never has to predict a continuous quantity. Variants refine *which* positions to corrupt: instead of independent random tokens, they mask contiguous spans / n-grams of tokens, which forces the model to use longer-range context rather than copying from immediate neighbors.

**The continuous-target obstacle, observed empirically.** When this recipe is translated literally to images, there is no pre-existing vocabulary for the input unit (a patch). The most direct substitute is to treat the masked-patch prediction as a *regression*: predict the raw pixels of the masked patches and train with a pixel-reconstruction loss. Diagnostic findings about this regression target are discouraging. A pixel-level reconstruction objective spends the model's capacity on short-range dependencies and high-frequency detail — neighboring pixels are highly correlated, so much of the loss is reducible by local smoothing/copying that carries little semantic content. Preliminary explorations of masked-patch prediction for vision Transformers (predicting the 3-bit mean color of a masked patch, a coarse regression-flavored target) were reported to *underperform*, even though it is the most straightforward port of the NLP recipe. So the field had the diagnostic in hand: continuous / low-level pixel targets are the wrong thing to predict.

**Discrete latent representations of images.** Separately, generative-model research had produced ways to map an image to a grid of *discrete* codes. A discrete variational autoencoder learns an encoder ("tokenizer") `q_φ(z|x)` that maps image pixels to a grid of indices into a learned codebook (vocabulary) of size `|V|`, and a decoder `p_ψ(x|z)` that reconstructs the image from those codes; it is trained to maximize reconstruction likelihood `E_{z∼q_φ(z|x)}[log p_ψ(x|z)]`. Because the latent codes are discrete, the sampling step is non-differentiable; a Gumbel-softmax relaxation provides a continuous, differentiable surrogate of categorical sampling so the encoder/decoder can be trained end-to-end. A uniform prior is placed over the codes during this training. One such tokenizer (trained at scale for a text-to-image generator) downsamples by a factor of 8; if the tokenizer view is `112×112`, it emits a `14×14` grid of codes drawn from a codebook of `8192` entries. Related two-stage vector-quantized autoencoders (VQ-VAE, VQ-VAE-2) established the recipe of *first* learning a discrete code space, *then* learning a model over those codes. These tokenizers were built to *compress / generate* images; the point relevant here is that they turn a continuous image into a grid of discrete symbols that abstract away pixel-level detail and retain higher-level structure.

**Prior self-supervised vision pretraining.** Two broad families existed. (1) *Generative / autoregressive on pixels:* one line clusters RGB pixels into a small palette via k-means to get "pixel tokens," shrinks the image, and trains a GPT/BERT-style Transformer over that flattened low-resolution sequence. This loses pixel-level spatial information (the tokens are coarse colors used as *both* input and output) and needs enormous models. (2) *Discriminative:* contrastive methods and self-distillation treat two augmentations of an image as a positive pair and pull their representations together (often needing many negatives, large batches/memory banks, or asymmetric stop-gradient tricks to avoid collapse). These learn a good global image vector but are not denoising/auto-encoding objectives.

## Baselines

- **Masked-token denoising for text (BERT; Devlin et al., 2019).** Mask ~15% of the discrete tokens, predict them with a softmax over the vocabulary using bidirectional context, transfer the encoder by fine-tuning a small task head. Core math: cross-entropy `−Σ_{i∈M} log p(token_i | corrupted sequence)` over masked positions only. The intermediate-fine-tuning practice (pretrain → fine-tune on a data-rich intermediate task → fine-tune on the target) also comes from this line. **Gap:** assumes the input is already a sequence of discrete symbols from a fixed vocabulary; gives no recipe for an input made of continuous pixels.

- **Span/n-gram masking (SpanBERT; UniLMv2; T5).** Instead of masking independent random tokens, mask contiguous spans. Forces the model to integrate longer-range context rather than exploiting a single adjacent token. **Gap:** again defined over discrete language tokens; the *idea* (mask blocks, not isolated units) is portable, but the target is not.

- **Patchify-and-Transformer for images (ViT; Dosovitskiy et al., 2021).** Establishes the patch-embedding interface and shows a plain Transformer can match CNNs given enough labeled data. Includes a preliminary self-supervised attempt: predict the (3-bit, mean) color of masked patches. **Gap:** data-hungry under supervised training; the masked-patch color-prediction objective is a coarse regression target that was reported to lag behind, leaving open what the *right* prediction target is.

- **Pixel-level masked auto-encoding (the regression baseline).** Mask patches, regress their raw pixels with an L1/L2 reconstruction loss. Core math: `Σ_{i∈M} ‖x_i − x̂_i‖²` on masked patches. **Gap:** the loss is dominated by locally-predictable high-frequency detail; capacity goes to short-range pixel correlations instead of semantics. This is the failure mode a denoising solution would have to avoid.

- **Discrete VAE image tokenizer (dVAE; Ramesh et al., 2021, building on VQ-VAE, van den Oord et al., 2017).** Learns `q_φ(z|x)` (pixels → grid of `8192`-way codes) and `p_ψ(x|z)` (codes → image) by maximizing reconstruction likelihood with a Gumbel-softmax relaxation and a uniform code prior. **Gap (as used here it is a *component*, not a competitor):** it is a generative/compression model; on its own it provides a discrete code space but no representation-learning objective for a downstream encoder. It supplies the missing "vocabulary" for images.

- **Pixel-cluster GPT/BERT for vision (iGPT; Chen et al., 2020).** k-means RGB pixels into a 9-bit palette, flatten a downsized image into a sequence of these color tokens, train GPT/BERT objectives. **Gap:** clustered color tokens are used as both input *and* output, discarding fine spatial/pixel information; needs billions of parameters to compete.

- **Contrastive / self-distillation SSL for ViT (MoCo v3; DINO; SwAV).** Augmentation-invariance objectives that learn a global image representation; not denoising/auto-encoding and tend to require either many negatives, large batches, or two model copies. **Gap:** a different paradigm; heavier pretraining footprint, and not a masked-prediction objective.

- **VAE / ELBO (Kingma & Welling, 2014).** The evidence-lower-bound framework `log p(x) ≥ E_q[log p(x|z)] − KL[q(z|x)‖p(z)]` used to *interpret* a tokenizer-plus-predictor pipeline as one variational objective. **Gap:** a modeling lens, not itself a vision pretraining method.

## Evaluation settings

- **Pretraining data:** ImageNet-1K (~1.2M images, 1k classes) used *without labels*. The patch view uses `224×224` images split into a `14×14` grid of `16×16` patches; a paired tokenizer view can use `112×112` pixels so a factor-8 tokenizer also emits `14×14` codes. Larger labeled corpora such as ImageNet-22K, and much larger proprietary corpora such as JFT-300M, are natural supervised-pretraining baselines.
- **Downstream — image classification:** ImageNet-1K top-1 accuracy (and CIFAR-100, 100 classes / 60k images). Also linear-probe accuracy (encoder frozen, single linear layer on pooled features).
- **Downstream — semantic segmentation:** ADE20K (25k images, 150 categories), mean Intersection-over-Union (mIoU), single- and multi-scale inference, with a segmentation task head on top of the encoder.
- **Protocol:** fine-tune the whole pretrained encoder plus a small task head; for classification, follow the standard ViT/DeiT augmentation and optimization recipe (RandAugment, mixup/cutmix, label smoothing, layer-wise learning-rate decay). Diagnostics of interest: convergence speed and training stability versus from-scratch training; what the self-attention attends to after pretraining.

## Code framework

The scaffold starts from a patchify-and-Transformer encoder, positional-signal modules, a pretrained discrete image tokenizer that maps an image to a grid of codebook indices, an image data pipeline, and an AdamW training loop. The open slots are the patch-grid corruption procedure, the paired image views consumed by the pipeline, how the corrupted input is formed and consumed by the encoder, what target is predicted, and the head that predicts it.

```python
import torch, torch.nn as nn

# --- patchify + Transformer encoder (the backbone interface) ---
class PatchEmbed(nn.Module):
    """Conv stem: image -> (N = (H/P)*(W/P)) patch embeddings of dim D."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768): ...
    def forward(self, x): ...  # -> (B, N, D)

class Block(nn.Module):
    """Standard Transformer block: MSA + MLP with residuals/norm."""
    def forward(self, x, rel_pos_bias=None): ...

class RelativePositionBias(nn.Module):
    """Optional position-bias module for a patch grid."""
    def forward(self): ...

# --- pretrained discrete image tokenizer (frozen) ---
class DiscreteImageTokenizer:
    """Maps pixels -> a grid of integer codebook labels.
    Trained separately by reconstruction; used here only for inference."""
    @torch.no_grad()
    def get_codebook_indices(self, images):  # -> (B, h, w) longs in [0, |V|)
        ...

# --- TODO: how to corrupt the patch grid ---
class PatchGridMaskGenerator:
    def __init__(self, input_size, num_masking_patches,
                 min_num_patches=None, max_num_patches=None,
                 min_aspect=None, max_aspect=None):
        # TODO: decide the masking ratio, the spatial structure of masked regions,
        #       and how overlapping regions should count.
        pass
    def _mask(self, mask, max_mask_patches):
        # TODO: add one region to an existing h-by-w mask.
        pass
    def __call__(self):
        # TODO: return an h-by-w boolean mask over patch positions.
        pass

# --- TODO: turn one image into the inputs the objective needs ---
class TwoViewPretrainingTransform:
    def __init__(self, args):
        self.common_transform = ...
        self.patch_transform = ...
        # TODO: define the target-producing view and the patch-grid mask generator.
        pass
    # TODO: produce the paired views/targets the pretraining objective consumes,
    # plus a corruption mask. The target shape is part of the contribution.
    def __call__(self, image):
        pass

# --- TODO: the self-supervised encoder + prediction head ---
class PatchSequencePretrainer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 vocab_size=None, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., **kwargs):
        self.patch_embed = PatchEmbed(...)
        self.cls_token = nn.Parameter(...)
        self.pos_drop = nn.Dropout(...)
        self.rel_pos_bias = RelativePositionBias(...)
        self.blocks = nn.ModuleList([Block(...) for _ in range(12)])
        self.norm = nn.LayerNorm(768)
        # TODO: what replaces a corrupted patch in the input sequence?
        # TODO: what prediction head is needed, and over what target space?
        pass
    def fix_init_weight(self):
        # TODO: define any depth-aware stabilization needed at initialization.
        pass
    def _init_weights(self, m):
        # TODO: initialize linear, convolutional, and normalization layers.
        pass
    @torch.jit.ignore
    def no_weight_decay(self):
        pass
    def forward_features(self, x, bool_masked_pos):
        # TODO: embed patches; substitute corrupted positions; run encoder;
        #       return contextual patch outputs.
        pass
    def forward(self, x, bool_masked_pos, return_all_tokens=False):
        # TODO: emit predictions at the corrupted positions unless all are requested.
        pass

# --- TODO: the pretraining loss ---
def train_one_step(model, tokenizer, batch, optimizer):
    # TODO: derive targets from the tokenizer view; run the encoder on the patch
    #       view plus corruption mask; compute the loss at selected positions; step.
    pass

# --- optimizer / schedule ---
def build_optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1.5e-3,
                             betas=(0.9, 0.999), weight_decay=0.05)
```
