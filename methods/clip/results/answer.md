# CLIP — Contrastive Language–Image Pre-training

## Problem

Standard vision models predict a fixed, predetermined set of categories through a static softmax head, so their supervision is capped by a label vocabulary and they have no native way to recognize concepts they were not explicitly trained on. CLIP learns visual representations directly from natural language so that (1) supervision is drawn from the open vocabulary of web text rather than a closed label set, and (2) a classifier for arbitrary new categories can be built at inference time from text descriptions, with no labeled data — enabling zero-shot transfer.

## Key idea

Train an image encoder and a text encoder jointly so that, given a batch of `N` (image, text) pairs, the model identifies which of the `N×N` possible pairings actually occurred. This replaces the expensive task of *generating* a caption with the much cheaper proxy of *matching* the correct whole text to each image: one batch softmax in each direction, with the other examples serving as free in-batch negatives. Empirically, bag-of-words prediction learns about `3×` faster than caption generation, and the contrastive matching objective learns about another `4×` faster than bag-of-words prediction, for roughly an order-of-magnitude efficiency gain over caption generation.

At test time, the trained text encoder acts as a hypernetwork that *generates a linear classifier from language*: embed each class name (wrapped in a prompt) to get a weight vector, embed the image, and classify by scaled cosine similarity. Zero-shot classification is the same matching operation used in pretraining, asked once per class name.

## Final Objective

Encode the batch, project each modality into a shared `d_e`-dimensional space with a learned linear map, and L2-normalize so that a dot product is a cosine similarity. Form the `N×N` matrix of similarities, scale it by a learned inverse temperature, and apply a symmetric cross-entropy whose correct answers lie on the diagonal:

- `I_e = l2_normalize(image_encoder(I) · W_i)`, `T_e = l2_normalize(text_encoder(T) · W_t)`
- `logits = (I_e · T_eᵀ) · min(exp(t), 100)`, with `logits[i, j]` comparing image `i` to text `j`
- `loss = ½ · [ CE(logits, arange(N)) + CE(logitsᵀ, arange(N)) ]`

The inverse temperature is stored as a log-parameter `t` and applied as `exp(t)` (always positive); it is initialized to `log(1/0.07)` and clamped so `exp(t) ≤ 100` to prevent the logit scale from running away and destabilizing training. The cosine logits *must* be scaled because raw cosine values in `[−1, 1]` give the softmax too little dynamic range, especially for large batches: even a positive at `1` and negatives at `−1` cannot produce a confident diagonal probability unless the similarities are scaled.

## Zero-Shot Classification

1. For each class, form text prompt(s) such as `"a photo of a {label}."`, encode and normalize them to get one weight vector per class (averaging several prompts' embeddings per class — "prompt ensembling" — done in embedding space so inference cost is unchanged). Prompts disambiguate polysemous words and match the sentence-like distribution of pretraining text.
2. Encode and normalize the image.
3. Predict `argmax_k min(exp(t), 100) · (image · class_weightsᵀ)`.

This is exactly a multinomial logistic regression with L2-normalized inputs and weights, no bias, and inverse-temperature scaling — but the weights are produced by the text encoder from language rather than fit on labeled data.

## Architecture Notes

- **Image encoder**: either a modified ResNet (ResNet-D tweaks, antialiased blur-pool, and a final transformer-style attention-pooling layer in place of global average pooling) or a Vision Transformer (with an extra layer-norm on the patch+position embeddings). ViTs are more compute-efficient at scale.
- **Text encoder**: a GPT-2-style Transformer over byte-pair-encoded tokens (≈49k vocab, max length 76), with causal masking (kept to allow LM initialization or an auxiliary LM objective if useful). The activation at the `[EOS]` token of the top layer is layer-normed and linearly projected into the joint space.
- **Simplifications**: linear (not non-linear) projection heads; trained from scratch (no pretrained init); only a random square crop for augmentation — overfitting is not a concern at hundreds of millions of pairs.
- **Scaling / optim**: compound scaling of width+depth+resolution for the image tower (text tower scales width only); AdamW with decoupled weight decay (not on gains/biases), cosine LR schedule, very large batch (e.g. 32,768), mixed precision.

## Working Code

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualEncoderModel(nn.Module):
    def __init__(self, image_encoder, text_encoder, d_image, d_text, d_embed):
        super().__init__()
        self.image_encoder = image_encoder          # CNN or ViT -> [n, d_image]
        self.text_encoder = text_encoder            # Transformer -> [n, d_text]
        self.image_proj = nn.Linear(d_image, d_embed, bias=False)   # W_i
        self.text_proj = nn.Linear(d_text, d_embed, bias=False)     # W_t
        # learned inverse temperature, stored in log space so exp() is always positive,
        # initialized to the equivalent of temperature 0.07
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_image(self, image):
        return F.normalize(self.image_proj(self.image_encoder(image)), dim=-1)

    def encode_text(self, text):
        return F.normalize(self.text_proj(self.text_encoder(text)), dim=-1)

    def forward(self, image, text):
        image_features = self.encode_image(image)   # [n, d_embed]
        text_features = self.encode_text(text)      # [n, d_embed]
        return image_features, text_features


def training_objective(image_features, text_features, model):
    # scaled pairwise cosine similarities; diagonal entries are the true pairs
    logit_scale = model.logit_scale.exp().clamp(max=100.0)
    logits_per_image = logit_scale * (image_features @ text_features.t())  # [n, n]
    logits_per_text = logits_per_image.t()
    # the matched (image, text) pair for example i is on the diagonal -> label i
    n = logits_per_image.shape[0]
    labels = torch.arange(n, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, labels)   # each image picks its text
    loss_t = F.cross_entropy(logits_per_text, labels)    # each text picks its image
    return (loss_i + loss_t) / 2                          # symmetric


def train(model, dataloader, optimizer):
    for images, texts in dataloader:
        image_features, text_features = model(images, texts)
        loss = training_objective(image_features, text_features, model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()


def _format_prompt(template, description):
    if "{label}" in template:
        return template.format(label=description)
    return template.format(description)


@torch.no_grad()
def build_classifier_from_descriptions(model, class_descriptions, tokenize, templates=("a photo of a {label}.",)):
    """Generate a linear classifier from class names via the text encoder."""
    weights = []
    device = next(model.parameters()).device
    for description in class_descriptions:
        prompts = tokenize([_format_prompt(t, description) for t in templates]).to(device)
        embeds = model.encode_text(prompts)                          # [n_templates, d_embed]
        class_vec = F.normalize(embeds.mean(dim=0), dim=-1)          # ensemble in embedding space
        weights.append(class_vec)
    return torch.stack(weights, dim=0)                               # [K, d_embed]


@torch.no_grad()
def classify(model, image, classifier):
    image_features = model.encode_image(image)                       # [1, d_embed]
    logit_scale = model.logit_scale.exp().clamp(max=100.0)
    logits = logit_scale * (image_features @ classifier.t())         # [1, K]
    return logits.argmax(dim=-1)                                     # no labels were fit
```

## Causal Chain

Language-shaped, uncapped supervision + a language-specified classifier → needs web scale → per-example cost is decisive → caption generation is too expensive (it models irrelevant phrasing) → relax to *matching* the right text among the batch (one batch softmax in each direction, free negatives) → reading the similarity matrix by rows and columns gives a symmetric InfoNCE/N-pair cross-entropy with diagonal labels → raw cosine logits have too little dynamic range for a large-batch softmax, so scale by a learned, log-parameterized, clamped inverse temperature → the trained text encoder is a hypernetwork emitting classifier weights from descriptions, so zero-shot classification is the same scaled-cosine matching, asked once per prompted class name.
