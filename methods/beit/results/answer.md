# BEiT: BERT Pre-Training of Image Transformers

## Problem

Vision Transformers (ViT) are data-hungry: trained from scratch on ImageNet-scale data they underperform and are unstable. We want a BERT-style self-supervised pretraining objective for a ViT, using only unlabeled images, so the encoder transfers well after fine-tuning. The obstacle: BERT's masked-language-modeling predicts discrete word tokens with a softmax over a fixed vocabulary, but image patches are continuous pixels — there is no vocabulary — and the naive substitute, regressing raw masked pixels, wastes capacity on short-range dependencies and high-frequency detail.

## Key idea — Masked Image Modeling (MIM)

Give each image **two views**:
1. **Image patches** (`16×16`), linearly embedded — the continuous input to the ViT (preserves raw pixels).
2. **Visual tokens** — a grid of discrete codebook indices produced by a frozen, pretrained discrete VAE (dVAE) image tokenizer (the DALL-E tokenizer; codebook size `|V| = 8192`, factor-8 downsampling). These are the **prediction target**.

A `224×224` image gives a `14×14` grid of both patches and visual tokens (`N = 196`), with one-to-one spatial correspondence.

Pretraining: randomly mask ~40% of patches using **blockwise masking** (mask contiguous rectangular blocks, not isolated patches, so the task can't be solved by local copying), replace masked patches with a learnable mask embedding `e_[M]`, feed the corrupted sequence to the ViT, and predict the visual token at each masked position with a softmax classifier over the codebook (cross-entropy). Predicting *discrete* tokens abstracts away pixel detail and keeps the objective semantic.

## Objective

Softmax classifier at each masked position over the codebook:

`p_MIM(z' | x^M) = softmax(W_c · h_i^L + b_c)`,  `W_c ∈ R^{|V|×D}`, `b_c ∈ R^{|V|}`.

Maximize the log-likelihood of the correct visual tokens at masked positions `M` only:

`max Σ_{x∈D} E_M [ Σ_{i∈M} log p_MIM(z_i | x^M) ]`.

**Variational view.** With `x` the image, `x̃` the masked image, `z` the visual tokens, the ELBO of `log p(x|x̃)` is

`log p(x|x̃) ≥ E_{z∼q_φ(z|x)}[log p_ψ(x|z)] − KL[q_φ(z|x) ‖ p_θ(z|x̃)]`,

where `q_φ(z|x)` is the tokenizer, `p_ψ(x|z)` the dVAE decoder, and `p_θ(z|x̃)` the MIM network. Two stages (à la VQ-VAE): stage 1 learns the tokenizer by reconstruction (uniform prior); stage 2 freezes `q_φ, p_ψ`, takes the one-point posterior `ẑ = argmax_z q_φ(z|x)`, and the bound reduces to `Σ_i ( E[log p_ψ(x_i|z)] + log p_θ(ẑ_i|x̃_i) )` — the second term is exactly the MIM cross-entropy.

## Blockwise masking (target ~0.4N masked)

```
M ← {}
repeat
    s ← Rand(16, 0.4N − |M|)          # block area, ≥ 16 patches
    r ← Rand(0.3, 1/0.3)             # aspect ratio (log-uniform)
    a ← √(s·r) ;  b ← √(s/r)         # block height, width
    t ← Rand(0, h − a) ; l ← Rand(0, w − b)
    M ← M ∪ { (i,j) : i∈[t,t+a), j∈[l,l+b) }
until |M| > 0.4N
```

## Architecture / setup

- Backbone: standard ViT-Base — 12 layers, hidden 768, 12 heads, FFN 3072, patch `16×16`; prepend `[S]` token, learnable 1D position embeddings. (Large: 24 layers, 1024 hidden, 16 heads.)
- MIM head: single `Linear(768 → 8192)` softmax classifier, applied at masked positions only.
- Stabilizing init: small uniform init, then rescale the `l`-th block's attention-output and FFN-output projections by `1/√(2l)`.
- Pretraining: ImageNet-1K (no labels), 224², 800 epochs, batch 2048, AdamW (β=(0.9,0.999)), lr 1.5e-3, 10-epoch warmup, cosine decay, weight decay 0.05, stochastic depth 0.1, no dropout, ~75 patches masked (~40%).
- Fine-tuning: drop `lm_head`, keep encoder, attach a task head. Classification: average-pool patch outputs → linear softmax, `softmax(avg({h_i^L}) · W_c)`, `W_c ∈ R^{D×C}`, end-to-end. Segmentation: deconv/upsampling decoder over the patch grid. Optional intermediate fine-tuning on a data-rich labeled set before the target task.

## Code

```python
import math, random
import numpy as np
import torch, torch.nn as nn


class MaskingGenerator:
    """Blockwise masking over the h×w patch grid until ~num_masking_patches are masked."""
    def __init__(self, input_size, num_masking_patches,
                 min_num_patches=16, max_num_patches=None,
                 min_aspect=0.3, max_aspect=None):
        self.height, self.width = (input_size, input_size) if isinstance(input_size, int) else input_size
        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches
        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)   # s
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))        # r
            h = int(round(math.sqrt(target_area * aspect_ratio)))                  # a = sqrt(s*r)
            w = int(round(math.sqrt(target_area / aspect_ratio)))                  # b = sqrt(s/r)
            if w < self.width and h < self.height:
                top  = random.randint(0, self.height - h)
                left = random.randint(0, self.width  - w)
                num_masked = mask[top:top + h, left:left + w].sum()
                if 0 < h * w - num_masked <= max_mask_patches:        # count newly-masked (overlap ok)
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1; delta += 1
                if delta > 0:
                    break
        return delta

    def __call__(self):
        mask = np.zeros((self.height, self.width), dtype=int)
        mask_count = 0
        while mask_count < self.num_masking_patches:
            max_mask_patches = min(self.num_masking_patches - mask_count, self.max_num_patches)
            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            mask_count += delta
        return mask                                                   # (h, w) boolean grid


class VisionTransformerForMaskedImageModeling(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, vocab_size=8192,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., **kw):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token  = nn.Parameter(torch.zeros(1, 1, embed_dim))      # [S]
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))      # learnable e_[M]
        self.pos_embed  = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, **kw) for _ in range(depth)])
        self.norm    = nn.LayerNorm(embed_dim, eps=1e-6)
        self.lm_head = nn.Linear(embed_dim, vocab_size)                  # softmax over codebook

        nn.init.trunc_normal_(self.pos_embed,  std=0.02)
        nn.init.trunc_normal_(self.cls_token,  std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.lm_head.weight, std=0.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        # rescale the l-th block's output projections by 1/sqrt(2*l) for stability
        for layer_id, blk in enumerate(self.blocks):
            blk.attn.proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
            blk.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)

    def forward(self, x, bool_masked_pos, return_all_tokens=False):
        x = self.patch_embed(x)                                          # (B, N, D)
        B, N, _ = x.shape
        mask_token = self.mask_token.expand(B, N, -1)
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w                                 # substitute e_[M] at masked patches
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)[:, 1:]
        if return_all_tokens:
            return self.lm_head(x)
        return self.lm_head(x[bool_masked_pos])                          # logits at masked positions only


def train_one_step(model, tokenizer, batch, optimizer):
    samples, token_view, bool_masked_pos = batch
    bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)          # (B, N)

    with torch.no_grad():
        input_ids = tokenizer.get_codebook_indices(token_view).flatten(1)  # argmax codes ẑ
        labels = input_ids[bool_masked_pos]                                # targets at masked positions

    logits = model(samples, bool_masked_pos=bool_masked_pos)               # (#masked, |V|)
    loss = nn.CrossEntropyLoss()(logits, labels)                           # = -Σ_{i∈M} log p_MIM(ẑ_i | x^M)

    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return loss.item()
```

The data pipeline produces two views per image (the normalized patch view for the ViT and a view in the tokenizer's expected input range for the codes) plus a blockwise mask; the frozen dVAE supplies `get_codebook_indices` (argmax over its encoder logits). The ViT-Base config sets `embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, vocab_size=8192`.
