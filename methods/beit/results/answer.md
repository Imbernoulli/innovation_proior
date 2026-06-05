# BEiT

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

where `q_φ(z|x)` is the tokenizer, `p_ψ(x|z)` the dVAE decoder, and `p_θ(z|x̃)` the MIM network. Two stages (à la VQ-VAE): stage 1 learns the tokenizer by reconstruction with a uniform prior; stage 2 freezes `q_φ, p_ψ`, takes the one-point posterior `ẑ = argmax_z q_φ(z|x)`, and the bound becomes `Σ_{(x,x̃)∈D} [ E_{z∼q_φ(z|x)} log p_ψ(x|z) + log p_θ(ẑ|x̃) ]`. If `p_θ` is factorized over masked positions, `log p_θ(ẑ|x̃) = Σ_{i∈M} log p_MIM(ẑ_i|x^M)`, which is the MIM cross-entropy term with the correct sign.

## Blockwise masking (target ~0.4N masked)

```
M ← {}
repeat
    s ← Rand(16, 0.4N − |M|)          # block area, ≥ 16 patches
    log r ← Rand(log 0.3, log(1/0.3))
    r ← exp(log r)                   # aspect ratio
    a ← √(s·r) ;  b ← √(s/r)         # block height, width
    t ← Rand(0, h − a) ; l ← Rand(0, w − b)
    M ← M ∪ { (i,j) : i∈[t,t+a), j∈[l,l+b) }
until |M| reaches the target count
```

## Architecture / setup

- Backbone: standard ViT-Base — 12 layers, hidden 768, 12 heads, FFN 3072, patch `16×16`; prepend `[S]` token, learnable 1D position embeddings. (Large: 24 layers, 1024 hidden, 16 heads.)
- MIM head: single `Linear(768 → 8192)` softmax classifier, applied at masked positions only.
- Stabilizing init: small bounded initialization (the code uses truncated normal with std 0.02 and bounds ±0.02), then rescale the `l`-th block's attention-output and FFN-output projections by `1/√(2l)`.
- Pretraining: ImageNet-1K (no labels), 224², 800 epochs, batch 2048, AdamW (β=(0.9,0.999)), lr 1.5e-3, 10-epoch warmup, cosine decay, weight decay 0.05, stochastic depth 0.1, no dropout, ~75 patches masked (~40%).
- Fine-tuning: drop `lm_head`, keep encoder, attach a task head. Classification: average-pool patch outputs → linear softmax, `softmax(avg({h_i^L}) · W_c)`, `W_c ∈ R^{D×C}`, end-to-end. Segmentation: deconv/upsampling decoder over the patch grid. Optional intermediate fine-tuning on a data-rich labeled set before the target task.

## Code

```python
import math, random
from functools import partial
import numpy as np
import torch, torch.nn as nn
from timm.models.layers import trunc_normal_ as _trunc_normal

def trunc_normal_(tensor, std=0.02):
    return _trunc_normal(tensor, std=std, a=-std, b=std)


class PatchGridMaskGenerator:
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


class TwoViewPretrainingTransform:
    def __init__(self, args):
        mean = IMAGENET_INCEPTION_MEAN if not args.imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not args.imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
        self.common_transform = transforms.Compose([
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(p=0.5),
            RandomResizedCropAndInterpolationWithTwoPic(
                size=args.input_size, second_size=args.second_input_size,
                interpolation=args.train_interpolation,
                second_interpolation=args.second_interpolation),
        ])
        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        ])
        if args.discrete_vae_type == "dall-e":
            self.visual_token_transform = transforms.Compose([transforms.ToTensor(), map_pixels])
        elif args.discrete_vae_type == "customized":
            self.visual_token_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
            ])
        else:
            raise NotImplementedError()
        self.masked_position_generator = PatchGridMaskGenerator(
            args.window_size, num_masking_patches=args.num_mask_patches,
            max_num_patches=args.max_mask_patches_per_block,
            min_num_patches=args.min_mask_patches_per_block)

    def __call__(self, image):
        for_patches, for_tokens = self.common_transform(image)
        return (self.patch_transform(for_patches),
                self.visual_token_transform(for_tokens),
                self.masked_position_generator())


class PatchSequencePretrainer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, vocab_size=8192,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 init_std=0.02, **kw):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_features = self.embed_dim = embed_dim
        self.init_std = init_std
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token  = nn.Parameter(torch.zeros(1, 1, embed_dim))      # [S]
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))      # learnable e_[M]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim)) if use_abs_pos_emb else None
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.rel_pos_bias = (
            RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
            if use_shared_rel_pos_bias else None
        )
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                  norm_layer=norm_layer, init_values=init_values,
                  window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                  attn_head_dim=attn_head_dim)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)                  # softmax over codebook

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=self.init_std)
        trunc_normal_(self.cls_token,  std=self.init_std)
        trunc_normal_(self.mask_token, std=self.init_std)
        trunc_normal_(self.lm_head.weight, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        # rescale the l-th block's output projections by 1/sqrt(2*l) for stability
        for layer_id, blk in enumerate(self.blocks):
            blk.attn.proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
            blk.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward_features(self, x, bool_masked_pos):
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)          # (B, N, D)
        B, N, _ = x.shape
        mask_token = self.mask_token.expand(B, N, -1)
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w                                 # substitute e_[M] at masked patches
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)
        return self.norm(x)

    def forward(self, x, bool_masked_pos, return_all_tokens=False):
        x = self.forward_features(x, bool_masked_pos)[:, 1:]
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
