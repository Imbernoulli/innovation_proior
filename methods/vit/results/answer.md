# Vision Transformer (ViT)

## Problem it solves

Image classification was the exclusive domain of convolutional networks, whose locality, 2D-neighborhood, and
translation-equivariance priors make them sample-efficient but also constrain them. ViT asks whether a *standard*
Transformer — the low-prior sequence model that scales so well in language — can be applied to images with almost
no modification, and whether large-scale pre-training can compensate for the missing convolutional prior. The
thesis: **at sufficient data scale, learned spatial structure can compete with hard-wired inductive bias.**

## Key idea

Treat an image as a short sequence of patch tokens and run an unmodified Transformer encoder on it.

1. **Patchify.** Split `x ∈ ℝ^{H×W×C}` into `N = HW/P²` non-overlapping `P×P` patches (e.g. `P=16` ⇒ `196` tokens
   at 224²). This collapses self-attention's per-pixel `O((HW)²·D)` cost down to language-scale `O(N²·D)` while
   leaving the attention mechanism untouched. `N ∝ 1/P²`, so patch size is the compute knob.
2. **Linear patch embedding.** Flatten each patch (`P²·C` values) and apply one shared trainable projection
   `E ∈ ℝ^{(P²·C)×D}` to the model width `D`. (Equivalently a Conv2d with kernel = stride = `P`.)
3. **Class token.** Prepend one learnable vector `x_class`; its final-layer state is the pooled image
   representation. This gives the sequence encoder a dedicated readout slot instead of forcing a fixed average
   over patch states.
4. **Learned 1D position embeddings.** Add a learned table `E_pos ∈ ℝ^{(N+1)×D}`; attention is permutation-
   equivariant and needs position injected. A 1D table keeps the spatial prior minimal while giving each raster-order
   slot an identity.
5. **Standard pre-norm Transformer encoder.** Alternating multi-head self-attention and a `~4×`-wide GELU MLP,
   each with a residual and layer norm applied to the sublayer input.

## Final algorithm

Embedding and encoder:

```
z_0  = [ x_class ; x_p^1 E ; … ; x_p^N E ] + E_pos      E ∈ ℝ^{(P²C)×D},  E_pos ∈ ℝ^{(N+1)×D}
z'_ℓ = MSA( LN(z_{ℓ-1}) ) + z_{ℓ-1}                     ℓ = 1 … L
z_ℓ  = MLP( LN(z'_ℓ) )    + z'_ℓ                         ℓ = 1 … L
y    = LN( z_L^0 )
```

Self-attention (per head, `D_h = D/k`):

```
[q, k, v] = z · U_qkv        U_qkv ∈ ℝ^{D×3D_h}
A         = softmax( q kᵀ / √D_h )                      ← /√D_h keeps logits unit-variance, softmax unsaturated
SA(z)     = A v
MSA(z)    = [ SA_1(z) ; … ; SA_k(z) ] · U_msa            U_msa ∈ ℝ^{kD_h×D}
```

Standard sizes: Base (L=12, D=768, heads=12, MLP=3072), Large (L=24, D=1024, heads=16, MLP=4096), Huge
(L=32, D=1280, heads=16, MLP=5120). Notation "L/16" = Large with `16×16` patches.

Training: pre-train on large data (Adam, `β=(0.9,0.999)`, batch 4096, weight decay 0.1, linear LR warmup+decay),
with an MLP+tanh head; transfer by replacing the head with a fresh **zero-initialized** linear `D×K` and fine-tuning
(SGD+momentum), typically at higher resolution — keeping `P` fixed lengthens the sequence, so the pre-trained
position embeddings are **2D-interpolated** to the new grid. The patch cut and this interpolation are the only
hand-injected pieces of 2D structure.

## Working code

A compact PyTorch implementation that preserves the architecture choices above:

```python
import torch
from torch import nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        assert dim % heads == 0, 'model dimension must be divisible by number of heads'
        dim_head = dim // heads
        inner_dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5            # 1 / sqrt(D_h)
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        x = self.norm(x)
        b, n, _ = x.shape
        q, k, v = (t.reshape(b, n, self.heads, -1).transpose(1, 2)
                   for t in self.to_qkv(x).chunk(3, dim=-1))
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.attend(dots))
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, heads=heads, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout),
            ]) for _ in range(depth)
        ])

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads,
                 mlp_dim, channels=3, dropout=0., emb_dropout=0.,
                 representation_size=None):
        super().__init__()
        ih, iw = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        ph, pw = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        assert ih % ph == 0 and iw % pw == 0, 'image size must be divisible by patch size'
        num_patches = (ih // ph) * (iw // pw)            # N = HW / P^2
        self.grid_size = (ih // ph, iw // pw)

        self.patch_embedding = nn.Conv2d(
            channels, dim, kernel_size=(ph, pw), stride=(ph, pw)
        )                                                # E, written as stride-P projection

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.empty(1, num_patches + 1, dim))
        nn.init.normal_(self.pos_embedding, std=0.02)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        head_dim = representation_size or dim
        self.pre_logits = (
            nn.Sequential(nn.Linear(dim, representation_size), nn.Tanh())
            if representation_size else nn.Identity()
        )
        self.head = nn.Linear(head_dim, num_classes)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _position_embedding(self, h, w):
        if (h, w) == self.grid_size:
            return self.pos_embedding
        cls_pos = self.pos_embedding[:, :1]
        patch_pos = self.pos_embedding[:, 1:]
        gh, gw = self.grid_size
        patch_pos = patch_pos.reshape(1, gh, gw, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(h, w), mode='bicubic', align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, h * w, -1)
        return torch.cat((cls_pos, patch_pos), dim=1)

    def forward(self, img):
        x = self.patch_embedding(img)                    # (b, D, H/P, W/P)
        h, w = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)                 # (b, N, D)
        b = x.shape[0]
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls, x), dim=1)                   # (b, N+1, D)
        x = x + self._position_embedding(h, w)
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.pre_logits(x[:, 0])                     # class-token readout
        return self.head(x)


# Base/16 at 224 resolution
def vit_base_patch16_224(num_classes=1000):
    return ViT(image_size=224, patch_size=16, num_classes=num_classes,
               dim=768, depth=12, heads=12, mlp_dim=3072, representation_size=768)
```
