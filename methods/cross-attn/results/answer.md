# Cross-attention conditioning, distilled

Cross-attention conditioning turns an unconditional diffusion denoiser into a conditional one by
inserting attention sublayers in which the **image feature map provides the queries** and the
**condition provides the keys and values**. With a multi-token condition, each spatial position reads
from the condition with content-dependent weights, so the conditioning is spatially varying and
content-aware (unlike a per-channel FiLM affine). The same layer also accepts the class-label special
case: one learnable class embedding token, where the one-key softmax degenerates to a learned residual
class injection rather than spatial routing.

## Problem it solves

Control a diffusion denoiser `eps_theta(x_t, t)` on a side input `c` (a class label, a caption, a
layout), i.e. learn `eps_theta(x_t, t, c)`, with conditioning strong enough to steer the output and
spatially/content-aware, carried by a single mechanism across condition shapes, without destabilizing the
tuned backbone and without an auxiliary model at sampling time.

## Key idea

Insert, after UNet blocks, a cross-attention layer. Flatten the feature map's `H*W` positions into query
tokens; take keys and values from the condition. With `Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V`,

```
Q = W_Q . phi(z_t),   K = W_K . tau(c),   V = W_V . tau(c),
```

where `phi(z_t)` is the (flattened) feature map and `tau(c)` is the condition encoder's output (`M`
tokens; for a class label `tau` is one learnable embedding, `M = 1`). The output has one row per spatial
position, so it returns to the convolutional stream as a feature map. Three design points make it work:

- **Scale `1/sqrt(d_head)`.** Logits `q . k` have variance `d_head` under unit-variance components; without
  the scale, softmax saturates and gradients vanish for wide heads. Dividing restores unit-variance logits.
- **Multi-head** (per-head width `= channels / num_heads`): read several aspects of the condition in
  parallel instead of averaging them into one weighting; cost stays ~ one full-width head.
- **Zero-initialized output projection inside a residual.** With `W_out = 0` the block is the identity at
  init, so it can be stapled onto an already-tuned denoiser without disturbing it; conditioning strength
  grows from zero as training learns it. GroupNorm precedes the layer, matching the backbone.

The class information flows entirely through these attention layers, so the timestep embedding stays a
pure noise-level signal (`prepare_conditioning` returns it unchanged). The training objective keeps the
eps-prediction MSE, now conditioned:

```
L = E_{x_0, c, eps, t} || eps - eps_theta( sqrt(abar_t) x_0 + sqrt(1-abar_t) eps,  t,  tau(c) ) ||^2,
```

with `tau` and the attention projections trained jointly with the backbone; sampling feeds the fixed `c`
at every reverse step.

Note on the single-class case: with `M = 1`, softmax over one key is identically 1, so the value
broadcasts to every position and the trivial attention is, in effect, a learned residual injection of the
class; the content-dependent, spatially-varying behavior is what the *same* operator delivers once `M > 1`
(text/layout). The point of the design is that one operator covers all condition shapes.

## Working code

The cross-attention conditioning layer uses queries from features, keys/values from the condition,
GroupNorm(32), q/k/v linear projections, scaled dot-product multi-head attention, a zero-initialized
output projection, and a residual add:

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def zero_module(module):
    """Zero out a module's parameters so the block starts as the identity."""
    for p in module.parameters():
        p.detach().zero_()
    return module


class CrossAttentionLayer(nn.Module):
    """Features attend to the condition as key/value. Inserted after each UNet block."""
    def __init__(self, channels, context_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.q_proj = nn.Linear(channels, channels)        # query from image features
        self.k_proj = nn.Linear(context_dim, channels)     # key   from condition
        self.v_proj = nn.Linear(context_dim, channels)     # value from condition
        self.out_proj = zero_module(nn.Linear(channels, channels))  # identity at init

    def forward(self, x, context):
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, -1).transpose(1, 2)    # [B, H*W, C] query tokens
        ctx = context.unsqueeze(1) if context.dim() == 2 else context  # [B, M, context_dim]
        q = self.q_proj(h).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(ctx).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(ctx).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)                      # over the M condition tokens
        out = torch.matmul(attn, v)                         # [B, heads, H*W, head_dim]
        out = out.transpose(1, 2).reshape(B, H * W, C)      # merge heads
        out = self.out_proj(out)                            # zero at init
        return x + out.transpose(1, 2).view(B, C, H, W)     # residual


def prepare_conditioning(time_emb, class_emb):
    # time embedding stays a pure noise-level signal; class info flows through cross-attention
    return time_emb


class ClassConditioner(nn.Module):
    """Cross-attention to the class embedding, applied after each UNet block."""
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.cross_attn = CrossAttentionLayer(channels, cond_dim, num_heads=4)

    def forward(self, h, class_emb):
        return self.cross_attn(h, class_emb)
```

Full-strength transformer-block form: `GroupNorm -> 1x1 conv proj_in -> flatten ->
[ self-attention, cross-attention(context), GEGLU feed-forward ] x depth -> 1x1 conv proj_out
(zero-initialized) -> reshape -> residual`, where the cross-attention sublayer is exactly the
queries-from-features, keys/values-from-condition attention above. For a class label the context is one
learnable embedding token (e.g. dimension 512); for text it is the sequence of caption-token embeddings.
