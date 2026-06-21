The problem is how to make a diffusion denoiser conditional on a side input such as a class label, text caption, or spatial layout. The denoiser is a UNet that already predicts noise from the noisy image and the noise level, and its backbone, group normalization, training recipe, and sampling schedule are already tuned. The only design freedom is the operator that injects the condition into the network. Existing options all hit a wall. FiLM and adaptive group normalization condition every spatial position with the same per-channel scale and shift, so the modulation is global, content-blind, and too low-bandwidth to steer complex class-specific structure. Concatenating the condition to the input only works when the condition is itself an image-shaped map aligned to the output, so it cannot handle a class index or a variable-length caption. Classifier guidance solves the bandwidth problem outside the network by training a separate noise-robust classifier and adding its gradient at sampling time, but that costs an extra model and extra compute. What is needed is a single operator inside the denoiser that is content-dependent, spatially varying, accepts conditions of different shapes and lengths, and can be inserted into the tuned backbone without destroying it.

The method is cross-attention conditioning. The idea is to insert one cross-attention layer after each UNet block, where the image feature map provides the queries and the condition provides the keys and values. The feature map is flattened into a sequence of spatial tokens, each of which projects to a query vector. The condition is encoded into a set of key and value vectors. With scaled dot-product attention, the output at each spatial position is a weighted sum of the condition values, and the weights depend on dot products between that position's query and the condition keys. Because the query comes from the local feature content, the conditioning is content-dependent and spatially varying, unlike a global affine. Because attention is agnostic to the number of keys, the same layer accepts a single class token, a sequence of text tokens, or any other token set without changing its wiring.

Several design details matter. First, the attention logits are scaled by one over the square root of the per-head dimension. Without this scaling, logits grow with head width, the softmax saturates toward one-hot, and gradients vanish; the scaling keeps logits at unit variance so the layer remains trainable. Second, the layer uses multiple attention heads so that different heads can read different aspects of the condition in parallel, rather than forcing everything into a single weighting. Third, the output projection is zero-initialized and wrapped in a residual connection, so the block starts as the identity and can be stapled onto an already-tuned denoiser. Conditioning strength then grows smoothly from zero as training learns to use it. Group normalization is applied before the projections, matching the rest of the UNet and keeping query magnitudes controlled.

It is worth being honest about the single-class-label case. A class label reduces to one condition token, so the softmax over a single key is identically one and the content-dependence through the query disappears. In that corner, the layer acts as a learned, zero-init-gated residual injection of the class rather than as a spatial router. The residual gain over simpler affine conditioning comes from the extra projection, normalization, and capacity, but the full spatial-routing benefit only appears once the condition contains multiple tokens, as with text or layout. The reason to build the operator this way anyway is generality: the identical layer handles class, text, and layout through the same wiring.

The training objective keeps the standard eps-prediction mean squared error, now with the condition as an additional argument. Sampling uses the same DDIM reverse process, feeding the fixed condition at every step. The timestep embedding is left untouched as a pure noise-level signal; all class or text information flows through the cross-attention layers, giving a clean separation between "how noisy" and "what to draw."

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
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(context_dim, channels)
        self.v_proj = nn.Linear(context_dim, channels)
        self.out_proj = zero_module(nn.Linear(channels, channels))

    def forward(self, x, context):
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, -1).transpose(1, 2)
        ctx = context.unsqueeze(1) if context.dim() == 2 else context
        q = self.q_proj(h).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(ctx).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(ctx).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, H * W, C)
        out = self.out_proj(out)
        return x + out.transpose(1, 2).view(B, C, H, W)


def prepare_conditioning(time_emb, class_emb):
    return time_emb


class ClassConditioner(nn.Module):
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.cross_attn = CrossAttentionLayer(channels, cond_dim, num_heads=4)

    def forward(self, h, class_emb):
        return self.cross_attn(h, class_emb)
```