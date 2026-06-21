The analysis starts from a basic limitation of the standard Transformer attention head. A single softmax attention map is a strictly positive probability distribution over every context position, so it can never assign exactly zero weight to an irrelevant token. On a long context, each irrelevant token only receives a small weight, but with thousands of positions those small weights add up to a large fraction of the total attention mass. The result is a noisy average in which the signal is diluted by a common floor of irrelevant information, which shows up empirically as attention leakage and the lost-in-the-middle failure. Existing fixes do not remove this floor. Sparse or local attention patterns cut out whole regions of context by construction, but they are content-blind and change the positions that are even computed rather than the attention scores themselves. QK normalization and logit soft-capping keep the softmax in a healthy dynamic range, but they still leave a strictly positive distribution with no subtraction mechanism. The core problem is therefore operator-level: a single softmax can only add positive mass, never cancel it.

The right fix is to build attention like a differential amplifier or a pair of noise-cancelling headphones: form two correlated attention maps and subtract one from the other so that the common noise floor cancels while the signal survives. The method is the Differential Transformer. It replaces the single positive softmax in each head with the difference of two softmax maps, DiffAttn(X) = (softmax(Q1 K1^T / sqrt(d)) - lambda * softmax(Q2 K2^T / sqrt(d))) V, where Q and K are each split into two halves, Q = [Q1; Q2] and K = [K1; K2]. Both maps are positive distributions and both carry a correlated noise floor over the irrelevant tokens. Because the floor is common, subtracting a scaled copy of the second map removes it. The result is a signed attention pattern that can drive an irrelevant token's contribution toward zero, which a single softmax structurally cannot do. On relevant tokens the model learns to make the two maps disagree, so the difference remains large and the signal is preserved.

Several details are needed to make this a practical, drop-in replacement. First, lambda must be learned in a stable way. Instead of learning lambda directly, it is reparameterized as lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init, using four learnable head-dimension vectors initialized from a normal distribution with small standard deviation. At initialization the two exponentials nearly cancel, so lambda starts near lambda_init, and the exponential form gives well-scaled signed gradients. The initialization lambda_init is depth-dependent: lambda_init = 0.8 - 0.6 * exp(-0.3 * (l - 1)), starting around 0.2 in early layers and rising toward 0.8 in deep layers. This gives gentle cancellation while the model is still doing broad mixing and stronger cancellation once the layers need focused retrieval.

Second, the parameter and FLOP budget must stay matched to a vanilla Transformer. The doubled query and key are absorbed by halving the head dimension. Each logical head uses head_dim = d_model / n_heads / 2, with 2 * n_heads query and key sub-heads of that smaller dimension and n_heads value heads of dimension 2 * head_dim. Total query and key width is 2 * n_heads * (d_model / n_heads / 2) = d_model, and value width is n_heads * (d_model / n_heads) = d_model, exactly the same as a standard multi-head attention block. Third, the subtraction makes different heads heterogeneous in output scale, so a per-head RMSNorm is applied to each head's 2 * head_dim output before concatenation. After normalization, the output is rescaled by the fixed constant (1 - lambda_init) to compensate for the gain lost by the subtraction. Using the fixed init rather than the learned lambda keeps the compensation stable and lets the model reuse standard Transformer hyperparameters. Causal masking, RoPE position encoding, and the output projection are left unchanged.

The Differential Transformer is therefore a drop-in change to the score-formation step of multi-head attention. It keeps the same projections, the same parameter count, the same FLOPs, and the same residual placement, but it replaces one positive softmax with a difference of two softmax maps, giving the attention operator a learnable way to cancel its own noise floor.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def lambda_init_fn(depth):
    # depth: 0-based layer index. Paper schedule (1-based l): 0.8 - 0.6*exp(-0.3*(l-1)).
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x):
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            out = out * self.weight
        return out


class MultiheadDiffAttn(nn.Module):
    def __init__(self, d_model, n_heads, depth):
        super().__init__()
        self.n_heads = n_heads
        # Halve the head dim so the doubled q/k is parameter/FLOP matched to a vanilla head.
        self.head_dim = d_model // n_heads // 2
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))

        # Per-head normalization over the 2*head_dim head output.
        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, x, rope, attn_mask=None):
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, 2 * self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, 2 * self.n_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_heads, 2 * self.head_dim)

        q, k = rope(q, k)                       # relative position on the doubled q/k sub-heads
        q = q.transpose(1, 2)                   # (B, 2H, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)                   # (B,  H, T, 2*head_dim)

        q = q * self.scaling
        att = torch.matmul(q, k.transpose(-1, -2))           # (B, 2H, T, T)
        if attn_mask is None:
            attn_mask = torch.triu(
                torch.full((T, T), float("-inf"), device=x.device, dtype=att.dtype), 1)
        att = att + attn_mask
        att = F.softmax(att, dim=-1, dtype=torch.float32).type_as(att)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        att = att.view(B, self.n_heads, 2, T, T)
        att = att[:, :, 0] - lambda_full * att[:, :, 1]      # (B, H, T, T), signed

        o = torch.matmul(att, v)                             # (B, H, T, 2*head_dim)
        o = self.subln(o)                                    # per-head normalization
        o = o * (1.0 - self.lambda_init)                     # fixed gain compensation
        o = o.transpose(1, 2).reshape(B, T, self.n_heads * 2 * self.head_dim)
        return self.out_proj(o)
```
