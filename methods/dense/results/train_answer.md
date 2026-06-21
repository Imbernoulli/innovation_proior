The problem is to build a single layer that mixes information across all positions of a sequence. A sequence model needs each position to read from the others, but the dominant recurrent stack passes information through a hidden state that is overwritten at every step. That makes the computation inherently sequential, with O(N) sequential operations, and it forces every long-range dependency to survive an O(N) chain of state overwrites, which is exactly the regime where gradients decay and distant relationships become hard to learn. Convolution recovers parallelism, but a finite kernel only sees a local window, so two far-apart positions still need a deep stack to meet; the path length still grows with distance. What is missing is a layer that connects every pair of positions in a single hop while staying cheap enough to be the main mixing operation inside a deep model.

The natural candidate is attention, which already lets a decoder read any source position directly. The additive attention used in early translation systems computes every pairwise score with a small feed-forward network. It is robust, but it does not collapse into a single large matrix multiply; for n positions that means evaluating a per-pair MLP n^2 times per layer, which is the wrong cost profile for a layer that must run dozens of times. Dot-product attention is much cheaper because all n^2 scores become one dense GEMM, QK^T, but it was used unscaled. As the key dimension grows, the variance of the dot product grows with it: if query and key entries have variance 1, then Var(q^T k) = d_k, so the logits have standard deviation sqrt(d_k). That pushes the softmax toward a sharp one-hot distribution and drives the softmax gradient to nearly zero, which explains the observed collapse of unscaled multiplicative attention at large width.

The method is scaled dot-product attention. It keeps the single-GEMM efficiency of multiplicative attention and removes the width-dependent instability by dividing the scores by sqrt(d_k). The read is Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V. Each position emits a query, a key, and a value. The score between a query and a key is their dot product, scaled so the logit standard deviation is 1 regardless of width. Softmax turns each query's score row into a normalized, nonnegative, differentiable weighting, and the output is the weighted blend of values. For autoregressive decoding, the future is masked out by adding negative infinity to illegal entries before the softmax, so they receive zero weight and the legal entries renormalize automatically. With no mask, every position attends to every position, giving the dense, fully-connected read with O(1) sequential depth and O(1) maximum path length.

The implementation below shows the canonical explicit form and a thin module wrapping PyTorch's fused scaled_dot_product_attention, which computes exactly the same operation. Both forms use the derived scale 1/sqrt(d_k) and support an optional additive mask or causal flag.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    """Reference scaled dot-product attention: softmax(Q K^T / sqrt(d_k) + mask) V."""
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    if dropout is not None:
        p = dropout(p)
    return torch.matmul(p, v)


class ScaledDotProductAttention(nn.Module):
    """Dense scaled dot-product attention.

    Computes softmax(Q K^T / sqrt(d_k) + mask) V as one fused primitive.
    """

    def __init__(self, dropout_p=0.0):
        super().__init__()
        self.dropout_p = dropout_p

    def forward(self, q, k, v, attn_mask=None, is_causal=False, scale=None):
        d_k = q.size(-1)
        scale = scale if scale is not None else 1.0 / math.sqrt(d_k)
        dropout_p = self.dropout_p if self.training else 0.0
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )
```
