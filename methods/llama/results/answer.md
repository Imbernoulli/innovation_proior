# LLaMA

## Problem

A deployed language model is trained once but served billions of times, so its lifetime cost is dominated by *inference*, which scales with parameter count $N$. The established scaling recipes (Kaplan-style, and the Chinchilla compute-optimal balance $N\propto C^{1/2}, D\propto C^{1/2}$) minimize the one-time *training* compute and stop at the training-optimal token count. For a fixed target quality, the model you actually want to ship is the smallest one that reaches it — even if reaching it costs more training tokens.

## Key idea

Trade extra (one-time) training tokens for a permanently smaller, faster-to-serve model: pick a relatively small $N$ and train it on far more tokens than compute-optimality assigns. This is viable because a small model's loss is still falling well past 1T tokens — it has not saturated. LLaMA is a family of dense decoder Transformers (7B–65B) trained on 1.0–1.4T tokens of *publicly available* data, so it can be released openly. The architecture is the standard Transformer with three targeted improvements chosen for long-horizon stability and budget-neutral capacity:

- **Pre-normalization with RMSNorm.** Normalize each sub-layer's *input* (clean residual highway → stable deep optimization). Use root-mean-square normalization $x/\sqrt{\tfrac1d\sum x_j^2+\epsilon}\cdot g$ — re-scaling only, no mean subtraction, no bias — which is cheaper than LayerNorm and keeps the half of normalization that matters.
- **SwiGLU feed-forward.** Replace $W_2\,\text{ReLU}(W_1x)$ with $W_2(\text{SiLU}(W_1x)\odot W_3x)$. A GLU has three matrices instead of two, so the hidden width is shrunk from $4d$ to $\tfrac23\cdot 4d$ to hold parameters and FLOPs constant.
- **Rotary position embeddings (RoPE).** Rotate each 2D pair of $q,k$ by an angle $m\theta_i$ with $\theta_i=10000^{-2(i-1)/d}$. Because rotations compose by adding angles, $\langle R(m\theta)q, R(n\theta)k\rangle = q^\top R((n-m)\theta)k$ depends only on the relative offset $m-n$, with a natural decay of attention with distance. Applied to $q,k$ at every layer; no learned absolute position embedding.

Optimization: AdamW ($\beta_1=0.9$, $\beta_2=0.95$ — short second-moment memory for large-batch stability), weight decay $0.1$, gradient clip $1.0$, 2000 warmup steps, cosine decay to 10% of peak LR, batch 4M tokens, BPE/SentencePiece tokenizer (digits split, byte fallback). Efficiency: memory-efficient causal attention (no full score matrix, masked scores skipped), selective activation checkpointing with a hand-written transformer backward, tensor/sequence parallelism with communication overlapped.

Model family: 7B ($d{=}4096$, 32 heads, 32 layers), 13B ($5120/40/40$), 33B ($6656/52/60$), 65B ($8192/64/80$).

## Code

```python
import math
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    vocab_size: int = -1
    multiple_of: int = 256
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    shape = [d if i == 1 or i == xq_.ndim - 1 else 1 for i, d in enumerate(xq_.shape)]
    freqs_cis = freqs_cis.view(*shape)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.cache_k = torch.zeros(args.max_batch_size, args.max_seq_len, self.n_heads, self.head_dim)
        self.cache_v = torch.zeros(args.max_batch_size, args.max_seq_len, self.n_heads, self.head_dim)

    def forward(self, x, start_pos, freqs_cis, mask):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)
        self.cache_k[:bsz, start_pos:start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos:start_pos + seqlen] = xv
        keys = self.cache_k[:bsz, : start_pos + seqlen].transpose(1, 2)
        values = self.cache_v[:bsz, : start_pos + seqlen].transpose(1, 2)
        xq = xq.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values).transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id, args):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args.dim, 4 * args.dim, args.multiple_of)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, start_pos, freqs_cis, mask):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        return h + self.feed_forward(self.ffn_norm(h))


class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList(TransformerBlock(i, args) for i in range(args.n_layers))
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.freqs_cis = precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len * 2)

    @torch.inference_mode()
    def forward(self, tokens, start_pos):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        return self.output(h).float()
```
