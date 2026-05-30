# Attention with Linear Biases (ALiBi)

## Problem

A transformer language model trained on inputs of length $L$ cannot be reliably run on longer inputs at inference: with the usual position representations, perplexity degrades sharply once the evaluation length exceeds $L$. Absolute tables are undefined past $L$, sinusoids expose the model to unseen phase combinations, and rotary-style relative phases still degrade after a limited extrapolation range. We want a position method that **extrapolates** to much longer inputs, costs no more than the cheapest existing method (sinusoidal), and adds no learned parameters.

## Key idea

Remove position embeddings entirely. Inject order only as a **static, non-learned, additive bias on the attention scores** that grows linearly with the distance between query and key. For zero-indexed query position $i$ attending to key position $j$ (with $0 \le j \le i$ under the causal mask):

$$
\text{softmax}\!\big(\mathbf{q}_i \mathbf{K}_{\le i}^\top + m \cdot [-i, -(i-1), \dots, -2, -1, 0]\big),
$$

i.e. the bias added to score$(i,j)$ is $-m\,(i-j)$. The scalar $m > 0$ is a fixed, head-specific slope. This is a relative-position method (only $i-j$ matters), applied at every layer, never added to the values.

Why it extrapolates: the bias is a parameter-free monotone function of relative distance. At a distance never seen in training, $-m\,d$ is a more negative value on the same line the model experienced for small $d$ — a forced continuation, not a new kind of position signal. Softmax is translation-invariant, so only relative gaps matter and no absolute coordinate can overflow. As distance grows, the bias also gives a **recency inductive bias**: distant keys are increasingly down-weighted, so extra far-away tokens at inference are gently suppressed rather than disruptive.

## The slopes

For $n$ heads, the slopes form a geometric sequence with start and ratio both equal to $2^{-8/n}$:

$$
m_h = 2^{-8h/n}, \qquad h = 1, \dots, n.
$$

- $n=8$: $\tfrac{1}{2}, \tfrac{1}{4}, \tfrac{1}{8}, \dots, \tfrac{1}{256}$ (start $=$ ratio $= 2^{-1}$).
- $n=16$: start $=$ ratio $= 2^{-1/2} = 1/\sqrt{2}$, giving $2^{-0.5}, 2^{-1}, 2^{-1.5}, \dots, 2^{-8}$ — the 8-head set with a geometric mean interposed between each consecutive pair.

The slopes lie in $(0,1)$ and are **dense near 0**: many heads with small slopes (long, near-uniform effective windows, finely spaced) and a few with large slopes (sharp recency). Geometric spacing gives this; linear spacing would waste resolution on the local heads and starve the valuable long-range ones. The slopes are **fixed before training and not learned** — making them trainable yields weak extrapolation (and a small slowdown). The method is robust to the exact set, so it is fixed once and reused across model sizes and datasets without retuning. For a head count that is not a power of two, take the nearest lower power-of-two's slopes and interpolate the remainder from the next set.

The bias is **not** multiplied by the $1/\sqrt{d_k}$ score scaling.

## Implementation

ALiBi fills the decoder's additive attention-mask slot. Memory grows only because the mask becomes per-head ($n \times L \times L$ instead of $L \times L$). Construction exploits softmax translation-invariance: the desired finite bias is $-m_h(i-j)=m_hj-m_hi$, so the per-head pattern $m_h \cdot [0,1,\dots,L-1]$ broadcast across all rows is the same scores plus the row constant $m_hi$, which cancels in the softmax.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_slopes(n_heads):
    # m_h = 2^{-8h/n}: geometric, start = ratio = 2^{-8/n}, dense near 0. Fixed, not learned.
    def slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))     # = 2^{-8/n}
        ratio = start
        return [start * ratio ** i for i in range(n)]  # = 2^{-8(i+1)/n}

    if math.log2(n_heads).is_integer():
        return slopes_power_of_2(n_heads)
    closest = 2 ** math.floor(math.log2(n_heads))       # non-power-of-2 fallback
    return (slopes_power_of_2(closest)
            + get_slopes(2 * closest)[0::2][: n_heads - closest])


def build_attn_mask(seq_len, n_heads, device=None):
    slopes = torch.tensor(get_slopes(n_heads), device=device)               # (n_heads,)
    # Per-head pattern m_h * [0,1,...,L-1]; rows identical after softmax row shifts cancel.
    bias = slopes[:, None, None] * torch.arange(seq_len, device=device)[None, None, :]
    bias = bias.expand(n_heads, seq_len, seq_len)                           # (n_heads, L, L)
    causal = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), 1)
    return causal[None] + bias                                             # (n_heads, L, L)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, attn_mask):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)        # bias NOT scaled
        scores = scores + attn_mask.to(device=scores.device, dtype=scores.dtype)
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, -1)
        return self.out(out)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask):
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.ff(self.ln2(x))
        return x


class LMModel(nn.Module):
    def __init__(self, vocab, d_model, n_heads, d_ff, n_layers):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, tokens):
        B, T = tokens.shape
        x = self.tok(tokens)  # no added position embedding
        mask = build_attn_mask(T, self.layers[0].attn.n_heads, device=x.device)
        for layer in self.layers:
            x = layer(x, mask)
        return self.head(self.ln_f(x))


def lm_loss(logits, targets):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
```

In a full decoder-only LM this replaces all positional encoding: the token embeddings enter the stack with no added position signal, and every self-attention sublayer receives `build_attn_mask(T, n_heads)`. To run on longer inputs at inference, build the mask at the longer length; the slopes are unchanged.

## What it buys

Train on short subsequences and evaluate on longer ones without adding learned position parameters or extra attention operations. Much of the intended gain when feeding longer nonoverlapping segments comes from reducing the *early token curse*: fewer predictions occur near a segment boundary with almost no left context, and the model can accept the longer segment without its position mechanism breaking.
