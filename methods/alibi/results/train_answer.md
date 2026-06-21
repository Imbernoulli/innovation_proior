Transformer language models are usually trained on fixed-length subsequences of size L, but the real goal is to run them on longer inputs at inference so each prediction sees more left context. The trouble is that standard position mechanisms do not generalize past the training length. Learned absolute embeddings are literally undefined beyond L, sinusoidal embeddings quickly expose the model to unseen phase combinations, and rotary or learned relative-bias methods either degrade after a modest extrapolation range or add runtime overhead and learned parameters. The failure is pinned on the position representation itself, not the transformer: hold everything else fixed and swap only the position method, and extrapolation behavior changes. So the right target is a position method that is relative rather than absolute, non-learned, parameter-free, cheap to apply, and naturally defined at every distance.

The method I would use is Attention with Linear Biases, or ALiBi. It removes position embeddings entirely and encodes order only as a fixed, additive score penalty in each attention layer. For a query at position i attending to a key at position j with j ≤ i, the score receives the bias −m(i − j), where m > 0 is a per-head slope fixed before training. Because the penalty depends only on the relative distance i − j, there is no absolute coordinate that can run out of range at inference. Because it is a straight line, any distance the model never saw during training is just a larger value on the same line it already learned to handle; it is a forced continuation rather than a new kind of signal. The linear growth also gives a built-in recency bias: distant keys are increasingly down-weighted after softmax, so extra far-away tokens at inference are gently suppressed instead of disrupting the computation.

The slopes are chosen to spread heads across different effective attention windows. For a power-of-two head count n, they form a geometric sequence mh = 2^(−8h/n) for h = 1, …, n. With eight heads this gives 1/2, 1/4, 1/8, …, 1/256; larger slopes yield sharply local heads, while slopes near zero yield long-range heads. Geometric spacing is important because it clusters resolution near zero, where small changes in slope produce large changes in effective window size. For other head counts, keep the nearest lower power-of-two set and append every other slope from the next power-of-two set. The slopes are not learned; making them trainable weakens extrapolation and reintroduces length-dependent overfitting. The bias is also not scaled by 1/√d_k, since it is a deliberate score-space penalty rather than part of the dot product.

Implementation is almost free. The additive bias folds directly into the causal attention mask the model already builds before softmax, so there are no extra matrix multiplications. The only cost is that the mask becomes per-head instead of shared, giving shape n_heads × L × L. Exploiting softmax translation invariance, the desired finite bias −m(i − j) equals m j − m i, and the −m i term is a row constant that cancels in softmax. So we can simply broadcast the row pattern m · [0, 1, …, L − 1] for each head and add the usual upper-triangular −∞ mask. The model below shows the full construction.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_slopes(n_heads):
    def slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))  # = 2^{-8/n}
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n_heads).is_integer():
        return slopes_power_of_2(n_heads)
    closest = 2 ** math.floor(math.log2(n_heads))
    return (slopes_power_of_2(closest)
            + get_slopes(2 * closest)[0::2][: n_heads - closest])


def build_attn_mask(seq_len, n_heads, device=None):
    slopes = torch.tensor(get_slopes(n_heads), device=device)
    # m_h * [0, 1, ..., L-1]; row constants cancel in softmax.
    bias = slopes[:, None, None] * torch.arange(seq_len, device=device)[None, None, :]
    bias = bias.expand(n_heads, seq_len, seq_len)
    causal = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), 1)
    return causal[None] + bias


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
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
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
        x = self.tok(tokens)  # no position embedding added
        mask = build_attn_mask(T, self.layers[0].attn.n_heads, device=x.device)
        for layer in self.layers:
            x = layer(x, mask)
        return self.head(self.ln_f(x))


def lm_loss(logits, targets):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
```

To use this, train the model at the desired short length L and at inference build the mask for any longer length L_valid. The token embeddings carry no position signal, and the same fixed slopes apply without retuning. This gives extrapolation without learned position parameters and without extra attention operations, so longer inputs can be accepted at essentially the same per-step cost as the cheapest baseline.
