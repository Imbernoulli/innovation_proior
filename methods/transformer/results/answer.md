# The Transformer

The Transformer is a sequence-transduction architecture that replaces recurrence and convolution entirely with attention. It keeps the encoder-decoder shape but builds both stacks out of one cross-position primitive — multi-head self-attention — interleaved with per-position feed-forward networks. Because each layer is a handful of large matrix multiplies and a softmax with no step-to-step dependency, all positions are processed in parallel: the per-layer sequential cost drops from `O(n)` (recurrent) to `O(1)`, and any two positions are connected in a single hop (`O(1)` maximum path length), so long-range dependencies are as easy to route as short ones.

## The problem it solves

Map a source sequence `(x_1, …, x_n)` to a target `(y_1, …, y_m)` autoregressively, `p(y) = ∏_t p(y_t | y_{<t}, x)`. Recurrent encoder-decoders compute `h_t = f(h_{t-1}, x_t)`, which is inherently sequential along the sequence and cannot be parallelized within an example — the binding throughput constraint at long sequence lengths. Attention already gave distance-independent routing but was bolted onto RNNs; convolutional alternatives parallelized but their path length grows with distance (linear for ConvS2S, log for ByteNet). The Transformer attains constant sequential depth *and* constant path length simultaneously, and is cheaper per layer than recurrence whenever `n < d` — the usual regime for subword-tokenized sentences.

## Key ideas

1. **Scaled dot-product attention.** `Attention(Q,K,V) = softmax(QKᵀ / √d_k) V`. Dot-product scoring maps to a single matmul (vs additive scoring's per-pair MLP). The `1/√d_k` is a variance correction: with unit-variance independent components, `q·k = Σ q_i k_i` has mean 0 and variance `d_k`, so logits scale like `√d_k`; large logits saturate the softmax to near one-hot, where its Jacobian `softmax_i(δ_ij − softmax_j) ≈ 0` and the attention weights stop receiving gradient. Dividing by `√d_k` restores unit-variance logits.
2. **Multi-head attention.** Run `h` attentions in parallel over learned projections to `d_k = d_model/h` dimensions, concat, project: `MultiHead(Q,K,V) = Concat(head_1,…,head_h)W^O`, `head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)`. A single head produces one weighting pattern and one average, blurring distinct relations together; separate heads keep distinct relations distinct because each has its own softmax. `W^O` mixes the heads' disjoint subspaces back into one `d_model` vector (concatenation alone never lets heads interact). With `d_k = d_v = d_model/h`, total cost equals one full-width head. Defaults: `d_model=512`, `h=8`, `d_k=d_v=64`.
3. **Three uses of the same block.** Encoder self-attention (Q,K,V from the previous encoder layer); masked decoder self-attention (Q,K,V from the previous decoder layer, future positions set to `−∞` before the softmax → `exp(−∞)=0`, preserving autoregression while keeping the single-matmul parallelism); encoder-decoder cross-attention (queries from the decoder, keys/values from the encoder output — Bahdanau's setup in the same mechanism).
4. **Position-wise feed-forward.** `FFN(x) = max(0, xW_1 + b_1)W_2 + b_2`, applied identically per position, `512 → 2048 → 512`. Attention's value mixing is essentially linear with a single softmax nonlinearity; the FFN is the only per-position nonlinear compute. The 4× inner width gives the ReLU room to carve real features before projecting back; it holds most of each layer's parameters.
5. **Positional encoding.** Pure attention is permutation-equivariant, so order must be injected. Add sinusoids to the input embeddings rather than concatenating position channels, because concatenation only buys separate first-layer projections while widening every downstream matrix: `PE(pos,2i)=sin(pos/10000^{2i/d})`, `PE(pos,2i+1)=cos(pos/10000^{2i/d})`. For any fixed offset `k`, `PE_{pos+k}` is a position-independent rotation of `PE_{pos}`, so heads can attend by relative position; sinusoids are defined for any `pos`, allowing extrapolation beyond training length (a learned table cannot). Embeddings are scaled by `√d_model` so their amplitude matches the `O(1)` sinusoids.
6. **Deep-stack glue.** The architectural wrap is `LayerNorm(x + Sublayer(x))`: the residual's `I + F'` derivative keeps a magnitude-1 gradient path through depth, and LayerNorm is used instead of BatchNorm because batch statistics are unreliable on variable-length padded sequences and at one-token-at-a-time decoding. The code below mirrors the canonical Annotated-Transformer implementation's norm-first variant, `x + Sublayer(LayerNorm(x))`, for code simplicity, with a final norm at each stack output. `N=6` layers each in encoder and decoder; with a shared source-target vocabulary, the source embedding, target embedding, and pre-softmax projection share one weight matrix.
7. **Training.** Adam (`β_1=0.9, β_2=0.98, ε=1e-9`) with warmup-then-inverse-sqrt learning rate `lrate = d_model^{−0.5} · min(step^{−0.5}, step·warmup^{−1.5})`, `warmup=4000` — warmup keeps early steps small while Adam's early second-moment estimate and the attention logits settle; `d_model^{−0.5}` shrinks the peak rate for wider models. Dropout `0.1` on sublayer outputs and embedding sums; label smoothing `ε_ls=0.1` discourages overconfident one-hot targets even if it raises perplexity. Inputs are subword tokens (BPE / word-piece).

## Working code

Grounded in a standard PyTorch implementation, with weight tying and LayerNorm variance made explicit.

```python
import math, copy, torch, torch.nn as nn
from torch.nn.functional import log_softmax

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Scaled dot-product attention."
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)   # W^Q, W^K, W^V, W^O
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nb = query.size(0)
        query, key, value = [
            lin(x).view(nb, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nb, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.a_2 * (x - mean) / torch.sqrt(var + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    "Norm-first residual connection, as in the canonical code for simplicity."
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

def subsequent_mask(size):
    "Causal mask: position i attends only to positions <= i."
    return torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.uint8) == 0

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class Generator(nn.Module):
    "Final projection shares the unscaled token matrix used by the embeddings."
    def __init__(self, tied_embedding):
        super().__init__()
        vocab, d_model = tied_embedding.lut.weight.shape
        self.proj = nn.Linear(d_model, vocab, bias=False)
        self.proj.weight = tied_embedding.lut.weight
    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

def make_model(src_vocab, tgt_vocab=None, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    if tgt_vocab is None:
        tgt_vocab = src_vocab
    if src_vocab != tgt_vocab:
        raise ValueError("three-way weight tying requires a shared source-target vocabulary")
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    src_embedding = Embeddings(d_model, src_vocab)
    tgt_embedding = Embeddings(d_model, tgt_vocab)
    tgt_embedding.lut.weight = src_embedding.lut.weight
    generator = Generator(tgt_embedding)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(src_embedding, c(position)),
        nn.Sequential(tgt_embedding, c(position)),
        generator,
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)   # Glorot init
    return model
```

Training plumbing: the warmup-then-inverse-sqrt schedule, and label smoothing.

```python
def rate(step, model_size, factor, warmup):
    "Warm up linearly, then decay ~ 1/sqrt(step)."
    if step == 0:
        step = 1
    return factor * (model_size ** -0.5 * min(step ** -0.5, step * warmup ** -1.5))

class LabelSmoothing(nn.Module):
    "Soft targets via KL divergence."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return self.criterion(x, true_dist.clone().detach())
```

Wired with Adam (`betas=(0.9, 0.98), eps=1e-9`) under a `LambdaLR(step → rate(step, d_model=512, factor=1, warmup=4000))` scheduler, this trains an `N=6`, `d_model=512`, `h=8` model end to end on subword-tokenized parallel text, with greedy or beam decoding at inference.
