I analyzed the sequence-transduction problem: mapping a variable-length source sequence to a variable-length target sequence autoregressively. Recurrent encoder-decoders compute hidden states one step at a time as h_t = f(h_{t-1}, x_t), so their per-layer computation is sequential in the sequence length and cannot be parallelized within a single example. At long sequence lengths this becomes the binding constraint, because memory limits batch size and there is no way to recover the lost intra-example parallelism on a throughput-oriented accelerator. Attention already provided any-to-any routing with constant path length, but it was always bolted onto recurrent networks. Convolutional alternatives like ByteNet and ConvS2S removed recurrence and achieved parallel per-layer computation, but they reintroduced a path-length tax: the number of layers needed to connect distant positions grows with distance. What is missing is a primitive that gives both O(1) sequential depth and O(1) maximum path length at the same time.

The method I propose is the Transformer. It is a sequence-transduction architecture that replaces recurrence and convolution entirely with attention. The encoder and decoder are both deep stacks built from multi-head self-attention and position-wise feed-forward networks. Self-attention forms every output position as a weighted mixture of all input positions in a single hop, using dot-product scores. Because the computation is a handful of large matrix multiplies and a softmax with no step-to-step dependency, all positions are processed in parallel. The decoder uses masked self-attention to preserve autoregression: future positions are set to negative infinity before the softmax so they receive zero weight, while the full score matrix is still computed in parallel.

The core primitive is scaled dot-product attention. Queries, keys, and values are linear projections of the layer input, and the output is softmax(Q K^T / sqrt(d_k)) V. The 1/sqrt(d_k) scaling is a variance correction: with unit-variance components the dot product q·k has variance d_k, so large d_k would push logits far from zero and saturate the softmax to a near one-hot distribution, collapsing its Jacobian and killing gradients to the attention weights. Dividing by sqrt(d_k) restores unit-variance logits and keeps the softmax in its responsive region. A single attention head produces one averaging pattern, which blurs distinct relations together, so the Transformer uses multi-head attention: h parallel heads each project to d_model/h dimensions, perform scaled dot-product attention in their own subspace, and their outputs are concatenated and mixed by a learned output projection. The cost is essentially the same as one full-width head because d_k = d_model/h.

Three attention roles appear in the architecture. Encoder self-attention lets every source position attend to every other source position. Masked decoder self-attention lets each target position attend only to earlier target positions. Encoder-decoder cross-attention lets the decoder query the final encoder representations, which is the Bahdanau attention mechanism expressed as dot-product attention. Pure attention is permutation-equivariant, so order is injected by adding sinusoidal positional encodings to the input embeddings rather than concatenating them, because concatenation would widen every downstream matrix while a learned linear can already separate content and position from a sum. The sinusoids are chosen so that shifting by a fixed offset corresponds to a position-independent rotation, allowing the model to learn relative-position patterns and extrapolate beyond training length. Embeddings are scaled by sqrt(d_model) so their amplitude matches the O(1) sinusoids. Each attention sublayer and each feed-forward sublayer is wrapped in residual connections and layer normalization; layer normalization is used instead of batch normalization because batch statistics are unreliable on variable-length padded sequences and during one-token-at-a-time decoding. The position-wise feed-forward network applies the same two-layer ReLU MLP independently to every position, providing the nonlinear per-position capacity that attention alone lacks. The standard size uses d_model=512, d_ff=2048, h=8 heads, and N=6 layers in both encoder and decoder.

The model is trained with Adam at beta1=0.9, beta2=0.98, epsilon=1e-9, using a warmup-then-inverse-square-root learning-rate schedule. The warmup keeps early steps small while Adam's adaptive variance estimate and the attention logits settle, and the d_model^-0.5 prefactor scales the peak rate down for wider models. Dropout of 0.1 is applied to sublayer outputs and embedding sums, and label smoothing with epsilon=0.1 discourages overconfident one-hot predictions.

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
        self.linears = clones(nn.Linear(d_model, d_model), 4)  # W^Q, W^K, W^V, W^O
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
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    "Norm-first residual connection."
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

class Batch:
    "Hold source/target tensors and build padding + causal masks."
    def __init__(self, src, tgt=None, pad=2):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()
    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)
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

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

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
        self.true_dist = None
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())
```
