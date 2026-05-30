# Reformer: The Efficient Transformer

## Problem

A standard Transformer trained on long sequences ($L\sim 64\text{K}$ tokens) runs
out of accelerator memory through three multiplicative costs: attention forms an
$L\times L$ matrix ($O(L^2)$ time and memory); activations are stored for all
$N$ layers for backpropagation (an $n_l$ multiplier); and the position-wise
feed-forward sublayer materializes an intermediate of width $d_{ff}\gg d_{model}$
(a $d_{ff}$ multiplier). Reformer removes all three while computing essentially
the same function, so a deep model fits and trains on a single device at
$L=64\text{K}$.

## Key ideas

**1. LSH attention ($O(L^2)\to O(L\log L)$).** Because
$\mathrm{softmax}(QK^\top)$ is dominated by the largest dot products, each query
only needs its nearest keys. Use **shared-QK** attention ($Q=K$, one projection;
keys unit-normalized, queries left unnormalized so their norm acts as a softmax
temperature) and **angular LSH** to find near neighbors: with a random rotation
$R\in\mathbb{R}^{d_k\times b/2}$,
$$h(x)=\arg\max\big([xR;\,-xR]\big)$$
buckets vectors with similar direction together. Sort tokens by
$(\text{bucket},\text{position})$ so bucket-mates become contiguous, cut the
sorted sequence into fixed chunks of size $m=2l/n_{buckets}$, and let each chunk
attend within itself and **one chunk back** (to cover buckets that straddle a
chunk boundary). Run $n_{rounds}$ independent hashes and take the **union** of
buckets to recover neighbors a single hash would miss.

**2. Reversible layers (removes the $n_l$ multiplier).** Split the activation in
two and interleave attention $F$ and feed-forward $G$:
$$Y_1=X_1+\mathrm{Attention}(X_2),\qquad Y_2=X_2+\mathrm{FeedForward}(Y_1),$$
which inverts exactly to
$$X_2=Y_2-\mathrm{FeedForward}(Y_1),\qquad X_1=Y_1-\mathrm{Attention}(X_2).$$
Store activations once (at the top of the stack) and recompute each layer's
inputs from its outputs during backprop. Normalization moves inside $F,G$;
both halves are width $d_{model}$ to match the baseline parameter count. RNG
state is saved/restored so dropout recomputes identically.

**3. Chunked feed-forward (removes the $d_{ff}$ multiplier).** The FFN is
position-wise, so split the sequence into $c$ chunks and process one at a time:
$$Y_2=\big[X_2^{(1)}+\mathrm{FF}(Y_1^{(1)});\dots;X_2^{(c)}+\mathrm{FF}(Y_1^{(c)})\big],$$
numerically identical but with peak memory $d_{ff}/c$. The output log-prob/loss is
chunked over the sequence the same way for large vocabularies.

## Multi-round combination

With $N_{i,j}=|\{r:j\in\mathcal{P}_i^{(r)}\}|$ counting how many rounds make $j$ a
neighbor of $i$, the union attention is assembled from per-round outputs
$o_i^{(r)}$ and per-round log-partitions $z(i,\mathcal{P}_i^{(r)})$ as
$$o_i=\sum_{r=1}^{n_{rounds}}\exp\!\big(z(i,\mathcal{P}_i^{(r)})-z(i,\mathcal{P}_i)\big)\,o_i^{(r)},$$
with the per-round mask
$$m^{(r)}_{i,j}=\begin{cases}\infty & j\notin\mathcal{P}_i^{(r)}\\ 10^5 & i=j\\ \log N_{i,j} & \text{otherwise}\end{cases}$$
where $\infty$ drops out-of-bucket pairs, $\log N_{i,j}$ (i.e. a $1/N_{i,j}$
weight) deduplicates pairs appearing in multiple rounds, and the finite $10^5$
self-penalty forbids self-attention except when a token has no other valid target.

## Complexity

For length $l$, batch $b$, heads $n_h$, $n_r$ rounds, $n_c=l/32$ chunks ($c=128^2$):

| Model | Memory |
|---|---|
| Transformer | $\max(bld_{ff}, bn_hl^2)\,n_l$ |
| Reversible Transformer | $\max(bld_{ff}, bn_hl^2)$ |
| Chunked Reversible | $\max(bld_{model}, bn_hl^2)$ |
| LSH Transformer | $\max(bld_{ff}, bn_hln_rc)\,n_l$ |
| Reformer | $\max(bld_{model}, bn_hln_rc)$ |

## Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states

TOKEN_SELF_ATTN_VALUE = -5e4  # large but finite self-attn penalty

def sort_key_val(t1, t2, dim=-1):
    values, indices = t1.sort(dim=dim)
    t2 = t2.expand_as(t1)
    return values, t2.gather(dim, indices)

def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))

class LSHAttention(nn.Module):
    def __init__(self, bucket_size=64, n_hashes=8, causal=False, dropout=0.):
        super().__init__()
        self.bucket_size, self.n_hashes, self.causal = bucket_size, n_hashes, causal
        self.dropout = nn.Dropout(dropout)

    def hash_vectors(self, n_buckets, vecs):
        b, device = vecs.shape[0], vecs.device
        assert n_buckets % 2 == 0
        R = torch.randn((1, vecs.shape[-1], self.n_hashes, n_buckets // 2),
                        device=device).expand(b, -1, -1, -1)
        rotated = torch.einsum('btf,bfhi->bhti', vecs, R)
        rotated = torch.cat([rotated, -rotated], dim=-1)
        buckets = torch.argmax(rotated, dim=-1)
        offsets = (torch.arange(self.n_hashes, device=device) * n_buckets).reshape(1, -1, 1)
        return torch.reshape(buckets + offsets, (b, -1))

    def forward(self, qk, v, query_len=None, input_mask=None):
        b, seqlen, dim, device = *qk.shape, qk.device
        assert seqlen % (self.bucket_size * 2) == 0
        n_buckets = seqlen // self.bucket_size
        H = self.n_hashes

        buckets = self.hash_vectors(n_buckets, qk)
        ticker = torch.arange(H * seqlen, device=device).unsqueeze(0).expand_as(buckets)
        buckets_and_t = (seqlen * buckets + (ticker % seqlen)).detach()
        sbuckets_and_t, sticker = sort_key_val(buckets_and_t, ticker, dim=-1)
        _, undo_sort = sticker.sort(dim=-1)

        st = sticker % seqlen
        sqk = batched_index_select(qk, st)
        sv  = batched_index_select(v,  st)

        chunk_size = H * n_buckets
        bq_t = bkv_t = torch.reshape(st, (b, chunk_size, -1))
        bqk = torch.reshape(sqk, (b, chunk_size, -1, dim))
        bv  = torch.reshape(sv,  (b, chunk_size, -1, dim))

        bq = bqk
        bk = F.normalize(bqk, p=2, dim=-1).type_as(bq)

        def look_one_back(x):
            x_extra = torch.cat([x[:, -1:, ...], x[:, :-1, ...]], dim=1)
            return torch.cat([x, x_extra], dim=2)
        bk, bv, bkv_t = look_one_back(bk), look_one_back(bv), look_one_back(bkv_t)

        dots = torch.einsum('bhie,bhje->bhij', bq, bk) * (dim ** -0.5)
        masked_value = -torch.finfo(dots.dtype).max

        if self.causal:
            dots.masked_fill_(bq_t[:, :, :, None] < bkv_t[:, :, None, :], masked_value)
        dots.masked_fill_(bq_t[:, :, :, None] == bkv_t[:, :, None, :], TOKEN_SELF_ATTN_VALUE)

        dots_logsumexp = torch.logsumexp(dots, dim=-1, keepdim=True)
        dots = self.dropout(torch.exp(dots - dots_logsumexp))

        bo = torch.einsum('buij,buje->buie', dots, bv)
        so = torch.reshape(bo, (b, -1, dim))
        slogits = torch.reshape(dots_logsumexp, (b, -1,))

        o = batched_index_select(so, undo_sort)
        logits = slogits.gather(1, undo_sort)
        o = torch.reshape(o, (b, H, seqlen, dim))
        logits = torch.reshape(logits, (b, H, seqlen, 1))

        probs = torch.exp(logits - torch.logsumexp(logits, dim=1, keepdim=True))
        return torch.sum(o * probs, dim=1)

class Deterministic(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.cpu_state = self.cuda_in_fwd = self.gpu_devices = self.gpu_states = None
    def record_rng(self, *args):
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(*args)
    def forward(self, *args, record_rng=False, set_rng=False, **kwargs):
        if record_rng:
            self.record_rng(*args)
        if not set_rng:
            return self.net(*args, **kwargs)
        rng_devices = self.gpu_devices if self.cuda_in_fwd else []
        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_in_fwd:
                set_device_states(self.gpu_devices, self.gpu_states)
            return self.net(*args, **kwargs)

class ReversibleBlock(nn.Module):
    def __init__(self, f, g):
        super().__init__()
        self.f, self.g = Deterministic(f), Deterministic(g)
    def forward(self, x, f_args={}, g_args={}):
        x1, x2 = torch.chunk(x, 2, dim=2)
        with torch.no_grad():
            y1 = x1 + self.f(x2, record_rng=self.training, **f_args)
            y2 = x2 + self.g(y1, record_rng=self.training, **g_args)
        return torch.cat([y1, y2], dim=2)
    def backward_pass(self, y, dy, f_args={}, g_args={}):
        y1, y2 = torch.chunk(y, 2, dim=2)
        dy1, dy2 = torch.chunk(dy, 2, dim=2)
        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.g(y1, set_rng=True, **g_args)
            torch.autograd.backward(gy1, dy2)
        with torch.no_grad():
            x2 = y2 - gy1
            dx1 = dy1 + y1.grad
            y1.grad = None
        with torch.enable_grad():
            x2.requires_grad = True
            fx2 = self.f(x2, set_rng=True, **f_args)
            torch.autograd.backward(fx2, dx1, retain_graph=True)
        with torch.no_grad():
            x1 = y1 - fx2
            dx2 = dy2 + x2.grad
            x2.grad = None
            x = torch.cat([x1, x2.detach()], dim=2)
            dx = torch.cat([dx1, dx2], dim=2)
        return x, dx

class _ReversibleFunction(Function):
    @staticmethod
    def forward(ctx, x, blocks, kwargs):
        ctx.kwargs = kwargs
        for block in blocks:
            x = block(x, **kwargs)
        ctx.y, ctx.blocks = x.detach(), blocks
        return x
    @staticmethod
    def backward(ctx, dy):
        y, kwargs = ctx.y, ctx.kwargs
        for block in ctx.blocks[::-1]:
            y, dy = block.backward_pass(y, dy, **kwargs)
        return dy, None, None

class Chunk(nn.Module):
    def __init__(self, chunks, fn, along_dim=-2):
        super().__init__()
        self.chunks, self.fn, self.dim = chunks, fn, along_dim
    def forward(self, x, **kwargs):
        if self.chunks == 1:
            return self.fn(x, **kwargs)
        cs = x.chunk(self.chunks, dim=self.dim)
        return torch.cat([self.fn(c, **kwargs) for c in cs], dim=self.dim)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.w1 = nn.Linear(dim, dim * mult); self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout); self.w2 = nn.Linear(dim * mult, dim)
    def forward(self, x, **kwargs):
        return self.w2(self.dropout(self.act(self.w1(x))))

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__(); self.norm = nn.LayerNorm(dim); self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class LSHSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, bucket_size=64, n_hashes=8, causal=False):
        super().__init__()
        self.heads = heads
        self.toqk = nn.Linear(dim, dim, bias=False)   # shared Q=K
        self.tov  = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.lsh = LSHAttention(bucket_size, n_hashes, causal)
    def forward(self, x, **kwargs):
        b, t, e, h = *x.shape, self.heads
        qk, v = self.toqk(x), self.tov(x)
        merge = lambda z: z.view(b, t, h, -1).transpose(1, 2).reshape(b * h, t, -1)
        out = self.lsh(merge(qk), merge(v), **kwargs)
        out = out.view(b, h, t, -1).transpose(1, 2).reshape(b, t, -1)
        return self.to_out(out)

class ReversibleSequence(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList([ReversibleBlock(f, g) for f, g in blocks])
    def forward(self, x, **kwargs):
        return _ReversibleFunction.apply(x, self.blocks, {'f_args': kwargs, 'g_args': {}})

class Reformer(nn.Module):
    def __init__(self, dim, depth, heads=8, bucket_size=64, n_hashes=8,
                 ff_chunks=100, causal=False):
        super().__init__()
        blocks = []
        for _ in range(depth):
            f = PreNorm(dim, LSHSelfAttention(dim, heads, bucket_size, n_hashes, causal))
            g = PreNorm(dim, Chunk(ff_chunks, FeedForward(dim), along_dim=-2))
            blocks.append((f, g))
        self.layers = ReversibleSequence(blocks)
    def forward(self, x, **kwargs):
        x = torch.cat([x, x], dim=-1)
        x = self.layers(x, **kwargs)
        return torch.stack(x.chunk(2, dim=-1)).mean(dim=0)

class ReformerLM(nn.Module):
    def __init__(self, num_tokens, dim, depth, max_seq_len, heads=8,
                 bucket_size=64, n_hashes=8, ff_chunks=100, causal=True):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.reformer = Reformer(dim, depth, heads, bucket_size, n_hashes, ff_chunks, causal)
        self.norm = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens)
    def forward(self, x, **kwargs):
        t = torch.arange(x.shape[1], device=x.device)
        x = self.token_emb(x) + self.pos_emb(t)
        x = self.reformer(x, **kwargs)
        return self.to_logits(self.norm(x))
```
