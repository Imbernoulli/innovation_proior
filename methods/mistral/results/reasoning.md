OK, let me think this through from scratch. The reflex in the field is to chase quality by growing the model, but every parameter I add makes the thing more expensive and slower to *serve*, and serving is where a deployed model spends its whole life. So I want to flip the objective: fix a small budget — call it around 7B parameters — and ask how to make that model both good and genuinely cheap at inference. "Cheap at inference" isn't one thing, so before I touch the architecture I want to know precisely what makes autoregressive decoding expensive, because that's the regime I'm paying for.

I generate one token at a time. Each step, the machine reads the model weights and reads the cached keys and values for every position I've already produced, then does a little arithmetic to emit the next distribution. Two costs jump out and they're different in character. First, the key/value cache: at step $i$ I'm holding keys and values of shape $[\text{heads}, i, \text{head\_dim}]$ per sequence, and I reload them every step. That traffic grows with the number of heads *and* with the history length $i$, and it's what makes decoding memory-bandwidth-bound — it caps how many sequences I can batch and how long they can get. Second, the attention itself: a query at position $i$ attends to all $i$ past keys, so over a full sequence of length $n$ the compute is $O(n^2)$ and the stored cache is $O(n)$ per token and growing without bound. Two separate bottlenecks: one is "the cache is fat per position" (set by head count), the other is "the cache and the attention grow with $n$" (set by sequence length). I'll attack them separately.

Start with the per-position fatness. The cache holds one key and one value vector *per head per position*. Standard multi-head attention learns a separate key, value, and query projection for each of the $h$ heads — so $h$ keys and $h$ values cached at every position. But notice the asymmetry in what gets cached: the *query* I bring to a decode step is just the current token's query, tiny, never cached; it's the keys and values that accumulate. So the cache cost is set by how many distinct *key/value* heads I store, not by how many query heads I run. That's a lever. Keep all $h$ query heads — they do the representational work of attending $h$ different ways, and they're cheap because they're not cached — but cut the number of key/value heads. If I drop from $h$ key/value heads to $g$ of them, each shared by a group of $h/g$ query heads, the cache shrinks by the factor $h/g$. The extreme $g=1$ — one shared key/value head — minimizes the cache but forces all query heads through a single key/value subspace, which costs quality and makes training brittle. So I don't go all the way: pick an intermediate $g$. With $h=32$ query heads and $g=8$ key/value heads, every key/value head serves $32/8 = 4$ query heads, the cache is $4\times$ smaller, and the capacity is spread over eight subspaces instead of collapsed into one. In the layer this means the query projection emits $32$ heads' worth while the key and value projections emit only $8$ heads' worth; the eight cached heads are then repeated four times each, *after* loading from the cache and just before the score matmul, so I compute against $32$ but only ever store $8$. That's the first bottleneck cut by four — for free on quality, with the grouping.

Now the harder one: the cache and the attention growing with sequence length $n$. Cutting head count is a constant factor; it does nothing about the $O(n^2)$ compute and the unbounded $O(n)$ cache that a long input forces. For the long contexts I care about, $n$ in the thousands or tens of thousands, this is the real wall. The standard move is to make attention *local*: let a query at position $i$ attend only to a fixed window of recent keys, say $[i-W, i]$, instead of all of the past. That instantly makes attention $O(n\cdot W)$ instead of $O(n^2)$ — linear in length — and, crucially, it means a query never looks further back than $W$.

But the obvious objection is fatal if true: if every query only sees $W$ tokens back, haven't I thrown away all long-range information? A token at position $10000$ would be blind to position $0$. That would be unacceptable — language needs long-range dependence. So local attention seems like a non-starter. Let me stare at it for a second, because there's something I'm not using: the model is a *stack* of layers, and each layer's local attention feeds the next.

Trace what one position can see as I go up the stack. At layer 1, the hidden state at position $i$ attends to input-layer positions in $[i-W, i]$ — it sees $W$ back. At layer 2, position $i$ attends to *layer-1* hidden states in $[i-W, i]$. But each of those layer-1 states already summarized a window $W$ wide below it — the layer-1 state at position $i-W$ saw input positions down to $i-2W$. So the layer-2 state at $i$ depends, transitively, on input positions down to $i-2W$. By induction, after $k$ layers the hidden state at position $i$ depends on input positions as far back as $i - kW$. The information isn't blocked by the window; it *propagates* upward one window per layer, exactly like the receptive field of a deep stack of small-kernel convolutions widening with depth. So a token outside any single layer's window still influences a far-future prediction — it just takes a few layers for its influence to climb. The effective span isn't $W$, it's $k\,W$. With $W = 4096$ and $32$ layers, the last layer can reach back about $32\times4096 \approx 131{,}000$ tokens. So I get linear-cost local attention *and* a six-figure effective context — the depth buys back the range that the window gave up. With a memory-efficient kernel that supports the local mask, this is a real wall-clock speedup over dense attention, not just an asymptotic one.

Now the part I like, because it's a free consequence rather than a new mechanism. Local attention with a fixed window $W$ means a query at position $i$ attends only to $[i-W+1,\, i]$ — at most $W$ keys. So any key/value at a position $\le i - W$ is *dead*: no current or future query within the window will ever attend to it again. (Future queries at $j > i$ attend to $[j-W+1, j]$, which only moves the window forward.) If those entries are never read again, why am I storing them? At any moment I only need the most recent $W$ keys and values per sequence — so cap the cache at size $W$ and stop it from growing. That's the third mechanism, and it falls straight out of the fixed span.

How do I physically bound it? A ring buffer of length $W$. Store the key/value for position $i$ at slot $i \bmod W$. When $i$ reaches $W$ and beyond, writing position $i$ lands on slot $i \bmod W$, which currently holds position $i - W$ — and $i - W$ is exactly the entry that just fell out of every window, so overwriting it loses nothing. The cache size stops growing at $W$ regardless of how long the sequence gets. For a 32k-token sequence with $W = 4096$, that's $32768/4096 = 8\times$ less cache memory, with no quality cost because the discarded entries were unreachable anyway. One bookkeeping subtlety: the ring stores entries out of chronological order once it wraps (slot $0$ might hold the newest, slot $W-1$ an older one), so when I hand the cache to the attention kernel I "unrotate" it — if the write position has wrapped to $p = \text{seqlen} \bmod W$, the chronological order is the concatenation of the slots from $p$ onward followed by the slots before $p$. And when ingesting a chunk I only keep the last $W$ of its tokens for the cache, since earlier ones are already out of window.

One more inference cost I can kill, and it's about the *prompt*. Decoding is inherently one-token-at-a-time because each token conditions on the previous, but the prompt is fully known up front — there's no reason to feed it through one token at a time. Pre-fill the key/value cache with the whole prompt in a single parallel pass. If the prompt is long, don't pre-fill it all at once (that would blow up memory with a giant attention matrix); chunk it, and the natural chunk size is the window $W$ itself, since nothing beyond $W$ back matters anyway. For each chunk the attention splits into three parts against three masks: the chunk attends to *itself* with a causal mask (a token in the chunk sees earlier tokens in the same chunk), it attends to the *cache* of previous chunks with a sliding-window mask (only the last $W$ positions before it), and it does *not* attend to anything older than $W$ back, which is masked off entirely. So a long prompt is ingested in bounded-memory pieces, each doing exactly the attention the window allows.

Everything else is the standard decoder substrate — pre-norm residual blocks, RMSNorm, a SwiGLU feed-forward, RoPE on the queries and keys — so let me write the attention and cache, which are where all four ideas live, grounded in how the layer is actually wired. The grouped-query attention first: project to $32$ query heads but only $8$ key/value heads, apply rotary embeddings, read/write the bounded cache, repeat the $8$ key/value heads up to $32$ to match the queries, and attend through the kernel with the window mask:

```python
import torch
import torch.nn as nn
from typing import Optional


def repeat_kv(keys, values, repeats, dim):
    # The cache holds only n_kv_heads heads; repeat each one `repeats = n_heads // n_kv_heads`
    # times to serve its group of query heads. Done AFTER loading from cache, so the
    # cache stays small while the score matmul runs over all query heads.
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.n_heads            # 32 query heads
        self.n_kv_heads = args.n_kv_heads       # 8 key/value heads -> 4x smaller cache
        self.head_dim = args.head_dim
        self.repeats = self.n_heads // self.n_kv_heads     # 4 query heads per kv head
        self.sliding_window = args.sliding_window
        self.scale = self.head_dim ** -0.5
        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

    def forward(self, x, freqs_cis, cache):
        seqlen_sum, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(seqlen_sum, self.n_heads, self.head_dim)
        xk = xk.view(seqlen_sum, self.n_kv_heads, self.head_dim)
        xv = xv.view(seqlen_sum, self.n_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if cache is None:
            key, val = xk, xv
        elif cache.prefill:                         # prompt pre-fill: cache holds prev chunks
            key, val = cache.interleave_kv(xk, xv)
            cache.update(xk, xv)
        else:                                       # single-token decode
            cache.update(xk, xv)
            key, val = cache.key, cache.value
            key = key.view(seqlen_sum * cache.sliding_window, self.n_kv_heads, self.head_dim)
            val = val.view(seqlen_sum * cache.sliding_window, self.n_kv_heads, self.head_dim)

        key, val = repeat_kv(key, val, self.repeats, dim=1)    # 8 kv heads -> 32 to match queries
        xq, key, val = xq[None, ...], key[None, ...], val[None, ...]
        # the mask encodes the sliding window: each query attends to <= W recent keys
        output = memory_efficient_attention(xq, key, val, None if cache is None else cache.mask)
        return self.wo(output.view_as(x))
```

The rolling buffer cache — a fixed-size-$W$ ring per sequence per layer. Writing position $i$ goes to slot $i \bmod W$, overwriting position $i-W$ which has just left every window; reading "unrotates" the ring back into chronological order:

```python
def unrotate(cache, seqlen):
    # cache: (W, H, D) ring buffer. Recover chronological order.
    position = seqlen % cache.shape[0]
    if seqlen < cache.shape[0]:
        return cache[:seqlen]            # not wrapped yet
    elif position == 0:
        return cache                     # wrapped exactly: already in order
    else:
        return torch.cat([cache[position:], cache[:position]], dim=0)   # oldest..newest


class RotatingBufferCache:
    # Cache size fixed at the window W, not the sequence length -> bounded memory.
    def __init__(self, n_layers, max_batch_size, sliding_window, n_kv_heads, head_dim):
        self.sliding_window = sliding_window
        shape = (n_layers, max_batch_size, sliding_window, n_kv_heads, head_dim)
        self.cache_k = torch.empty(shape)
        self.cache_v = torch.empty(shape)
        self.kv_seqlens = None

    def cache_positions(self, positions):
        # position i is stored at slot (i mod W); when i >= W it overwrites i - W.
        return positions % self.sliding_window

    def to_cache_mask(self, seqlen):
        # of an ingested chunk, keep only its last W tokens -- earlier ones are out of window.
        return torch.tensor([x >= seqlen - self.sliding_window for x in range(seqlen)])
```

The block is the usual pre-norm residual, and the SwiGLU feed-forward and RMSNorm are the standard substrate:

```python
class FeedForward(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)
    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
    def forward(self, x, freqs_cis, cache):
        h = x + self.attention(self.attention_norm(x), freqs_cis, cache)
        return h + self.feed_forward(self.ffn_norm(h))
```

The causal chain, start to end: autoregressive decode is bottlenecked two ways — a key/value cache that's fat per position (set by head count) and a cache-plus-attention that grows with sequence length. The first I cut by keeping all 32 query heads but only 8 key/value heads, each shared by 4 query heads, shrinking the cache $4\times$ at almost no quality cost. The second I cut by making attention local to a window $W$ — $O(n\,W)$ instead of $O(n^2)$ — which seems to throw away long-range information until I notice that local attention stacked over $k$ layers has receptive field $kW$, so 32 layers with $W=4096$ still reach back ~131K tokens. The fixed window then makes any key/value older than $W$ unreachable, so the cache can be capped at size $W$ in a ring buffer (slot $i \bmod W$, overwriting the just-expired position $i-W$), cutting cache memory ~$8\times$ at 32k length with no quality loss; and since the prompt is known in advance it's pre-filled in parallel, chunked at size $W$, each chunk attending causally to itself and within-window to the cache. The reductions compose: a compact model that decodes fast, holds a bounded cache, and handles long sequences cheaply.
