Let me start from what actually hurts when I serve one of these decoders. I have a Transformer that generates token by token, and at every step each attention layer has to attend back over the whole history. I don't want to recompute the keys and values of every past token on every step, so I cache them — the KV cache — and reload it each step. And that cache is the thing that's killing me. Standard multi-head attention stores, per token, a key and a value for every head in every layer: `2 · n_h · d_h · l` scalars. It grows linearly in sequence length and in batch size, and once the context is long it's the cache, not the matmuls, that decides how much I can serve. It caps the batch size and the context length a given chunk of accelerator memory can hold. So I want to cache fewer bytes per token.

But I should be precise about *why* fewer bytes matters, because it's not just "memory is full." The sharp version of the argument is about bandwidth, not capacity. During training I batch over all positions at once, and if I tally the memory I touch against the arithmetic I do, the ratio is small — something like `1/d_h + 1/(b·n)` — so I'm compute-bound, fine. But incremental generation is different in kind: the output at one position feeds the input at the next, so I cannot process positions in parallel. I generate one token, do one step of attention, generate the next. Across `n` such steps the arithmetic is still `Θ(b n d^2)`, but the memory I move is dominated by reloading the K and V tensors — size `b n^2 d_h n_h`-ish — at every step. Divide the two and the ratio comes out like `Θ(n/d + 1/b)`. When the context `n` gets near the model width `d`, or the batch `b` is small, that ratio approaches one, and on hardware where compute throughput beats memory bandwidth by two orders of magnitude, a ratio near one means I'm stalled waiting on memory the whole time. So the lever is concrete: shrink the K and V that get reloaded each step. Shrink the cache, and I fix both the capacity wall and the bandwidth wall at once.

What do people already do? The obvious knob is *how many* key/value heads to keep. Multi-query attention is the extreme: keep all `n_h` query heads, but share a single key head and a single value head across all of them. In code it's literally dropping the head index from K and V. The cache drops to `2 d_h l` — a factor `n_h` smaller — which is exactly the `n/d → n/(d·n_h)` improvement the bandwidth analysis was asking for. Algebraically beautiful. But I've seen what it costs: collapsing every key and value into one shared head is a brutal cut to the attention module's capacity. One key vector and one value vector now have to serve all `n_h` query heads. Quality drops on the hard benchmarks, and at the most aggressive setting training even gets unstable. So MQA buys the bytes by spending capacity.

Grouped-query attention softens that — partition the `n_h` query heads into `n_g` groups, each group sharing one key head and one value head. `n_g = 1` is MQA again, `n_g = n_h` is full multi-head, and in between you dial the cache `2 n_g d_h l` up or down. People build the group heads by mean-pooling the original heads. It's a genuinely useful interpolation. But stare at what it actually is: the saving still comes *only* from reducing the count of materialized key/value heads. The cached object is always a pile of realized per-head keys and values; the only knob is how many. Each cached key head is one `d_h`-vector, reused verbatim by every query in its group. So the trade is monotone — fewer groups, fewer distinct keys reaching the queries, less capacity, always. The byte budget is welded to the head count times the head dimension. I cannot make the cache small while still letting every query head see something it didn't have to share.

That welding is the thing I want to break. Let me say the constraint out loud and then poke at it: every one of these methods caches the *realized* keys and values, and so the number of bytes is forced to equal (number of cached heads) × (head dim) × 2. Why should that be true? The cache is just a summary of the past token that lets me reconstruct the attention contribution. It does not have to *be* the keys and values. What if I cache something smaller — some compact per-token state — and *reconstruct* the per-head keys and values from it on the fly, inside the layer, when I actually need them?

Let me try the cheapest possible version of that. Take the input `h_t ∈ R^d` and send it down to a small vector `c_t = W^{D} h_t ∈ R^{d_c}` with `d_c ≪ n_h d_h`. Cache *that*. Then when I need head `i`'s key and value, bring them back up: `k_{t,i}^C = W^{UK}_i c_t`, `v_{t,i}^C = W^{UV}_i c_t`. The cache is now `d_c l` — completely decoupled from how many heads I have. And — this is the part that distinguishes it from grouped sharing — each head gets its *own* up-projection `W^{UK}_i, W^{UV}_i`, so even though they all read the same cached `c_t`, every head reconstructs a *different* key and value. I'm not replicating one shared head; I'm giving every head a distinct low-rank view of a shared latent. That's exactly the diversity GQA had to throw away to save bytes.

Should K and V share one latent, or get separate ones? They're two different functions of the same token, sure, but they're both summaries of *that one context token's* content. If I cache a separate `c^K` and `c^V` I've doubled the cached bytes for not-obviously-double the information. Let me cache one joint latent `c_t^{KV} = W^{DKV} h_t` and let the two up-projections `W^{UK}, W^{UV}` specialize out of it. One down-projection, two up-projections. The down-projection learns "what about this token is worth keeping," the up-projections learn "how the key needs it" and "how the value needs it." That halves the bottleneck versus separate latents, and the up-projections recover the specialization. Good — joint compression it is.

Now, is the latent expensive at inference, since I have to up-project before I can attend? Let me actually write the score and look. For query head `i` against cached token `j`, the logit is `q_{t,i}^T k_{j,i}^C`. The query I can also make low-rank if I want — set `q_{t,i}^C = W^{UQ}_i c_t^Q` from a query latent `c_t^Q = W^{DQ} h_t` — but let me first do the key side with a plain query `q_{t,i}`. The logit is

  `q_{t,i}^T k_{j,i}^C = q_{t,i}^T (W^{UK}_i c_j^{KV}) = (W^{UK}_i{}^T q_{t,i})^T c_j^{KV}`.

Look at that. The scalar identity is simply `(W^{UK}_i{}^T q_{t,i})^T c_j^{KV} = q_{t,i}^T W^{UK}_i c_j^{KV}`, and the left form is the useful one because `W^{UK}_i{}^T q_{t,i}` is something I can compute from the *current* query alone — it doesn't depend on `j`. So I fold `W^{UK}_i` into the query projection once, define a modified query `q'_{t,i} = W^{UK}_i{}^T q_{t,i}`, and then the logit against every cached token is just `q'_{t,i}{}^T c_j^{KV}` — an inner product directly against the cached latent. I never have to materialize the keys at all. The up-projection `W^{UK}` got *absorbed* into the query side. The same trick works on the value/output side: the head output is `o_{t,i} = sum_j a_{ij} v_{j,i}^C = sum_j a_{ij} W^{UV}_i c_j^{KV} = W^{UV}_i (sum_j a_{ij} c_j^{KV})`, so I can attend directly over the cached latents to get `sum_j a_{ij} c_j^{KV}`, and the `W^{UV}_i` (and the output projection `W^O`) get applied afterward — `W^{UV}_i` absorbs into `W^O`. So at generation time I never reconstruct K or V; I attend over the small cached latent as if it were a single shared key/value head. The latent costs nothing extra to use — it behaves like one big MQA-style shared head, except the per-head distinctness lives in the absorbed up-projections rather than being lost. That's the whole point clicking into place: I get MQA-style cache traffic, but I keep per-head key/value diversity that MQA destroyed.

While I'm here — the query latent `c^Q`. It doesn't shrink the KV cache (queries aren't cached; they're recomputed for the current token). So why bother? Because during training I materialize queries for every position at once, and a full `W^Q ∈ R^{n_h d_h × d}` is a big activation. Routing queries through a low-rank `W^{DQ}` then `W^{UQ}` cuts the activation memory of the query path during training. It's a pure training-memory optimization, orthogonal to the cache story, and it costs nothing at inference. I'll keep it.

So the picture is: cache a small joint latent, reconstruct distinct per-head K/V from it, and the up-projections vanish into the query and output projections at inference so the latent is free to use. Cache `d_c l`, preserve per-head projections, and move the generation traffic toward the single-shared-head regime. Let me start wiring it up.

And immediately I hit a wall, because I forgot about position. These decoders use rotary embeddings, and rotary embeddings are not additive — they *rotate* the query and key by an angle set by the token's absolute position. Concretely `q_t → R_t W_q x_t` and `k_j → R_j W_k x_j`, with a block-diagonal rotation matrix whose 2×2 blocks rotate by the position times the RoPE frequency. The reason everyone uses RoPE is the relative-position identity: `R_t^T R_j = R_{j-t}` because rotations compose and are orthogonal, so the score becomes `q_t^T k_j = x_t^T W_q^T R_{j-t} W_k x_j` — it depends only on `j − t`. Great property, and exactly the thing that breaks me.

Watch what happens to the absorb trick if I rotate the reconstructed key. The cached-token key is `k_{j,i}^C = W^{UK}_i c_j^{KV}`, and RoPE turns it into `R_j W^{UK}_i c_j^{KV}`. The query is `R_t W^{UQ}_i c_t^Q`. The logit is

  `(R_t W^{UQ}_i c_t^Q)^T (R_j W^{UK}_i c_j^{KV}) = c_t^Q{}^T W^{UQ}_i{}^T R_t^T R_j W^{UK}_i c_j^{KV} = c_t^Q{}^T W^{UQ}_i{}^T R_{j-t} W^{UK}_i c_j^{KV}`.

There's the wall. The relative rotation `R_{j-t}` now sits *between* `W^{UQ}_i{}^T` and `W^{UK}_i`. And it depends on `j` — on which cached token I'm scoring against — and on `t`, the current step, so it changes every single decoding step. I can no longer precompute `W^{UQ}_i{}^T W^{UK}_i` into one absorbed matrix, because a position-dependent matrix has wedged itself in the middle and matrix multiplication doesn't commute. The only way to honor RoPE on the reconstructed keys is to actually reconstruct every key, rotate each by its own `R_j`, and score — which means recomputing all the prefix keys at every step, which is exactly the work I cached the latent to avoid. The latent saving and RoPE are, naively, mutually exclusive.

Let me not panic and instead ask what RoPE actually needs. It needs *some* channels of the query and key to carry the rotation so the score can depend on relative position. It does not need *all* of them to. What if position lives in a small, separate slice that is allowed to be position-dependent, while the bulk of the key — the part I want to absorb — stays position-free?

So decouple it. Alongside the compressed, position-free content keys, add a small extra key slice that carries RoPE and *only* RoPE. Concretely: produce a few extra query dimensions per head, `q_{t,i}^R = RoPE(W^{QR} c_t^Q)_i` of dimension `d^R_h`, and an extra key slice `k_t^R = RoPE(W^{KR} h_t)` also of dimension `d^R_h`. Concatenate them onto the content parts: the full query for head `i` is `q_{t,i} = [q_{t,i}^C ; q_{t,i}^R]` and the full key is `k_{t,i} = [k_{t,i}^C ; k_t^R]`. Now the score splits cleanly:

  `q_{t,i}^T k_{j,i} = q_{t,i}^C{}^T k_{j,i}^C + q_{t,i}^R{}^T k_j^R`.

The first term is the position-free content score — `W^{UK}` still absorbs into the query there, no rotation in the middle, latent still free. The second term carries all the relative-position information through `q^R` and `k^R`, which *do* get rotated, but they're a tiny `d^R_h`-dimensional slice I compute explicitly. RoPE is satisfied (the score depends on `j − t` through the second term), and absorption survives (untouched on the first term). The two requirements that looked mutually exclusive now coexist because I gave position its own channel.

One more decision in there I should justify rather than wave at: I made the RoPE key `k_t^R` a *single shared* slice across all heads, not one per head. Why? Because it has to be cached — it carries position, it can't be reconstructed from the position-free latent — so every dimension of it adds to the cache I'm fighting to shrink. A per-head RoPE key would be `n_h · d^R_h` extra cached scalars; a shared one is just `d^R_h`. And position is genuinely a property of the *token*, not of the head — all heads want to know "how far back is this token" — so sharing the positional key across heads costs almost nothing in capacity while keeping the cache tiny. The decoupled query `q^R`, by contrast, is per-head and not cached, so I let it be per-head for free expressivity. So the total cache becomes `(d_c + d^R_h) l` — the content latent plus the one shared positional slice. Still decoupled from head count, still tiny.

Let me sanity-check the cache arithmetic against the alternatives to make sure I actually won. Per token, per layer: MHA caches `2 n_h d_h`, MQA caches `2 d_h`, GQA caches `2 n_g d_h`, and mine caches `d_c + d^R_h`. With the kind of numbers a real model uses — say the content latent at `d_c = 4 d_h` and the positional slice at `d^R_h = d_h/2` — mine is `4.5 d_h` per token per layer. Compare to GQA: `4.5 d_h` is `2 n_g d_h` at `n_g = 2.25`. So my cache is the size of GQA with barely more than two groups — near the MQA end of the spectrum — and yet I haven't collapsed any heads, because each head still reconstructs its own key and value from the latent. That's the trade I couldn't get before: MQA-scale bytes with multi-head-scale capacity.

I want to be honest about *why* I believe the capacity is really there and I'm not fooling myself with a low-rank bottleneck. Take GQA at some cache budget and write its shared key head as if it were an up-projection from a cached vector. The shared head is one `d_h`-vector that gets handed, unchanged, to every query head in the group — so as a map from "cached state" to "the `n_h` per-head keys," GQA's effective up-projection is forced to be a *replication*: rank-deficient, the same vector copied out. My up-projection `W^{UK}` from the same-sized latent is unconstrained — full low-rank, every head's key a genuinely different linear image of the latent. At equal cached bytes, the family of attention patterns GQA can express is a strict subset of what I can express, because "replicate one head" is a special case of "give each head its own projection" (just choose `W^{UK}` to be the replication map). MQA is the one-replica corner, GQA is the grouped-replica corner, and full multi-head attention appears if I let the cached state grow all the way to the realized per-head keys and values. So I'm not gambling that low rank is enough; I'm choosing a *richer* parameterization of the cache than counting heads, one that contains the head-counting methods as corners while separating the cache budget from the number of query heads.

There's a softer hypothesis floating around too — that the low-rank bottleneck itself acts as a mild regularizer and could *help* generalization, the way any information bottleneck on the context might. I can't prove that; the lossy compression could just as easily hurt. I'll hold it as a maybe and lean on the expressivity argument, which I can actually justify, as the real reason this works.

Now a numerical-stability point I'd regret skipping. I've stuck a narrow low-rank bottleneck `c^{KV}` (and `c^Q`) into the middle of the layer. Bottlenecks like that change the scale of the activations flowing through them — the variance of the latent isn't going to match what the up-projection expects, and across 24 layers small scale mismatches compound into training blow-ups. The fix is the standard one: normalize the latent before up-projecting. RMSNorm — `x · rsqrt(mean(x^2) + ε)` times a learned gain, no mean subtraction, cheap — right after `c^{KV}` and right after `c^Q`. It pins the scale entering the up-projections so the bottleneck doesn't destabilize the stack. Cheap insurance, and it's the kind of thing that's invisible in the equations but decides whether the thing trains at all.

And the softmax temperature. My key is now the concatenation `[k^C ; k^R]` of dimension `d_h + d^R_h`, not `d_h`. The standard `1/sqrt(d_h)` scaling was calibrated to the variance of a `d_h`-dimensional dot product; since the score is a sum over `d_h + d^R_h` channels now, I scale by `1/sqrt(d_h + d^R_h)` so the logits keep their intended magnitude and the softmax doesn't saturate. Small thing, easy to get wrong.

OK. I think I have the whole structure. Let me now actually land it in the nanoGPT training substrate, because that's where I have to make it concrete, and the substrate has its own small constraints that force a few choices. The block is the usual `pre-LN attention + MLP` residual stack; the only thing I'm redesigning is the attention module. The head dimension here is `head_dim = n_embd // n_head`, which on the small config is modest, so I can't use the big absolute dimensions a large production instance would; I have to scale the ranks to the tiny hidden size while keeping the *relative* schedule that makes the method work.

Let me fix the dimensions. The content key per head should keep the original head dimension — call it `qk_nope_head_dim = head_dim` — that's the position-free part that gets compressed and reconstructed. On top of it goes the rotary slice `qk_rope_head_dim`. I want it even (RoPE rotates pairs of dimensions, so an odd count is meaningless), and I want it a real fraction of the head but not so large it dominates a tiny head: clamp it to `[16, 64]`, take `min(64, head_dim)`, force even. The full per-head query/key dimension is then `qk_head_dim = qk_nope_head_dim + qk_rope_head_dim`, and the value keeps `v_head_dim = head_dim`. For the ranks: the KV latent should stay below the realized per-head K+V payload on the target small config, so set `kv_lora_rank = max(16, head_dim // 2)`: the `//2` gives a real compression at the intended head size, and the `max(16, ·)` keeps tiny heads from collapsing to an unusable rank. The query latent can be generous (it's not cached) — mirror the real model's roughly-twelve-times-head-dim query-latent schedule, `q_lora_rank = min(n_embd, 12 * head_dim)`, capped by the hidden size.

Now the projections, straight from the structure. Query path: `q_a_proj` takes `n_embd → q_lora_rank`, then RMSNorm on that latent, then `q_b_proj` takes `q_lora_rank → n_head · qk_head_dim` (producing both the nope and rope query parts for every head). KV path: a single `kv_a_proj_with_mqa` takes `n_embd → kv_lora_rank + qk_rope_head_dim` — that one matrix produces *both* the content latent `c^{KV}` (first `kv_lora_rank` dims) *and* the shared rotary key `k^R` (last `qk_rope_head_dim` dims), which is why it's named "with mqa": the rotary key is the one MQA-style shared head riding alongside the latent. Then RMSNorm on the `kv_lora_rank` latent slice only, then `kv_b_proj` takes `kv_lora_rank → n_head · (qk_nope_head_dim + v_head_dim)`, up-projecting the latent into every head's content key and its value at once. Finally `o_proj` maps `n_head · v_head_dim → n_embd`.

In the forward pass: project and split the query into its nope and rope parts per head. Project the input once through `kv_a_proj_with_mqa`, split off the content latent and the rope key. RMSNorm the latent, up-project through `kv_b_proj`, split into content keys `k_nope` and values per head. The rope key is a single head — view it as `[bsz, 1, seq_len, qk_rope_head_dim]` so it broadcasts across all query heads. Build the rotary cos/sin cache for `qk_rope_head_dim` with base `10000`, apply RoPE to `q_rot` and `k_rot`. Then assemble the full query `[q_nope ; q_rot]` and full key `[k_nope ; k_rot]` per head, with `k_rot` broadcast across heads (it's shared). Run causal scaled-dot-product attention with scale `qk_head_dim^{-0.5}`, transpose back, and `o_proj`.

A small RoPE implementation note, because I want to get the convention right and not pay for a permutation I don't need. I build the cache with the half-split convention — `cat((freqs, freqs), -1)` — and pair it with `rotate_half(x) = cat((-x2, x1), -1)` and the `q·cos + rotate_half(q)·sin` formula. That's already a correct, self-consistent RoPE. There's a tempting `view → transpose → reshape` re-interleave step you see in code that loads pretrained weights in an interleaved layout — but that exists *only* to match that stored layout. Training from scratch, there are no such weights to match, so the permutation buys nothing and just materializes an extra copy of every Q and K each forward. Drop it; use the half-split form directly.

And one efficiency choice for assembling the concatenated query/key. The natural way is `cat([k_nope, k_rot.expand(...)], -1)`, but `expand` to all heads then `cat` materializes an expanded contiguous tensor plus a fresh concat buffer — wasteful for the broadcast rope key. Instead allocate `query_states`/`key_states` with `new_empty` and slice-assign the nope part and the rope part into their column ranges. Slice `__setitem__` is autograd-safe — the backward just scatters gradients into `q_nope`, `q_rot`, `k_nope`, `k_rot`, broadcasting along the head axis for the shared `k_rot` — and it avoids the expand-and-cat intermediates.

I can now fill the empty attention slot with this:

```python
import torch
import torch.nn as nn
from torch.nn import functional as F


def build_kv_heads(config):
    # The latent behaves like a single shared KV head; per-head distinctness
    # comes from the up-projections inside the attention block, not the count.
    head_dim = config.n_embd // config.n_head
    return 1, head_dim


def cross_layer_share(layer_idx, config):
    return False


def latent_kv_project(k, v, config):
    # The real compression lives inside CausalSelfAttention; this hook is a no-op.
    return k, v, 1.0


class MLARMSNorm(nn.Module):
    """RMSNorm on the low-rank latent: pins its scale before the up-projection
    so the bottleneck does not destabilize the deep stack."""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def build_rotary_cache(seq_len, dim, device, dtype, theta=10000.0):
    # theta_i = theta^{-2i/dim}: the standard RoPE frequency schedule.
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)          # half-split convention
    cos = emb.cos().to(dtype).view(1, 1, seq_len, dim)
    sin = emb.sin().to(dtype).view(1, 1, seq_len, dim)
    return cos, sin


def apply_rotary_pos_emb_interleave(q, k, cos, sin):
    # Half-split cache + rotate_half is already correct; no re-interleave needed
    # when training from scratch (that permutation only matched pretrained layouts).
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.layer_idx = layer_idx
        self.n_kv_head, self.head_dim = build_kv_heads(config)
        self.share_across_layers = False

        # RoPE rides on a small even slice on top of the content head dim,
        # not by splitting the head dim in half.
        self.qk_rope_head_dim = min(64, self.head_dim)
        self.qk_rope_head_dim = max(16, self.qk_rope_head_dim)
        if self.qk_rope_head_dim % 2 != 0:
            self.qk_rope_head_dim -= 1
        self.qk_nope_head_dim = self.head_dim                       # position-free content key dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = self.head_dim
        # Keep the relative rank schedule (query latent ~ 12x head dim), capped by the tiny hidden size.
        self.q_lora_rank = min(self.n_embd, 12 * self.head_dim)     # query latent (not cached)
        self.kv_lora_rank = max(16, self.head_dim // 2)             # KV latent: < head_dim so cache shrinks

        # Query path: down -> RMSNorm -> up (produces nope + rope query parts).
        self.q_a_proj = nn.Linear(config.n_embd, self.q_lora_rank, bias=False)
        self.q_a_layernorm = MLARMSNorm(self.q_lora_rank)
        self.q_b_proj = nn.Linear(
            self.q_lora_rank, self.n_head * self.qk_head_dim, bias=config.bias
        )

        # One matrix makes both the cached content latent c^{KV} and the shared
        # rotary key k^R (the "with_mqa" tail): only these two get cached.
        self.kv_a_proj_with_mqa = nn.Linear(
            config.n_embd, self.kv_lora_rank + self.qk_rope_head_dim, bias=config.bias
        )
        self.kv_a_layernorm = MLARMSNorm(self.kv_lora_rank)
        # Up-project the latent into every head's content key and value.
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.n_head * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(self.n_head * self.v_head_dim, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )
        self.use_pos_emb = False                                    # position is internal (RoPE), not added embeddings
        self.head_sharing_ratio = float(self.n_head)
        self.scaling = self.qk_head_dim ** -0.5                     # 1/sqrt(d_h + d^R_h): score sums both channels

    def forward(self, x):
        bsz, seq_len, _ = x.size()

        # Query: down-project, normalize, up-project, split into nope/rope parts.
        q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q_states = q_states.view(bsz, seq_len, self.n_head, self.qk_head_dim).transpose(1, 2)
        q_nope, q_rot = torch.split(
            q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # KV: one projection yields the cached content latent and the shared rope key.
        compressed_kv = self.kv_a_proj_with_mqa(x)
        kv_latent, k_rot = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        # Up-project the normalized latent into per-head content keys and values.
        kv_states = self.kv_b_proj(self.kv_a_layernorm(kv_latent))
        kv_states = kv_states.view(
            bsz, seq_len, self.n_head, self.qk_nope_head_dim + self.v_head_dim
        ).transpose(1, 2)
        k_nope, value_states = torch.split(
            kv_states, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        # The rope key is a single shared head; apply RoPE to it and to q_rot.
        k_rot = k_rot.view(bsz, seq_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        cos, sin = build_rotary_cache(
            seq_len, self.qk_rope_head_dim, x.device, q_rot.dtype
        )
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)

        # Assemble [content ; rope] for query and key via new_empty + slice-assign
        # (avoids expand-then-cat materialization; slice setitem is autograd-safe,
        #  k_rot broadcasts over the head axis since it is shared).
        query_states = q_states.new_empty(bsz, self.n_head, seq_len, self.qk_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_rot

        key_states = q_states.new_empty(bsz, self.n_head, seq_len, self.qk_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_rot       # broadcasts over n_head

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
                scale=self.scaling,
            )
        else:
            att = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scaling
            att = att.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = torch.matmul(att, value_states)

        # Report the realized cache geometry: cached bytes ∝ kv_lora_rank + qk_rope_head_dim.
        latent_ratio = self.kv_lora_rank / self.qk_head_dim
        storage_ratio = (self.kv_lora_rank + self.qk_rope_head_dim) / (2 * self.head_dim)
        self._last_latent_rank_ratio = float(latent_ratio)
        self._last_kv_storage_ratio = float(storage_ratio)
        self._uses_latent_compression = True

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_head * self.v_head_dim)
        y = self.resid_dropout(self.o_proj(y))
        return y
```

Let me trace the causal chain one more time so I'm sure nothing's load-bearing-but-unjustified. I started cache-bound: decoding is memory-bandwidth-limited because every step reloads the full K and V, and the cache is welded to (head count × head dim). MQA cuts the count to one and pays in capacity; GQA interpolates but the saving is still only a head-count reduction, so the byte-quality trade stays monotone. The move that breaks the welding is to cache not the realized K and V but a small joint latent and reconstruct distinct per-head keys and values from it — distinct because each head gets its own up-projection, which is what keeps capacity that MQA threw away; and the up-projections absorb into the query and output projections at inference, so the latent is as cheap to attend over as a single shared head. RoPE wrecks that absorption because its position-dependent rotation lands in the middle of the query-key product, so I split position off into a small decoupled slice — a per-head rotary query and a single shared rotary key — leaving the content part position-free and absorbable; the score is the sum of the content term and the positional term. The KV latent must be smaller than the realized K+V payload to shrink the cache, RMSNorm on the latent keeps the bottleneck from destabilizing training, and the softmax scale becomes `1/sqrt(d_h + d^R_h)` because the key is now the concatenation of content and rotary channels. Against GQA at equal cached bytes the structure is more expressive — GQA's shared head is the rank-deficient "replicate one vector" special case of a per-head up-projection — which is the principled reason to separate cache size from head diversity. And it all drops into the nanoGPT block as a redesigned attention module: down-project to latents, normalize, up-project to per-head K/V, carry RoPE on a tiny shared slice, attend, project out — caching only the latent and the shared rotary key, a cache decoupled at last from how many heads the model uses.
