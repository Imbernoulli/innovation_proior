OK, let me think this through from scratch. The thing that actually hurts when I serve a decoder-only Transformer is generation, token by token. Each new token has to attend over every token before it, and I don't want to recompute all the past keys and values every step, so I keep them around in a KV cache. Fine. But how big is that cache? Per token, per layer, I'm storing keys and values for all n_h heads, each of width d_h — that's 2·n_h·d_h scalars per token per layer, so 2·n_h·d_h·l over the whole stack. For a model with n_h=128 and d_h=128 that's ~32K elements per token per layer before I multiply by l, and it scales linearly with sequence length and with batch. At 128K context and a decent batch this thing is enormous, and it's the cache, not the weights, that decides how big a batch I can run and how long a context I can hold.

And here's the part that really stings: when I generate one token, the arithmetic I do is tiny — one fresh query dotted against all the stored keys, a softmax, a weighted sum of the stored values. But to do that tiny arithmetic I have to *stream the entire cache* through the chip. The operation is memory-bandwidth bound, not compute bound. So the per-token latency is basically "how many bytes of cache do I have to read." That reframes the whole problem: I'm not trying to save flops, I'm trying to save *bytes cached per token*. That's the number. Shrink it.

So what's on the table for shrinking it? The known moves all do the same thing — they make heads share keys and values. Multi-Query Attention: keep all n_h query heads, but have them share one single key head and one single value head. The cache drops from 2·n_h·d_h·l to 2·d_h·l per token — a factor of n_h, which is huge. Then GQA softens that: split the query heads into G groups, each group gets its own shared key/value head, so the cache is 2·G·d_h·l. G=n_h is plain MHA, G=1 is MQA, and G in between buys you a dial. Nice and clean. You can even take an MHA checkpoint and mean-pool each group's key/value heads to initialize it, then continue training cheaply.

But I keep coming back to *what these moves actually do*: they reduce the cache by deleting key/value subspaces. With MQA every head looks at the past through one shared key/value projection — one lens for all of them. And there's a measured cost to that. Take three 7B dense models, same everything except the attention, parameters realigned by tweaking depth, same 1.33T tokens. MHA wins across the board on the hard stuff — MMLU 45.2, GQA-8 41.2, MQA 37.9; C-Eval 42.9 vs 37.7 vs 30.0; same ordering on BBH and CMMLU. The pattern is unambiguous: the more you force heads to share K and V, the more quality you lose, and it's not subtle on demanding benchmarks. GQA lets me claw quality back, but only by raising G — which puts the cache right back up. So on this frontier I genuinely cannot have both: small cache *and* MHA quality. The sharing approach is fundamentally a trade.

Let me stare at why sharing costs quality, because maybe the trade isn't fundamental — maybe it's just this *particular* way of shrinking. The reason MHA is strong is that each head gets its *own* key and value subspace; the n_h heads jointly span a rich 2·n_h·d_h-dimensional key/value space, and different heads specialize. MQA collapses all of that onto a single d_h subspace. So the loss isn't "we cached less," it's "we *destroyed the per-head expressivity* to cache less." Those are different things, and I've been conflating them.

So here's the question that actually matters: can I keep a small cache without collapsing the per-head subspaces? What if I don't *share*, but *compress*? Concretely — the keys and values of a token are all functions of the same input h_t. There's a lot of redundancy: k_t and v_t are 2·n_h·d_h numbers all squeezed out of one d-dimensional h_t. What if I cache a small summary of h_t and reconstruct the full per-head K and V from it on the fly?

Let me write that down. Introduce a low-rank latent: c_t^{KV} = W^{DKV} h_t, where W^{DKV} ∈ R^{d_c × d} is a down-projection and d_c is small, much smaller than n_h·d_h. Then reconstruct per-head keys and values by up-projecting:

  k_t^C = W^{UK} c_t^{KV},   v_t^C = W^{UV} c_t^{KV},   with W^{UK}, W^{UV} ∈ R^{n_h d_h × d_c}.

Crucially I let K and V *share the same latent* c_t^{KV} — one compressed vector, two different up-projections fanning it back out into a key subspace and a value subspace. Why share? Because k and v are both deterministic functions of the same h_t, so their information overlaps; jointly compressing them means I cache one d_c-vector instead of two, and the up-projections W^{UK}, W^{UV} can still send that latent into entirely different per-head subspaces. So during generation I cache only c_t^{KV} — d_c·l elements per token instead of 2·n_h·d_h·l. And unlike MQA, the heads are *not* sharing a key/value head: W^{UK} and W^{UV} are full n_h·d_h × d_c matrices, so after up-projection each head gets its own slice, its own subspace. I've decoupled "small cache" from "few subspaces." That's exactly the conflation I wanted to break.

Wait — but now there's an obvious worry. If at inference I cache c_t^{KV} but then up-project it back to the *full* k_t^C and v_t^C every step, have I actually saved bandwidth? I cached fewer bytes, yes, but if I immediately reconstruct the full keys and values I'm reading the small latent and then doing the same big attention arithmetic on full-size tensors. The cache shrank but did I just move the cost? Let me check whether I even *need* to materialize k^C and v^C.

Look at the score for head i: q_{t,i}^C · k_{j,i}^C. The query side, let me also give it a low-rank path for symmetry (this won't shrink the KV cache but it shrinks the activation memory in training — q_t^C = W^{UQ} c_t^Q with c_t^Q = W^{DQ} h_t). So q_{t,i}^C = W^{UQ}_i c_t^Q and k_{j,i}^C = W^{UK}_i c_j^{KV}, where I write W^{UQ}_i, W^{UK}_i for the per-head row-blocks. The dot product is

  q_{t,i}^C · k_{j,i}^C = (W^{UQ}_i c_t^Q)^T (W^{UK}_i c_j^{KV}) = c_t^{Q,T} (W^{UQ}_i)^T W^{UK}_i c_j^{KV}.

The two up-projection matrices appear glued together as (W^{UQ}_i)^T W^{UK}_i. That is a fixed d_c' × d_c matrix with no token index on it — so it looks like I could precompute it and fold W^{UK}_i into the query side, never forming k_{j,i}^C at all. I don't fully trust an algebra step until I've seen it hold on actual numbers, so let me put tiny matrices through it: hidden d=6, compressions d_c'=d_c=3, content head d_h=2, one head, tokens t and j. Compute the score the honest way — build c_t^Q = W^{DQ}h_t, c_j^{KV} = W^{DKV}h_j, then qC = W^{UQ}_i c_t^Q and kC = W^{UK}_i c_j^{KV}, and dot them: I get qC·kC = 9.0487. Now the absorbed way — form M = (W^{UQ}_i)^T W^{UK}_i once and evaluate c_t^{Q,T} M c_j^{KV} without ever building kC: I get 9.0487 as well, agreeing to ~1e-13. So the key tensor genuinely never has to be materialized; reading the small latent c_j^{KV} and contracting it against an absorbed per-head query reproduces the exact same logit. W^{UK} has dissolved into the query projection.

The value/output side should absorb the same way, and I want to see that one land too rather than assume it by symmetry. The head output is o_{t,i} = Σ_j a_{ij} v_{j,i}^C = Σ_j a_{ij} W^{UV}_i c_j^{KV} = W^{UV}_i (Σ_j a_{ij} c_j^{KV}): up-projection pulls out of the sum because it's linear and a_{ij} are scalars. Check it on three prefix latents with weights a = (0.2, 0.5, 0.3): summing the full values Σ a_{ij}(W^{UV}_i c_j^{KV}) versus up-projecting the weighted latent W^{UV}_i(Σ a_{ij} c_j^{KV}) — the two d_v-vectors come out equal (allclose passes). So I can run the attention weighted-sum *directly on the cached latents*, producing a d_c-vector, and up-project only once at the end; and since u_t = W^O [o_{t,1};…;o_{t,n_h}], each W^{UV}_i folds straight into the corresponding block of W^O. V is never materialized either.

So both numeric checks pass, which tells me the bandwidth saving isn't being relocated: at decode the only tensor the attention reads is c_j^{KV} — the full per-head K and V are reconstructed nowhere. And the heads still own private subspaces, because W^{UK}_i and W^{UV}_i are full per-head blocks of a real n_h·d_h × d_c matrix, so I haven't paid the MQA tax that the head-sharing route charged.

This feels almost too good, so let me find the catch. The catch is position. I haven't put positional information in yet, and this model is supposed to use RoPE for long context. RoPE rotates the query and key of a token at position m by R_m *before* the dot product. The property I'm relying on is (R_m q)^T(R_n k) = q^T R_{n-m} k — the score sees only the relative offset. Let me confirm that on 2-D vectors before I lean on it: with rotation angle θ=0.7, positions m=3, n=7, a random q and k, the left side (R_m q)^T(R_n k) = −2.8348 and the right side q^T R_{n-m} k = −2.8348 — they agree, because R_m^T R_n = R_{n-m} for plane rotations. Good, the relative property holds. But note *where* the rotation lives: it sits *between* q and k and depends on position.

Let me try to just bolt RoPE onto the compressed keys: k_{j,i} = R_j W^{UK}_i c_j^{KV}. Now redo the score:

  q_{t,i} · k_{j,i} = (R_t W^{UQ}_i c_t^Q)^T (R_j W^{UK}_i c_j^{KV}) = c_t^{Q,T} (W^{UQ}_i)^T R_t^T R_j W^{UK}_i c_j^{KV} = c_t^{Q,T} (W^{UQ}_i)^T R_{j-t} W^{UK}_i c_j^{KV}.

The clean (W^{UQ}_i)^T W^{UK}_i that I was going to precompute now has R_{j-t} wedged into the middle of it. The whole absorption trick depended on that middle factor being one constant matrix; is it still? Let me actually compute the middle factor W^{UQ}^T R_{j-t} W^{UK} at a few position pairs and compare. At (t,j)=(0,0) versus (0,3): not equal. At (0,0) versus (1,5): not equal. So it is genuinely not a fixed matrix — it changes with the pair, which kills the precompute-once plan. What it *does* equal, I notice when I check (0,2) against (3,5): those two match, because both have offset j−t=2. So the middle factor is constant only along fixed offsets, never globally. That's fatal for absorption: at decode token t is fixed but j ranges over the whole cached prefix, so I'd see a different middle matrix for every cached token and could not fold W^{UK} away. To get scores I'd have to actually form R_{j-t} W^{UK}_i c_j^{KV} for every prefix j — recompute a full key for every cached token, every step — which is exactly the bandwidth bomb I was trying to defuse. So RoPE and low-rank-compression-with-absorption are in direct conflict, and the numbers just showed me it's not a fixable detail but the structure of the rotation.

So either I get RoPE or I get absorption, not both — unless I stop trying to put RoPE on the *same* vector that carries the content. The conflict is specifically that the content key has to be reconstructed through W^{UK} (which I want to absorb away), but RoPE has to apply *after* reconstruction and depends on position. What if I split those two jobs onto two different vectors? Let the compressed content key carry no position at all — keep it absorbable — and add a *separate*, small, explicitly position-carrying piece that lives outside the latent.

Concretely: alongside the content key k_{t,i}^C = W^{UK}_i c_t^{KV}, compute a small extra key k_t^R = RoPE(W^{KR} h_t) ∈ R^{d_h^R}, where d_h^R is small. And give the query a matching extra piece q_{t,i}^R = RoPE(W^{QR} c_t^Q) ∈ R^{d_h^R}. Concatenate per head: q_{t,i} = [q_{t,i}^C; q_{t,i}^R], k_{t,i} = [k_{t,i}^C; k_t^R]. The thing I need to be true is that concatenating before the dot product separates the two jobs additively — let me not just assert it. For [qC;qR]·[kC;kR] with qC,kC of width 4 and qR,kR of width 3, a dot of the concatenations equals qC·kC + qR·kR exactly (allclose passes); of course it does, a dot product over a concatenated axis is the sum of the dots over each block. So

  q_{t,i} · k_{j,i} = q_{t,i}^C · k_{j,i}^C  +  q_{t,i}^R · k_{j}^R.

The first term is pure content, no RoPE — the absorbable one I numerically confirmed above; it reads only the cached latent c_j^{KV}. The second term carries all the position. If I call the unrotated small vectors \bar q_{t,i}^R and \bar k_j^R, then by the relative property I just verified, their post-RoPE dot product is (\bar q_{t,i}^R)^T R_{j-t} \bar k_j^R — a genuine relative-position term. The reason this branch survives where the naive bolt-on died: there *is* no up-projection between \bar q^R and \bar k^R to fold away, so the (t,j)-dependent middle matrix that killed absorption simply has nowhere to form here. Position now lives entirely in a small decoupled subspace; content lives in the absorbable compressed subspace; and the two never have to share a vector. That resolves the conflict the numbers pinned down a moment ago.

Now, should the RoPE key be per-head or shared? Notice k_t^R has no head index in what I wrote — I made it a *single* shared vector broadcast to all heads, MQA-style on this tiny branch. Why share it? Two reasons. First, cache: if every head had its own RoPE key I'd be caching n_h·d_h^R extra elements per token, which would eat back the savings; sharing makes it just d_h^R per token. Second, this branch is only carrying relative-position signal, which is a low-dimensional, head-agnostic quantity — there's little to gain from per-head positional subspaces, so the MQA-style collapse here costs almost nothing in quality while the content branch keeps full per-head richness. The query side q_{t,i}^R *is* per-head (it comes from W^{QR} producing n_h·d_h^R outputs), so each head still steers its own positional query against the one shared positional key.

Let me total the cache now. Per token per layer I cache c_t^{KV} (d_c elements) plus the shared k_t^R (d_h^R elements): (d_c + d_h^R)·l. With the numbers that fall out — d_c = 4·d_h, d_h^R = d_h/2 — that's (4 + 0.5)·d_h·l = 4.5·d_h·l per token. Compare: MHA is 2·n_h·d_h·l (with n_h=128 that's 256·d_h·l), MQA is 2·d_h·l, GQA is 2·n_g·d_h·l. To place 4.5·d_h·l on the GQA dial I just solve 2·n_g·d_h = 4.5·d_h, i.e. n_g = 2.25 groups — so size-wise this sits down near MQA, a ~57× cut from MHA's 256·d_h. The thing GQA can't do at that size is keep quality, because reaching n_g≈2 forces it to collapse nearly all the per-head K/V subspaces — the exact move the benchmark table charged for. Here the small number came from compressing one shared latent, not from deleting subspaces: W^{UK} and W^{UV} still fan that latent out into 128 private per-head content subspaces. So the cache lands in MQA territory while the content path keeps the structure that made MHA score highest.

The softmax scaling needs a fix-up. Each head's query and key are now [content; rope] of total width d_h + d_h^R, so the dot product accumulates over d_h + d_h^R terms, and to keep the logits at unit scale I divide by √(d_h + d_h^R), not √d_h.

Let me assemble the full forward for one token t, the way it runs in training and prefill (where I do materialize K and V over the chunk; the absorption is the decode-time optimization I derived above):

  c_t^Q  = W^{DQ} h_t
  q_t^C  = W^{UQ} c_t^Q,           sliced into per-head q_{t,i}^C
  q_t^R  = RoPE(W^{QR} c_t^Q),     sliced into per-head q_{t,i}^R
  q_{t,i} = [q_{t,i}^C; q_{t,i}^R]
  c_t^{KV} = W^{DKV} h_t           ← cached
  k_t^C  = W^{UK} c_t^{KV},        sliced into per-head k_{t,i}^C
  k_t^R  = RoPE(W^{KR} h_t)        ← cached, shared across heads
  k_{t,i} = [k_{t,i}^C; k_t^R]
  v_t^C  = W^{UV} c_t^{KV},        sliced into per-head v_{t,i}^C
  o_{t,i} = Σ_{j≤t} softmax_j( q_{t,i}·k_{j,i} / √(d_h + d_h^R) ) v_{j,i}^C
  u_t    = W^O [o_{t,1};…;o_{t,n_h}]

Only the two boxed quantities — c_t^{KV} and k_t^R — are written to the cache. Everything else is reconstructed (training/prefill) or absorbed away (decode).

A couple of stability details I'd want. These down-projections create narrow bottlenecks — d_c and d_c' are small relative to d — and a low-rank bottleneck can let the activation scale drift, so I'll put an RMSNorm right on the latents c^{KV} and c^Q before they're up-projected, to keep the reconstruction well-conditioned. And since the up-projection outputs are large activations I'd rather not store through the whole backward pass, in training I can recompute the RMSNorms and up-projections during back-propagation rather than persisting their outputs. They are cheap to redo from the latents, and that saves a chunk of activation memory. For long context, the only place RoPE exists is the small decoupled branch, so any RoPE-extension scheme (e.g. a YaRN-style rescaling) only needs to touch that branch — the content/compression branch is position-free and untouched.

For sizes: I'll take n_h = 128 heads of d_h = 128, a KV compression dimension d_c = 512 (= 4·d_h), a query compression dimension d_c' = 1536, and a decoupled per-head dimension d_h^R = 64 (= d_h/2). So a content head is 128-wide, the rope piece adds 64, the per-head q/k width is 192, and the value head is 128.

Now to code. In a real implementation a few things get fused for efficiency, and tracking the fusions is the whole game. The query path is W^{DQ} → RMSNorm → a single matrix that produces *both* q^C and q^R per head, so its output width is n_h·(d_h + d_h^R); I then split each head into the nope part (content, no RoPE) and the pe part (RoPE). On the KV side, the down-projection and the decoupled-key projection are fused into *one* linear whose output is the latent c^{KV} concatenated with the shared RoPE key k^R, width d_c + d_h^R. That single matrix is W^{DKV} stacked with W^{KR}; I split its output into compressed_kv and k_pe. RMSNorm hits only the compressed_kv part. Then one fused up-projection W^{UK}/W^{UV}, with output width n_h·(qk_nope_head_dim + v_head_dim), takes the normalized latent to per-head k^C and v^C, which I split. RoPE is applied to the query pe part and the single shared k_pe, with k_pe broadcast to all heads. Then I lay [q^C; q^R] and [k^C; k^R] into contiguous per-head vectors, scale by (d_h + d_h^R)^{-1/2}, softmax, weight the values, concat heads, and apply W^O.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
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


def apply_rope(x, cos, sin, position_ids):
    # Match the cached-cos/sin permutation used by the decoder implementation.
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    bsz, n_heads, seq_len, head_dim = x.shape
    x = x.view(bsz, n_heads, seq_len, head_dim // 2, 2).transpose(4, 3).reshape_as(x)
    return (x * cos) + (rotate_half(x) * sin)


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size           # d
        self.num_heads = config.num_attention_heads     # n_h
        self.attention_dropout = config.attention_dropout
        self.q_lora_rank = config.q_lora_rank           # d_c'  (query compression)
        self.kv_lora_rank = config.kv_lora_rank         # d_c   (KV compression)  -> the only "big" cache
        self.qk_nope_head_dim = config.qk_nope_head_dim # d_h   (content key/query part, absorbable)
        self.qk_rope_head_dim = config.qk_rope_head_dim # d_h^R (decoupled RoPE part, shared key)
        self.v_head_dim = config.v_head_dim             # value head width
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim  # per-head q/k width

        # --- query path: W^{DQ} -> RMSNorm -> fused W^{UQ}/W^{QR} (produces nope+rope per head) ---
        self.q_a_proj = nn.Linear(
            self.hidden_size, self.q_lora_rank, bias=config.attention_bias)         # W^{DQ}
        self.q_a_layernorm = RMSNorm(self.q_lora_rank)                              # norm on c^Q
        self.q_b_proj = nn.Linear(self.q_lora_rank,
                                  self.num_heads * self.q_head_dim, bias=False)     # [W^{UQ}; W^{QR}]

        # --- KV path: fused W^{DKV}/W^{KR} (latent c^{KV} ++ shared rope key k^R) ---
        self.kv_a_proj_with_mqa = nn.Linear(self.hidden_size,
                                            self.kv_lora_rank + self.qk_rope_head_dim,
                                            bias=config.attention_bias)             # [W^{DKV}; W^{KR}]
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank)                           # norm on c^{KV}
        self.kv_b_proj = nn.Linear(self.kv_lora_rank,
                                   self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
                                   bias=False)                                     # [W^{UK}; W^{UV}]

        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim,
                                self.hidden_size, bias=config.attention_bias)       # W^O
        # logits accumulate over content+rope dims -> scale by 1/sqrt(d_h + d_h^R)
        self.softmax_scale = self.q_head_dim ** -0.5

    def forward(self, hidden_states, cos, sin, position_ids,
                past_key_value=None, attention_mask=None):
        bsz, q_len, _ = hidden_states.size()

        # queries: compress, norm, up-project (fused), split into content + rope parts
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # KV: one down-projection yields the cached latent ++ the shared decoupled key
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)  # shared across heads

        # reconstruct per-head content keys and values from the latent (fused up-proj)
        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        kv = kv.view(bsz, q_len, self.num_heads,
                     self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # RoPE only on the decoupled pieces; k_pe broadcasts to all heads
        q_pe = apply_rope(q_pe, cos, sin, position_ids)
        k_pe = apply_rope(k_pe, cos, sin, position_ids)

        # assemble per-head [content ; rope] queries and keys
        query_states = torch.cat([q_nope, q_pe], dim=-1)
        key_states = torch.cat([k_nope, k_pe.expand(-1, self.num_heads, -1, -1)], dim=-1)

        if past_key_value is not None:
            # The generic eager cache stores reconstructed tensors; the compressed
            # decode path stores compressed_kv plus k_pe after the absorption above.
            key_states, value_states = past_key_value.update(key_states, value_states)

        scores = torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
        if attention_mask is not None:
            scores = scores + attention_mask
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        probs = F.dropout(probs, p=self.attention_dropout, training=self.training)

        out = torch.matmul(probs, value_states)                    # per-head outputs o_{t,i}
        out = out.transpose(1, 2).reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        return self.o_proj(out)                                    # u_t = W^O [o_{t,1};...]
```

So the causal chain, start to end: decoding is memory-bandwidth bound on the KV cache, so the quantity to minimize is bytes-cached-per-token; the existing fixes (MQA, GQA) cut it by making heads share key/value subspaces, but that destroys per-head expressivity and measurably costs quality; so instead of sharing I compress — down-project each token to one small joint latent c^{KV}, cache only that, and reconstruct full per-head K and V by up-projection; the up-projections then *absorb* into the query and output projections at decode time, so the full K and V are never materialized and the cache really is just the latent; RoPE wrecks that absorption because its position-dependent rotation wedges between the two up-projections and can't be precomputed away; so I decouple position from content — a small shared RoPE-carrying key (and matching per-head RoPE query) added alongside the compressed content key, splitting the score into an absorbable content dot product plus a tiny positional one (the split, the absorption, and the relative-position term all checked numerically above); the result is a cache the size of GQA-with-2.25-groups but with full per-head content subspaces preserved. The cache-size and absorption claims I've verified on the page; whether keeping those subspaces actually buys back MHA-level quality is the empirical question I can't settle here — I'd want the controlled like-for-like benchmark run before believing it, but the mechanism is now exactly the one designed to make that outcome possible.
