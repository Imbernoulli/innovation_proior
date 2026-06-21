When I serve a decoder-only Transformer, the cost that ends up dominating is not the arithmetic but the KV cache. The model generates one token at a time, and at every step each attention layer has to attend back over the whole history; to avoid recomputing the keys and values of every past token on every step I cache them and reload them each step. Standard multi-head attention stores, per token, a key and a value for every head in every layer — $2\,n_h\,d_h\,l$ scalars — and that grows linearly in sequence length and batch size. Once the context is long it is the cache, not the matmuls, that caps how much I can serve. The sharp version of why this matters is about bandwidth, not just capacity: during training I batch over all positions and am comfortably compute-bound, but incremental generation cannot parallelize across positions, so across $n$ steps the arithmetic stays $\Theta(b\,n\,d^2)$ while the memory I move is dominated by reloading the K and V tensors at every step, giving a memory-to-compute ratio of roughly $\Theta(n/d + 1/b)$. When the context $n$ approaches the model width $d$, or the batch $b$ is small, that ratio approaches one, and on hardware where compute throughput beats memory bandwidth by two orders of magnitude that means I am stalled waiting on memory the entire time. The lever is therefore concrete: shrink the K and V that get reloaded each step.

The existing ways to do that all turn one knob — *how many* key/value heads to keep. Multi-query attention keeps all $n_h$ query heads but shares a single key head and a single value head across every one of them, dropping the cache to $2\,d_h\,l$, a factor $n_h$ smaller; algebraically it is exactly the reduction the bandwidth analysis asks for, but collapsing every key and value into one shared head is a brutal cut to capacity — one key and one value vector now serve all $n_h$ query heads, quality drops on hard benchmarks, and at the most aggressive setting training goes unstable. Grouped-query attention softens this by partitioning the $n_h$ query heads into $n_g$ groups that each share one key and value head, with cache $2\,n_g\,d_h\,l$, recovering MQA at $n_g=1$ and full multi-head at $n_g=n_h$. It is a useful interpolation, but the saving still comes *only* from reducing the count of materialized heads: the cached object is always a pile of realized per-head keys and values, each one a $d_h$-vector reused verbatim by every query in its group, so the trade between cached bytes and per-head diversity is monotone. The byte budget is welded to (head count) $\times$ (head dim) $\times 2$, and I cannot make the cache small while still letting every query head see something it did not have to share. That welding is the thing to break.

I propose Multi-head Latent Attention (MLA). The observation underneath it is that the cache does not have to *be* the keys and values — it only has to be a summary of the past token from which the attention contribution can be reconstructed. So I cache a small joint low-rank latent per token, $c_t^{KV} = W^{DKV} h_t \in \mathbb{R}^{d_c}$ with $d_c \ll n_h d_h$, and reconstruct each head's key and value from it on the fly, $k_{t,i}^C = W^{UK}_i c_t^{KV}$ and $v_{t,i}^C = W^{UV}_i c_t^{KV}$. The cache is now $d_c\,l$, decoupled from how many heads I have. The decisive difference from grouped sharing is that *each head gets its own up-projection* $W^{UK}_i, W^{UV}_i$: all heads read the same cached latent, but every head reconstructs a *different* key and value, so I am not replicating one shared head, I am giving every head a distinct low-rank view of a shared latent — precisely the per-head diversity GQA had to discard. I cache one joint latent rather than separate $c^K$ and $c^V$ because both are summaries of the same context token; caching two would double the cached bytes for not-obviously-double the information, and letting the two up-projections specialize out of a single down-projection halves the bottleneck while recovering the specialization — the down-projection learns what about the token is worth keeping, the up-projections learn how the key and value each need it.

What makes the latent free to use at inference is that the up-projections absorb away. For query head $i$ against cached token $j$ the content logit is $q_{t,i}^T (W^{UK}_i c_j^{KV}) = (W^{UK}_i{}^T q_{t,i})^T c_j^{KV}$, and the right-hand form is the useful one because $W^{UK}_i{}^T q_{t,i}$ depends only on the *current* query, not on $j$. So I fold $W^{UK}_i$ into the query projection once, define a modified query, and the logit against every cached token becomes a plain inner product against the cached latent — I never materialize the keys at all. With the query itself routed through a low-rank path ($c_t^Q = W^{DQ} h_t$, $q^C = W^{UQ} c^Q$) this reads $(W^{UQ}_i c_t^Q)^T (W^{UK}_i c_j^{KV}) = c_t^Q{}^T (W^{UQ}_i{}^T W^{UK}_i)\, c_j^{KV}$, so $W^{UK}$ collapses into the query projection as one precomputed matrix. The same trick works on the output side: the head output $o_{t,i} = \sum_j a_{ij} W^{UV}_i c_j^{KV} = W^{UV}_i \sum_j a_{ij} c_j^{KV}$ lets me attend directly over the latents and apply $W^{UV}_i$ (absorbed into $W^O$) afterward. So at generation time MLA attends over the small cached latent as if it were a single shared key/value head — MQA-style cache traffic — while the per-head distinctness lives in the absorbed up-projections rather than being lost. The query low-rank path, incidentally, does not shrink the KV cache (queries are not cached), but routing queries through $W^{DQ}$ then $W^{UQ}$ cuts the activation memory of the query path during training, a pure training-memory optimization that costs nothing at inference, so I keep it.

The wall is position. These decoders use RoPE, which is not additive — it rotates each query and key by an angle set by absolute position, $q_t \to R_t W_q x_t$, $k_j \to R_j W_k x_j$, with the relative-position identity $R_t^T R_j = R_{j-t}$ that makes the score depend only on $j-t$. If I rotate the reconstructed content key, the content logit becomes $c_t^Q{}^T W^{UQ}_i{}^T R_t^T R_j W^{UK}_i c_j^{KV} = c_t^Q{}^T W^{UQ}_i{}^T R_{j-t} W^{UK}_i c_j^{KV}$, and the position-dependent $R_{j-t}$ now sits *between* $W^{UQ}_i{}^T$ and $W^{UK}_i$. Because it depends on $j$ and $t$ and changes every decoding step, I can no longer precompute $W^{UQ}_i{}^T W^{UK}_i$ into one absorbed matrix; honoring RoPE naively would force me to rebuild and rotate every prefix key at every step, which is exactly the work the latent was meant to avoid. The resolution is that RoPE does not need *all* channels to carry the rotation, only some. So I decouple position: alongside the position-free content keys I add a small extra slice that carries RoPE and only RoPE — a per-head rotary query $q_{t,i}^R = \mathrm{RoPE}(W^{QR} c_t^Q)_i$ and a single shared rotary key $k_t^R = \mathrm{RoPE}(W^{KR} h_t)$, each of dimension $d^R_h$ — and concatenate them onto the content parts so the score splits cleanly,

$$q_{t,i}^T k_{j,i} = (q_{t,i}^C)^T k_{j,i}^C + (q_{t,i}^R)^T k_j^R.$$

The first term is the position-free content score where $W^{UK}$ still absorbs into the query with no rotation in the middle; the second term carries all the relative-position information through the tiny rotated slice I compute explicitly. The two requirements that looked mutually exclusive now coexist because position has its own channel. I make the rotary key a *single shared* slice across all heads rather than one per head for a deliberate reason: it must be cached (it carries position and cannot be reconstructed from the position-free latent), so every dimension of it adds to the cache I am fighting to shrink — a per-head rotary key would cost $n_h \cdot d^R_h$ extra cached scalars while a shared one costs $d^R_h$ — and position is genuinely a property of the token, not the head. The decoupled query, not cached, stays per-head for free expressivity. The total cache is therefore $(d_c + d^R_h)\,l$, still decoupled from head count.

Checking the arithmetic against the alternatives: per token per layer, MHA caches $2\,n_h d_h$, MQA caches $2\,d_h$, GQA caches $2\,n_g d_h$, and MLA caches $d_c + d^R_h$. With realistic ratios $d_c = 4 d_h$ and $d^R_h = d_h/2$, MLA caches $4.5\,d_h$ per token per layer, which equals GQA at $n_g = 2.25$ groups — near the MQA end of the spectrum — yet no heads have been collapsed, because each head still reconstructs its own key and value. The reason I trust the capacity is really there, and am not just hiding behind a bottleneck, is an expressivity argument. Written as an up-projection from a cached vector, GQA's shared key head is a *replication map*: one $d_h$-vector handed unchanged to every query head in the group, rank-deficient. MLA's $W^{UK}, W^{UV}$ from the same-sized latent are unconstrained low-rank up-projections, every head a genuinely different linear image of the latent. At equal cached bytes, MQA and GQA are constrained replication special cases of MLA's parameterization (choose $W^{UK}$ to be the replication map), and full MHA reappears only when the cached state grows to the realized per-head K/V. I am choosing a richer parameterization of the cache than counting heads — one that contains the head-counting methods as corners while separating cache budget from query-head count. There is a softer hypothesis that the low-rank bottleneck might also act as a mild regularizer, but I cannot prove that and lean instead on the expressivity argument as the real reason it works.

Two numerical details are load-bearing even though they are invisible in the score equations. First, a narrow low-rank bottleneck sits in the middle of every layer, and bottlenecks shift the scale of the activations flowing through them; across 24 layers small scale mismatches compound into training blow-ups, so I apply RMSNorm — $x \cdot \mathrm{rsqrt}(\mathrm{mean}(x^2)+\epsilon)$ times a learned gain, no mean subtraction — to $c^{KV}$ and to $c^Q$ right before up-projecting, pinning the scale entering the up-projections. Second, the key is now the concatenation $[k^C; k^R]$ of dimension $d_h + d^R_h$, not $d_h$, so the standard $1/\sqrt{d_h}$ softmax temperature, calibrated to a $d_h$-dimensional dot product, would be wrong; I scale by $1/\sqrt{d_h + d^R_h}$ so the logits keep their intended magnitude and the softmax does not saturate.

Landing this in the nanoGPT substrate forces the ranks to scale to the small hidden size while keeping the relative schedule that makes the method work. The content key keeps the original head dimension ($\texttt{qk\_nope\_head\_dim} = \texttt{head\_dim}$); the rotary slice is clamped even and into $[16,64]$ via $\min(64, \texttt{head\_dim})$ so it is a real fraction of the head without dominating a tiny one; the KV latent is $\texttt{kv\_lora\_rank} = \max(16, \texttt{head\_dim}//2)$ so it stays below the realized per-head K+V payload and the cache actually shrinks; the query latent is generous since it is not cached, $\min(\texttt{n\_embd}, 12\cdot\texttt{head\_dim})$, mirroring the roughly twelve-times-head-dim query-latent schedule of a full-scale instance (where $n_h = d_h = 128$, $d_c = 512$, query latent $1536$, $d^R_h = 64$, RoPE base $\theta = 10000$). A single $\texttt{kv\_a\_proj\_with\_mqa}$ produces both the cached content latent and the shared rotary key — that is what its name records, the rotary key being the one MQA-style shared head riding alongside the latent. For RoPE I use the half-split cache ($\texttt{cat((freqs, freqs))}$) paired with $\texttt{rotate\_half}$ and the $q\cos + \texttt{rotate\_half}(q)\sin$ form, which is already correct from scratch; the $\texttt{view}\to\texttt{transpose}\to\texttt{reshape}$ re-interleave one sees elsewhere exists only to match a stored pretrained layout, so training from scratch it buys nothing and just copies every Q and K, and I drop it. Finally I assemble the concatenated query and key by allocating with $\texttt{new\_empty}$ and slice-assigning the content and rotary columns rather than $\texttt{expand}$-then-$\texttt{cat}$, since slice $\texttt{\_\_setitem\_\_}$ is autograd-safe — the backward scatters gradients into the parts and broadcasts along the head axis for the shared rotary key — and it avoids the expand-and-cat intermediates.

```python
import torch
import torch.nn as nn
from torch.nn import functional as F


def build_kv_heads(config):
    head_dim = config.n_embd // config.n_head
    return 1, head_dim                       # latent = one shared KV head


def cross_layer_share(layer_idx, config):
    return False


def latent_kv_project(k, v, config):
    return k, v, 1.0                         # compression lives inside the block


class MLARMSNorm(nn.Module):
    """RMSNorm on the latent: stabilizes the low-rank bottleneck."""
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
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype).view(1, 1, seq_len, dim)
    sin = emb.sin().to(dtype).view(1, 1, seq_len, dim)
    return cos, sin


def apply_rotary_pos_emb_interleave(q, k, cos, sin):
    # half-split cache + rotate_half is the correct from-scratch form
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

        self.qk_rope_head_dim = min(64, self.head_dim)
        self.qk_rope_head_dim = max(16, self.qk_rope_head_dim)
        if self.qk_rope_head_dim % 2 != 0:
            self.qk_rope_head_dim -= 1
        self.qk_nope_head_dim = self.head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = self.head_dim
        self.q_lora_rank = min(self.n_embd, 12 * self.head_dim)
        self.kv_lora_rank = max(16, self.head_dim // 2)

        self.q_a_proj = nn.Linear(config.n_embd, self.q_lora_rank, bias=False)
        self.q_a_layernorm = MLARMSNorm(self.q_lora_rank)
        self.q_b_proj = nn.Linear(
            self.q_lora_rank, self.n_head * self.qk_head_dim, bias=config.bias
        )

        self.kv_a_proj_with_mqa = nn.Linear(
            config.n_embd, self.kv_lora_rank + self.qk_rope_head_dim, bias=config.bias
        )
        self.kv_a_layernorm = MLARMSNorm(self.kv_lora_rank)
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
        self.use_pos_emb = False
        self.head_sharing_ratio = float(self.n_head)
        self.scaling = self.qk_head_dim ** -0.5

    def forward(self, x):
        bsz, seq_len, _ = x.size()

        q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q_states = q_states.view(bsz, seq_len, self.n_head, self.qk_head_dim).transpose(1, 2)
        q_nope, q_rot = torch.split(
            q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        compressed_kv = self.kv_a_proj_with_mqa(x)
        kv_latent, k_rot = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        kv_states = self.kv_b_proj(self.kv_a_layernorm(kv_latent))
        kv_states = kv_states.view(
            bsz, seq_len, self.n_head, self.qk_nope_head_dim + self.v_head_dim
        ).transpose(1, 2)
        k_nope, value_states = torch.split(
            kv_states, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        k_rot = k_rot.view(bsz, seq_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        cos, sin = build_rotary_cache(seq_len, self.qk_rope_head_dim, x.device, q_rot.dtype)
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)

        # new_empty + slice-assign: avoids expand/cat of the shared rope key;
        # slice setitem is autograd-safe (k_rot broadcasts over the head axis).
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

        latent_ratio = self.kv_lora_rank / self.qk_head_dim
        storage_ratio = (self.kv_lora_rank + self.qk_rope_head_dim) / (2 * self.head_dim)
        self._last_latent_rank_ratio = float(latent_ratio)
        self._last_kv_storage_ratio = float(storage_ratio)
        self._uses_latent_compression = True

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_head * self.v_head_dim)
        y = self.resid_dropout(self.o_proj(y))
        return y
```
