When a decoder-only Transformer generates text one token at a time, the cost that actually hurts at serving time is not the parameter count but the KV cache. To produce token $t$ each attention layer must attend over the keys and values of all preceding tokens, and to avoid recomputing them every step I keep them around. Multi-head attention stores $2\,n_h\,d_h\,l$ scalars per token — keys and values, across all $n_h$ heads of width $d_h$, in every one of the $l$ layers. At long context and large batch this grows until it no longer fits, and worse, it makes decoding memory-bandwidth bound: each generation step does a tiny amount of arithmetic — one fresh query dotted against the stored keys, a softmax, a weighted sum of the stored values — but to do it I have to stream the entire cache through the chip. So per-token latency is essentially "how many bytes of cache do I read," and the quantity to minimize is bytes cached per token, not flops.

The known ways to shrink it all do the same thing: they make heads share key and value subspaces. Multi-Query Attention keeps all $n_h$ query heads but gives them a single shared key head and a single shared value head, dropping the cache to $2\,d_h\,l$ per token — a factor of $n_h$, but every head now looks at the past through one low-rank lens. Grouped-Query Attention softens this by splitting the query heads into $G$ groups, each with its own shared key/value head, so the cache is $2\,G\,d_h\,l$ and $G$ becomes a dial between MQA ($G=1$) and MHA ($G=n_h$). The trouble is that this dial is a genuine trade. A controlled comparison of three 7B models, identical except for the attention block and realigned to the same parameter count by adjusting depth, trained on the same 1.33T tokens, shows MHA uniformly best on hard benchmarks — MMLU 45.2 versus 41.2 for GQA-8 and 37.9 for MQA, C-Eval 42.9 versus 37.7 versus 30.0, with the same ordering on BBH and CMMLU. The more you force heads to share K and V, the more quality you lose, and to claw it back with GQA you must raise $G$, which puts the cache straight back up. On that frontier you cannot have both a small cache and MHA-level quality.

The crucial realization is that the quality loss is not caused by caching less; it is caused by destroying per-head expressivity in order to cache less, and those are different things that the baselines conflate. MHA is strong precisely because each head gets its own key and value subspace, so the $n_h$ heads jointly span a rich $2\,n_h\,d_h$-dimensional space and specialize. MQA collapses all of that onto a single $d_h$ subspace. So the question is whether I can keep a small cache without collapsing the subspaces — and the answer is to stop sharing and start compressing. I propose Multi-Head Latent Attention (MLA). The keys and values of a token are all deterministic functions of the same input $h_t$, so they are hugely redundant: $2\,n_h\,d_h$ numbers squeezed out of one $d$-dimensional vector. Instead of caching them, I down-project $h_t$ into one small joint latent and cache only that, then reconstruct the full per-head keys and values on the fly:
$$ c_t^{KV} = W^{DKV} h_t \in \mathbb{R}^{d_c}, \qquad k_t^C = W^{UK} c_t^{KV}, \qquad v_t^C = W^{UV} c_t^{KV}, $$
where $d_c \ll n_h d_h$ and $W^{UK}, W^{UV} \in \mathbb{R}^{n_h d_h \times d_c}$. Keys and values share the one latent $c_t^{KV}$ — two different up-projections fan it back out into a key subspace and a value subspace — because their information overlaps, so jointly compressing them caches one $d_c$-vector instead of two. And unlike MQA, the heads are not sharing a key or value head: $W^{UK}$ and $W^{UV}$ are full-rank fan-outs, so after up-projection each head receives its own slice, its own subspace. I have decoupled "small cache" from "few subspaces." During generation I cache only $c_t^{KV}$, which is $d_c\,l$ elements per token instead of $2\,n_h\,d_h\,l$.

The obvious worry is whether this actually saves bandwidth or merely relocates the cost: if I cache the small latent but then up-project it back to full-size $k^C$ and $v^C$ every step, I have done the same big arithmetic on full tensors. The answer is that I never need to materialize them. Write the per-head score with a symmetric low-rank query path ($c_t^Q = W^{DQ} h_t$, $q_t^C = W^{UQ} c_t^Q$, which shrinks training activation memory though not the cache):
$$ q_{t,i}^C \cdot k_{j,i}^C = (W^{UQ}_i c_t^Q)^{\!\top}(W^{UK}_i c_j^{KV}) = c_t^{Q\top}\,(W^{UQ}_i)^{\!\top} W^{UK}_i\, c_j^{KV}. $$
The two up-projections appear glued as the fixed matrix $(W^{UQ}_i)^{\!\top} W^{UK}_i$, which does not depend on the token: I precompute it, absorb $W^{UK}_i$ into the query-side projection once, and never form $k_{j,i}^C$ — scoring reads only the cached latent $c_j^{KV}$. The value side absorbs the same way, since $o_{t,i} = \sum_j a_{ij} W^{UV}_i c_j^{KV} = W^{UV}_i \sum_j a_{ij} c_j^{KV}$, so the weighted sum runs directly on the cached latents and $W^{UV}_i$ folds into the corresponding block of the output projection $W^O$. Both up-projections evaporate into the surrounding query and output matrices at decode time; the cache holds only $c_j^{KV}$ and that is literally all attention reads. The saving is real, and the per-head subspaces survive, so I do not pay the MQA tax.

The catch is position. The model uses RoPE, which rotates the query and key of a token at position $m$ by $R_m$ before the dot product, with the relative-offset property $(R_m q)^{\!\top}(R_n k) = q^{\!\top} R_{n-m} k$. If I bolt RoPE onto the compressed key, $k_{j,i} = R_j W^{UK}_i c_j^{KV}$, the score becomes
$$ q_{t,i}\cdot k_{j,i} = c_t^{Q\top}\,(W^{UQ}_i)^{\!\top} R_{j-t}\, W^{UK}_i\, c_j^{KV}, $$
and now $R_{j-t}$ is wedged between the two up-projections. That rotation depends on the pair $(t,j)$, so it changes for every cached token at every step; matrix multiplication does not commute, so it cannot be folded into a fixed matrix, and the absorption is dead — I would be back to reconstructing a full key per cached token per step. RoPE and absorbable low-rank compression are in direct conflict, and the conflict is specifically that the content key must be reconstructed through $W^{UK}$ (which I want to absorb) while RoPE must be applied after reconstruction and depends on position. The fix is to stop putting RoPE on the vector that carries content, and instead split those two jobs onto two vectors. I keep the compressed content key position-free, and alongside it add a small explicitly position-carrying key $k_t^R = \mathrm{RoPE}(W^{KR} h_t) \in \mathbb{R}^{d_h^R}$ with a matching per-head positional query $q_{t,i}^R = \mathrm{RoPE}(W^{QR} c_t^Q)$. Concatenating per head, $q_{t,i} = [q_{t,i}^C; q_{t,i}^R]$ and $k_{t,i} = [k_{t,i}^C; k_t^R]$, the dot product splits cleanly:
$$ q_{t,i}\cdot k_{j,i} = q_{t,i}^C\cdot k_{j,i}^C \;+\; q_{t,i}^R\cdot k_j^R. $$
The first term is pure content with no RoPE — fully absorbable as derived, reading only $c_j^{KV}$ — and the second carries all the position as $(\bar q_{t,i}^R)^{\!\top} R_{j-t}\,\bar k_j^R$, where there is no up-projection to obstruct, so nothing breaks. Position lives in a small decoupled subspace; content lives in the absorbable compressed subspace.

The decoupled key is a single vector shared across all heads, MQA-style on this tiny branch, for two reasons: a per-head RoPE key would cache $n_h\,d_h^R$ extra elements per token and eat the savings, whereas sharing makes it just $d_h^R$; and this branch only carries low-dimensional, head-agnostic relative-position signal, so the collapse costs almost nothing while the content branch keeps full per-head richness. The query side $q_{t,i}^R$ stays per-head, so each head still steers its own positional query against the one shared positional key. Because each head's query and key are now $[\text{content};\text{rope}]$ of total width $d_h + d_h^R$, the softmax divides by $\sqrt{d_h + d_h^R}$ rather than $\sqrt{d_h}$. The full per-token computation is
$$ o_{t,i} = \sum_{j\le t} \mathrm{softmax}_j\!\Big(\tfrac{q_{t,i}\cdot k_{j,i}}{\sqrt{d_h + d_h^R}}\Big)\, v_{j,i}^C, \qquad u_t = W^O[o_{t,1};\dots;o_{t,n_h}], $$
and only $c_t^{KV}$ and the shared $k_t^R$ are written to the cache. The per-token-per-layer cache is therefore $(d_c + d_h^R)\,l$. With the sizes that fall out — $d_c = 4 d_h$ and $d_h^R = d_h/2$ — that is $4.5\,d_h\,l$, the same cache as GQA with 2.25 groups and far below MHA's $2\,n_h\,d_h\,l$ (which at $n_h=128$ is $256\,d_h\,l$), yet with full per-head content subspaces intact, so the quality target stays at the MHA level rather than at GQA's small-cache region.

Two stability details matter. The down-projections create narrow bottlenecks where the activation scale can drift, so I apply an RMSNorm to the latents $c^{KV}$ and $c^Q$ before they are up-projected to keep reconstruction well-conditioned. And because the up-projection outputs are large activations, in training I recompute the RMSNorms and up-projections during back-propagation rather than persisting them — they are cheap to redo from the latents and this saves activation memory. For long context, RoPE lives only on the small decoupled branch, so any extension scheme such as a YaRN-style rescaling touches that branch alone, leaving the position-free content branch untouched. At scale I use $n_h=128$ heads of $d_h=128$, KV compression $d_c=512$, query compression $d_c'=1536$, and decoupled width $d_h^R=64$, so a per-head query/key is $128+64=192$ wide and a value head is $128$. In the implementation the query path fuses $W^{DQ}$, RMSNorm, and a single up-projection producing both $q^C$ and $q^R$ per head; the KV down-projection and the decoupled-key projection fuse into one linear whose output is $c^{KV}$ concatenated with the shared $k^R$; RMSNorm hits only the $c^{KV}$ slice; and a single fused up-projection takes the normalized latent to per-head $k^C$ and $v^C$. The eager forward below materializes K and V for training and prefill and for a generic reconstructed-tensor cache; in the compressed decode path the up-projections fold away as derived, so the durable cache is just $c^{KV}$ plus the rotated $k_t^R$.

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
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    bsz, n_heads, seq_len, head_dim = x.shape
    x = x.view(bsz, n_heads, seq_len, head_dim // 2, 2).transpose(4, 3).reshape_as(x)
    return (x * cos) + (rotate_half(x) * sin)


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.attention_dropout = config.attention_dropout
        self.q_lora_rank = config.q_lora_rank           # d_c'
        self.kv_lora_rank = config.kv_lora_rank         # d_c  -> the cached latent
        self.qk_nope_head_dim = config.qk_nope_head_dim # d_h   (content, absorbable)
        self.qk_rope_head_dim = config.qk_rope_head_dim # d_h^R (decoupled RoPE, shared key)
        self.v_head_dim = config.v_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        # query: W^{DQ} -> RMSNorm -> fused [W^{UQ}; W^{QR}]
        self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=config.attention_bias)
        self.q_a_layernorm = RMSNorm(self.q_lora_rank)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)

        # KV: fused [W^{DKV}; W^{KR}] -> latent ++ shared rope key
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=config.attention_bias)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)  # [W^{UK}; W^{UV}]

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim, self.hidden_size, bias=config.attention_bias)
        self.softmax_scale = self.q_head_dim ** -0.5

    def forward(self, hidden_states, cos, sin, position_ids,
                past_key_value=None, attention_mask=None):
        bsz, q_len, _ = hidden_states.size()

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)  # shared across heads

        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
        kv = kv.view(bsz, q_len, self.num_heads,
                     self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        q_pe = apply_rope(q_pe, cos, sin, position_ids)
        k_pe = apply_rope(k_pe, cos, sin, position_ids)

        query_states = torch.cat([q_nope, q_pe], dim=-1)
        key_states = torch.cat([k_nope, k_pe.expand(-1, self.num_heads, -1, -1)], dim=-1)

        if past_key_value is not None:
            # Generic eager caches store reconstructed tensors.
            key_states, value_states = past_key_value.update(key_states, value_states)

        scores = torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
        if attention_mask is not None:
            scores = scores + attention_mask
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        probs = F.dropout(probs, p=self.attention_dropout, training=self.training)

        out = torch.matmul(probs, value_states)
        out = out.transpose(1, 2).reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        return self.o_proj(out)
```
