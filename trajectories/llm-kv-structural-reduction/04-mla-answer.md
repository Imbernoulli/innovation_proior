**Problem.** The three sharing runs trace a tilted Pareto line: dense (4096, 2.275), GQA-4 (1024, 2.313),
MQA (256, 2.338) — quality and bytes move together, and no point on it is both small-cache and
dense-quality. Head-sharing recovers quality only by putting subspaces back, which puts bytes back. To
get *below* the line I must stop deleting per-head subspaces and instead shrink what is stored.

**Key idea (latent KV compression).** Multi-head latent attention: keys and values are all linear views
of one residual vector `h_t`, so cache a small low-rank *latent* of `h_t` (dim `kv_lora_rank`) and
up-project it back into full per-head K and V on the fly. The up-projections are full-rank in the head
axis — every query head keeps its own subspace — so this decouples "small cache" from "few subspaces."
Position is decoupled: the content key is position-free (compressible), and a separate small *shared*
rotary key carries RoPE alongside it; the per-head score splits into a content dot-product plus a tiny
positional one.

**Why.** The cached footprint is `2·(kv_lora_rank + qk_rope_head_dim)`, independent of head count, so it
can drop below MQA's 256 while per-head content subspaces stay at full dense width — small cache *and*
dense-level quality at once, the point off the sharing line. This is the design that answers the task's
central comparison: latent compression vs head sharing.

**What the harness exposes/omits.** From-scratch pretraining only — the decode-time *absorption* of the
up-projections into Q/O (which is why the latent is the real cache) is not exercised, and `kv_bytes` is
read analytically as `2·(kv_lora_rank + qk_rope_head_dim)`. This fill keeps `qk_nope_head_dim = head_dim`
(content at full dense width, rotary slice added on top), `kv_lora_rank = max(16, head_dim//2) = 32`
(capped by the small nanoGPT width), one shared rotary key broadcast to all heads, `new_empty` +
slice-assign assembly, softmax scale `(qk_nope+qk_rope)^{-1/2}`. No YaRN, no recompute-in-backward, no
interleave permutation (dropped — only needed for loading pretrained weights), no decode KV cache.

**Hyperparameters.** 345M, head_dim 64 → `qk_rope_head_dim = 64`, `qk_nope_head_dim = 64`,
`qk_head_dim = 128`, `kv_lora_rank = 32`, `q_lora_rank = min(1024, 12·64) = 768`,
`kv_bytes_per_token = 2·(32+64) = 192`, `latent_rank_ratio = 32/128 = 0.25`, `head_sharing_ratio = 16.0`.
Frozen loop: ~7.1B tokens (13535 steps), 2-GPU DDP, LR 3e-4, bf16, seed 42.

**What to watch.** `kv_bytes_per_token` 192 (below MQA's 256), `latent_rank_ratio` 0.25 (deterministic);
`val_loss` better than both MQA (2.337850) and GQA (2.312553) at *fewer* bytes — i.e. strictly inside the
sharing Pareto line. If it lands on or above the line, rank 32 is acting like a subspace deletion and the
latent is too narrow. That `val_loss`-at-192-bytes vs the sharing line is the task's decisive measurement.

```python
# EDITABLE region of custom_pretrain.py — step 4: multi-head latent attention (latent KV compression)
def build_kv_heads(config):
    head_dim = config.n_embd // config.n_head
    return 1, head_dim


def cross_layer_share(layer_idx, config):
    return False


def latent_kv_project(k, v, config):
    return k, v, 1.0


class MLARMSNorm(nn.Module):
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
    # build_rotary_cache uses the half-split convention (cat((freqs, freqs), -1)),
    # so rotate_half + the *cos/+sin formula below is already the correct form.
    # The original view->transpose(4,3)->reshape re-interleave was needed only when
    # loading DeepSeek-V2 pretrained weights in interleaved layout; for a from-scratch
    # nanoGPT this permutation just adds a per-forward materialization per Q and K
    # (~640MB total activation across 24 layers at B=32 T=1024). Drop it.
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

        # DeepSeek/TransMLA treat qk_nope as the original dense head dimension
        # and add a separate rotary slice on top, rather than partitioning the
        # original head dim into two halves.
        self.qk_rope_head_dim = min(64, self.head_dim)
        self.qk_rope_head_dim = max(16, self.qk_rope_head_dim)
        if self.qk_rope_head_dim % 2 != 0:
            self.qk_rope_head_dim -= 1
        self.qk_nope_head_dim = self.head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = self.head_dim
        # Preserve the relative rank schedule used in DeepSeek-V2 style MLA
        # while capping by the tiny nanoGPT hidden size.
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
        cos, sin = build_rotary_cache(
            seq_len, self.qk_rope_head_dim, x.device, q_rot.dtype
        )
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)

        # DeepSeek-V2 official pattern: new_empty + slice-assign.
        # Avoids k_rot.expand(-1, n_head, -1, -1) materialization (saves the
        # expanded-contiguous intermediate) and the subsequent torch.cat's
        # transient output buffer. slice __setitem__ is autograd-safe — the
        # backward scatters gradients back into q_nope / q_rot / k_nope / k_rot
        # (broadcast along head axis for k_rot).
        query_states = q_states.new_empty(bsz, self.n_head, seq_len, self.qk_head_dim)
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_rot

        key_states = q_states.new_empty(bsz, self.n_head, seq_len, self.qk_head_dim)
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_rot  # broadcasts over n_head

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
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
