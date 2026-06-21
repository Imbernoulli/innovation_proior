GQA-4 did what I bet it would and, in doing so, drew the sharing trade in full. It recovered most of the quality MQA gave up — `val_loss` 2.312553, well below MQA's 2.337850 and creeping back toward the dense 2.275425, with `heldout_loss` 3.969 essentially matching dense's 3.967 and ARC-Easy even edging past dense at 55.01. So the capacity story held: spreading the collapse over four subspaces instead of one recovers nearly all the quality. But the *cost* of that recovery is the whole point. GQA-4 paid 1024 bytes — four times MQA's 256, only a quarter below the dense 4096. The grouped layer recovered dense-level quality essentially by becoming most of the way back to dense on the byte axis. The three sharing points (256 / 1024 / 4096 bytes at 2.338 / 2.313 / 2.275 loss) trace a tilted line where quality and bytes move together, and no point on it is both small-cache *and* dense-quality. Every byte saved by sharing heads was a byte of key/value subspace deleted; to get the quality back I had to put the subspaces back, and putting the subspaces back put the bytes back. I cannot beat this line by picking a different `G` — any `G` is just another point on it.

That reframes the problem. It was never "too many KV heads," it was "the cache stores one full key and one full value per head." Head-sharing attacks that by reducing the number of heads, which necessarily reduces the number of subspaces. The alternative is to keep all sixteen per-head subspaces but store something *smaller than the keys and values themselves* — to attack the bytes without touching the head count. The structural observation that opens the door: the keys and values of a token are all functions of the same input vector $h_t$, the residual stream at that position. The dense layer projects $h_t$ up into $2 \cdot 16 \cdot 64 = 2048$ numbers of key plus value, all squeezed out of one 1024-dimensional source — enormous redundancy, because the per-head keys and values are not sixteen independent objects but sixteen linear views of one $h_t$.

So I propose **multi-head latent attention** (DeepSeek-V2). Instead of caching the full per-head keys and values, or sharing fewer of them, cache a *small compressed latent* of $h_t$ of dimension $d_c$ much smaller than $2 \cdot 16 \cdot 64$, and reconstruct the full per-head K and V from it on the fly: down-project $h_t$ to the latent, cache only the latent, up-project it back into full per-head keys and values when needed. Crucially the up-projection matrices are full-rank in the head dimension — they fan the one small latent back out into sixteen distinct per-head subspaces — so unlike head-sharing I am *not* collapsing subspaces. Each query head still gets its own key and value subspace; I have only made the *stored* object small. That decouples "small cache" from "few subspaces," exactly the conflation the three sharing runs were trapped in.

I have to be careful about what the harness measures, because latent attention has a serving-side trick this fixed loop does not exercise, and conflating the two would claim a saving the metric does not read. The full argument has two layers. The *training/prefill* layer: down-project to a latent, cache the latent, up-project back to full per-head K and V to do attention. And a *decode-time* layer where the up-projection matrices algebraically *absorb* into the query and output projections, so the full keys and values are never materialized and the cache really is just the latent. This task never runs that decode loop — it is from-scratch pretraining, and `GPT.structural_metrics()` computes `kv_bytes_per_token` analytically. For a latent block (one exposing `kv_a_proj_with_mqa` and `kv_b_proj`) the metric reads `2 · (kv_lora_rank + qk_rope_head_dim)` — the cached latent plus the decoupled positional key. So the harness rewards precisely the *footprint of the cached latent*, the first-layer saving; absorption is the serving-time realization of why that footprint is the real cache, but it is not separately measured. My fill is therefore the training/prefill form: materialize K and V from the latent every forward, set the latent dimensions small, let the metric read the footprint. No absorption code, no decode path.

The one subtlety where the naive latent scheme breaks is *position*, and the fill encodes the fix. The model needs rotary position embeddings; RoPE rotates the query and key of a token by a position-dependent matrix before their dot product, so the score sees only the relative offset. If I rotate the *content* keys — the ones reconstructed from the latent via the up-projection — the position-dependent rotation wedges itself between the latent and the up-projection, and because it depends on the *pair* of positions it changes every step, which is fatal to absorption and, more concretely for a from-scratch trainer, tangles position into the compressed content path where it does not belong. The fix is to split the two jobs onto two vectors: keep the compressed content key *position-free* (purely content, from the latent), and add a *separate, small, explicitly position-carrying* rotary key outside the latent that gets RoPE applied. Concatenate per head — the per-head key is $[\text{content key from latent}\;;\;\text{shared rotary key}]$ and the per-head query is $[\text{content query}\;;\;\text{per-head rotary query}]$ — and the score splits cleanly into a content dot-product (absorbable, position-free) plus a tiny positional dot-product that carries all the relative-position signal. Position lives entirely in a small decoupled subspace; content lives in the compressible one.

This fill makes specific choices that differ from the textbook latent recipe, and they are the load-bearing ones. The content-key dimension: rather than partitioning the head dimension into a content part and a rotary part, this fill treats `qk_nope_head_dim` as the *full* original head dimension (64) and adds the rotary slice *on top*, so the per-head query/key width is `qk_nope (64) + qk_rope (64) = 128`, wider than a dense head, and the value head stays at the full 64. That keeps the content subspace at full dense width — the whole point is to *not* shrink per-head expressivity. The rope dimension is `qk_rope_head_dim = min(64, head_dim)`, floored at 16 and forced even, which is 64 here. The latent rank is `kv_lora_rank = max(16, head_dim // 2) = 32` — deliberately tiny, capped by the small nanoGPT hidden size rather than the large multiple-of-head-dim a big model would use; that 32 is the number that drives the cache down. The KV down-projection `kv_a_proj_with_mqa` emits `kv_lora_rank + qk_rope_head_dim = 32 + 64 = 96` — the 32-dim latent concatenated with the 64-dim *shared* rotary key (one rotary key broadcast to all heads, MQA-style on the tiny positional branch, so it costs `qk_rope_head_dim` per token, not `n_head · qk_rope_head_dim`). The cached footprint is thus `2 · (32 + 64) = 192` bytes — *below* MQA's 256, with full per-head content subspaces intact. The query side has its own low-rank path (`q_a_proj` → RMSNorm → `q_b_proj`) producing both content and rotary query parts per head; `kv_b_proj` reconstructs per-head content keys and values from the normalized latent; an RMSNorm sits on each latent before its up-projection to keep the narrow bottleneck well-conditioned.

A few from-scratch implementation specifics the fill pins. The rotary cache is built fresh per forward with the half-split convention, and the fill *drops* the view→transpose→reshape re-interleave permutation the reference uses only when loading interleaved pretrained weights — for from-scratch training that permutation is pure overhead (a per-forward materialization of every query and key) and changes nothing. The full per-head query and key are assembled with `new_empty` plus slice-assignment rather than `cat`, avoiding the expanded-then-concatenated intermediate and remaining autograd-safe: the backward scatters gradients into the content and rotary pieces, broadcasting along the head axis for the shared rotary key. The softmax is scaled by $(qk_{\text{nope}} + qk_{\text{rope}})^{-1/2} = 128^{-1/2}$, not $64^{-1/2}$, because the per-head dot product now accumulates over 128 dims. The output uses a dedicated `o_proj` on the concatenated `n_head · v_head_dim` value outputs. None of this is the absorption/long-context machinery — no YaRN, no recompute-in-backward, no decode KV cache — because the harness measures a from-scratch pretraining footprint. `build_kv_heads` returns `(1, head_dim)` and `head_sharing_ratio` is set to `n_head = 16`, but the actual projections are MLA's own; the byte metric ignores `n_kv_head` for this block and reads the latent path.

The falsifiable claim against every prior rung. On the efficiency axis, `kv_bytes_per_token` should read `2 · (32 + 64) = 192` and `latent_rank_ratio` should read `kv_lora_rank / qk_head_dim = 32 / 128 = 0.25`, both deterministic from the latent dimensions, with `head_sharing_ratio` 16.0. That 192 is the headline — *below* MQA's 256, the previous floor, an order of magnitude below dense's 4096, the smallest cache of any rung. On the quality axis, because the per-head content subspaces are kept at full dense width and only the *storage* is compressed, `val_loss` should land *better* than both MQA (2.337850) and GQA-4 (2.312553) despite spending *fewer* bytes than either — something in the low 2.30s, strictly inside the Pareto frontier the three sharing points traced. A sharing method cannot beat both MQA's bytes and GQA's quality at once, because they sit on a trade-off line; compression can, because it attacks the cache without deleting subspaces. If MLA instead lands *on* the line — worse loss than GQA at its lower bytes, or no better than MQA — then the latent at rank 32 is too narrow and the compression is itself acting like a subspace deletion. If it lands below the line, the decoupling of small-cache from few-subspaces worked, and that single number — `val_loss` at 192 bytes versus the dense-MQA-GQA line — settles the task's central comparison in favor of latent compression over head sharing.

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
