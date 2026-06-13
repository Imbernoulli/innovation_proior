**Problem.** The dense control paid the architectural-maximum KV footprint — `kv_bytes_per_token`
4096 = `2·16·64·2` — to reach its quality ceiling. The byte count's only free variable is `n_kv_head`;
the diagnostic's offending cache term carries one key and one value *per head*. The first reduction is
to cut that head count, and the most informative single probe is the extreme.

**Key idea (the maximum head-sharing corner).** Multi-query attention: keep all 16 query heads, collapse
the KV side to a *single* shared key head and value head. `build_kv_heads` returns `(1, head_dim)`. Every
query head still computes its own attention pattern but they all read the same one key/value subspace.
The stored KV is `[b, 1, n, 64]`; right before the score matmul `expand_kv_to_q_heads` repeats it out to
16 heads (a local computation, never cached). This is the `G = 1` corner of the head-sharing axis, with
the dense layer as the `G = 16` corner.

**Why.** The cache is `∝ n_kv_head`, so one head is the smallest footprint pure head-sharing can reach:
`2·1·64·2 = 256`, a 16× cut from dense. The capacity cost is real — sixteen per-head key/value subspaces
become one shared subspace, so the query heads can differ in *how* they weight keys but not in *what* the
keys and values are. It is the cheapest and most aggressive sharing design; measuring it frames the whole
interpolation before spending a run on an intermediate group count.

**What the harness exposes/omits.** No decode loop is timed; `GPT.structural_metrics()` reads
`kv_bytes_per_token` analytically off the realized structure (`2·n_kv_head·head_dim·2`). This is a
*from-scratch* pretraining run — there is no dense checkpoint to convert and no uptraining/mean-pool
initialization, so the single shared subspace must be learned from random init, unprotected.

**Hyperparameters.** 345M: 24 layers, 16 heads, width 1024, head_dim 64 → `n_kv_head = 1`,
`kv_bytes_per_token = 256`, `head_sharing_ratio = 16.0`, `latent_rank_ratio = 1.0`. Frozen loop: ~7.1B
tokens (13535 steps), 2-GPU DDP, LR 3e-4, bf16, seed 42.

**What to watch.** `kv_bytes_per_token` exactly 256 and `head_sharing_ratio` 16.0 (deterministic);
`val_loss` clearly above the dense 2.275425 (the unprotected collapse). The size of that gap decides
whether the next rung interpolates the group count.

```python
# EDITABLE region of custom_pretrain.py — step 2: multi-query attention (single shared KV head)
def build_kv_heads(config):
    """Use one shared KV head for all query heads."""

    n_kv_head = 1
    head_dim = config.n_embd // config.n_head
    return n_kv_head, head_dim


def cross_layer_share(layer_idx, config):
    return False


def latent_kv_project(k, v, config):
    return k, v, 1.0


def expand_kv_to_q_heads(tensor, target_heads):
    current_heads = tensor.size(1)
    if current_heads == target_heads:
        return tensor
    full_repeats = target_heads // current_heads
    remainder = target_heads % current_heads
    parts = []
    if full_repeats > 0:
        parts.append(tensor.repeat_interleave(full_repeats, dim=1))
    if remainder > 0:
        parts.append(tensor[:, :remainder, :, :])
    return torch.cat(parts, dim=1)


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

        q_dim = config.n_embd
        kv_dim = 2 * self.n_kv_head * self.head_dim
        self.c_attn = nn.Linear(config.n_embd, q_dim + kv_dim, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
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
        self.use_pos_emb = True
        self.head_sharing_ratio = float(self.n_head)

    def forward(self, x):
        bsz, seq_len, channels = x.size()
        qkv = self.c_attn(x)
        q, kv = qkv.split(
            [self.n_embd, 2 * self.n_kv_head * self.head_dim],
            dim=2,
        )
        k, v = kv.chunk(2, dim=2)
        q = q.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        k = expand_kv_to_q_heads(k, self.n_head)
        v = expand_kv_to_q_heads(v, self.n_head)
        self._last_latent_rank_ratio = 1.0
        self._last_kv_storage_ratio = 1.0
        self._uses_latent_compression = False
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, channels)
        y = self.resid_dropout(self.c_proj(y))
        return y
```
