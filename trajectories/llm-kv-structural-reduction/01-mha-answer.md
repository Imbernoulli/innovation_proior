**Problem.** GPT decoding is memory-bandwidth bound on the KV cache; the memory-to-compute ratio
$n/d + 1/b$ has its offending $n/d$ term coming from $b\,h\,n^2k$ — one key and one value cached *per
head*. Before reducing that, I need the unreduced control: the quality ceiling at the maximum KV cost,
the number every later rung is measured against.

**Key idea (the dense control).** Standard multi-head attention: `build_kv_heads` returns
`n_kv_head = n_head`, so every query head has its own key and value subspace. No head sharing, no
cross-layer reuse, no latent compression — `cross_layer_share` is `False` and `latent_kv_project` is
the identity. Full per-head capacity (best quality), and `kv_bytes_per_token = 2·n_kv_head·head_dim·2`
at its architectural maximum (worst efficiency). It is the weakest *KV-efficient* design by
construction and the strongest quality reference.

**Step-1 edit.** Fill the editable region with the dense form: KV head count equal to query head count,
drop the generic sharing/compression branches, call SDPA with `is_causal=True`, and set the diagnostic
attributes to "no reduction" (`head_sharing_ratio = 1.0`, latent ratio `1.0`).

**Hyperparameters.** 345M: 24 layers, 16 heads, width 1024, head_dim 64 → `n_kv_head = 16`,
`kv_bytes_per_token = 2·16·64·2 = 4096`. Frozen loop: ~7.1B tokens (13535 steps), 2-GPU DDP, LR 3e-4,
bf16, seed 42.

**What to watch.** Best `val_loss`/`heldout_loss` of any fill, paid for with the largest KV footprint
(4096 bytes/token). That maximum cost is what forces the first head-sharing cut at step 2.

```python
# EDITABLE region of custom_pretrain.py — step 1: dense multi-head attention (control)
def build_kv_heads(config):
    """Dense control: one KV head per query head."""

    n_kv_head = config.n_head
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
        self.head_sharing_ratio = 1.0

    def forward(self, x):
        bsz, seq_len, channels = x.size()
        qkv = self.c_attn(x)
        q, kv = qkv.split([self.n_embd, 2 * self.n_kv_head * self.head_dim], dim=2)
        k, v = kv.chunk(2, dim=2)
        q = q.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        k, v, latent_ratio = latent_kv_project(k, v, self)
        self._last_latent_rank_ratio = float(latent_ratio)
        self._last_kv_storage_ratio = 1.0
        self._uses_latent_compression = False
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, channels)
        y = self.resid_dropout(self.c_proj(y))
        return y
```
