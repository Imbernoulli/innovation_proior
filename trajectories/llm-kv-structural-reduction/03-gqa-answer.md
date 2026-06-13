**Problem.** The two sharing-axis corners are measured: dense (4096 bytes, `val_loss` 2.275425) and MQA
(256 bytes, `val_loss` 2.337850). The collapse to one shared KV head cost ~0.062 cross-entropy — visible,
not free, so per-head KV subspaces carry real work. The trade is wildly asymmetric (16× bytes for 0.062
loss), which says the operating point is too extreme. The fix is to interpolate, not to revert.

**Key idea (the head-sharing dial).** Grouped-query attention: make `n_kv_head` a dial. Keep all 16 query
heads, partition them into `G` groups of `16/G`, each group sharing one key/value head. `G = 1` is MQA,
`G = 16` is dense, intermediate `G` spreads the capacity over `G` subspaces instead of one. This fill
fixes `G` via `build_kv_heads`: `n_kv_head = max(1, n_head // 4)` adjusted down to divide `n_head`
evenly → `G = 4` for 16 heads. The four stored KV heads are expanded to 16 query heads with
`k.repeat_interleave(4, dim=1)` in the forward.

**Why.** The cache is `∝ n_kv_head`, so `G = 4` costs `2·4·64·2 = 1024` — four subspaces should recover
most of the quality MQA lost (the bottleneck was one shared subspace) while bytes rise only linearly.
`G = 4` is the geometric mean of 1 and 16: the single most informative interior point. Its position
relative to the dense–MQA line measures the steepness of the sharing trade.

**What the harness exposes/omits.** Trained *from scratch* — the conversion half of grouped attention
(mean-pool each group's trained dense key/value projections into one, then continue-pretrain a few
percent of steps to adapt) is entirely absent; there is no checkpoint hook and no uptraining knob. So
this is the unprotected, from-scratch `G = 4`, apples-to-apples with the from-scratch MQA. `kv_bytes`
is read analytically off `n_kv_head`. `G` is fixed at `H//4`, not the general recipe's recommended 8.

**Hyperparameters.** 345M: 24 layers, 16 heads, width 1024, head_dim 64 → `n_kv_head = 4`,
`kv_bytes_per_token = 1024`, `head_sharing_ratio = 4.0`, `latent_rank_ratio = 1.0`. Frozen loop: ~7.1B
tokens (13535 steps), 2-GPU DDP, LR 3e-4, bf16, seed 42.

**What to watch.** `kv_bytes_per_token` exactly 1024, `head_sharing_ratio` 4.0 (deterministic);
`val_loss` between dense 2.275425 and MQA 2.337850, expected nearer the dense end if four subspaces
recover what one lost. How far loss recovers per byte spent vs MQA is the steepness of the sharing
trade — and the argument for switching to latent compression at the endpoint.

```python
# EDITABLE region of custom_pretrain.py — step 3: grouped-query attention (G = n_head // 4)
def build_kv_heads(config):
    """Use fewer KV heads than query heads, preserving query expressivity."""

    n_kv_head = max(1, config.n_head // 4)
    while config.n_head % n_kv_head != 0:
        n_kv_head -= 1
    head_dim = config.n_embd // config.n_head
    return n_kv_head, head_dim


def cross_layer_share(layer_idx, config):
    return False


def latent_kv_project(k, v, config):
    return k, v, 1.0


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
        self.head_sharing_ratio = self.n_head / max(self.n_kv_head, 1)

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
        repeat_factor = self.n_head // self.n_kv_head
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)
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
