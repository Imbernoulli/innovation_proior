## Research question

GPT-style autoregressive decoding is dominated not by arithmetic but by memory bandwidth: at every
decode step the accelerator must stream the entire stored history of attention keys and values — the
KV cache — across the bus to produce a single token. That cache grows with sequence length, batch, and
the number of materialized KV heads, and it is what caps context length and batch size at serving time.
The single thing being designed here is the **attention block's KV structure** — how the keys and values
are materialized, shared, or compressed — under one fixed nanoGPT-style pretraining loop. The question
is sharp and two-sided: how much language-model quality survives a reduction in the *realized* KV state,
and whether **head sharing** (collapse the number of KV heads) or a **latent KV bottleneck** (compress
K/V into a low-rank vector decompressed on the fly) gives the better quality-per-byte tradeoff at a
fixed small-scale pretraining budget. Everything outside the attention block is frozen.

## Prior art before the first rung (the attention lineage)

The first rung is dense multi-head attention itself — the unreduced control the whole task reacts
against. The KV-reduction designs that follow it are responses to one structural fact: the KV cache is
large *because* it stores a separate key and value per head. The lineage that precedes the ladder:

- **Scaled dot-product attention (Vaswani et al., 2017).** A head projects the input into queries,
  keys, and values and returns $\mathrm{softmax}(QK^\top/\sqrt{k})V$; $h$ heads run in parallel, each
  with its own projections, so the layer attends to several relationships at once. Gap: it says nothing
  about *decoding* cost — the per-head K/V are exactly what must be cached and reloaded.
- **The KV cache.** During generation, position $i$ attends over all $j\le i$; recomputing past K/V
  each step is wasteful, so they are computed once and cached. This makes per-step arithmetic linear in
  history but introduces a stored tensor whose size grows with the sequence — and, critically, with the
  head count. Gap: nothing here bounds how many KV heads must be stored.
- **The memory-bandwidth diagnostic (roofline; Williams, Waterman, Patterson, 2009).** Counting bytes
  vs ops over a decode of length $n$, batch $b$, width $d$, $h$ heads, head dim $k=d/h$: arithmetic is
  $\Theta(b\,n\,d^2)$ (the projections dominate), while the cached K/V touched over all steps is
  $\Theta(b\,h\,n^2 k)=\Theta(b\,n^2 d)$. The memory-to-compute ratio is $n/d + 1/b$; the $n/d$ term —
  the cost of streaming the cache — is the offender, and it carries an $h$ inside ($b\,h\,n^2k$). Gap:
  the diagnostic pins the cost on *one key and one value per head*, but does not say how to shrink it.

The fixed substrate below is the standard dense layer those three converged on, and the editable
interface is precisely the region where the $h$ in the cache term can be cut or compressed away.

## The fixed substrate

A nanoGPT pretraining loop is frozen and must not be touched: token + learned absolute position
embeddings, a stack of `n_layer` pre-LayerNorm `Block`s (attention then a 4× GELU MLP, residual around
each), a tied LM head, AdamW with weight decay split (decay on 2-D params only), cosine LR schedule with
warmup, bf16 autocast, DDP. Training is on ClimbMix; held-out cross-entropy is measured on
WikiText-2/103 + LAMBADA, and 0-shot downstream accuracy via lm-eval. The loop also computes the KV
footprint itself: `GPT.structural_metrics()` walks every block's attention module and reports
`head_sharing_ratio = n_head / n_kv_head`, `latent_rank_ratio`, and the headline efficiency number
`kv_bytes_per_token`. That last is **derived from the realized attention structure**, not measured at
runtime — for a plain block it is `2 * n_kv_head * head_dim * 2`; for a latent-compression block (one
exposing `kv_a_proj_with_mqa` and `kv_b_proj`) it is `2 * (kv_lora_rank + qk_rope_head_dim)`; for a
layer that borrows the previous layer's K/V (`share_across_layers`) it is `0`. So the design choices in
the attention block *are* what the efficiency metric reads off.

## The editable interface

Exactly one region is editable — the span between the read-only `# BEGIN/END KV EDITABLE REGION`
markers in `custom_pretrain.py`. An AST validator enforces that only the allowed helper functions plus
`CausalSelfAttention` appear at the top level of that span; the contract is the three required helpers
and the attention class:

- `build_kv_heads(config)` → `(n_kv_head, head_dim)`: how many KV heads are materialized relative to
  the `config.n_head` query heads.
- `cross_layer_share(layer_idx, config)` → `bool`: an optional structural hook to reuse a previous
  layer's K/V (auxiliary; not the main axis).
- `latent_kv_project(k, v, config)` → `(k, v, latent_ratio)`: an optional latent KV bottleneck.
- `CausalSelfAttention(nn.Module)`: how the above choices are instantiated inside the block — the
  internal Q/KV projection, any per-head expansion, and the attention mixing path. Its contents are
  flexible (it may define its own RMSNorm / rotary helpers from the allowed name list), but it must set
  `self.n_kv_head`, `self.head_dim`, and the `_last_*` diagnostic attributes the metric reads.

The starting point is the scaffold default: **dense multi-head attention** — one KV head per query
head, no sharing, no compression. Each rung on the ladder replaces exactly this region.

```python
# EDITABLE region of custom_pretrain.py — default fill (dense MHA)
def build_kv_heads(config):
    """Return the number of KV heads and per-head dimension."""

    n_kv_head = config.n_head
    head_dim = config.n_embd // config.n_head
    return n_kv_head, head_dim


def cross_layer_share(layer_idx, config):
    """Optionally reuse KV structure across layers. Default: no sharing."""

    return False


def latent_kv_project(k, v, config):
    """Optional latent KV bottleneck. Default: identity projection."""

    return k, v, 1.0


def expand_kv_to_q_heads(tensor, target_heads):
    """Expand KV heads to query heads while remaining safe for any head count."""

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
    _shared_kv_cache = {}

    def __init__(self, config, layer_idx=0):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.layer_idx = layer_idx
        self.n_kv_head, self.head_dim = build_kv_heads(config)
        self.share_across_layers = cross_layer_share(layer_idx, config)

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

        reused_previous = False
        if self.share_across_layers and (self.layer_idx - 1) in self._shared_kv_cache:
            k, v = self._shared_kv_cache[self.layer_idx - 1]
            reused_previous = True
        else:
            self._shared_kv_cache[self.layer_idx] = (k.detach(), v.detach())

        if self.n_kv_head != self.n_head:
            k = expand_kv_to_q_heads(k, self.n_head)
            v = expand_kv_to_q_heads(v, self.n_head)

        k, v, latent_ratio = latent_kv_project(k, v, self)
        self._last_latent_rank_ratio = float(latent_ratio)
        self._last_kv_storage_ratio = 0.0 if reused_previous else float(latent_ratio)
        self._uses_latent_compression = bool(latent_ratio < 0.999)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :seq_len, :seq_len] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, channels)
        y = self.resid_dropout(self.c_proj(y))
        return y
```

## Evaluation settings

Primary evaluation at **345M** scale (24 layers, 16 heads, width 1024), seed 42, on the Chinchilla-optimal
~7.1B-token schedule (13535 steps, 2-GPU DDP, LR 3e-4, bf16). Metrics, with directions:

- `val_loss` (primary; cross-entropy on the ClimbMix validation split, **lower is better**).
- `kv_bytes_per_token` (the efficiency axis; evaluator-derived from the realized attention structure,
  **lower is better**).
- `heldout_loss` (mean cross-entropy on WikiText-2/103 + LAMBADA at the final checkpoint, lower is
  better), with per-corpus breakdowns.
- `arc_easy`, `hellaswag` (0-shot downstream accuracy via lm-eval, higher is better).
- `head_sharing_ratio`, `latent_rank_ratio` (structural descriptors of the chosen design).

The research question is the **quality-per-byte tradeoff**, so a rung is "stronger" when it Pareto-improves
the relevant prior: lower KV bytes at equal-or-better loss, or better loss at equal-or-lower bytes.
