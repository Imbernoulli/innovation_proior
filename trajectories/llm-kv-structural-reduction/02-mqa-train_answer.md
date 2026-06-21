The dense control landed where it had to: `val_loss` 2.275425, `heldout_loss` 3.967285, ARC-Easy 54.88, HellaSwag 33.42 — the quality ceiling — bought at a `kv_bytes_per_token` of 4096, the architectural maximum. That 4096 is `2 · n_kv_head · head_dim · 2` with `n_kv_head = 16`, `head_dim = 64`, and the only free variable in it is the sixteen: one key head and one value head materialized for every one of the sixteen query heads. `head_dim` is pinned at `n_embd / n_head = 64`, and the two trailing twos are "K and V are separate" and "bf16 is two bytes," both immovable. The diagnostic from the dense run said the offending cache term was $b\,h\,n^2 k$ and the $h$ inside it is the per-head KV; the measured 4096 is that $h$ made concrete. So the first reduction is forced: cut $h$.

The only question is *how far* to cut on the first try, and I choose the extreme deliberately rather than timidly. I could shave to eight or four and creep down the axis, but the most informative single probe is the corner. I propose **multi-query attention** (Shazeer, 2019): keep all sixteen query heads, and collapse the entire KV side to a *single* shared key head and value head that all sixteen query heads read. This is the $G = 1$ corner of the head-sharing axis, with the dense layer as the $G = h$ corner; measuring both corners frames the whole interpolation before I spend a run on anything between them.

Mechanically, in this scaffold: I keep all sixteen query heads — they are cheap, they do the representational work, and they accumulate nothing across decode steps, so cutting them would buy no bytes. `build_kv_heads` returns `(1, head_dim)`: one KV head, head dimension still 64. The combined `c_attn` projection now emits queries of width `n_embd` (1024, sixteen query heads) plus keys and values of width `2 · 1 · 64 = 128` — a single key head and a single value head. The query reshapes to $[b, 16, n, 64]$, key and value to $[b, 1, n, 64]$. Then, right before the score matmul, the one KV head has to line up with the sixteen query heads, because scaled-dot-product attention wants matching head counts on both sides. That is what `expand_kv_to_q_heads` does: it repeats the single stored KV head out to sixteen so every query head attends over the same keys and values. The *ordering* is the whole point — the *stored* tensor is the small $[b, 1, n, 64]$ one, and the expansion to sixteen is a local computation for the matmul that never gets cached. In a real decode loop that is the entire saving: cache one head, expand for the arithmetic, discard the expansion. Here, with no decode loop, the saving shows up as `kv_bytes_per_token` reading `n_kv_head = 1` and reporting `2 · 1 · 64 · 2 = 256` — a clean sixteen-fold cut and the floor any pure-sharing fill can reach, since one KV head of full width is the smallest a shared-head structure has.

I want to be honest about what the harness measures versus what the head-sharing idea is *supposed* to carry, because it bounds what I can claim. The original argument for one shared KV head is a decode-bandwidth argument: the cache is streamed every step, the cache is $\propto n_{kv}$, so one head is a sixteen-fold bandwidth win at serving time. This task never runs an autoregressive decode loop and never times one — deliberately, because in pure-PyTorch eager mode wall-clock generation tracks per-layer op count more than KV-structure merit. Instead `GPT.structural_metrics()` computes `kv_bytes_per_token` analytically from the realized structure, reading `n_kv_head` and `head_dim` off the module. So the bandwidth win is captured as a structural footprint, not a measured latency — the right call for isolating the design choice from kernel noise, but it means I am optimizing the footprint the structure implies and the quality I preserve at it.

Now what I actually care about: what this costs in quality, and why. Collapsing sixteen KV subspaces into one is not small. In the dense layer each query head attended through its *own* key and value projection — its own subspace defining "what matches what" and "what gets read out" — and the sixteen jointly spanned a rich $2 \cdot 16 \cdot 64$-dimensional key/value space where different heads specialized. The single-shared-head collapse forces all sixteen query heads to read and write through *one* 64-dimensional key subspace and one 64-dimensional value subspace. The heads keep their own query projections, so they can still differ in *how* they weight a fixed set of keys — but they can no longer disagree about *what the keys and values are*. That is a genuine loss of capacity, concentrated entirely where the cache lives. So I expect `val_loss` to rise above the dense 2.275425; how much is exactly the experiment. A small rise would mean head sharing is cheap and the dense layer carried redundant subspaces; a large rise would mean the per-head KV expressivity was doing real work.

I expect the rise to be *visible* here, and for a specific reason: this is a *from-scratch* pretraining run, not a conversion. There is no trained dense checkpoint being adapted — `build_kv_heads` returns one head and the model learns from random init with that one head from step zero, for the full ~7.1B-token schedule. The harness gives no uptraining knob: no mean-pool initialization from a dense checkpoint, no short continuation run. So the MQA number is the honest "what can one KV head learn from scratch at this budget" figure, with none of the cushioning a conversion would provide — the unprotected version of the collapse.

So the falsifiable expectations against the dense numbers. On the efficiency axis, `kv_bytes_per_token` should read exactly 256 (`2 · 1 · 64 · 2`) and `head_sharing_ratio` exactly 16.0 — deterministic consequences of `n_kv_head = 1`; anything else means the fill is wrong. On the quality axis, `val_loss` should rise clearly above 2.275425, with `heldout_loss` and the downstream accuracies softening in step. The size of that gap decides the next rung: if small, sharing is cheap and the interesting question becomes how low compression can push bytes; if large, the gap itself argues for an intermediate group count that recovers quality by spending a little of the byte budget back. Either way MQA establishes the cheap-and-lossy corner — minimum sharing-axis footprint, maximum sharing-axis quality cost — against the expensive-and-faithful dense corner.

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
