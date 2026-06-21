The whole task is about how to shrink the KV cache, but I cannot measure how much quality a reduction costs until I have measured the quality with *no* reduction. So the first thing to run is the dense, unreduced attention layer itself — the control. It is simultaneously the thing every later rung must beat on bytes and the quality ceiling every later rung is measured against, and holding both of those facts at once is the entire frame for the climb.

Let me ground the cost I am about to count in the arithmetic rather than a slogan. A single attention head projects each position into a query, a key, and a value, then returns a softmax-weighted average of the values with weights from scaled query-key dot products, $\mathrm{softmax}(QK^\top/\sqrt{k})V$, where $k$ is the per-head dimension. A layer runs $h$ such heads in parallel, each with its own projections, so it attends to several relationships at once; the per-head outputs are concatenated and passed through one output projection. With model width $d$ the head dimension is $k = d/h$, so the $h$ heads tile the full width. In this scaffold $d = 1024$, $h = 16$, $k = 64$. The dense layer materializes one key head and one value head for *every* query head — sixteen of each — and that is exactly what makes it expensive to decode.

The cost that decides deployability is not training, which processes a whole sequence in parallel and is compute-bound, but autoregressive decoding, where I emit one token, append it, and feed it back. To avoid recomputing the keys and values of every past position I cache them and append the new token's K/V each step. That makes per-step arithmetic linear in history length rather than quadratic, but it introduces a stored tensor whose size grows with the sequence and — the part that bites — with the head count. Counting bytes against ops over a decode of length $n$, batch $b$: the arithmetic per token is dominated by the four $d\times d$ projections, $\Theta(d^2)$ per token and $\Theta(b\,n\,d^2)$ over the generation; the score-and-mix is smaller until the history is very long, so the projections dominate compute. On the memory side, two things stream: the projection matrices, reloaded each step because there is only one token's activation to reuse them on, costing $\Theta(n\,d^2)$; and the cache, which at step $i$ has shape $[b, h, i, k]$ for keys and the same for values, so the bytes touched over all steps sum to $\sum_i b\,h\,i\,k = \Theta(b\,h\,n^2 k)$, and since $hk = d$ that is $\Theta(b\,n^2 d)$. The ratio of memory access to arithmetic is therefore

$$\frac{b\,n^2 d + n\,d^2}{b\,n\,d^2} = \frac{n}{d} + \frac{1}{b}.$$

The whole motivation for the task lives in that ratio. The $1/b$ term I beat by batching more requests — a throughput knob, not an architecture problem. The $n/d$ term is the killer: as the generated length climbs toward the model width it approaches and then exceeds one, roughly one full memory load per arithmetic operation, deep in the memory-bound regime where the compute units starve. And I can read off precisely where the $n/d$ comes from — the $b\,n^2 d$ memory term, the cost of streaming the cached keys and values, which carries an $h$ inside it ($b\,h\,n^2 k$): one key and one value per head per position. The cache is big *because every head keeps its own keys and values*. That localizes the entire problem to one design decision inside the editable region — how many KV heads `build_kv_heads` returns — and the dense fill returns $h$, the maximum, so it sits at the worst point of the very axis the task is about.

So what I run first is plain dense multi-head attention, and I want to be precise that I run it *as the control* rather than skipping straight to a reduction. The dense layer is the quality reference: each query head attends through its own key and value subspace, the $h$ heads jointly span a rich $2hk$-dimensional key/value space and specialize, and that per-head expressivity is exactly what every reduction trades away. It is also the most stable to train — no bottleneck to squeeze gradients through, no shared subspace forcing heads to agree — so it gives the cleanest read on what the rest of the loop achieves before any attention surgery. It is the ceiling on quality and the floor on efficiency at once.

The fill is the scaffold default, made deliberately minimal so later rungs read as clean deltas. `build_kv_heads` returns `(config.n_head, config.n_embd // config.n_head)` — KV head count equal to query head count, head dimension the full $d/h$. `cross_layer_share` returns `False`: every layer materializes its own K/V. `latent_kv_project` is the identity returning latent ratio `1.0`: no compression. Inside `CausalSelfAttention` the combined `c_attn` projection emits queries of width $d$ plus keys and values of width $2\,n_{kv}\,k = 2d$ (since $n_{kv} = h$), the three are reshaped to $[b, h, n, k]$, and because the dense control wants no helper to fire I drop the cross-layer cache branch and call SDPA directly with `is_causal=True`. I set the diagnostic attributes to "no reduction" — `_last_latent_rank_ratio = 1.0`, `_last_kv_storage_ratio = 1.0`, `_uses_latent_compression = False` — and `head_sharing_ratio = 1.0`, one query head per KV head. The control should differ from the generic template only in that it hard-wires "no reduction" rather than routing through the sharing/compression branches, so the baseline is unambiguous.

One harness detail makes the dense layer's weakness *visible in the score*. The loop does not time decoding; it computes `kv_bytes_per_token` analytically in `GPT.structural_metrics()` from the realized structure. For a plain block that is `2 * n_kv_head * head_dim * 2` — the leading two for separate K and V, the trailing two for bf16 bytes per element — which for $n_{kv} = 16$, $k = 64$ is $2\cdot16\cdot64\cdot2 = 4096$, the largest KV footprint any fill of this region can produce. So the whole climb is a search over what `build_kv_heads` and `latent_kv_project` return, and the metric reports the consequence directly. I expect dense MHA to confirm exactly the split it is built to show: the best `val_loss` and `heldout_loss` of any fill, competitive downstream numbers, paid for with a `kv_bytes_per_token` of 4096 that is an order of magnitude worse than anything that shares or compresses. That 4096 — the $h$ in $b\,h\,n^2 k$ made concrete — is what forces the first head-sharing cut at the next rung.

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
