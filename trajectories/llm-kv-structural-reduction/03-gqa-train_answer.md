MQA measured the two corners of the sharing axis cleanly, and the gap is the one I was watching for. The single shared KV head cut `kv_bytes_per_token` from the dense 4096 to 256 — the clean sixteen-fold reduction, `head_sharing_ratio` 16.0 — but the price showed on quality: `val_loss` rose from 2.275425 to 2.337850, a gap of about 0.062 in cross-entropy, with `heldout_loss` 3.999 against dense's 3.967 and the downstream numbers softening too. So the collapse to one shared KV head is *not free*, and from a from-scratch run with no checkpoint to mean-pool from, that is the tell: the per-head KV expressivity was carrying real work, not redundancy. But read the *shape* of the trade, because the shape says it is mis-calibrated rather than fundamental. MQA moved sixteen-fold on bytes and only 0.062 on loss — a wildly asymmetric ratio that says the operating point is too extreme. `val_loss` is the primary metric, and a sixteen-fold byte cut costing 0.062 loss is not obviously a win if a *milder* cut costs far less loss for still-large byte savings. I have the two endpoints of one axis; what I lack is the *curvature* between them.

So the move is to make the number of KV heads a *dial* rather than a binary. I propose **grouped-query attention** (Ainslie et al., 2023). The cache cost scales with `n_kv_head` — that is the entire content of `2 · n_kv_head · head_dim · 2` — so `n_kv_head` is a discrete lever settable anywhere from 1 to 16. I keep all sixteen query heads and give them `G` shared KV heads, partitioning the sixteen query heads into `G` groups of `16/G`, each group sharing one key head and one value head. Query heads in the same group attend over a common key/value subspace; different groups get different ones. The two designs I have measured are literally the endpoints: `G = 1` is one big group — the MQA collapse — and `G = 16` is sixteen groups of one — the dense layer. The grouping *is* the interpolation, and the dense-vs-MQA gap is the interval it spans.

Why should an intermediate `G` sit *above* the line connecting the two corners rather than on it? Two reasons, both following from what MQA's failure actually was. First, capacity: instead of cramming all sixteen query heads through one shared 64-dimensional subspace, I now have `G` subspaces, so the representational damage is spread over `G` buckets rather than concentrated in one. MQA showed the single subspace was the bottleneck; a handful of subspaces should recover most of the lost quality because the heads no longer all have to agree on one notion of what the keys are. Second, the byte cost stays *linear* in `G` while the quality cost should be *sub-linear* in how much I share — the first few subspaces buy back most of the expressivity and additional ones have diminishing returns. So a small `G` should recover a disproportionate share of the dense quality for a still-large byte cut. That is the curvature I am betting on: quality recovering fast as `G` rises off 1, bytes rising only linearly.

This fill fixes the dial at one point. `build_kv_heads` returns `n_kv_head = max(1, n_head // 4)`, then walks `n_kv_head` down until it divides `n_head` evenly. For sixteen heads that is `16 // 4 = 4`, and 4 already divides 16, so `G = 4` — four groups of four query heads, four KV heads. The byte count becomes `2 · 4 · 64 · 2 = 1024`, `head_sharing_ratio = 16 / 4 = 4`. This is a sensible single probe: 4 is the geometric mean of 1 and 16, the midpoint of the axis on a log scale, so it is the single most informative interior point. The expansion from four stored KV heads to sixteen query heads happens in the forward via `k.repeat_interleave(repeat_factor, dim=1)` with `repeat_factor = n_head // n_kv_head = 4`, each stored KV head repeated four times so the four-group structure lines up with the sixteen query heads for the score matmul.

I want to name what this fill *omits* relative to the full grouped recipe, because it changes what the number means. The general motivation for grouped KV heads comes with a second, equally important half: you do not train from scratch — you take an existing dense checkpoint, *mean-pool* each group's trained key and value projections into one shared projection (the most information-preserving merge, better than picking one head or random init), and then continue pretraining for a small fraction of the original steps to let the averaged heads re-coordinate with the untouched query and output projections. That conversion-plus-adaptation recipe is the cheap way to get a grouped model, and it is also what *protects* the quality — the grouped model starts from a sensible average of trained subspaces, not from noise. This task's harness exposes none of it: no checkpoint-conversion hook, no mean-pool initialization, no continued-pretraining knob. The model trains from random init with four KV heads from step zero, exactly like the MQA run trained with one. That actually makes the comparison cleaner: MQA and GQA-4 are apples-to-apples, both from scratch, and the only difference is the structural one, `G = 1` versus `G = 4` — no conversion artifacts in the way of "does spreading capacity over four subspaces help."

I should be honest about *where* on the Pareto picture `G = 4` will land. Its byte count is 1024 — four times MQA's 256, only a quarter below the dense 4096 — so on the efficiency axis GQA-4 is much *closer to dense* than MQA is; it gives back three-quarters of MQA's hard-won byte savings to buy back quality. Whether that is a Pareto improvement depends entirely on how much quality it recovers. If `val_loss` drops from 2.337850 back toward 2.275425 — say the low 2.31s — then GQA-4 is a real interior point: most of the dense quality at a quarter of the dense bytes, a Pareto-better balance than either corner. If instead loss barely moves off 2.338 despite quadrupling the bytes, that says head-sharing capacity is not the binding constraint at this scale and the four extra subspaces were wasted — which would itself argue that the next rung should stop sharing heads and attack the bytes a different way. So GQA-4 is a discriminating probe: its position relative to the dense-MQA line decides whether the sharing axis has a useful interior at all.

That is the bridge I most want to set up. The sharing axis, even at its best interior point, is fundamentally a *trade*: every byte saved by sharing heads is a byte of key/value subspace deleted, so small-cache regions force few subspaces and lose quality, and quality-preserving regions keep many subspaces and large caches. GQA-4 quantifies how steep that trade is, but `G = 4` cannot escape it — it is just a less-bad point on the same line. The single number that matters most is how far `val_loss` recovers per byte spent relative to MQA: that ratio is the steepness of the sharing trade, and if even the recovered point still sits visibly above dense, the honest conclusion is that *sharing* heads cannot give both small cache and dense quality, and the only way past is to stop deleting subspaces and instead *compress* the per-token key/value information into a small latent decompressed back into full per-head subspaces on the fly — the final rung.

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
