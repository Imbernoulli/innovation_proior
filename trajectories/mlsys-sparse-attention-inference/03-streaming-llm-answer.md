**Problem (from rung 2).** BigBird collapsed NIAH to `0.2` (chance) and kept only ~60% of Qasper F1
(`0.0871` vs oracle `0.1406`), while its densities ran *over* budget (`0.2601` / `0.2421` / `0.2693`). It
failed not for lack of budget but for *where* it spent it: a large share went to **random** blocks chosen
blind to the question, which buy abstract connectivity but at inference are almost always irrelevant —
and they starved the parts that reliably carry signal. Fix the static floor before getting adaptive.

**Key idea.** Spend the whole static budget on the two reliably-useful pieces BigBird diluted: a **recent
sliding window** (where the local language-modeling signal and contiguous QA evidence live) plus a few
**attention sinks** (the first `num_sinks` columns). Sinks matter mechanistically: softmax must allocate a
full unit of mass, so under causal masking the always-visible first tokens become the universal dump for
surplus attention — keeping them holds the softmax denominator in its trained shape. Drop the random
gamble entirely.

**Why it works (and how it differs from the paper here).** The paper's StreamingLLM is a *KV-cache
eviction* policy with within-cache position re-indexing (rotate cached keys by cache-position each decode
step). None of that applies: the harness runs `use_cache=False`, every forward is a full parallel pass,
positions are the model's own RoPE. So here StreamingLLM is a **static sink+window mask** over the full
`(N,N)` causal matrix — only the *pattern* transfers, not the cache/latency story. The window is sized
from the harness's exact density: row keeps `num_sinks + W`, density `= 2(num_sinks+W)/(N+1)`, so
`W ≈ round(budget·(N+1)/2) − num_sinks` — derived from `mask_sum/denom`, which lands *on* budget where
BigBird's naive row-count sizing over-shot.

**What to watch.** Expect Qasper/MultiFieldQA F1 to **rise** (reclaimed random budget → local window) and
all three densities to drop just under `0.25`. Expect NIAH to **stay ~0.2**: a static sink+window still
cannot reach a mid-haystack needle. That residual NIAH gap to the oracle's `1.0` is the case for a
content-adaptive selection in the next rung.

**Hyperparameters.** `num_sinks=4`, `W` solved per `N` from the budget (causal:
`round(0.25·(N+1)/2) − 4`), fp32 score/softmax, density measured from the realized mask.

```python
# EDITABLE region of custom_sparse_attn.py — rung 3: StreamingLLM (sink + sliding window)
class SparseAttention(nn.Module):
    """StreamingLLM-style sink + sliding window attention.

    Causal mode (LLM): row-relative last-W window (``i-W+1 <= j <= i``) plus
    the first ``num_sinks`` columns.
    Non-causal mode (ViT/DiT): symmetric window (``|i-j| <= W``) plus the
    first ``num_sinks`` columns.
    """

    def __init__(self, head_dim, num_heads, block_size=64, density_budget=0.25):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.density_budget = density_budget
        # Paper default: 4 attention sinks. Xiao et al. show 4 is enough to
        # recover almost all the dense-attention quality on streaming inputs.
        self.num_sinks = 4
        self.last_density = None

    def _build_mask(self, N, device, is_causal):
        idx = torch.arange(N, device=device)
        if is_causal:
            # Solve density = 2*(num_sinks+W)/(N+1) = budget for W.
            W = max(1, int(round(self.density_budget * (N + 1) / 2.0)) - self.num_sinks)
            di = idx[:, None] - idx[None, :]
            local = (di >= 0) & (di < W)
        else:
            # Solve density = (num_sinks + 2W + 1)/N = budget for W.
            W = max(1, (int(round(self.density_budget * float(N))) - self.num_sinks - 1) // 2)
            di = idx[:, None] - idx[None, :]
            local = di.abs() <= W
        sinks = (idx[None, :] < min(self.num_sinks, N))
        mask = local | sinks
        if is_causal:
            mask = mask & (idx[:, None] >= idx[None, :])
        return mask

    def forward(self, q, k, v, is_causal=False, scale=None):
        B, H, N, D = q.shape
        scale = scale if scale is not None else 1.0 / math.sqrt(D)

        mask = self._build_mask(N, q.device, is_causal)  # (N, N)
        denom = (N * (N + 1) / 2.0) if is_causal else float(N * N)
        self.last_density = float(mask.sum().item()) / max(denom, 1.0)

        # Broadcast (N,N) mask across (B,H).
        attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
        attn = attn.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out = torch.matmul(attn, v.float())
        return out.to(q.dtype)
```
