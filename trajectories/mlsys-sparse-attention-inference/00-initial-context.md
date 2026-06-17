## Research question

A pretrained 1.5B-param causal LLM has to run long-context tasks at inference time under a hard
sparsity budget. The single thing being designed is the **`SparseAttention` module** — one instance is
monkey-patched into every attention layer of `Qwen/Qwen2.5-1.5B-Instruct`, replacing the attention
forward and nothing else. No retraining, no fine-tuning, no weight edits, no architectural surgery.
For each query token the module decides which of the preceding keys receive non-zero attention, and the
fraction of `(q, k)` pairs it keeps — the **density** — must stay at or under `0.25` (plus a small
`0.02` slack) averaged across all 24 layers, or the harness aborts the run. The question is how to
spend that 25% of the attention matrix so long-context retrieval and QA quality survive.

## Prior art before the first rung (the lineage the ladder reacts to)

The first rung is the **dense oracle** — the unmodified full-attention forward, density `1.0`, the only
baseline allowed to break the budget. It exists to fix the quality target the sparse rungs must climb
toward. The methods it reacts to are the long-context attention families that try to recover that
target under a budget:

- **Full scaled-dot-product attention (Vaswani et al., 2017).** `softmax(QKᵀ/√d)V` — every query reads
  every key in one hop, `O(1)` path length, `O(n²)` cost. The trustworthy reference, and exactly the
  quadratic the budget forbids: at N=8K it forms an N×N score matrix per head per layer. Gap: cost
  scales with the square of context, so it cannot be the deployed module — only the oracle.
- **Sliding-window / local attention.** Keep only the last `W` keys per query. Linear, trivial, fast —
  but content-blind: anything older than the window is invisible, so a needle far back in the context
  cannot be retrieved. Gap: a fixed local pattern drops long-range information by construction.
- **Static block-sparse patterns (sink+window; global+window+random).** Add always-attended anchor
  tokens (and, in the expander variants, random long-range links) to a local window. Cheap and
  decode-friendly under parallel forward, but the pattern does not depend on what the query is asking,
  so the relevant middle block is reached only by luck. Gap: a fixed mask, however clever, cannot route
  to query-specific evidence.
- **KV-cache compression keyed off the decode window (H2O, SnapKV, Quest).** Pick important keys from
  the accumulating attention of a mutable KV cache during autoregressive decode. Genuinely
  content-adaptive, but their importance signal shifts with the observation window and drifts under this
  benchmark's **parallel-forward** setting (every forward processes the full prefix in one shot; the
  same module replays at every generation step, `use_cache=False`). Gap: deliberately out of scope here
  — they do not transfer cleanly without cache plumbing the harness does not implement.

## The fixed substrate

The loop is frozen and must not be touched. `harness.py` loads `Qwen2.5-1.5B-Instruct` (native 32K
context, so no RoPE rescaling is needed at 8K), monkey-patches the agent's `SparseAttention` into every
attention layer, runs the model with `use_cache=False` so every forward is a single parallel pass over
the whole prefix, and replays the same module at every generation step. The backbone is GQA with 12
query heads and 2 KV heads; **the harness applies `repeat_kv` before calling the module**, so the
module always sees 12 heads on both Q and K/V — it never has to handle GQA replication itself. After
every forward the harness reads `self.last_density` and aggregates it across the 24 layers;
`enforce_budget` aborts the run if the mean exceeds `0.25 + 0.02` for any non-`dense` baseline, and a
missing, NaN, infinite, negative, or `>1` density report is treated as a harness error (not as zero).
RoPE and the GQA replication are applied by the loop *before* `q, k, v` reach the module.

## The editable interface

Exactly one region is editable — lines 31–103 of `sparse-attn-eval/custom_sparse_attn.py`, the body of
the `SparseAttention` class. `harness.py` and `run_llm.py` are read-only (model loading, patching,
density tracking, metrics). The contract:

```python
class SparseAttention(nn.Module):
    def __init__(self, head_dim, num_heads, block_size=64, density_budget=0.25): ...
    def forward(self, q, k, v, is_causal=False, scale=None) -> torch.Tensor: ...
```

`q, k, v` arrive as `(B, H, N, D)` in float16/bfloat16, GQA already replicated to `H=12`.
`is_causal=True` for this LLM (always). `scale` defaults to `1/√D`. The forward returns the attention
output in the same shape and dtype, and **must set `self.last_density`** to the fraction of `(q, k)`
pairs that received non-zero attention — causal-adjusted, dividing the kept count by `N(N+1)/2` when
`is_causal=True`. Every method on the ladder is a fill of this one class.

The starting point is the scaffold default below: a sink + sliding-window mask, built once per `N` and
cached across the 24 layers, run through fused SDPA. Each later method replaces exactly this class body.

```python
# EDITABLE region of custom_sparse_attn.py — default fill (sink + sliding window)
class SparseAttention(nn.Module):
    """Default: sliding window + sink, fused SDPA with a cached bool mask."""

    def __init__(self, head_dim, num_heads, block_size=64, density_budget=0.25):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.block_size = block_size
        self.density_budget = density_budget
        self.window = 1024          # local window radius (tokens per side)
        self.num_sinks = 4          # "always attended" sink tokens
        self.last_density = None
        self._mask_cache: dict = {}  # (N, is_causal, device) -> (mask, density)

    @torch.no_grad()
    def _get_mask(self, N: int, device: torch.device, is_causal: bool):
        key = (N, is_causal, device)
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        idx = torch.arange(N, device=device)
        di = idx[:, None] - idx[None, :]
        mask = (di.abs() <= self.window) | (idx[None, :] < self.num_sinks)
        if is_causal:
            mask = mask & (idx[:, None] >= idx[None, :])
        denom = N * (N + 1) / 2.0 if is_causal else float(N * N)
        density = float(mask.sum().item()) / max(denom, 1.0)
        self._mask_cache[key] = (mask, density)
        return mask, density

    def forward(self, q, k, v, is_causal=False, scale=None):
        B, H, N, D = q.shape
        scale = scale if scale is not None else 1.0 / math.sqrt(D)
        mask, self.last_density = self._get_mask(N, q.device, is_causal)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask.view(1, 1, N, N),
            dropout_p=0.0, is_causal=False, scale=scale,
        )
        return out
```

## Evaluation settings

`Qwen2.5-1.5B-Instruct`, single A100 80GB, FP16 only (no FP8), pure PyTorch ops (no Triton;
`torch.nn.attention.flex_attention` permitted if available). One seed, `42`. Three long-context
environments, all higher-is-better:

| Env | Metric | Notes |
|---|---|---|
| `niah_8k` | retrieval accuracy | Synthetic Needle-In-A-Haystack at 8K context |
| `longbench_qasper` | QA F1 | LongBench Qasper single-doc scientific-paper QA |
| `longbench_multifieldqa_en` | QA F1 | LongBench MultiFieldQA-EN long-document multi-field QA |

`density_budget = 0.25` for every non-`dense` rung; the `dense` oracle reports `last_density = 1.0` and
runs with `--allow-dense` so the budget check is skipped for it alone.
