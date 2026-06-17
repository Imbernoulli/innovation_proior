**Problem.** A frozen 1.5B causal LLM must run long-context tasks, but full attention is `O(N²)` —
forbidden as a deployed module. Before cutting anything I need the quality ceiling each sparse rung is
graded against, so rung 1 is the full-attention oracle: the only baseline allowed to exceed the `0.25`
density budget.

**Key idea (the reference).** Run the unmodified attention forward — `softmax(QKᵀ/√D + M)V` with the
causal `−∞` bias — and report the true density `1.0`. The GQA replication, RoPE, and `is_causal=True`
are all applied by the fixed loop before the module is called, so the oracle is the *empty* edit: hand
`q, k, v` straight to fused SDPA.

**Why it works.** Dense attention routes any query to any key in one hop, so it both retrieves the NIAH
needle perfectly and covers the full distributed evidence of the QA documents — it is the upper bound by
construction. The harness recognizes it via `ALLOW_DENSE_FLAG=1`, which forwards `--allow-dense` so
`enforce_budget` skips the budget check for this rung alone.

**What to watch.** Three target numbers, one per env. Expect near-perfect NIAH (clean position-coverage
discriminator) but only modest QA F1 (the 1.5B model's own competence caps it). The NIAH-vs-QA split is
the diagnosis the sparse rungs inherit: NIAH tests whether the mask covers the needle's block, QA tests
whether it covers enough relevant spans.

**Hyperparameters.** None — full attention, `scale = 1/√D`, `last_density = 1.0`.

```python
# EDITABLE region of custom_sparse_attn.py — rung 1: dense oracle (density 1.0)
class SparseAttention(nn.Module):
    """Dense attention oracle. Reports true full-attention density.

    The point of this baseline is to give an upper-bound quality reference;
    it is not meant to satisfy a real sparsity constraint.
    """

    def __init__(self, head_dim, num_heads, block_size=64, density_budget=0.25):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.density_budget = density_budget
        self.last_density = None

    def forward(self, q, k, v, is_causal=False, scale=None):
        B, H, N, D = q.shape
        scale = scale if scale is not None else 1.0 / math.sqrt(D)
        # Use PyTorch's fused SDPA for efficient dense attention.
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0,
            is_causal=is_causal, scale=scale,
        )
        self.last_density = 1.0
        return out
```
