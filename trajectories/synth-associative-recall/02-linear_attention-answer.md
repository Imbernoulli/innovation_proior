**Problem.** The recurrent floor failed even at 8 key-value pairs because it has no operation that *compares*
a query against the set of earlier keys — only a running summary. The mixer needs an **explicit query-key
score** so the lookup is content-addressed, but it must stay sub-quadratic (no growing `N×N` cache).

**Key idea (linear attention).** Read attention as a kernel smoother and replace the softmax similarity with
a factored, non-negative feature map: `sim(q,k) = φ(q)^T φ(k)` with `φ(x) = elu(x) + 1` (positive, cheap,
gradient-alive). Factoring lets associativity collapse the over-keys part into a fixed-shape running state
`S_i = Σ_{j≤i} φ(k_j) v_j^T`, so the read is `y_i = φ(q_i)^T S_i / (φ(q_i)^T z_i)` — an explicit per-query,
per-key score with no growing cache. Trained with the **quadratic view** (materialize `A = φ(Q)φ(K)^T`,
causal-mask, multiply by `V`, normalize by masked row-sums), which is term-for-term identical to the
recurrent view but GPU-friendly at these lengths.

**Why it beats step 1 but is still limited.** The explicit score is the comparison the LSTM lacked, so it
should solve the easy setting. But `elu+1` induces a *gentle, high-entropy* kernel that spreads weight
broadly; recall needs a *spiky* distribution that concentrates on the one matching key. As the key count
grows the non-matching smear swamps the matching value and accuracy collapses.

**Hyperparameters.** `head_dim=32` (projection below `d_model=64`); bias-free `q/k/v/out` projections;
denominator clamped at `1e-6`; causal `tril` mask (attention is not causal by default). No short conv inside
the mixer — the substrate's fixed first-layer conv handles local shifts.

```python
# EDITABLE region of custom_strategy.py — step 2: linear attention (elu+1 feature map)
class CustomMixer(nn.Module):
    """Causal linear attention with phi(x) = elu(x) + 1 (Katharopoulos 2020)."""

    def __init__(self, d_model: int, seq_len: int, head_dim: int = 32):
        super().__init__()
        self.head_dim = head_dim
        self.q_proj = nn.Linear(d_model, head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    @staticmethod
    def phi(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q = self.phi(self.q_proj(x))                  # [B, T, F]
        k = self.phi(self.k_proj(x))                  # [B, T, F]
        v = self.v_proj(x)                            # [B, T, D]
        # Quadratic-view causal linear attention (training time): build the
        # T x T causal kernel matrix <q_t, k_s> and apply to v. This matches
        # the recurrent / prefix-sum view exactly but uses simple matmuls.
        scores = torch.matmul(q, k.transpose(-1, -2))  # [B, T, T]
        causal = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal, 0.0)
        denom = scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = scores / denom
        out = torch.matmul(weights, v)
        return self.out_proj(out)
```
