**Problem.** Linear attention gave the mixer an explicit query-key score but a *gentle* kernel: the
matching value is diluted by a high-entropy smear of every other stored value, so accuracy collapses as the
key count grows (0.813 → 0.368 → 0.163). Recall needs a *spiky* kernel that concentrates weight on the one
matching key.

**Key idea (BASED — Taylor-2 linear attention).** Keep the factored sub-quadratic structure, swap the
feature map for one that tracks `exp(q^T k)`: the second-order Taylor kernel
`k(q,k) = 1 + q^T k + (q^T k)²/2`, which equals `φ(q)^T φ(k)` for the explicit map `φ(x) = [1, x, (x⊗x)/√2]`.
It is strictly positive (`= ½((q^Tk+1)²+1) ≥ ½`) and grows *quadratically* in `q^T k`, so a matching key's
weight blows up past the rest — softmax-like spikiness from a finite deterministic feature map, no random
noise. Scale `q, k` by `feature_dim^{-1/2}` to keep `q^T k` in the radius where dropping `x³/6` is benign,
and keep `feature_dim` small (16) so the expansion stays cheap. Pair the spiky *global* read with a cheap
*local* short causal conv (used both to feed locally-shifted features in and as an output residual), since a
global average is blunt at the fine local shifts recall also needs.

**Why it beats step 2 but not the ceiling.** The quadratic sharpening closes the selectivity gap at moderate
scale (near-perfect at 128/512). But at `mqar-2048` the fixed-shape state must still hold 128 bindings, and
the recall-state floor — exact recall of *all* bindings needs state that grows with their number — bites, so
it stays far below the short settings. Only full softmax attention, paying `N×N` cost so it never compresses
the bindings, is expected to hold accuracy there.

**Hyperparameters.** `feature_dim=16`, `scale = feature_dim^{-1/2}`; bias-free `q/k/v/out` projections; an
internal depthwise causal conv (`kernel_size=3`, left-padded); normalizer = masked row-sum of the full
Taylor-2 kernel, clamped at `1e-6`; quadratic training view with a causal `tril` mask.

```python
# EDITABLE region of custom_strategy.py — step 3: BASED (Taylor-2 spiky linear attention)
class CustomMixer(nn.Module):
    """BASED-style short convolution + 2nd-order Taylor linear attention.

    Implementation uses BASED's `train_view="quadratic"` formulation: at
    training time we materialise the T x T attention matrix using the
    Taylor-2 kernel k(q,k) = 1 + q^T k + (q^T k)^2 / 2 (which equals
    <phi(q), phi(k)>) and apply a causal mask. This is mathematically
    identical to the recurrent / cumulative-sum view but uses memory
    O(B*T^2) instead of O(B*T*F*D), which is what the official BASED
    repo also does during training (see
    https://github.com/HazyResearch/zoology/blob/main/zoology/mixers/based.py
    `train_view`).
    """

    def __init__(self, d_model: int, seq_len: int, feature_dim: int = 16):
        super().__init__()
        self.d_model = d_model
        self.feature_dim = feature_dim
        self.q_proj = nn.Linear(d_model, feature_dim, bias=False)
        self.k_proj = nn.Linear(d_model, feature_dim, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.conv_kernel_size = 3
        self.local_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=self.conv_kernel_size,
            groups=d_model,
            padding=0,
            bias=True,
        )
        # Scale q,k so q^T k stays in the radius of convergence of exp().
        self.scale = feature_dim ** -0.5

    def short_conv(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        h = F.pad(h, (self.conv_kernel_size - 1, 0))
        h = self.local_conv(h)
        return h.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        local = self.short_conv(x)
        h = x + local
        q = self.q_proj(h) * self.scale
        k = self.k_proj(h) * self.scale
        v = self.v_proj(h)
        # Taylor-2 kernel score: 1 + q^T k + (q^T k)^2 / 2
        qk = torch.matmul(q, k.transpose(-1, -2))           # [B, T, T]
        kernel = 1.0 + qk + 0.5 * qk * qk                   # spiky, non-negative
        causal = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
        kernel = kernel.masked_fill(~causal, 0.0)
        denom = kernel.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = kernel / denom
        out = torch.matmul(weights, v)                      # [B, T, D]
        return self.out_proj(out) + local
```
