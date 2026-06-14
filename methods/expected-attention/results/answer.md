# Expected Attention, distilled

Expected Attention is a training-free KV-cache compression method. It evicts the cached
`(k_i, v_i)` pairs that contribute least to the transformer's output, where each pair's
contribution is its additive update to the residual stream, `||Δh_i|| = a_i · ||W_o v_i||`.
The obstruction — the attention weight `a_i` depends on future queries that don't exist at
compression time and that Flash-Attention kernels never materialize — is solved by *predicting*
it: hidden states in modern LLMs are approximately Gaussian, so queries are Gaussian, and the
expected unnormalized attention to each key has a closed form via the Gaussian
moment-generating function. Rank pairs by the expected contribution and keep the top budget.

## Problem it solves

Compress the KV cache of a frozen, unmodified transformer for long-context inference — drop a
subset of cached key-value pairs to fit a fixed budget with minimal quality loss — using a
score that (a) reflects each pair's true effect on the model output, (b) is computable at
compression time without future queries, (c) needs no materialized attention matrix
(Flash-Attention compatible), and (d) works for both one-shot prefill compression and
streaming-decode compression.

## Key idea

1. **Exact importance from the residual stream.** With `h_out = h + sum_i a_i W_o v_i`, pair
   `i` adds `Δh_i = a_i W_o v_i`, so its importance is `||Δh_i|| = a_i · ||W_o v_i||`.
2. **The attention factor is unknowable directly**, so estimate it in expectation over future
   queries. Hidden states are approximately `h ~ N(mu, Sigma)`; since `q = R W_Q h` is linear,
   `q ~ N(R W_Q mu, R W_Q Sigma W_Q^T R^T)`.
3. **Average RoPE over the future window** to get one tractable query distribution:
   `R̄ = (1/T) sum_{j=1}^T R_{t+j}` (an average of rotations — a contraction, not a rotation),
   giving `q̄ ~ N(μ̄_q, Σ̄_q)` with `μ̄_q = R̄ W_Q mu`, `Σ̄_q = R̄ W_Q Sigma W_Q^T R̄^T`.
4. **Closed-form expected score via the Gaussian MGF.** For `q̄ ~ N(μ̄_q, Σ̄_q)` and a fixed key,
   using `E[exp(s^T X)] = exp(s^T m + ½ s^T C s)` with `s = k_i/sqrt(d)`:
   ```
   ẑ_i = E[ exp(q̄^T k_i / sqrt(d)) ] = exp( μ̄_q^T k_i / sqrt(d) + k_i^T Σ̄_q k_i / (2d) ).
   ```
   The mean term is the ordinary attention logit at the mean query; the covariance term is the
   Jensen boost — keys aligned with high-variance future-query directions score higher.
5. **Normalize and weight by value norm.** `â_i = ẑ_i / sum_j ẑ_j`, then the expected
   contribution magnitude is `||Δĥ_i|| = (â_i + ε) · ||W_o v_i||`. The implementation uses
   `||v_i||` as a cheaper value-magnitude proxy, avoiding an output projection for every cached
   value.
6. **Evict the lowest-scoring pairs**; keep the top `(1 - r)` fraction. The same scores can also
   feed an adaptive head-wise budget wrapper; the score formula itself stays unchanged.

## Defaults and why

- `n_future_positions T = 512` — averages the rotation over a future generation horizon, so one
  query distribution stands in for many future positions instead of committing to one offset.
- `n_sink = 4` — the first tokens are massive-activation / attention-sink outliers: **excluded
  from the mean/covariance** (they would corrupt the Gaussian fit) and **force-kept in the
  cache** (the model relies on them regardless of content) by padding their score above the max.
- `use_covariance = True` — keeps the second-order MGF term; setting it False gives the cheaper
  mean-only estimate `ẑ_i ≈ exp(μ̄_q^T k_i / sqrt(d))`.
- `use_vnorm = True` — keeps the value-magnitude factor corresponding to the `||W_o v_i||`
  term, implemented as `||v_i||` to avoid the projection cost.
- `epsilon` defaults to `0.0`; a small positive value such as `0.02` floors near-zero expected
  attention so the value norm still orders the otherwise-ignored keys, while `0.0` lets attention
  dominate fully.
- Statistics come from the hidden states on hand (prompt for prefill; a small rolling buffer for
  decode); queries are taken **pre-RoPE** (`W_Q h`), with RoPE applied analytically via `R̄`.

## Final algorithm

```
exclude first n_sink tokens from statistics
q = W_Q h                                            # pre-RoPE queries, Gaussian
mu  = mean_s(q);  Sigma = cov_s(q)                   # per (batch, head)
R-bar = mean_{j=1..T} R_{t+j}                        # averaged future RoPE
mu  <- R-bar @ mu ;  Sigma <- R-bar @ Sigma @ R-bar^T
for each cached key k_i:
    log_z_i = mu^T k_i / sqrt(d) + k_i^T Sigma k_i / (2d)      # log Gaussian MGF
a_i = softmax_i(log_z_i)                              # same as z_i / sum_j z_j
score_i = (a_i + eps) * ||v_i||                      # residual-stream contribution proxy
score(sink tokens) = max(score) + 1                  # never evict sinks
keep the top (1 - r) fraction of pairs by score; evict the rest
```

## Working code

Filling the `score` slot of the per-layer eviction hook:

```python
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv

from kvpress.presses.scorer_press import ScorerPress
from kvpress.utils import get_prerope_query_states


@dataclass
class ExpectedAttentionPress(ScorerPress):
    """Score KV pairs by expected residual-stream contribution."""

    compression_ratio: float = 0.0
    n_future_positions: int = 512
    n_sink: int = 4
    use_covariance: bool = True
    use_vnorm: bool = True
    epsilon: float = 0.0

    def apply_avg_rope(self, module: nn.Module, mu: torch.Tensor, cov: torch.Tensor, q_len: int):
        # R-bar = (1/T) sum_{j=1..T} R_{t+j}; push query mean/cov through it.
        pos = torch.arange(q_len, q_len + self.n_future_positions, device=mu.device).unsqueeze(0)
        head_dim = module.head_dim
        cos, sin = module.rotary_emb(mu, pos)
        cos, sin = cos[0], sin[0]
        Id = torch.eye(head_dim, device=cos.device, dtype=cos.dtype)
        P = torch.zeros((head_dim, head_dim), device=cos.device, dtype=cos.dtype)
        half = head_dim // 2
        eye_half = torch.eye(half, device=cos.device, dtype=cos.dtype)
        P[half:, :half] = eye_half                         # first half moves to second half
        P[:half, half:] = -eye_half                        # second half moves to first with sign flip
        R = cos.unsqueeze(1) * Id + sin.unsqueeze(1) * P   # per-future-position rotation
        R = R.mean(dim=0).to(mu.device)                    # averaged -> a contraction
        mu = torch.matmul(mu, R.T)
        if cov is not None:
            cov = torch.matmul(R, torch.matmul(cov, R.T))
        return mu, cov

    def get_query_statistics(self, module: nn.Module, hidden_states: torch.Tensor):
        q_len = hidden_states.shape[1]
        h = hidden_states[:, self.n_sink:]                 # drop sink outliers from stats
        query_states = get_prerope_query_states(module, h)
        mu = query_states.mean(dim=2, keepdim=True)
        cov = None
        if self.use_covariance:
            centered = query_states - mu
            cov = torch.einsum("bnsi,bnsj->bnij", centered, centered) / h.shape[1]
        mu = mu.squeeze(2)
        return self.apply_avg_rope(module, mu, cov, q_len)

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        assert keys.size(2) > self.n_sink, f"Input should contain more tokens than n_sink={self.n_sink}"
        keys = keys[:, :, self.n_sink:]
        values = values[:, :, self.n_sink:]

        mean_query, cov_query = self.get_query_statistics(module, hidden_states)

        bsz, num_key_value_heads, q_len, d = keys.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads
        keys = repeat_kv(keys, num_key_value_groups).transpose(2, 3)

        # log E[exp(q^T k / sqrt(d))] = mu^T k / sqrt(d) + k^T Sigma k / (2d)
        log_scores = torch.matmul(mean_query.unsqueeze(2), keys).squeeze(2) / math.sqrt(d)
        if self.use_covariance:
            log_scores += torch.einsum("bhin,bhij,bhjn->bhn", keys, cov_query, keys) / d / 2
        scores = F.softmax(log_scores, dim=-1)             # softmax over keys

        scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, q_len)
        scores = scores.mean(dim=2)                        # average query heads sharing a KV head

        if self.use_vnorm:
            scores = (scores + self.epsilon) * values.norm(dim=-1)

        # Re-attach sinks with top score so the scorer path keeps them.
        return F.pad(scores, (self.n_sink, 0), value=scores.max().item() + 1)
```

The eviction step is then the generic keep-highest top-`k`: `n_kept = int(seq * (1 - r))`,
`idx = scores.topk(n_kept, dim=-1).indices`, gather keys/values along the sequence axis.
