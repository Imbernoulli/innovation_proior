# LagKV, distilled

LagKV is an attention-free, query-independent KV-cache token-eviction rule. It scores each
cached prefill token by how *incoherent* it is with the lag window that immediately follows it,
using only the cached Key and Value tensors — never the attention weights and never the query —
and keeps the highest-scoring tokens plus a fixed attention-sink and a recent sliding window.
Because it touches only K and V, it is compatible with FlashAttention and reusable across
turns/instructions.

## Problem it solves

Long-context inference makes the KV cache dominate both memory (linear in sequence length) and
compute (decode attends over the whole cache each step; prefill is quadratic). The goal is to
keep, after prefill, only a small fixed-budget subset of cached tokens without losing task
quality, subject to three constraints the dominant prior art violates: (1) no attention-matrix
access (FlashAttention never materializes it), ruling out H2O / SnapKV scoring; (2) query
independence, so the compressed context is reusable; (3) an actual compute reduction (drop
tokens), which quantization does not give since it keeps every token.

## Key idea

The model is autoregressive, so adjacent KV vectors barely change (token-wise locality). Hence
the *next* contiguous chunk is a faithful external reference for the current chunk's local
distribution. Normalize the current chunk's K and V channel-by-channel using the *next* chunk's
per-channel min/max (taken over the token axis); this removes the persistent per-channel key
outliers (KIVI) and re-expresses each token in the next chunk's frame. The surviving
channel-wise standard deviation of a normalized token measures how off-distribution — how
incoherent with what comes next — that token is; high std = a discontinuity carrying
information that local flow does not predict = keep it. Softmax over the window's tokens
separates outliers; summing the Key and Value scores uses both axes of structure (keys are
channel-organized, values token-organized). Attention sinks and the final reference-less chunk
(the recent sliding window) are always kept.

## Method

Partition the cache after the first `S` sink tokens into contiguous windows of size `L` (the
lag). For window `p`, head `i`, and `Z ∈ {K, V}` (a chunk of `L` tokens × `d_h` channels),
reference the *next* window `Z_i^{p+1}` and compute per channel, over the token (`seq`) axis:

```
min_i^{p,Z} = min_seq( Z_i^{p+1} ),        max_i^{p,Z} = max_seq( Z_i^{p+1} )
Zbar_i^p    = ( Z_i^p - min_i^{p,Z} ) / ( max_i^{p,Z} - min_i^{p,Z} )      # channel-wise normalize
score(Z_i^p) = Softmax_tokens( Std_channel( Zbar_i^p ) )                   # one scalar per token
score_i^p   = score(K_i^p) + score(V_i^p)
```

The code below uses `(key_score + value_score) / 2`; multiplying by a positive constant does not
change the top-K order or the within-window rank order.

Keep the top `rL` tokens of each window (per head). The sinks (first `S`) and the last full
window plus the remainder (the sliding window; the last window has no next chunk to reference)
are always kept. Skip compression when the sequence is shorter than `S + 2L`.

Compression-ratio bookkeeping for sequence length `L_s ≥ S + 2L`, with per-window retention `r`:

```
L_R = S + rL ( floor((L_s - S)/L) - 1 ) + L + Mod(L_s - S, L)
C   = 1 - L_R / L_s
```

(`C = 0` for `L_s < S + 2L`.) The `+ L + Mod(...)` is the always-kept sliding tail.

**Defaults:** `n_sink = 4`, `lag_size L = 128`, `cross_scoring = False`. Choose `L` short enough
that token-wise locality is plausible and large enough to estimate per-channel min/max; choose
`r` so `rL` can cover the contiguous information span the application needs to preserve.

## Why a single global top-K equals per-partition top-K

The harness applies one global `topk(n_kept)` over the whole cache, but the method wants a
per-partition quota. With `cross_scoring = False`, replace each window's scores by their
within-window rank, normalized by `L`:

```
score = score.argsort(dim=-1).argsort(dim=-1) / L      # uniform in [0,1) per window
```

Every window now has the identical rank distribution `{0, 1/L, ..., (L-1)/L}`. When `C` is set
from the retained-length formula, the sinks and tail take the guaranteed `1.0` slots first, and
the remaining scored-token budget is the aligned integer `rL` per window; a global top-K then
keeps exactly `rL` from each scored window. `cross_scoring = True` skips the rank step and lets
budget flow toward windows with larger raw outliers.

## Why each choice

- **Next-chunk reference (not own-chunk):** locality makes the next chunk a valid *external*
  yardstick, measuring genuine incoherence; own-chunk min/max is self-referential and pins the
  envelope-defining tokens to 0/1.
- **Min-max along the token axis, per channel:** removes KIVI's persistent large-magnitude key
  channels, which would otherwise dominate the std.
- **Channel-wise std as importance:** after channel norms are gone, residual spread =
  off-distribution = informative.
- **Sum K and V scores:** keys are channel-organized, values token-organized; each catches
  different discontinuities.
- **Softmax over window tokens:** common per-window scale + outlier separation.
- **Sinks + reference-less last window:** StreamingLLM anchor and recent window, kept for free.

## Working code

Filling the `score()` slot of the score-then-keep eviction harness; the base class supplies the
global top-K and the gather.

```python
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ScorerPress:
    """Base: score tokens, keep the top n_kept globally. Subclass supplies score()."""

    compression_ratio: float = 0.0

    def __post_init__(self):
        assert 0 <= self.compression_ratio < 1, "Compression ratio must be between 0 and 1"

    def score(self, module: nn.Module, hidden_states, keys, values, attentions, kwargs) -> torch.Tensor:
        raise NotImplementedError

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        if self.compression_ratio == 0:
            return keys, values
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
        k_len = keys.shape[2]
        n_kept = int(k_len * (1 - self.compression_ratio))
        indices = scores.topk(n_kept, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()
        return keys, values


@dataclass
class LagKVPress(ScorerPress):
    """LagKV: lag-relative, attention-free, query-free KV eviction score."""

    compression_ratio: float = 0.0
    n_sink: int = 4
    lag_size: int = 128
    cross_scoring: bool = False

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        bsz, num_key_value_heads, q_len, d = keys.shape

        if q_len < self.n_sink + 2 * self.lag_size:
            # too short for a scored chunk + a reference chunk: skip compression,
            # but ramp the post-sink scores so any forced top-K keeps the recent tail.
            score = torch.ones((bsz, num_key_value_heads, q_len), dtype=keys.dtype, device=keys.device)
            if q_len > self.n_sink:
                score[:, :, self.n_sink:] = (
                    torch.arange(q_len - self.n_sink, device=keys.device) / (q_len - self.n_sink)
                ).to(keys.dtype)
            return score

        end_idx = self.n_sink + ((q_len - self.n_sink) // self.lag_size) * self.lag_size
        tail_len = self.lag_size + q_len - end_idx   # last full window + remainder = sliding tail

        key_score = self._get_states_score(
            keys[:, :, self.n_sink:end_idx].view(bsz, num_key_value_heads, -1, self.lag_size, d))
        value_score = self._get_states_score(
            values[:, :, self.n_sink:end_idx].view(bsz, num_key_value_heads, -1, self.lag_size, d))
        score = (key_score + value_score) / 2        # same ordering as summed K+V

        if not self.cross_scoring:
            # within-window rank / L: under the aligned budget,
            # one global top-K becomes a per-window top-K at ratio r
            score = score.argsort(dim=-1).argsort(dim=-1) / self.lag_size
            score = score.to(keys.dtype)

        sink_score = torch.ones((bsz, num_key_value_heads, self.n_sink), dtype=score.dtype, device=score.device)
        tail_score = torch.ones((bsz, num_key_value_heads, tail_len), dtype=score.dtype, device=score.device)
        return torch.cat((sink_score, score.reshape(bsz, num_key_value_heads, -1), tail_score), dim=-1)

    def _get_states_score(self, target_v):
        # target_v: (b, h, num_partitions, L, d_h). Score chunk p by chunk p+1's stats (the lag).
        ref = target_v[:, :, 1:, :, :]    # next chunk = reference
        v = target_v[:, :, :-1, :, :]     # current chunk = scored
        min_r = ref.min(dim=-2).values.unsqueeze(-2).expand(-1, -1, -1, self.lag_size, -1)
        max_r = ref.max(dim=-2).values.unsqueeze(-2).expand(-1, -1, -1, self.lag_size, -1)
        # channel-wise normalize into the next chunk's frame; channel-wise std per token; softmax over tokens
        return ((v - min_r) / (max_r - min_r)).std(dim=-1).softmax(dim=-1)
```

The score uses only `keys` and `values` (the `attentions` argument is never read) and no query
state, so it runs under FlashAttention and the same compressed cache serves any later query.
