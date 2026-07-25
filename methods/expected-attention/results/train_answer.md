The problem is KV-cache compression for a frozen, unmodified decoder-only transformer during autoregressive generation. Every processed token stores a key and a value in each layer, so the cache grows linearly with sequence length and soon dominates memory: a 70B model on a one-million-token context can need hundreds of gigabytes just for cached activations. The goal is to keep only a small fixed budget of cached pairs while losing as little generation quality as possible, without retraining and without changing the architecture. The central question is which (k_i, v_i) pairs are safe to evict.

Existing approaches each cover only part of the requirement. Attention-score eviction methods such as H2O, SnapKV and TOVA rank pairs by the attention mass they receive, but that signal comes from past queries, not the future queries that will actually read the compressed cache, and production Flash-Attention kernels never materialize the attention matrix, so the score is unavailable at deployment. Position heuristics such as StreamingLLM keep attention sinks and a recent window; they are cheap and compatible with fused kernels, but they discard the middle of the context blindly and cannot keep a distant fact that happens to be the answer. Norm and embedding heuristics such as KNorm and KeyDiff are also kernel-compatible, yet they score keys by geometric proxies that have no principled tie to a pair's actual effect on the model output. What is missing is a per-pair score that reflects the true contribution to the residual stream, can be computed from the cache at compression time, and works under Flash Attention.

The method I propose is Expected Attention. It starts from the exact additive update that each cached pair makes to one attention head: h_out = h + sum_i a_i W_o v_i, so pair i contributes Δh_i = a_i W_o v_i. The magnitude of that contribution is ||Δh_i|| = a_i · ||W_o v_i||. This is the quantity that should determine eviction: a pair is unimportant when both its attention weight and its value-induced update are small. The value norm ||W_o v_i|| is available immediately from the cached value and the output projection. The difficulty is a_i, which depends on future queries that do not yet exist.

Expected Attention handles the missing attention by predicting it in expectation over future queries. Empirically, the hidden states feeding an attention block in modern LLMs are approximately zero-mean, unimodal and Gaussian. Since the query is a linear transform of the hidden state, q = R W_Q h, the query distribution is also Gaussian. The future contains many positions, each carrying a different RoPE rotation R_t, so the method averages those rotations over a future window of T positions to obtain a single contracting operator R̄ = (1/T) sum_j R_{t+j}. Pushing the Gaussian hidden-state statistics through R̄ W_Q gives one tractable query distribution q̄ ~ N(μ̄_q, Σ̄_q). The averaging is intentional: high-frequency directions whose phase churns across the future window are damped, while slow directions survive, so R̄ captures what a typical future query looks like.

With q̄ Gaussian and a fixed key k_i, the expected unnormalized attention score is a multivariate Gaussian moment-generating function: E[exp(q̄^T k_i / sqrt(d))] = exp(μ̄_q^T k_i / sqrt(d) + k_i^T Σ̄_q k_i / (2d)). The first term is the ordinary attention logit at the mean query; the second is a Jensen boost for keys aligned with high-variance directions of the future-query distribution, ensuring that keys some future queries will attend strongly are not discarded just because the average query is lukewarm. After computing these log expected scores for every key, a softmax over the key dimension yields expected attention weights â_i that sum to one, and the final per-pair score is (â_i + ε) · ||v_i||, using ||v_i|| as a cheap proxy for ||W_o v_i||. The small ε prevents near-zero attention from annihilating the value-norm signal. The first few tokens are attention sinks with massive activations, so they are excluded from the Gaussian statistics to avoid corrupting the mean and covariance, then force-kept by padding their scores above the maximum.

This score can be plugged directly into the standard per-layer eviction hook: compute scores from hidden states, keys and values, keep the top (1 - r) fraction by top-k selection, and gather the kept keys and values along the sequence dimension. It reads no materialized attention matrix, requires no future queries, works for one-shot prefill compression and for repeated trimming during long decoding, and can feed either a uniform per-head budget or an adaptive head-wise budget wrapper.

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
