# StreamingLLM (Attention Sinks)

## Problem

Deploy a pretrained LLM on effectively infinite-length input streams at constant memory and per-token latency, without fine-tuning, and without the quality collapse that hits once the sequence exceeds the pretraining attention window. The KV cache grows linearly with tokens seen (unbounded memory/latency); window attention bounds it but collapses the moment the first tokens are evicted.

## Key idea

Trained LLMs allocate a large, content-independent share of attention to the first few tokens — **attention sinks**. Cause: softmax forces attention weights to sum to one, so the model must dump surplus attention somewhere; under causal masking the initial tokens are visible to every later token, so they get trained to absorb it. Evicting them removes a big part of the softmax denominator and pushes the attention distribution out of distribution — that is why window attention breaks. Fix: **keep the first few tokens' KV permanently (the sinks) alongside the rolling recent window**, and assign positions *within the cache* so they never exceed the trained range.

## Final method (inference, no fine-tuning)

- **Cache layout:** retain `start_size` initial sink tokens (≈4 suffices; 1–2 do not, because these models had no single consistent start token in pretraining) + the most recent `recent_size` tokens; evict the middle. Constant memory and latency.
- **Positions within the cache:** if the cache holds text positions [0,1,2,3,6,7,8] and decodes the next token, assign positions [0,1,2,3,4,5,6] then 7 — contiguous, bounded by cache size, always in-distribution. Requires relative positional encoding (RoPE/ALiBi). For RoPE, cache keys *before* rotation and rotate by cache-position at each step; for ALiBi, apply a contiguous (not jumping) linear bias.

## Pretraining-time variants (for future models)

- **Sink Token:** prepend one dedicated learnable token to every training sample; a single sink then suffices for streaming.
- **Zero Sink / SoftMax₁:** SoftMax₁(x)_i = e^{x_i} / (1 + Σ_j e^{x_j}) lets weights sum to <1, absorbing surplus mass — equivalent to prepending a token with all-zero Key and Value.

## Code

```python
import torch

class StartRecentKVCache:
    """Keep `start_size` sink tokens at the front + the last `recent_size` tokens."""
    def __init__(self, start_size=4, recent_size=2000, k_seq_dim=2, v_seq_dim=2):
        self.start_size, self.recent_size = start_size, recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim, self.v_seq_dim = k_seq_dim, v_seq_dim
        self.k_slice = lambda x, a, b: x[:, :, a:b, ...]
        self.v_slice = lambda x, a, b: x[:, :, a:b, ...]

    def __call__(self, past_key_values):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values
        return [
            [torch.cat([self.k_slice(k, 0, self.start_size),
                        self.k_slice(k, seq_len - self.recent_size, seq_len)], self.k_seq_dim),
             torch.cat([self.v_slice(v, 0, self.start_size),
                        self.v_slice(v, seq_len - self.recent_size, seq_len)], self.v_seq_dim)]
            for k, v in past_key_values
        ]

    def evict_for_space(self, past_key_values, num_coming):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values
        end_recent = seq_len - self.recent_size + num_coming
        return [
            [torch.cat([self.k_slice(k, 0, self.start_size),
                        self.k_slice(k, end_recent, seq_len)], self.k_seq_dim),
             torch.cat([self.v_slice(v, 0, self.start_size),
                        self.v_slice(v, end_recent, seq_len)], self.v_seq_dim)]
            for k, v in past_key_values
        ]

@torch.no_grad()
def streaming_generate(model, token_stream, kv_cache):
    past = None
    for token in token_stream:
        past = kv_cache.evict_for_space(past, num_coming=1)
        # positions assigned within the (bounded) cache; RoPE keys cached pre-rotation
        logits, past = model(token, past_key_values=past, use_cache=True)
        yield sample(logits)
```

This is the StreamingLLM `StartRecentKVCache`: keep 4 sink tokens + a recent window, with cache-relative positions, letting models like Llama-2, MPT, Falcon, and Pythia model millions of tokens at constant cost with no fine-tuning.
