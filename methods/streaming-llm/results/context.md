# Context

## Research question

Large language models are increasingly deployed in *streaming* settings — a chatbot expected to run over a day-long conversation, an assistant consuming an unbounded input stream — where text arrives continuously and generation must continue indefinitely. We want such a model to keep generating fluent, coherent text over inputs far longer than anything it saw in training, at constant memory and constant per-token latency, forever.

The precise problem: **can we deploy a pretrained LLM on inputs of effectively infinite length without sacrificing efficiency or output quality, and without any fine-tuning?** Two hard constraints make this nontrivial: (1) a Transformer's KV cache grows linearly with the number of tokens seen, so memory and decoding latency blow up without bound; and (2) models have limited length extrapolation — quality degrades once the sequence exceeds the attention window length they were pretrained with (e.g. 4K for Llama-2). A solution must bound the cache (constant memory/latency) *and* keep perplexity stable past the pretraining window.

## Background

**Autoregressive decoding and the KV cache.** A decoder-only Transformer, when generating token by token, caches the Key and Value states of every previous token so each new token can attend to the full history without recomputation. This cache grows with sequence length: unbounded memory and growing latency, the direct cause of constraint (1).

**Length extrapolation fails.** Even ignoring memory, feeding a model a sequence longer than its pretraining attention window degrades quality — positions beyond the trained window are out of distribution. Much work has tried to extend the window (positional-interpolation methods, etc.) or to make long-input training/inference cheaper (FlashAttention), but the usable length remains intrinsically finite, which forbids persistent deployment.

**The softmax normalization constraint.** In attention, the scores over the attended tokens are pushed through a softmax, so the attention weights are forced to sum to one. For scores x₁…x_N, the weight on key i is SoftMax(x)_i = e^{x_i} / Σ_j e^{x_j}. The denominator must be nonzero and the weights must sum to one on every step, for every head.

**Two candidate streaming strategies and how they behave:**

- *Dense (full) attention*: cache everything. Strongest use of available context but O(T) memory and rising latency, with degradation once T exceeds the pretraining window — unusable for streaming.
- *Window attention*: keep only a fixed sliding window of the most recent L tokens' KV; evict the rest. Constant memory and latency once the cache fills — exactly the efficiency we want. But the model *collapses* the instant the sequence exceeds the cache size, i.e. the moment the very first tokens' KV are evicted: perplexity spikes hard — a cliff, not a gentle drift, even though the evicted tokens are the oldest ones, far from what is currently being predicted.
- *Sliding window with recomputation*: rebuild the KV of the recent window from scratch for every generated token. Strong quality, but recomputing quadratic attention over the window per token is far too slow for real streaming.

**Attention-map observations.** Visualizing attention maps across all layers and heads of trained LLMs (Llama-2, MPT, Falcon, Pythia) reveals that, beyond the first couple of layers, a surprisingly large fraction of attention mass is allocated to the *first few tokens of the sequence* — regardless of whether those tokens are semantically relevant. Two probes characterize this behavior: (a) replacing the first four tokens with a meaningless linebreak "\n" token does *not* remove the effect — the model still places high attention on those positions and perplexity is restored — so it is the *absolute position*, not the semantics, that matters; (b) it generally takes about four initial tokens to fully restore window-attention perplexity, while one or two are insufficient. A related observation in the quantization literature noted similar persistent-outlier behavior and motivated a SoftMax-off-by-one variant.

## Baselines

- **Dense attention.** Cache all tokens; full causal attention. Limitation: O(T) memory, growing latency, and quality degradation beyond the pretraining window.

- **Window attention (Beltagy et al. 2020, Longformer-style sliding window).** Keep a fixed window of the L most recent tokens' KV, evict everything older. Constant memory/latency, but perplexity explodes the moment the earliest tokens are evicted — the limitation that demands an explanation.

- **Sliding window with recomputation.** For each new token, recompute the KV cache of the recent window. Recovers quality but recomputes quadratic attention per token — far too slow; the only quality-competitive baseline, and the speed yardstick to beat.

## Evaluation settings

- **Models.** Off-the-shelf pretrained LLMs spanning families and sizes: Llama-2 (7/13/70B), MPT (7/30B), Falcon (7/40B), Pythia (2.8/6.9/12B); positional schemes include RoPE and ALiBi. No fine-tuning. Separately, 160M-parameter models pretrained from scratch under identical settings to test pretraining-time variants.
- **Tasks / data.** Long-text language modeling (perplexity over very long concatenated texts, e.g. PG19/books and similar), with streams extended to million-token scale; streaming question answering / instruction-following over long contexts.
- **Metrics.** Language-modeling perplexity as a function of position/length; decoding latency and memory per token; speedup relative to the sliding-window-with-recomputation baseline.
- **Protocol.** Cache size fixed; sequence streamed token by token well beyond the pretraining window; for the pretraining-variant study, compare vanilla softmax against candidate pretraining-time modifications under matched settings.

## Code framework

The pre-existing pieces: a pretrained decoder-only Transformer with cached Key/Value states (`past_key_values`), positional handling in attention (RoPE rotates query/key pairs; ALiBi adds a distance bias to the scores), and an autoregressive decode loop that appends each step's KV to the cache. The scaffold is a generic KV-cache manager with an empty eviction policy, plus the decode loop.

```python
import torch

class KVCacheManager:
    """Keeps the decoder's KV cache bounded for streaming generation."""
    def __init__(self, cache_size, k_seq_dim=2, v_seq_dim=2):
        self.cache_size = cache_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim

    def evict(self, past_key_values, num_incoming):
        # TODO: when seq_len + num_incoming exceeds cache_size, decide the
        #       eviction policy.
        ...
        return past_key_values

def apply_position(keys, positions):
    # RoPE position handling and ALiBi distance bias live inside attention.
    # TODO: decide how positions are assigned to the cached keys.
    ...

@torch.no_grad()
def stream_generate(model, token_stream, cache_manager):
    past_key_values = None
    for token in token_stream:
        past_key_values = cache_manager.evict(past_key_values, num_incoming=1)
        logits, past_key_values = model(token, past_key_values=past_key_values)
        yield sample(logits)
```
