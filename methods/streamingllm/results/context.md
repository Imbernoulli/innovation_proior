## Research question

A decoder-only Transformer LLM is deployed in a setting where the input never stops: a
multi-round chat assistant that stays up over day-long conversations, a log monitor, a
running transcript. Tokens arrive one after another and the model must keep decoding, in
principle forever. Two hard walls make this impossible with a vanilla model.

First, **memory and latency grow without bound.** Autoregressive decoding caches the Key and
Value tensors of every past token in every attention layer (the KV cache) so that each new
query can attend to all of history without recomputing it. The cache therefore grows linearly
with the number of tokens seen, and the per-token attention cost grows with it. On a stream of
millions of tokens this exhausts GPU memory and the per-token latency climbs until the system
is unusable.

Second, **the model cannot generalize past its pretraining length.** An LLM is trained with a
fixed attention window (e.g. 4K tokens for Llama-2). When the input exceeds that length its
quality degrades sharply, regardless of how the KV cache is managed — the position encodings
and the learned attention patterns were never exercised at those distances.

The precise goal is an inference-time scheme that lets an *already trained* model — no
finetuning — decode from a **fixed-budget** subset of the KV cache (constant memory, constant
per-token latency) while keeping language-modeling quality essentially intact, for sequence
lengths far beyond the pretraining window, ideally without bound. The scheme must work with
the relative position encodings real LLMs use (RoPE, ALiBi) and must run on standard dense
attention kernels, not a bespoke sparse-attention retraining. Each existing approach below
achieves a subset of these; none achieves all at once.

## Background

By this time decoder-only Transformer LLMs (GPT-family, OPT, Llama, Llama-2, MPT, Falcon,
Pythia) are the substrate for chat assistants, summarization, code completion, and QA. The
dominant serving recipe is dense causal attention with a growing KV cache, and the field has
poured effort into pushing the *usable context length* up — better training-time efficiency
(FlashAttention), approximate/sparse attention, and position-encoding tricks to extrapolate
beyond the trained window — but the acceptable length stays intrinsically finite, which rules
out persistent deployment.

The load-bearing concepts:

- **The KV cache and causal attention.** In a causal Transformer, token *t*'s query attends
  only to keys/values at positions `<= t`. Caching those K,V avoids recomputing them, so
  decoding is O(1) compute per token *given* the cache — but the cache itself is O(t) in size.

- **The softmax normalization constraint.** Within a head, attention weights are
  `SoftMax(x)_i = e^{x_i} / sum_j e^{x_j}`, and they are forced to sum to one over all attended
  tokens. There is no "abstain" option: even when a query has no strong match anywhere in its
  context, the leftover probability mass must be placed on *some* tokens. A related line of work
  on quantization had already observed persistent **outlier activations** and traced them to
  exactly this forced-allocation behavior of softmax (SmoothQuant, Xiao et al. 2023;
  "Quantizable Transformers", Bondarenko et al. 2023), prompting the proposal of a
  "SoftMax-off-by-one" variant `SoftMax_1(x)_i = e^{x_i} / (1 + sum_j e^{x_j})` (Miller 2023)
  whose extra `+1` lets the weights sum to less than one.

- **Relative position encodings.** RoPE (Su et al. 2021) rotates the query and key of each head
  by an angle proportional to absolute position: `q_m = R_{Θ,m} W_q x_m`, `k_n = R_{Θ,n} W_k x_n`,
  with `R_{Θ,m}` a block-diagonal rotation by frequencies `θ_i = 10000^{-2(i-1)/d}`. The rotation
  matrix is orthogonal and additive in the position index, `R_{Θ,m} R_{Θ,m'} = R_{Θ,m+m'}` and
  `R_{Θ,m}^T = R_{Θ,-m}`, so the query-key inner product depends only on the *relative* position:
  `q_m^T k_n = x_m^T W_q^T R_{Θ, n-m} W_k x_n`. A numerically efficient realization is
  `R_{Θ,m} x = x ⊙ cos(m θ) + rotate_half(x) ⊙ sin(m θ)`, with
  `rotate_half([x_1, x_2]) = [-x_2, x_1]`. ALiBi (Press et al. 2022) instead adds a
  distance-proportional bias `-slope * (m - n)` directly to the attention logits. Both inject
  relative position; both were observed to degrade when the test length far exceeds the training
  length.

- **The diagnostic findings that define the problem.** Two measured phenomena about existing
  systems are the empirical heart of this context:
  - *Window attention fails as a cliff, not a slope.* Plotting language-modeling perplexity
    against position on a long text, a model that keeps only a fixed recent window has flat,
    healthy perplexity until the sequence length reaches the cache size — and then spikes
    abruptly at the exact step where the **first** token is evicted. The collapse coincides with
    evicting the *initial* tokens specifically, not with losing recent context.
  - *Models pour attention onto the initial tokens regardless of content.* Visualizing attention
    maps across all layers and heads of Llama-2-7B/70B, MPT, Falcon, and Pythia, beyond the
    bottom couple of layers almost every head concentrates a large fraction of its attention mass
    on the first few token positions, irrespective of what those tokens are. Quantitatively, on
    sequences of length 4096 averaged over hundreds of samples, the attention from the last token
    to the very first token often exceeds half of the total mass in most layers. Substituting the
    first four tokens with a linebreak token does not move this concentration, and reintroducing
    them restores perplexity — so it is the *absolute position* of those slots, not their
    semantic content, that the model depends on. A concurrent observation in Vision Transformers
    (Darcet et al. 2023) found the same attention concentration on uninformative patch tokens,
    handled by adding dedicated "register" tokens.

## Baselines

The prior methods a fixed-budget streaming scheme is measured against and reacts to.

**Dense (full) attention.** Cache every past token's KV; each query attends to all of history.
Highest quality inside the trained window. *Gap:* cache size and per-token latency grow without
bound (out-of-memory on long streams), and quality still collapses once the input passes the
pretraining window length, so it cannot stream.

**Window (sliding) attention** (Beltagy et al. 2020, as used for streaming). Keep only the KV
of the most recent `L` tokens; evict the oldest as new ones arrive. Memory and per-token latency
are constant once the window is full, and it scales linearly with sequence length. *Gap:* the
moment the sequence outgrows the cache — i.e. the very first token is evicted — perplexity
spikes and generation degenerates. The failure is tied specifically to dropping the *initial*
tokens; keeping a fixed window of recent tokens is not enough to preserve the attention
computation.

**Sliding window with re-computation.** For each generated token, rebuild the KV states of the
recent window from scratch so that the recent tokens are always re-encoded with valid positions.
Quality is strong — it matches dense attention within the window and serves as the practical
quality oracle. *Gap:* it recomputes O(L^2) attention within the window per generated token, so
decoding latency rises quadratically with the window size, which is impractically slow for
real-time streaming.

**Heavy-hitter KV eviction (H2O, Zhang et al. 2023).** Observe that *accumulated* attention
scores across a generation follow a power law, so a small set of "heavy-hitter" tokens carries
most of the influence; keep the top-K tokens by accumulated attention score together with a
recent window, and evict the rest. Reduces the cache while preserving the influential tokens.
*Gap:* the eviction decision is driven by the realized attention matrix, so it needs per-step
attention scores to score tokens, and it is designed to compress within a window rather than to
extrapolate to unbounded length; it adds dynamic, attention-dependent machinery on top of the
decode loop.

**Length-extrapolation position encodings** (RoPE/ALiBi and their extensions: position
interpolation, Chen et al. 2023; NTK-aware scaling; YaRN, Peng et al. 2023). Modify or rescale
the position encoding so the model tolerates longer inputs. *Gap:* they extend the usable window
to a limited, still-finite extent, and extending the window does not by itself give constant
memory/latency or unbounded streaming; they are orthogonal to the cache-budget problem and
could be combined with a streaming scheme rather than replacing it.

## Evaluation settings

The natural yardsticks already in use, all predating any fixed-budget streaming scheme:

- **Long-text language modeling perplexity.** Concatenate a long-document corpus
  (e.g. the PG19 test set, 100 long books) into one stream and report perplexity as a function
  of position, with a fixed cache budget (e.g. 2048 for Llama-2, 1024 for Falcon/Pythia/MPT,
  about half the pretraining window for clarity). The position-resolved curve is what exposes the
  window-attention cliff and whether a fixed-cache decode remains stable as the stream grows.
- **Model families and scales.** Llama-2-[7,13,70]B and Falcon/Pythia (RoPE) plus MPT (ALiBi),
  to test that any conclusion holds across both dominant relative-position schemes and across
  model size.
- **Streaming question answering with instruction-tuned models.** Concatenate many QA pairs
  (e.g. ARC-Challenge/Easy) into one continuous stream fed to a chat model, scoring each answer
  by exact match; and a streaming-eval protocol where a query is issued periodically and its
  answer lies a fixed number of lines earlier, reflecting questions about recent context.
- **Long-range NLP benchmarks.** LongBench (Bai et al. 2023) — single-/multi-document QA and
  summarization — with the standard middle-truncation protocol that preserves the beginning and
  end of an over-length input.
- **Decoding efficiency.** Per-token decoding latency and peak memory vs cache size, measured in
  the Hugging Face Transformers library on a single GPU, against the recomputation baseline.
- **Controlled small-model pretraining protocol.** Train small (160M-parameter) models from scratch on a deduplicated
  large text corpus (the Pile) under a fixed recipe, to compare standard pretraining against
  variants, evaluated by training-loss convergence, zero-shot accuracy across standard NLP
  benchmarks (ARC, HellaSwag, LAMBADA, OpenBookQA, PIQA, Winogrande), and streaming perplexity.

## Code framework

The scheme plugs into a shared full-attention Hugging Face decoding harness. The harness owns
the model, the datasets, the prompt templates, the decode loop, the cache budget, and the
measurement. After the prefill pass it hands each layer's prefill KV tensors to a *selection
policy* whose job is to score the cached tokens and return the subset to retain. What that
scoring rule and that selection step should be is exactly what is undecided — the policy below
is empty stubs.

```python
import torch


class SelectionPolicy:
    """Decides which prefill KV entries to keep under a fixed cache budget.

    The harness calls retention_plan once per layer to get the policy's metadata,
    then score_tokens to rank the cached tokens, then select_cache to pick n_kept
    of them. keys/values have shape (bsz, n_kv_heads, k_len, head_dim); the harness
    enforces the budget (n_kept) at the call site.
    """

    method_name = "selection_policy"

    def retention_plan(self, layer_id, request_meta, cache_meta):
        # Per-layer metadata: at minimum the harness-supplied compression_ratio.
        # TODO: any additional policy metadata.
        return {"method": self.method_name,
                "compression_ratio": cache_meta["compression_ratio"]}

    def score_tokens(self, module, hidden_states, keys, values, kwargs, plan):
        # Return a score per cached token; the harness keeps the highest-scoring n_kept.
        # TODO: assign per-token scores.
        pass

    def select_cache(self, module, keys, values, scores, n_kept):
        # Keep n_kept tokens according to scores; return the pruned (keys, values).
        # TODO: gather the retained entries.
        pass


# existing decode harness the policy plugs into (sketch)
def decode_with_selection(model, policy, stream, compression_ratio):
    cache = []                                  # growing KV cache
    for layer_id, (keys, values, hs, kwargs, module) in enumerate(prefill(model, stream)):
        plan = policy.retention_plan(layer_id, request_meta(stream), {"compression_ratio": compression_ratio})
        scores = policy.score_tokens(module, hs, keys, values, kwargs, plan)
        n_kept = int(keys.shape[2] * (1.0 - plan["compression_ratio"]))
        keys, values = policy.select_cache(module, keys, values, scores, n_kept)
        cache.append((keys, values))
        # ... harness continues the decode loop attending over the pruned cache ...
```

The harness supplies the full prefill KV and a budget; `score_tokens` and `select_cache` are
the empty policy slots.
