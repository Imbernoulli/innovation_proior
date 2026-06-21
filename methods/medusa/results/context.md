# Context

## Research question

Large language models keep getting better as they grow, but bigger models make *inference latency* worse — and latency, not training cost, is what users feel when an LLM is hosted for interactive, single-request use. The bottleneck is structural: autoregressive decoding emits one token per forward pass, and each forward pass must stream the entire model's parameters from high-bandwidth memory to the accelerator. Generating a single token after moving gigabytes of weights leaves the accelerator's arithmetic units almost idle — LLM decoding is **memory-bandwidth-bound**, not compute-bound.

The question: how do we generate multiple tokens per forward pass, raising arithmetic intensity (FLOPs per byte moved) and cutting the number of sequential decode steps, without changing the model's output distribution?

## Background

**Why decoding is slow.** Each autoregressive step does one forward pass to produce one token; the pass is dominated by moving the model weights from HBM. The per-token cost is set by memory bandwidth, and the abundant FLOPs of the accelerator go unused. A single weight-load that already pays the memory cost has spare arithmetic capacity to spend on several token positions at once.

**Speculative decoding.** A small, fast **draft model** autoregressively guesses the next several tokens, then the large **target model** runs once over all the guesses in parallel — one forward pass verifies many positions, because a single pass with a longer input is still one weight-load. Accepted guesses are committed; the first rejection truncates. With **rejection sampling**, the accepted tokens are provably distributed exactly as the target model would have sampled them — output distribution preserved. Speedup comes from the target model confirming several draft tokens per pass instead of generating one.

**Parallel decoding with extra heads.** Earlier work (Stern et al. 2018, blockwise parallel decoding) attached extra output heads to a single model to predict several future positions at once, for tasks like translation and image super-resolution. Multi-position predictions come from one backbone with no second model.

**Rejection sampling.** Rejection sampling guarantees the target's exact distribution by comparing the draft and target probability for each candidate token and accepting with probability min(1, p_target / p_draft). It is used to achieve lossless speculative decoding.

**Truncation sampling.** A line of decoding work (e.g. truncation/typical sampling, Hewitt et al. 2022) accepts only tokens that are not *too improbable* under the model — thresholding by probability and by the entropy of the predictive distribution. The principle: high-probability tokens are meaningful, and when the distribution's entropy is high, many continuations are reasonable.

## Baselines

- **Vanilla autoregressive decoding.** One token per forward pass; memory-bandwidth-bound; the latency baseline.

- **Speculative decoding with a separate draft model (Leviathan et al. 2022; Chen et al. 2023).** Draft guesses several tokens; target verifies in one pass; rejection sampling preserves the target distribution.

- **Blockwise parallel decoding with extra heads (Stern et al. 2018).** Multiple feed-forward heads on one model predict several positions in parallel — no draft model. The structural precedent for adding future-token predictors to a single backbone and then verifying their continuations.

## Evaluation settings

- **Models.** Open chat/instruct LLMs of varying size and training recipe: Vicuna-7B/13B/33B and Zephyr-7B (covering public-data SFT, private-data SFT, and SFT+alignment/RLHF).
- **Regime.** Primarily batch size 1 — the locally-hosted, personal-use case where decoding is most bandwidth-bound and latency matters most.
- **Metrics.** End-to-end wall-clock speedup over vanilla decoding; expected number of tokens accepted per decoding step (acceptance length); generation-quality preservation (the speedup must not degrade quality).
- **Protocol / data.** Head training on instruction data aligned with the target's output distribution (e.g. ShareGPT), or a self-generated dataset when the original data is unavailable; comparison across prompt types; greedy and temperature sampling.

## Code framework

The pre-existing pieces: a pretrained decoder-only Transformer exposing its last hidden state h_t and an original LM head (`lm_head: ℝ^d → ℝ^V`); a KV cache; standard causal attention with an attention mask and positional indices; and a decode loop. One empty slot remains for the approach.

```python
import torch
import torch.nn as nn

# Pre-existing: a pretrained decoder-only Transformer with
#   model.hidden_states(tokens) -> last hidden state h_t  (one weight-load)
#   model.lm_head(h)            -> logits over the vocabulary for the next position
#   a KV cache, causal attention (mask + positional indices), and a decode loop.

@torch.no_grad()
def decode(model, prompt):
    tokens = prompt
    while not done(tokens):
        h = model.hidden_states(tokens)              # one weight-load
        base_logits = model.lm_head(h)               # next position
        # TODO: the approach goes here.
        ...
    return tokens
```
