# Context

## Research question

Large language models keep getting better as they grow, but bigger models make *inference latency* worse — and latency, not training cost, is what users feel when an LLM is hosted for interactive, single-request use. The bottleneck is structural: autoregressive decoding emits one token per forward pass, and each forward pass must stream the entire model's parameters from high-bandwidth memory to the accelerator. Generating a single token after moving gigabytes of weights leaves the accelerator's arithmetic units almost idle — LLM decoding is **memory-bandwidth-bound**, not compute-bound.

The precise problem: **how do we generate multiple tokens per forward pass — raising arithmetic intensity (FLOPs per byte moved) and cutting the number of sequential decode steps — without changing the model's output quality, and without the operational burden of training and serving a second model?** A good solution must (a) produce more than one accepted token per step, (b) be cheap to add to an *existing* model (ideally trainable on a single GPU, no full retrain), (c) preserve generation quality, and (d) not require running or co-locating a separate draft network in the serving stack.

## Background

**Why decoding is slow.** Each autoregressive step does one forward pass to produce one token; the pass is dominated by moving the model weights from HBM. So the per-token cost is set by memory bandwidth, and the abundant FLOPs of the accelerator go unused. The fix has to be to do *more useful prediction per weight-load* — i.e., process several token positions in the same pass that already pays the memory cost.

**Speculative decoding (the immediate ancestor).** Use a small, fast **draft model** to autoregressively guess the next several tokens, then run the large **target model** once over all the guesses in parallel (one forward pass verifies many positions, because a single pass with a longer input is still one weight-load). Accepted guesses are committed; the first rejection truncates. With **rejection sampling**, the accepted tokens are provably distributed exactly as the target model would have sampled them — output distribution preserved. Speedup comes from the target model confirming several draft tokens per pass instead of generating one.

*Its limitations, which motivate the next step:* obtaining a good draft model is hard — it must be small enough to be fast yet aligned enough that the target accepts its continuations, and the usual route is to *separately pretrain* a smaller model (reportedly hundreds of GPU-hours). A separately trained draft can also suffer distribution shift from the target, lowering acceptance. And serving two models — especially in a distributed setting — adds real system complexity.

**Parallel decoding with extra heads (the other ancestor).** Rather than a separate model, earlier work (Stern et al. 2018, blockwise parallel decoding) attached extra output heads to a single model to predict several future positions at once, for tasks like translation and image super-resolution. This sidesteps the draft-model problem — there is no second model — and is the seed Medusa revisits and refines for modern LLM decoding.

**Rejection sampling's efficiency wrinkle.** Rejection sampling guarantees the target's exact distribution, but its acceptance rate falls as sampling temperature rises, so it does not improve — and can hurt — speedup in the regimes where users actually sample. Intuitively, even if draft and target distributions match perfectly, independent sampling of the two means a draft token can still be rejected; only when the draft *is* the target under greedy decoding is everything accepted. This is the gap a cheaper acceptance rule could exploit, given that exact distribution matching is often unnecessary in practice.

**Truncation sampling.** A line of decoding work (e.g. truncation/typical sampling, Hewitt et al. 2022) accepts only tokens that are not *too improbable* under the model — thresholding by probability and by the entropy of the predictive distribution. The principle: high-probability tokens are meaningful, and when the distribution's entropy is high, many continuations are reasonable. This furnishes a notion of "acceptable" that does not require matching a target distribution exactly.

## Baselines

- **Vanilla autoregressive decoding.** One token per forward pass; memory-bandwidth-bound; the latency baseline.

- **Speculative decoding with a separate draft model (Leviathan et al. 2022; Chen et al. 2023).** Draft guesses several tokens; target verifies in one pass; rejection sampling preserves the target distribution. Limitation: needs a separate, well-aligned, separately-trained draft model (expensive, distribution-shift risk, two-model serving complexity).

- **Blockwise parallel decoding with extra heads (Stern et al. 2018).** Multiple feed-forward heads on one model predict several positions in parallel — no draft model. The structural precedent; Medusa modernizes it (head design, candidate verification, acceptance) for LLMs.

## Evaluation settings

- **Models.** Open chat/instruct LLMs of varying size and training recipe: Vicuna-7B/13B/33B and Zephyr-7B (covering public-data SFT, private-data SFT, and SFT+alignment/RLHF).
- **Regime.** Primarily batch size 1 — the locally-hosted, personal-use case where decoding is most bandwidth-bound and latency matters most.
- **Metrics.** End-to-end wall-clock speedup over vanilla decoding; expected number of tokens accepted per decoding step (acceptance length); generation-quality preservation (the speedup must not degrade quality).
- **Protocol / data.** Head training on instruction data aligned with the target's output distribution (e.g. ShareGPT), or a self-generated dataset when the original data is unavailable; comparison across prompt types; greedy and temperature sampling.

## Code framework

The pre-existing pieces: a pretrained decoder-only Transformer exposing its last hidden state h_t and an original LM head (`lm_head: ℝ^d → ℝ^V`); a KV cache; standard causal attention with an attention mask and positional indices; and a decode loop. The scaffold has empty slots for the auxiliary predictor(s), the parallel verification of candidates, and the acceptance rule.

```python
import torch
import torch.nn as nn

class AuxiliaryPredictor(nn.Module):
    """Predicts future token positions from the backbone's last hidden state.
    TODO: what architecture? It must be cheap to add to an existing model and
          trainable without a full retrain. A separate sequential model is the
          option we want to avoid."""
    def __init__(self, hidden_size, vocab_size, num_future):
        super().__init__()
        ...
    def forward(self, hidden_states):
        ...

def build_candidates(base_logits, aux_logits):
    # TODO: from the predictions for several future positions, assemble candidate
    #       continuations to be verified. How many, and how to verify many at once
    #       in a SINGLE forward pass without blowing up the batch?
    ...

def verify_candidates(model, candidates):
    # TODO: run the target model over all candidates in one pass; needs an
    #       attention mask / position indices that respect each candidate's history.
    ...

def accept(candidate_tokens, base_probs):
    # TODO: decide which prefix of a candidate to commit. Rejection sampling
    #       preserves the exact distribution but loses speed at high temperature.
    ...

@torch.no_grad()
def fast_decode(model, prompt, aux: AuxiliaryPredictor):
    tokens = prompt
    while not done(tokens):
        h = model.hidden_states(tokens)              # one weight-load
        aux_logits = aux(h)                          # predict several positions
        base_logits = model.lm_head(h)               # next position
        cands = build_candidates(base_logits, aux_logits)
        verified = verify_candidates(model, cands)   # one parallel pass
        tokens = tokens + accept(verified, base_probs=None)
    return tokens
```
