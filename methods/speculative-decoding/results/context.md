# Context

## Research question

Large autoregressive Transformers produce text one token at a time: to sample token `x_t` the model must first know `x_{<t}`, so generating a sequence of `K` tokens requires `K` *serial* forward passes of the model. As model sizes grew into the billions of parameters, this serial latency became the dominant cost of deployment — a single decode step from a very large model is far slower than a step from a small one, and the `K` steps cannot be overlapped.

The pressing question is how to reduce the *number of serial passes* through the large model during generation. A satisfying answer would have to (a) keep the model's output distribution exactly intact — for production systems that have been validated on a specific model, silently changing what the model emits is unacceptable; (b) require no retraining and no change to the model architecture, so it applies to off-the-shelf checkpoints; and (c) actually translate into lower wall-clock latency on real accelerators, not just a cleaner asymptotic count.

The setting that makes this hard, and also makes it interesting, is the hardware profile of inference. At batch size 1, a decode step from a large Transformer is **memory-bandwidth-bound**: most of the wall-time is spent streaming the model's weights and its attention key/value cache out of high-bandwidth memory, while the arithmetic units stay largely idle. So latency is serial, but compute is *cheap and underused* — there is spare capacity that the serial loop cannot exploit.

## Background

**Autoregressive Transformer decoding.** A Transformer language model defines a conditional distribution `p(x_t | x_{<t})` over the next token. Sampling proceeds by drawing `x_t`, appending it, and feeding the extended prefix back in. Practical sampling layers a choice of scheme on top of the raw logits: greedy/argmax, temperature scaling, top-k truncation, and nucleus (top-p) truncation. A useful observation is that every one of these can be expressed as drawing a sample from a *single adjusted categorical distribution* — argmax, for instance, is sampling from the distribution that puts all mass on the max; top-k is sampling from the renormalized truncated distribution. So one can reduce "all the ways people sample" to the canonical operation "sample `x` from a probability vector."

**Teacher-forced parallel scoring.** A Transformer with causal masking can score an entire sequence in a *single* forward pass: given `x_1, ..., x_K`, one pass returns `p(·|x_{<1}), p(·|x_{<2}), ..., p(·|x_{<K})` for all positions at once (this is exactly how the model is trained). The key cost fact at batch size 1: because the pass is memory-bound, scoring a block of `K` tokens in one pass costs roughly the same wall-time as scoring a single token — the weights are read from memory once either way. Generation does not use this, because it does not yet know the tokens to score; it discovers them one at a time.

**The memory-bandwidth bottleneck.** Decoding a single token touches every weight of the model exactly once and the full KV cache, but does only `O(model size)` arithmetic — an arithmetic intensity far below what modern accelerators are built for. Consequently, at low batch the wall-time of a decode step is set by bytes moved, not FLOPs performed, and adding parallel arithmetic within a step is nearly free until bandwidth or memory capacity is exhausted.

**"Not all steps are equally hard."** Empirically, many next-token decisions in a hard language-modeling task are easy — continuations of common words, closing brackets, formulaic phrasing — and a much smaller, cheaper model predicts them identically to the large one. Only a minority of positions genuinely require the large model's capacity. This unevenness is what adaptive-computation methods try to exploit.

**Sampling from a hard distribution via an easy one.** Two classical tools take samples from a tractable proposal `q` and turn them into something about a target `p`. *Rejection sampling* (von Neumann): draw `x ~ q` and `r ~ U(0,1)`, accept `x` if `r < p(x) / (M q(x))` where `M = max_x p(x)/q(x)`, otherwise retry from scratch; accepted samples are exactly `p`-distributed. Its weaknesses are that `M` can be enormous (driving the accept rate toward zero) and that every rejection throws away the draw entirely. *Importance sampling*: keep all `x ~ q` but reweight each by `p(x)/q(x)`; this corrects expectations but does not produce unweighted `p`-distributed *draws*, so it cannot stand in for sampling a token.

**Speculative execution.** A classic processor optimization: perform a task in parallel with checking whether it was actually needed (branch prediction is the canonical example). The payoff is increased concurrency whenever the guess is right; the prerequisite is a cheap, accurate mechanism for guessing what will be needed. In its standard form the guessed task is either needed or not — a deterministic predicate.

## Baselines

**Uniform-cost reduction (distillation, quantization, sparsification, architecture surgery).** These lower the cost of *every* token equally — e.g. distilling the large model into a smaller student, quantizing weights/activations, sparsifying, or replacing attention with a cheaper variant. They are effective but require training a new model or modifying the architecture, and they change the output distribution: the cheaper model is not the original model.

**Adaptive computation / early exit (Graves; Schuster et al.; Elbayad et al.; Mamou et al. "Wisdom of Committees").** These spend compute proportional to instance difficulty — attending to fewer inputs, exiting early at a shallow layer when a confidence criterion is met, or routing easy instances to a smaller model in a committee. They directly exploit "easy vs. hard steps," and the committee variant even reuses off-the-shelf small models. Their gap: they decide *when to take the shortcut* with a learned or heuristic criterion, which means they require a changed architecture or training procedure, and — fatally for the goal here — taking a shortcut changes what the model emits, so the output distribution is no longer that of the large model.

**Blockwise parallel decoding (Stern et al. 2018).** Trains auxiliary heads to predict several future tokens at once, then verifies the block with the large model and keeps the longest correct prefix — close in spirit to guess-then-verify. Gaps: it supports only greedy decoding (no stochastic sampling), it requires training the custom prediction heads, and it targets preserving downstream task quality rather than guaranteeing the exact output distribution.

**Shallow aggressive decoding (Sun et al.).** Also emits several tokens per step by copying spans from the input to the output and verifying them, which works only when input and output are nearly identical (e.g. grammatical error correction). Gaps: restricted to copy-style tasks, and greedy-only — no general proposal model and no stochastic sampling.

The open gap across all of these: none simultaneously (a) keeps the *exact* output distribution of the large model, (b) supports general stochastic sampling, and (c) works on unmodified off-the-shelf models without retraining.

## Evaluation settings

The natural yardsticks are standard conditional and unconditional generation tasks with public checkpoints, comparing against an optimized serial-decoding baseline:

- **Unconditional generation** with a GPT-like decoder-only Transformer trained on the One Billion Word benchmark (lm1b), using subword tokenization.
- **Machine translation**, English→German (WMT EnDe), with an encoder-decoder Transformer fine-tuned for the task.
- **Abstractive summarization** (CNN/DailyMail) with the same encoder-decoder family.
- **Dialog** with a very large decoder-only conversational model.

Across these, both argmax decoding (temperature 0) and standard stochastic sampling (temperature 1) are relevant settings. The metrics of interest are *wall-clock latency* (and the implied factor of speedup over the optimized serial baseline) at batch size 1 on a single accelerator, and — because any acceleration claim is only meaningful if outputs are unchanged — confirmation that the generated tokens are distributed identically to the baseline. A faithful comparison must use a strong, optimized implementation of standard decoding as the reference, not a naive loop.

## Code framework

The pre-existing pieces: a Transformer that returns per-position logits for a whole input sequence in one pass, and standard sampling utilities. We lay out the generic serial decoding loop, and leave one empty slot for the faster decoding procedure to be designed.

```python
import torch
from torch.nn import functional as F


def top_k_top_p_filter(logits, top_k=0, top_p=0.0):
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < kth[:, [-1]]] = float("-inf")
    if top_p > 0.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = 0
        logits[remove.scatter(1, sorted_idx, remove)] = float("-inf")
    return logits


def norm_logits(logits, temperature, top_k, top_p):
    # Fold any sampling scheme (argmax / top-k / nucleus / temperature)
    # into a single categorical distribution to sample from.
    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    return F.softmax(logits, dim=-1)


def sample(probs, num_samples=1):
    return torch.multinomial(probs, num_samples=num_samples)


@torch.no_grad()
def autoregressive_decoding(prefix, model, max_len, temperature=1.0, top_k=0, top_p=0.0):
    """Baseline: one serial forward pass of `model` per emitted token."""
    T = prefix.shape[1] + max_len
    while prefix.shape[1] < T:
        logits = model(prefix).logits
        probs = norm_logits(logits[:, -1, :], temperature, top_k, top_p)
        prefix = torch.cat((prefix, sample(probs)), dim=1)
    return prefix


@torch.no_grad()
def fast_decoding(prefix, model, max_len, **kwargs):
    """Emit the same distribution as `autoregressive_decoding` but with fewer
    serial passes of `model`, by exploiting the spare parallel compute."""
    # TODO: the decoding procedure we will design.
    pass
```
