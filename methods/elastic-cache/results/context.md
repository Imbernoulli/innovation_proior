## Research question

A masked diffusion language model generates by *denoising*: it starts from a sequence of
`[MASK]` tokens and, over many iterative steps, predicts and commits a few tokens at a time,
remasking the rest, until everything is filled in. The predictor is a Transformer with
**bidirectional (non-causal) attention** — at every step every position attends to every
other position, including positions that are still masked. This is what gives these models
their infilling ability. A standard decoder recomputes queries, keys, and values (QKV) for
*all* tokens at *all* layers at *every* denoising step; for a few-hundred-token answer over
hundreds of steps that is a large amount of repeated matrix work and memory traffic, and it
is the dominant cost of deployment.

The question: how to reuse key/value state across denoising steps in such a bidirectional
decoder at inference time, with no retraining, while keeping final-task accuracy unchanged —
using only signals the model already produces during a forward pass. The setting differs from
ordinary autoregressive decoding, where the property that makes caching trivial does not hold.

## Background

**Causal KV caching.** In a causal (left-to-right) Transformer, the key and value of a position
depend only on tokens to its left, which are already fixed once it is generated. So across
decoding steps the cached keys/values of earlier positions are *exactly* invariant,
`K^{t}_{[1:t-1]} = K^{t-1}_{[1:t-1]}`, and one simply caches them and reuses them forever (Pope
et al. 2023). Under bidirectional attention this invariance does not hold: a token's key/value
is a function of the *whole* sequence, including positions that are still masked. When the model
unmasks a new token, the context changes, and the keys/values of tokens committed earlier shift.
dKV-Cache (Ma et al. 2025) measured that token representations evolve across denoising steps,
with the largest jump in `K`/`V` happening *at* the step a token is decoded, after which it
stabilizes.

**Masked diffusion language models.** These are absorbing-state discrete diffusion models built
on D3PM (Austin et al. 2021) and its continuous-time form (Campbell et al. 2022). The forward
process replaces each token by `MASK` with probability `t`:
`q_{t|0}(x_t^i | x_0^i) = Cat(x_t^i; (1-t) δ_{x_0^i} + t δ_MASK)`, `t ∈ [0,1]`. Recent simplified
theory (MDLM, Sahoo et al. 2024; Shi et al. 2024; RADD, Ou et al. 2024) reduces training to a
reweighted cross-entropy over the masked positions,
`L = ∫_0^1 (1/t) E_{q_{t|0}}[ Σ_{i: x_t^i = MASK} -log p_θ(x_0^i | x_t) ] dt`. LLaDA (Nie et al.
2025) scaled this to an 8B model competitive with autoregressive LLMs; its instruction-tuned
variant decodes **semi-autoregressively** (left-to-right over blocks, diffusion within a block)
and uses *low-confidence remasking* — it keeps the high-confidence predictions and remasks the
rest. The base predictor attends bidirectionally throughout.

**Measurable phenomena during denoising.** Several facts about the model's dynamics, independent
of any caching scheme:

- **Step-to-step KV change is small most of the time, and grows with depth.** Define the KV
  *drift* of a token as the step-to-step change `||K^{t}-K^{t-1}||_2 + ||V^{t}-V^{t-1}||_2`. For
  most steps it is small; and averaged over tokens it is larger in deeper layers than in shallow
  layers. This lines up with the long-standing finding that early Transformer layers encode local
  lexical structure that settles quickly while deep layers encode global semantics that keep
  shifting (Kovaleva et al. 2019; Jawahar et al. 2019; Rogers et al. 2021).
- **Distant `MASK` tokens barely participate.** Masked positions near the active prediction region
  attend strongly to one another; masked positions far away receive very little attention and act
  essentially as a length-bias prior on the sequence.
- **The token that receives the most attention changes the least.** Among already-decoded tokens,
  the one with the largest incoming attention mass has the smallest KV drift across steps;
  empirically its attention-weight rows stay highly similar between consecutive steps, and changes
  in attention weights track changes in KV state closely.

**An analytic tool.** The softmax map `σ(z)_i = e^{z_i}/Σ_j e^{z_j}` is 1-Lipschitz in the `ℓ2`
norm: `||σ(z) - σ(z')||_2 ≤ ||z - z'||_2` (Gao & Pavel 2017, Prop. 2). This bounds how much
attention *weights* can move in terms of how much the attention *logits* (hence the hidden states)
move.

## Baselines

**Approximate block-wise KV cache, fixed schedule (Fast-dLLM; Wu et al. 2025).** Decode the
answer in fixed-size blocks (default 32). Before decoding a block, compute and store the KV of all
tokens *outside* it (the DualCache variant caches both the prefix prompt and the masked suffix);
within the block, reuse that frozen cache for every denoising step; recompute the entire cache at
the block boundary. The justification is that the cosine similarity of cached keys/values between
adjacent steps is consistently close to 1, so reuse loses little. It is paired with
*confidence-aware parallel decoding*: at each step unmask, in parallel, every token whose maximum
softmax probability exceeds a threshold `ε`, falling back to the single most-confident token if
none clears it. Their Theorem 1 supports this: if each of `n` candidate tokens has confidence
`p_j > 1-ε` with `(n+1)ε ≤ 1`, the `argmax` of the product of marginals equals the `argmax` of the
joint, so parallel factorized decoding agrees with greedy sequential decoding. The refresh is on a
fixed per-block clock — recompute at block boundaries — and recomputes all layers uniformly.

**Delayed conditioned caching (dKV-Cache; Ma et al. 2025).** Exploits that a token's KV is most
volatile at the moment it is decoded and stabilizes afterward: cache a decoded token's KV with a
*one-step delay* and recompute KV for tokens that are still masked/active. Two variants —
`dKV-Cache-Decode` refreshes the cache periodically (near-lossless, helps long sequences) and
`dKV-Cache-Greedy` restricts caching to recently decoded tokens plus a fixed local window for a
larger speedup at some accuracy cost. The refresh cadence is a fixed interval, applied uniformly
across all layers.

**Adaptive feature-similarity caching (dLLM-Cache; Liu et al. 2025).** Refreshes prompt features
on a long fixed interval and recomputes only the generated rows whose features have low similarity
to the cached ones, the similarity being measured on raw features, with layers treated alike.

**No-cache control.** Recompute QKV for every token at every layer every step. Exact, and the
accuracy ceiling.

## Evaluation settings

The natural yardstick is real end-to-end generation with a fixed masked-diffusion host model
(LLaDA-8B-Instruct), one predeclared inference policy used unchanged across all workloads (no
per-benchmark tuning), greedy/deterministic decoding, on a single A100-class GPU. Benchmarks and
metrics already standard for these models:

- **Mathematical reasoning** — MATH-500 (4-shot, final-answer match) and GSM8K (5-shot, flexible
  numeric extraction).
- **Code generation** — HumanEval (0-shot, `pass@1` by executing the completion) and MBPP
  (3-shot, `pass@1`).
- **Multiple-choice reasoning** — ARC-Challenge (exact answer-letter match).

Final-task accuracy is the quality metric (the soft gate that must be preserved); efficiency is
read off as decode throughput (tokens/sec, averaged until the answer is emitted), the fraction of
cache work reused vs. recomputed, and peak GPU memory. Prefill length and generation length are
natural stress tests because they change how much context the decoder must repeatedly process.

## Code framework

The substrate is the existing masked-diffusion decode loop around a fixed bidirectional
Transformer host. What already exists: a per-block KV buffer on each Transformer layer, an
attention routine that can return its attention weights, a confidence-based parallel unmasking
step, and the outer denoising loop that maintains the sets of decoded and still-masked positions.
The unresolved part is the **caching policy**: given the
signals already exposed by the bidirectional decoder and the current decoded/masked state, decide
how much cached state to reuse this step while still committing confident masked tokens.

```python
import torch
import torch.nn.functional as F


class CachePolicy:
    """Builds the cache plan used by a bidirectional masked-diffusion decoder.
    The host model already owns the cache buffers and returns the logits; this
    object supplies the missing policy."""

    def __init__(self, **hparams):
        self.hparams = hparams

    def plan_step(self, step, decoded_pos, masked_pos, cache_state, model_signals):
        # TODO: the caching decision we will design.
        pass

    def commit_tokens(self, logits, masked_pos, threshold):
        # Confidence-aware parallel unmasking: keep predictions whose max softmax
        # probability clears the threshold; always commit at least one.
        p = F.softmax(logits.to(torch.float64), dim=-1)
        conf, pred = torch.max(p, dim=-1)
        keep = conf >= min(threshold, conf.max())
        commit_pos = masked_pos[keep[0]]
        return pred[:, keep[0]], commit_pos


@torch.no_grad()
def diffusion_decode(model, prompt, gen_length, policy, threshold):
    # existing masked-diffusion loop the policy plugs into
    x = torch.full((1, prompt.shape[1] + gen_length), MASK_ID, dtype=torch.long)
    x[:, :prompt.shape[1]] = prompt
    decoded_pos = torch.arange(prompt.shape[1])
    masked_pos = torch.arange(prompt.shape[1], x.shape[1])
    step = 0
    while masked_pos.numel() > 0:
        plan = policy.plan_step(step, decoded_pos, masked_pos, cache_state=..., model_signals=...)
        logits = model(x, cache_plan=plan)
        tokens, commit_pos = policy.commit_tokens(logits, masked_pos[: ...], threshold)
        x[:, commit_pos] = tokens
        masked_pos = masked_pos[~torch.isin(masked_pos, commit_pos)]
        step += 1
    return x
```

The single open slot is the policy: `plan_step` must decide how the existing cache is used before
the unchanged decode loop and confidence-based commit consume the logits.
