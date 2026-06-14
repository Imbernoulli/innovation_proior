## Research question

Diffusion-based large language models (dLLMs) such as LLaDA (Nie et al. 2025) and Dream (Ye et
al. 2025) generate text not left-to-right but by iterative denoising of a fixed-length
sequence: the response region starts as `n` `[MASK]` tokens, and across `T` decoding steps the
model repeatedly re-predicts every still-masked position and commits a few of them, until none
remain. The forward pass uses **full bidirectional attention** over the whole sequence
(prompt + response), which is what lets a dLLM use right-context and decode in any order. The
cost is that at every step the model recomputes the key/value (KV) projections of the *entire*
sequence at *every layer*. An 8B dLLM that needs `T = 256` steps for a 256-token answer is
doing on the order of 256 full-sequence forwards, where an autoregressive model of the same
size would do one forward per generated token with almost all of its KV state reused from a
cache. The result is that dLLMs are roughly competitive with autoregressive models (ARMs) on
reasoning and instruction-following quality but markedly slower per token.

The precise problem: build a method that lets a dLLM **reuse most of its KV state across
decoding steps**, the way an ARM reuses its prefix cache, *without retraining the model and
without measurably hurting final-task accuracy*. A solution must (1) be training-free (a
drop-in inference-time wrapper, since the only public dense dLLMs are fixed pretrained
checkpoints); (2) respect that bidirectional attention makes the standard prefix cache invalid
— committing one masked token changes the context, and therefore the keys and values, of every
other token; (3) decide, at the granularity the dynamics actually demand, which tokens' KV
states are stale enough to recompute and which are stable enough to reuse; and (4) preserve the
benchmark score within a near-lossless margin while cutting redundant compute. Each existing
approach below achieves a subset of these; none achieves all four with fine enough resolution.

## Background

**Why the standard KV cache does not transfer.** In an ARM, attention is causal: token `i`
attends only to tokens `≤ i`, and generation appends one new token at the end. So once token
`i`'s keys and values are computed, they never change (no later token can affect them), and the
next step reuses them verbatim — the prefix KV cache is exact and append-only. A dLLM breaks
both assumptions. Attention is bidirectional, so token `i`'s representation depends on tokens
on *both* sides, including masked ones; and decoding does not append at the end but fills in
masked slots anywhere in a fixed-length sequence. When a `[MASK]` at position `j` becomes a
concrete token, every position that attends to `j` — i.e. the whole sequence — sees a changed
context, so its keys and values shift. Exact reuse is therefore impossible, and the entire
sequence's KV must in principle be recomputed each step.

**The redundancy that makes approximation possible.** Empirically, although exact reuse is
impossible, the KV states of many tokens are *nearly* identical across adjacent decoding steps.
A line of work measures this and reuses KV states approximately, dividing the sequence into a
**static segment** whose KV is cached for many steps and a **dynamic segment** whose KV is
refreshed often. This is the redundancy any acceleration method exploits; the open question is
how finely, and by what criterion, to draw the line between "reuse" and "recompute."

**Diagnostic finding 1 — masked-token KV states move in three phases.** Projecting the
layer-averaged key states of a single masked token to two dimensions with PCA and tracing them
across decoding steps (LLaDA-8B-Instruct on GSM8K, `L=328`, `n=256`, `T=256`) shows a
characteristic trajectory: a **gradual-change** phase over the early steps, a **rapid-change**
phase in the few steps immediately before that token is itself decoded, and a **stable** phase
after it has been decoded, where its KV barely moves for the rest of generation. Key and value
states of the same token track each other in both shape and magnitude. The change is largest
right before decoding because that is when the token's local context is filling in fastest: if
a nearby masked position is decoded, the masked token's neighbor flips from `[MASK]` to a
concrete embedding, and the closer that neighbor, the more it reshapes the model's
representation of the token. Crucially, a masked token's KV needs updating essentially only
during its rapid-change phase; in the gradual and stable phases it can be cached. This is a
*token-level* statement that segment-level caching cannot express, because within one segment
different masked tokens are in different phases at the same step.

**Diagnostic finding 2 — decoding order is spatially localized.** Sampling 64 GSM8K examples
and measuring, for each adjacent pair of decoding steps, the sequence distance between the
positions decoded, shows that a dLLM strongly prefers to decode the next token *near* the most
recently decoded one: 90% of next-decoded tokens fall within a distance of 10. So a masked
token surrounded by many already-known (prompt or decoded) tokens is the one about to be
decoded next — its imminence is readable from the density of known tokens in its local context.

**Diagnostic finding 3 — attention is concentrated, and stable across steps.** Prior ARM work
observed that transformer attention is not uniform but concentrated on a small set of salient
tokens — "attention sinks" (Xiao et al. 2023) — and exploited this to prune or budget KV state
by importance (Ada-KV, Feng et al. 2024; PyramidKV, Cai et al. 2024). Measuring the analogous
quantity in a dLLM with **attention rollout** (below) confirms the same concentration: queries
consistently attend to a small subset of key positions, overwhelmingly the **prompt and
decoded** tokens, while masked tokens receive negligible attention. Moreover, the rollout map
at one step is almost identical to the next step's, so the current step's attention is a good
predictor of which tokens will matter at the next step.

**Attention rollout (Abnar & Zuidema 2020).** To attribute how much each input token
influences a deep transformer's output, raw last-layer attention is unreliable: in deep nets
the per-layer attention weights become diffuse and do not track input importance. Rollout fixes
this by composing attention across layers while accounting for residual connections. Because a
transformer block computes `V_{l+1} = V_l + W_att V_l = (W_att + I) V_l`, the effective
per-layer mixing is `A = normalize(W_att + I)` — add the identity for the residual path and
renormalize each row to sum to one (for a single-head-averaged block this is the familiar
`A = 0.5 W_att + 0.5 I`). The end-to-end attribution is then the product of these per-layer
matrices, `C^{(l)} = A^{(l)} C^{(l-1)}` with `C^{(0)} = I`; a column sum of the final `C`
measures the total influence flowing into each token. Rollout was introduced for ARM/encoder
analysis; whether it is faithful for bidirectional dLLM attention is an open empirical
question.

**Premature overconfidence in dLLM decoding.** Confidence-based scheduling — at each step
commit the masked positions whose predicted token has the highest confidence — is the default,
but LLaDA-Instruct is observed to become *prematurely overconfident* about the end-of-sequence
token near the tail of the response in early steps, terminating too soon. To curb this, LLaDA
offers block-wise **semi-autoregressive (semi-AR)** decoding: split the response into blocks
and decode blocks left-to-right (with parallel decoding inside a block), which keeps a
roughly sequential order and preserves the model's reasoning ability — at the cost of
flexibility, since the block boundary is fixed regardless of where confident tokens actually
are.

## Baselines

**dKV-Cache (Ma et al. 2025).** A one-step *delayed* caching scheme: a token's KV is stored not
at the step it is decoded but at the following step, paired with a periodic refresh. This
acknowledges that a just-decoded token's KV is still settling, but the refresh is on a fixed
schedule rather than driven by which tokens are actually stale.

**Fast-dLLM (Wu et al. 2025).** Couples block-wise semi-AR decoding with two caches. PrefixCache
recomputes the current block and all blocks after it while reusing the KV of everything before
the current block; once a block is fully decoded, all positions are refreshed. DualCache goes
further, recomputing only the current block and caching everything else. PrefixCache
accelerates with little quality loss; DualCache is faster but degrades quality. **Limitation:**
it is welded to block-wise semi-AR decoding (so it inherits the rigid block boundary), and it
applies a single reuse/recompute rule to a whole block at once — a segment-level decision.
Within a block, a token whose KV has already stabilized after being decoded is still
recomputed; and a token in the *next* block that has already entered its rapid-change phase is
left stale until the current block finishes.

**dLLM-Cache (Liu et al. 2025).** Splits the sequence into a **prompt** segment and a
**response** segment and refreshes their KV at different fixed intervals: the prompt cache
every `K_p` steps, the response cache every `K_r` steps. On the steps in between, instead of
freezing the whole response, it does an adaptive partial update of the response: it computes
the current value (`V`) vector for each response token, takes the cosine similarity to that
token's cached `V`, and recomputes the KV of the `ρ`-fraction of response tokens whose `V` has
changed most (lowest similarity), since `V`-drift is found to correlate with the change in a
token's attention output and FFN output. **Limitation:** the two intervals `K_p`, `K_r` and the
ratio `ρ` must be retuned per model and per dataset (the published settings differ across every
benchmark); every token inside a segment shares the same refresh interval, so the method still
both reuses KV that should have been refreshed and refreshes KV that could have been reused;
and because the refresh is segment-wide, per-step cost grows with context length.

**ARM KV-cache pruning (Xiao et al. 2023; Feng et al. 2024; Cai et al. 2024).** For
autoregressive models, the observation that attention concentrates on few tokens motivates
keeping the KV of the salient tokens and evicting or budgeting the rest. **Limitation as it
stands for dLLMs:** this is an eviction strategy for *causal* attention with a growing prefix;
it is not designed for a fixed-length bidirectionally-attended sequence whose token roles
(masked vs decoded) change every step, and it offers no notion of *which masked tokens are
about to need their state recomputed*.

Across these, the common shape is a **coarse, segment-level** reuse/recompute decision driven
by **fixed schedules or intervals**. The finer, token-level structure that the diagnostic
findings reveal — different tokens being in different KV-dynamics phases, and a small,
shifting set of attention-salient tokens — is left on the table.

## Evaluation settings

The natural yardsticks already in use for dLLM acceleration, on the public dense dLLMs
LLaDA-8B (Base and Instruct) and Dream-v0-7B (Base and Instruct):

- **Math reasoning:** GSM8K (4-shot) and MATH-500 (4-shot), generation length 256, scored by
  exact final-answer accuracy.
- **Code generation:** HumanEval (0-shot) and MBPP (3-shot), generation length 512, scored by
  execution pass rate (pass@1).
- **Knowledge / multiple choice:** ARC-Challenge, GPQA (0-shot), MMLU-Pro (5-shot), generation
  length 256, scored by exact answer-letter / final-answer accuracy.
- **Decoding settings to compare under:** single-token-per-step decoding and parallel /
  block-wise semi-AR decoding (threshold 0.9 as in Fast-dLLM); generation length stress test at
  1024 for long-context behavior.
- **Efficiency yardsticks:** decode throughput (tokens/s), latency (s), peak GPU memory; the KV
  cache of a dLLM occupies the same `2·L·N·d` floats as an ARM of equal size, so any auxiliary
  state (e.g. an `L×L` matrix) must be checked against that budget.
- **Protocol:** one fixed decoding rollout per method, deterministic generation, identical host
  checkpoint and tokenizer; each method uses one predeclared configuration across all
  workloads (no per-benchmark hyperparameter search).

## Code framework

The method plugs into an existing dLLM denoising harness. The host model, the benchmark
loaders and scorers, and the outer block/step loop already exist; what does not exist is the
policy that decides, each step, which tokens to recompute and which to reuse from cache. The
substrate is therefore a generic approximate-cache object that the denoising loop calls through
a small set of hook points — it can intercept the model's input before the layers run, can
intercept each attention layer's query/key/value to substitute cached state, and is notified at
the end of every step so it can update whatever bookkeeping it keeps. The single empty slot is
the selection-and-update logic itself: how to choose the recompute set and how to maintain the
cache.

```python
import torch
import torch.nn as nn
from contextlib import contextmanager


class ApproxCache:
    """Generic approximate KV cache for a diffusion LM denoising loop. The loop
    drives one bidirectional forward per step over a fixed-length sequence; this
    object intercepts the forward to reuse cached K/V where it can, and is told
    at the end of each step what was decoded. All token-selection policy lives in
    the slot below."""

    def __init__(self, model_config):
        self.model_config = model_config
        self.key_cache: list[torch.Tensor] = []     # per-layer cached keys,  (B, L, ...)
        self.value_cache: list[torch.Tensor] = []   # per-layer cached values, (B, L, ...)

    @contextmanager
    def model_forward(self, x: torch.Tensor):
        """Wrap the whole-model forward. May narrow the rows fed to the layers
        and must restore full-length logits afterward."""
        ctx = ModelForwardContext(x=x)
        # TODO: decide which rows actually enter the layers this step,
        #       and scatter the layer outputs back to full sequence length.
        yield ctx

    @contextmanager
    def attention(self, layer_idx, x, attn_norm, q_proj, k_proj, v_proj,
                  attention_mask=None, position_ids=None):
        """Wrap one attention layer: project q/k/v, hand them to the outer code
        for the attention compute, and manage cached k/v."""
        residual = x
        x = attn_norm(x)
        q, k, v = q_proj(x), k_proj(x), v_proj(x)
        ctx = AttentionContext(q=q, k=k, v=v, residual=residual)
        # TODO: store fresh k/v into the cache, reuse cached k/v for the rows we
        #       chose not to recompute, and gather anything the policy needs from
        #       the attention output.
        yield ctx

    def on_step_end(self, block_mask, frame, delta):
        """Called after each decoding step, with what was just decoded.
        This is where the recompute set for the *next* step is decided."""
        # TODO: the token-selection-and-cache-update policy we will design —
        #       given the current confidence outputs, the current token layout,
        #       and whatever state the policy chooses to keep, decide
        #       which tokens' K/V to refresh next step and update the cache state.
        pass


# existing diffusion-LM denoising loop the cache plugs into
def denoise(model, initial_frame, num_blocks, block_length, cache):
    frame = initial_frame
    for block_idx in range(num_blocks):                 # block-wise schedule
        block_mask = make_block_mask(block_idx, block_length)
        while True:
            delta = generate_step(model, frame, block_mask,   # one bidirectional forward,
                                  past_key_values=cache)      # routed through cache hooks
            if delta is None:                           # block exhausted of masks
                break
            cache.on_step_end(block_mask, frame, delta) # update the recompute policy
            frame = frame.apply_delta(delta)            # commit decoded tokens
    return frame
```

The outer loop supplies one bidirectional forward per step and tells the cache what was
decoded; the `attention` / `model_forward` hooks and `on_step_end` are where the reuse policy
will live.
