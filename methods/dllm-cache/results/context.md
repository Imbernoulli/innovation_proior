## Research question

Diffusion-based large language models (dLLMs) such as LLaDA-8B and Dream-7B generate text
not left-to-right but by iteratively *denoising* a fully masked response: a bidirectional
Transformer takes the prompt plus the current partially-masked response, predicts a
distribution over the clean tokens at every position at once, commits a few high-confidence
tokens, and repeats. To produce a response of length `L`, the model runs on the order of `L`
denoising steps, and each step is a full forward pass over the entire prompt-plus-response
sequence, recomputing bidirectional attention and the feed-forward network in every layer.
The practical cost is therefore roughly cubic in the sequence length, `O(N^3)` for the whole
generation, against the roughly quadratic `O(N^2)` that autoregressive models (ARMs) achieve.

The question is how to run the denoising rollout of a fixed dLLM faster, without changing the
model weights or the intended decoded sequence. ARMs reach `O(N^2)` with a Key-Value cache,
but that technique assumes a causal mask and a fixed left-to-right order — neither of which a
bidirectional dLLM has.

## Background

**The denoising inference loop.** A dLLM is a masked diffusion model. In the forward
(corruption) process, each token of a clean sequence `x_0` is independently replaced by a
special `[MASK]` token with probability `t \in [0,1]`, so at `t=1` the sequence is fully
masked. The model `p_theta` — a bidirectional Transformer — is trained to reconstruct the
clean tokens at masked positions, minimizing negative log-likelihood over the masked
positions. At inference, given a prompt `c = (c_1,...,c_M)`, generation starts from
a fully masked response `y^(K) = ([MASK],...,[MASK])` of a chosen length `L`, and runs `K`
denoising steps `k = K, K-1, ..., 1`. At each step the model predicts the clean sequence,
greedily reads off the most likely tokens, and a transition function commits a subset of
them while re-masking the rest, yielding `y^(k-1)`. The default remasking rule keeps the
highest-confidence predictions (low-confidence remasking); generation can also run
semi-autoregressively, filling the response one fixed-size *block* at a time. The number of
steps `K` trades quality against latency.

**How the AR cache works and why dLLMs differ.** In an ARM, a causal mask restricts each
token to attend only to earlier positions, so once a token is processed its Key and Value
states are fixed for the rest of generation and can be cached; per-step cost drops and the
total falls to `O(N^2)`. A dLLM has **no causal mask**: every token attends to every other
token, including still-masked ones, at every step. Two structural properties follow, both
noted in the prior literature on dLLM efficiency. (1) *Timestep-variant K/V*: the Key and
Value of a token a model committed earlier keep changing as other positions get unmasked,
because the representation depends on the full (evolving) context — the K/V a position
presents at step `m` differ from those at step `n != m`. (2) *Non-sequential decoding order*:
any masked position may be the next one filled, so the order in which positions are computed
is not fixed in advance. Host dLLMs such as LLaDA were built with vanilla multi-head attention
(not grouped-query attention).

**Measurable behavior of the denoising rollout.** Across *adjacent* denoising steps, one can
measure the cosine similarity between a token's feature (Key, Value, attention output, or FFN
output) at the current step and at the previous step. The **prompt** tokens never change and
their internal representations drift only slowly over many steps. In the **response** region,
across adjacent steps most tokens are highly similar to their previous version, while some are
markedly dissimilar. For the response tokens, the adjacent-step similarity of a token's
**Value** (or Key) is correlated with the adjacent-step similarity of its attention output and
FFN output. These are properties of how an existing dLLM behaves during its rollout,
measurable independently of any acceleration mechanism.

**Feature caching for iterative denoisers (from image/video diffusion).** A separate line of
work accelerates *continuous* diffusion transformers (for images and video) by caching
intermediate features. The basic scheme: at the first timestep of a cache period compute all
layer features (self-attention, cross-attention, MLP) and store them, then for the next `N-1`
timesteps skip those computations and reuse the cache, giving roughly `N-1x` acceleration; it
relies on adjacent-step features differing very little, with accuracy depending on the cache
period `N`. Finer-grained variants observe that **different tokens have different temporal
redundancy** — the per-token feature distance between adjacent steps varies, with some tokens
moving far more than the mean — and select *which* tokens to cache versus recompute, scoring
tokens by their similarity to the cached value (cache the most-similar, recompute the
least-similar), updating the cache for computed tokens at every step rather than only at
period start, and using different cache ratios at different layer depths. This paradigm —
caching layer intermediate features and choosing tokens by temporal similarity — is
established for continuous-noise image/video DiTs, where the input is a grid of patch tokens
under a fixed noise schedule.

## Baselines

A new acceleration method for dLLM inference would be measured against the no-cache rollout
and the concurrent attempts to cache it.

**No-cache denoising (the control).** Run the full bidirectional forward over the entire
prompt-plus-response sequence at every one of the `K` steps. Per-step FLOPs are dominated by
attention and the FFN: for `T` layers, sequence length `n`, hidden size `d`, and FFN
intermediate size `m`, roughly `T*(8 n d^2 + 4 n^2 d + 6 n d m)` (the `6 n d m` reflecting a
three-matmul SwiGLU FFN), times `K` steps.

**Delayed KV reuse for dLLMs (dKV-Cache; Ma et al., 2025).** Observes that once a token is
*decoded* its Key/Value become relatively stable while still-masked tokens keep fluctuating,
with the largest K/V change happening at a token's own decoding step. It therefore caches K/V
only for already-decoded tokens, with a one-step *delay* (cache on the step after a token is
decoded), indexed by an arbitrary-order set rather than a contiguous prefix.

**Block-wise approximate KV cache for dLLMs (Fast-dLLM; Wu et al., 2025).** Decodes the
response block by block; computes a KV cache for the prompt once and reuses it across a
block's steps, and after a block finishes recomputes the KV cache for all blocks (a
"DualCache" variant also caches the masked suffix). Justified by the empirically high cosine
similarity of KV activations across adjacent steps within a block. It pairs this with
confidence-aware parallel decoding — unmask all tokens whose confidence exceeds a threshold,
rather than a fixed count — to address the conditional-independence assumption when many
tokens are unmasked at once.

**AR Key-Value caching (Pope et al., 2023).** Under a causal mask, append each new token's
K/V to a running buffer and attend over it, computing fresh states only at the current
decoding position; total cost `O(N^2)`. This is the standard reuse mechanism for ARMs.

## Evaluation settings

The natural yardsticks are the dLLM's host model run end to end on public final-task
benchmarks under deterministic decoding. The host is `LLaDA-8B-Instruct` (the related Dream-7B
is a second host in this line). Representative benchmarks and their native metrics: grade-school
and competition math by exact final-answer accuracy (e.g. GSM8K, MATH-500); code generation by
execution pass@1 (HumanEval, MBPP); multiple-choice knowledge and reasoning by exact
answer-letter accuracy (ARC-Challenge, MMLU, GPQA, BBH); and long-context tasks (LongBench,
including HotpotQA) by their native scores. Each task fixes a denoising configuration: number
of steps, block length, generation length, remasking rule (low-confidence by default), and
few-shot count — for instance generation lengths from 128 to 512 with steps to match, and
blocks from 8 up to the full generation length. Efficiency is read off in FLOPs (or FLOPs per
token), decode throughput (tokens/second), and peak GPU memory; quality is the benchmark-native
score, with the protocol holding the model, tokenizer, and decoding settings fixed across the
no-cache control and any cache method so only the reuse strategy differs.

## Code framework

The substrate is the existing dLLM denoising rollout: a fixed bidirectional Transformer host,
a generation loop that denoises a masked response block by block under low-confidence
remasking, and a per-layer Transformer block whose forward pass currently recomputes
everything from scratch every step. The only open engineering slot is an optional cross-step
state object plus a hook inside the block; before a new mechanism exists, that hook simply
falls back to the ordinary full forward.

```python
import torch
import torch.nn.functional as F


class CrossStepState:
    """Optional state carried by an acceleration hook."""

    def __init__(self):
        self.data = {}

    def reset(self, prompt_length: int):
        self.data = {}
        self.prompt_length = prompt_length
        self.current_step = 0

    # TODO: state update rule.


class TransformerBlock(torch.nn.Module):
    """One bidirectional block: attn_norm -> Q/K/V projections -> full attention
    -> attn_out, then ff_norm -> FFN -> ff_out, with residual adds. Today every
    sublayer is recomputed over the whole sequence on every denoising step."""

    def forward(self, x, state: CrossStepState, layer_id: int, attention_bias=None):
        residual = x
        h = self.attn_norm(x)
        q, k, v = self.q_proj(h), self.k_proj(h), self.v_proj(h)
        att = self.attention(q, k, v, attention_bias)      # full bidirectional attention
        x = residual + self.attn_out(att)
        residual = x
        h = self.ff_norm(x)
        x = residual + self.ffn(h)                          # FFN over the whole sequence
        # TODO: block-level acceleration rule.
        return x


def get_num_transfer_tokens(mask_index, steps):
    """Spread the masked positions to commit evenly across the steps of a block."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base, rem = mask_num // steps, mask_num % steps
    n = base.expand(-1, steps).clone()
    if rem.sum() > 0:
        idx = torch.arange(steps, device=mask_index.device)
        n[idx.unsqueeze(0) < rem] += 1
    return n.to(torch.int64)


@torch.no_grad()
def generate(model, input_ids, state, steps, gen_length, block_length, mask_id):
    """The existing dLLM denoising rollout: semi-autoregressive blocks, low-confidence
    remasking. The host model already calls TransformerBlock.forward per layer per step."""
    B, prompt_length = input_ids.shape
    x = torch.full((B, prompt_length + gen_length), mask_id, dtype=torch.long,
                   device=model.device)
    x[:, :prompt_length] = input_ids
    state.reset(prompt_length)
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks
    for b in range(num_blocks):
        s, e = prompt_length + b * block_length, prompt_length + (b + 1) * block_length
        block_mask = x[:, s:e] == mask_id
        num_transfer = get_num_transfer_tokens(block_mask, steps_per_block)
        for i in range(steps_per_block):
            mask_index = x == mask_id
            logits = model(x, state=state).logits[:, prompt_length:]   # full forward today
            x0 = torch.argmax(logits, dim=-1)
            conf = torch.where(mask_index[:, prompt_length:],
                               F.softmax(logits, dim=-1).gather(-1, x0[..., None])[..., 0],
                               -torch.inf)
            conf[:, (b + 1) * block_length:] = -torch.inf
            for j in range(B):                                          # commit top-conf
                sel = torch.topk(conf[j], k=num_transfer[j, i]).indices
                x[j, prompt_length:][sel] = torch.where(
                    mask_index[j, prompt_length:][sel], x0[j, sel],
                    x[j, prompt_length:][sel])
    return x[:, prompt_length:]
```
