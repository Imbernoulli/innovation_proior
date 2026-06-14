Elastic-Cache is a training-free, architecture-agnostic KV-cache policy for masked-diffusion
language models (e.g. LLaDA). Because these models attend bidirectionally, the cached keys/values
of already-decoded tokens drift as new tokens are unmasked, so the exact prefix invariance that
makes autoregressive KV caching free does not hold. Elastic-Cache decides, adaptively and from the
model's own attention, both **when** to refresh the cache (an attention-aware drift test on the
most-attended token) and **where** to refresh it (a depth-aware schedule that recomputes only from
a chosen boundary layer onward and reuses shallow-layer caches), while block-caching distant
`MASK` tokens outside a sliding prediction window.

## Problem it solves

Standard diffusion-LM decoders recompute QKV for all tokens at all layers at every denoising step,
even though step-to-step KV change is small most of the time and concentrated in deep layers. The
goal is to remove most of that redundant QKV work — preserving final-task accuracy and lowering
latency/memory — using only signals already available during a forward pass, with no retraining.
Fixed-schedule baselines (block-wise refresh, fixed-interval delayed caching) refresh on a clock
rather than on the model's state, and refresh all layers uniformly.

## Key ideas

- **Sliding-window decoding.** Predict only the leftmost window `M^t_β` of masked positions (size
  `β=16`); block-cache the KV of distant `MASK` tokens outside the window (they act as a
  length-bias prior and receive little attention). Nearby masks that strongly attend to one
  another stay live, unlike a rigid block that freezes masks near a block edge.
- **Attention-aware refresh (WHEN).** Per layer, track the most-attended decoded token
  `T^{t,ℓ} = argmax_{k∈D^{<t}} Σ_{q∈M^t_β} S^{t,ℓ}_{q,k}` (`track_num=1`). Recompute that token's
  attention-weight row and compare to the previous step by cosine similarity `σ^{t,ℓ}`. If
  `σ^{t,ℓ} < γ` (`γ=0.9`) the pattern has broken and a refresh is triggered. The most-attended
  token is used because it has the *smallest* drift, so it is a conservative lower bound on
  everyone else's change. Attention weights are watched (not hidden states) because the
  weight change is the *cause* of KV drift under bidirectional attention and is not polluted by the
  caching error that accumulates in reused hidden states.
- **Depth-aware refresh (WHERE).** The first layer whose tracked attention row breaks is the last
  layer allowed to finish in reuse mode; the next layer and all deeper layers recompute KV for all
  tokens (re-init from the cached full-sequence hidden, project, overwrite the cache — as at the
  initial step). In the implementation a trigger at zero-based `block_idx` mutates
  `lengths[1] = block_idx + 1`, so the next layer is the first full-recompute layer.
- **Confidence-aware parallel decoding.** Inside the window, commit in parallel every token whose
  top softmax probability clears `ε=0.9`; if none clears it, clamp the cutoff to the current window
  maximum so at least one position is committed.

The single knob `γ` trades speed for accuracy: higher `γ` triggers refreshes more often (higher
quality, lower throughput), lower `γ` triggers less. Overhead is negligible — finding the
most-attended token is `O(K²H)` versus `O(K²HD)` for full attention; the cosine test is `O(KH)`.

## Why it works (theory)

Let `Δ^{t,ℓ}_i = ||ΔK^{t,ℓ}_i||_2 + ||ΔV^{t,ℓ}_i||_2` be KV drift. Bounded projection norms give the
one-way bound `Δ^{t,ℓ}_i ≤ 2 W_max ||ΔH^{t,ℓ}_i||_2`. The average-drift and attention-drift bounds
below also require no-single-token-domination and projection-scale comparability where hidden and
KV drift are related on the relevant directions.

**Layer-wise drift monotonicity.** Under the average-drift regularity above, the residual recursion gives
`Δ̄^{t,ℓ+1} ≤ λ_ℓ Δ̄^{t,ℓ}` with `λ_ℓ = (1+L_FFN)[1 + W_max(1 + C_attn(ℓ))]` and
`C_attn(ℓ) = 2 W_max² R_ℓ² N / √d_k`, so `Δ̄^{t,ℓ} ≤ Δ̄^{t,1} ∏_{k<ℓ} λ_k`. Since `λ_ℓ > 1` this
upper bound grows with depth; it does not by itself prove that actual drift is monotone. The
ordering needs the specialization assumption in the cached quantity, or equivalently non-collapsing
K/V projections on the persistent deep-layer directions: shallow hidden drift gives
`E[Δ^{t,ℓ}] ≤ 2W_max f_ℓ(t) → 0` for `ℓ ≤ ℓ*`, while deep KV drift satisfies
`liminf_t E[Δ^{t,ℓ'}] ≥ c^{KV}_{ℓ'} > 0` for `ℓ' > ℓ*`. For large `t`,
`E[Δ^{t,ℓ}] < E[Δ^{t,ℓ'}]`, justifying shallow reuse and deep refresh.

**Attention gap bound ⇒ most-attended token has near-minimal drift.** Following the Lipschitz
chain, the change in total attention received by token `k` obeys
`|Δα^{t,ℓ}_k| ≤ |M^t_β| (W_max R_ℓ √N / √d_k) max_i Δ^{t,ℓ}_i` once hidden drift and KV drift are
comparable along the relevant projection directions; otherwise the same Lipschitz step is a
hidden-drift bound. If the most-attended token `T` had excess KV drift `ε` over the average, the
extra swing in its incoming attention is
`|M^t_β| (W_max R_ℓ √N / √d_k) ε`; for `T` to remain the argmax this differential must be no
larger than its previous attention gap, and the gap is at most a constant fraction of the window
mass, `Γ^{t-1,ℓ} ≤ c |M^t_β|` (with `c=1` as the loose total-mass bound). Solving,
`ε ≤ c √d_k / (W_max R_ℓ √N) = O(√d_k / (R_ℓ √N))`, so
`Δ^{t,ℓ}_{T^{t,ℓ}} ≤ Δ̄^{t,ℓ} + O(√d_k/(R_ℓ √N))` — the most-attended token's drift exceeds the
average only by a term that vanishes as `N` grows, making it a low-drift conservative trigger.

## Algorithm

```
Require: prompt, window size β, threshold γ, gen length N, confidence ε
x ← [prompt; MASK,...,MASK];  t ← 1;  D ← prompt positions;  M ← gen positions;  T ← ∅
while M ≠ ∅:
    M_β ← leftmost β of M;  Q ← T ∪ newly_decoded ∪ M_β  # tracked + fresh decoded + window
    if t == 1: start_reset ← -1                     # all zero-based block_idx satisfy update
    else:      start_reset ← L                      # start every layer in reuse
    for block_idx = 0..L-1:
        if block_idx >= start_reset:                # CACHE UPDATE
            recompute Q,K,V for all tokens I; overwrite KV cache; H^{ℓ+1} from full hidden
        else:                                       # CACHE REUSE
            recompute Q,K,V only for Q; write those rows into cache; reuse the rest
            σ^{t,ℓ} ← cos_sim(S^{t-1,ℓ}_T , S^{t,ℓ}_T)      # tracked-token attention drift
            if σ^{t,ℓ} < γ:  start_reset ← block_idx + 1      # in-place: next layer updates
        T^{t,ℓ} ← argmax_{k∈D} Σ_{q∈M_β} S^{t,ℓ}_{q,k}     # next-step probe (top-track_num)
    x, D_new ← confidence_decode(x, M_β, ε)         # parallel unmask above ε
    M ← M \ D_new;  T ← ∪_ℓ T^{t,ℓ};  t ← t + 1
return x
```

## Working code

Faithful to the canonical LLaDA implementation. The outer loop builds the query set (tracked
tokens + sliding window), forwards it, gathers each layer's most-attended token for next step, and
commits confident tokens. The per-layer routine carries the cached hidden, reuses or recomputes KV
against the mutable `start_reset` boundary in `lengths[1]`, and runs the cosine-similarity trigger
on the tracked token's attention row in the reuse branch.

```python
import torch
import torch.nn.functional as F

MASK_ID, EOS_ID = 126336, 126081


@torch.no_grad()
def generate_with_elastic_cache(model, prompt, gen_length=512, window_length=16,
                                mask_id=MASK_ID, eos_id=EOS_ID, threshold=0.9,
                                tokens_per_iter=1, gamma=0.9, track_num=1,
                                block_caching=True):
    """Elastic-Cache decoding for a bidirectional masked-diffusion LM.
       gamma: attention-similarity refresh trigger;  window_length: sliding window beta;
       track_num: number of most-attended tokens used as the drift probe."""
    for block in model.model.transformer.blocks:        # reset per-layer cache + probe
        block.x_cache = block.q_cache = block.k_cache = block.v_cache = None
        block.track_token = None

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    query_position = torch.arange(x.shape[1], device=model.device)
    track_position = query_position[:0].clone()
    new_decoded_position = query_position[:prompt.shape[1]].clone()
    masked_position = query_position[prompt.shape[1]:].clone()
    L = len(model.model.transformer.blocks)
    i, nfe, num_computed, total_computed = 0, 0, 0, 0
    decoded_eos = False

    while True:
        query_masked_position = masked_position[:window_length] if block_caching else masked_position

        if i == 0:                                       # first step: full forward, fill cache
            x_query, start_reset = x, -1
        else:                                            # later: window + tracked rows only
            query_position = torch.cat([track_position, new_decoded_position, query_masked_position], dim=0)
            x_query, start_reset = x[:, query_position], L

        positions = [query_position, track_position, query_masked_position, masked_position]
        lengths   = [x.shape[1], start_reset, gamma, track_num]   # attention mutates lengths[1]

        logits = model(x_query, use_cache=True, lengths=lengths, positions=positions).logits
        logits = (logits[:, query_masked_position, :] if logits.shape[1] == x.shape[1]
                  else logits[:, -query_masked_position.shape[0]:, :])
        if not block_caching:
            query_masked_position = query_masked_position[:window_length]
            logits = logits[:, :window_length]

        track_position = torch.cat([b.track_token for b in model.model.transformer.blocks],
                                   dim=0).unique(sorted=False)

        if threshold is not None:
            x, new_decoded_position, eos_pos = _commit_confident(logits, query_masked_position, x, threshold, eos_id)
        else:
            x, new_decoded_position, eos_pos = _commit_topk(logits, query_masked_position, x, tokens_per_iter, eos_id)
        masked_position = masked_position[~torch.isin(masked_position, new_decoded_position)]

        nfe += 1
        if not decoded_eos and eos_pos.shape[0] > 0:
            eos_pos = eos_pos.min().item()
            decoded_eos = True
            masked_position = masked_position[masked_position <= eos_pos]

        num_computed   += L - lengths[1]                 # canonical accounting from start_reset
        total_computed += L
        if masked_position.numel() == 0:
            break
        i += 1
    return x, nfe, num_computed / total_computed


def _commit_confident(logits, query_masked, x, threshold, eos_id=EOS_ID):
    p = F.softmax(logits.to(torch.float64), dim=-1)
    conf, pred = torch.max(p, dim=-1)
    keep = (conf >= min(threshold, conf.max()))          # clear threshold; never stall
    commit = query_masked[keep[0]]
    pred = pred[:, keep[0]]
    x[:, commit] = pred
    return x, commit, commit[pred.eq(eos_id)[0]]


def _commit_topk(logits, query_masked, x, num_transfer_tokens=1, eos_id=EOS_ID):
    p = F.softmax(logits.to(torch.float64), dim=-1)
    conf, pred = torch.max(p, dim=-1)
    _, keep = torch.topk(conf[0], k=min(num_transfer_tokens, pred.shape[1]), largest=True)
    commit = query_masked[keep]
    pred = pred[:, keep]
    x[:, commit] = pred
    return x, commit, commit[pred.eq(eos_id)[0]]


@torch.no_grad()
def elastic_attention(block, x, positions, lengths, block_idx, qkv_fn, attn_fn, scale):
    """Per-layer KV caching with the attention-drift trigger.
       lengths[1] is start_reset, the first zero-based layer to full-recompute."""
    query_position, track_position, query_masked, masked_position = positions
    key_len, start_reset, gamma, track_num = lengths

    if block_idx > start_reset:                  # already past boundary: full hidden input
        block.x_cache = x
    else:                                        # reuse: overwrite only queried rows of hidden
        block.x_cache[:, query_position, :] = x
        if block_idx == start_reset:             # first update layer restores full hidden
            x = block.x_cache

    q, k, v = qkv_fn(x)                           # (B, heads, T, d_head)

    if block_idx >= start_reset:                 # CACHE UPDATE: overwrite full KV cache
        block.q_cache, block.k_cache, block.v_cache = q, k, v
    else:                                         # CACHE REUSE: write queried rows, keep rest
        past_k = block.k_cache.clone()
        block.k_cache[:, :, query_position, :] = k
        block.v_cache[:, :, query_position, :] = v
        past_q = block.q_cache[:, :, track_position, :].clone()
        block.q_cache[:, :, query_position, :] = q
        k, v = block.k_cache, block.v_cache

    att, att_weight = attn_fn(q, k, v, need_weights=True)     # bidirectional attention + weights

    if block_idx >= start_reset:
        masked_att = att_weight[:, :, query_masked, :]
    else:                                                     # drift trigger in the reuse branch
        masked_att = att_weight[:, :, -query_masked.shape[0]:, :]
        cur_track_att = att_weight[:, :, :track_position.shape[0], :]
        past_att = torch.softmax(past_q @ past_k.transpose(-2, -1) * scale, dim=-1)
        sim = F.cosine_similarity(past_att, cur_track_att, dim=1).mean()
        if sim < gamma:                                       # next layer becomes first update layer
            lengths[1] = block_idx + 1

    masked_att = masked_att.sum(dim=(0, 1, 2))               # this layer's next-step probe
    masked_att[masked_position] = 0.0                         # only decoded tokens are candidates
    block.track_token = masked_att.topk(k=track_num, dim=0, largest=True)[1]
    return x, att
```

## Defaults

`γ = 0.9` (refresh trigger), `β = 16` (sliding window), `track_num = 1` (top-1 probe),
`ε = 0.9` (parallel-decode confidence), and block caching enabled for distant `MASK` tokens.
Higher `γ` requests more refreshes; lower `γ` requests fewer. Larger `β` keeps more masked tokens
live per step; smaller `β` caches more aggressively and takes more iterations.
