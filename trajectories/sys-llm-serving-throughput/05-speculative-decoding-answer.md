**Problem (from step 4).** After paging, continuous batching, chunked prefill, and prefix caching, the untouched bottleneck is the *decode step* itself, where a request spends almost all its life. Decode is memory-bandwidth-bound: each forward pass streams the whole model and KV cache yet advances each request by exactly **one** token, leaving the GPU's compute units idle. The fixed weight-load could do far more arithmetic before bandwidth re-binds — that slack is unclaimed.

**Key idea — speculative decoding (draft-and-verify).** Generation is sequential, but *verification is parallel*: one forward pass over k query positions gives the model's true distribution at every position at once. So cheaply **propose** k candidate next tokens (a draft), run the target model once over all k, and **verify** — accept the longest draft prefix that matches the target's own outputs, take the target's token at the first mismatch, fall back to normal decode after. Accepted tokens per pass = 1 + (correct-draft length). This is **lossless**: every emitted token is one the target would have produced (greedy: argmax-match per position; sampling: the analogous rejection-sampling rule preserves the target distribution).

**Why it works.** When drafts are accepted, several tokens come out of one big-model forward pass, so the dominant per-token decode cost (bandwidth) is amortized across the accepted tokens — claiming the idle-compute slack. Proposers span a spectrum: **n-gram / prompt-lookup** (zero model passes — find the longest recent n-gram in the request's own context via a KMP failure function, propose what followed it; great when output echoes input), and richer neural drafters — a small **draft model**, **EAGLE** (a lightweight autoregressive head over the target's hidden state), or **Medusa** (extra heads predicting several future tokens in one shot). It composes with the stack by construction: verification *is* the chunked-prefill multi-query attention over the paged cache, and "advance request by accepted count" *is* the scheduler's variable per-step advance. The gain is the **mean accepted length**, set by workload predictability + drafter quality, so it's config-sensitive — reproducible via the throughput benchmark with a speculative config, not a fixed multiplier.

**Change / code.** Add a proposer emitting k draft tokens per request; verify all k in one target pass over the paged cache; accept the longest matching prefix; advance by the accepted count. The cheapest proposer (n-gram lookup):

```python
# vllm/v1/spec_decode/ngram_proposer.py (excerpt) — zero model forward passes.
def _find_longest_matched_ngram_and_propose_tokens(
        origin_tokens, min_ngram, max_ngram, max_model_len, k):
    total_token = origin_tokens.shape[0]
    if total_token < min_ngram:
        return np.empty((0,), dtype=origin_tokens.dtype)
    k = min(k, max_model_len - total_token)
    if k <= 0:
        return np.empty((0,), dtype=origin_tokens.dtype)
    tokens = origin_tokens[::-1]                  # reverse: suffix-match -> prefix-match
    lps = np.zeros(max_ngram, dtype=np.int32)     # KMP failure function
    longest_ngram, position, prev_lps, i = 0, 0, 0, 1
    while i < total_token:
        if tokens[prev_lps] == tokens[i]:
            prev_lps += 1
            if prev_lps >= longest_ngram:
                longest_ngram, position = prev_lps, i
            if i < max_ngram:
                lps[i] = prev_lps
            if prev_lps == max_ngram:
                prev_lps = lps[max_ngram - 1]
            i += 1
        elif prev_lps != 0:
            prev_lps = lps[prev_lps - 1]
        else:
            i += 1
    if longest_ngram < min_ngram:
        return np.empty((0,), dtype=origin_tokens.dtype)
    # ... read the k tokens following the matched n-gram as the draft ...
```
