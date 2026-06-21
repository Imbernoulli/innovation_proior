Prefix caching closed the last redundancy on the prefill side — with paging, continuous batching, chunked prefill, and prefix caching in place, prefill is now cheap, shared, and non-blocking. The one thing none of those rungs touched is the *decode* step itself, and decode is where a request spends almost all of its life. Every decode forward pass produces exactly **one** token per request, and decode is memory-bandwidth-bound: the step's wall-clock is dominated by streaming the entire model and reading the KV cache while the actual arithmetic is tiny. So I pay the full cost of loading the whole model and cache merely to advance each request by a single token, and the compute units sit mostly idle during that load. There is enormous slack — the same weight-load could do far more arithmetic before bandwidth re-binds. The obstacle is structural to autoregression: token $t+2$ cannot be computed until token $t+1$ is known, so I generate strictly one token per pass. I cannot *generate* $k$ tokens in parallel. But I can *verify* $k$ tokens in parallel, and that is a different operation the transformer is already good at — a single forward pass over $k$ query positions computes the model's next-token distribution at *every* one of them at once, which is exactly what prefill does. So if I had a *guess* of the next $k$ tokens, I could feed all $k$ into one pass and check, position by position, whether the model would actually have produced my guess.

I propose **speculative decoding** (draft-and-verify). Cheaply **propose** $k$ candidate next tokens — a draft — then run the expensive target model once over all $k$ and **verify**, accepting the longest prefix of the draft that matches what the target would itself have generated and rejecting the rest. The accept rule is what makes it exact, not merely close. Running the target over the sequence with the drafts appended gives the target's distribution $p_\text{target}$ at every position; for greedy decoding I accept $d_1$ iff it equals $\arg\max p_\text{target}$ at position 1, then accept $d_2$ iff it equals the argmax at position 2 given $d_1$ was right, and so on, stopping at the first position where the draft disagrees with the target's argmax. At that stop position I take the target's *own* token — which I already computed in this same pass — so even a fully wrong draft still yields one correct token, never worse than plain decoding. (For sampling there is the analogous rejection-sampling rule that preserves the target's sampling distribution exactly; the greedy case is the clean way to see why it is **lossless**.) The number of accepted tokens per forward pass is $1 + (\text{length of the correct draft prefix})$, so if the drafter is right most of the time, several tokens come out of one big-model pass.

The real design question is where a *cheap, good* draft comes from — it must be much cheaper than the target (or the draft cost eats the win) and accurate enough to be accepted often. There is a spectrum, and the engine should support its useful points behind one interface: a drafter proposes $k$ tokens, the target verifies, and the accepted prefix advances the request. The cheapest possible drafter does no neural work at all — it just *looks up*. In much real generation the next few tokens have already appeared in the context: the model is quoting the prompt back, as in extractive QA, "repeat the document," code reusing earlier identifiers, or structured output echoing the input. So an **n-gram / prompt-lookup** drafter takes the last few generated tokens as a pattern, searches the existing context for the most recent earlier occurrence of that same pattern, and proposes the tokens that followed it last time — zero model forward passes, pure string matching over tokens the request already has. When the output echoes the input, acceptance is high and the draft is essentially free; when it does not, the guess is wrong but cost nothing, so I just fall back. Because this runs every decode step, the matching must be fast, which calls for an efficient longest-suffix match: reverse the token stream so "longest n-gram matching the current suffix" becomes "longest prefix," and use a Knuth-Morris-Pratt failure function (an LPS array) to find it in linear time, then read off the $k$ tokens after the matched n-gram. The richer drafters use a tiny model to *predict* rather than look up, for continuations that are not literal quotes: a small **draft model**, or **EAGLE** (a lightweight autoregressive head over the target's hidden state), or **Medusa** (extra heads predicting several future tokens in one shot). These cost a little compute per draft but generalize beyond echoing.

This slots cleanly into everything already built, which is the point. Verification is a forward pass over $k$ query positions for a request — *exactly the multi-query attention the chunked-prefill path already runs* against the paged cache — and "advance request $r$ by `num_new_tokens` this step" simply becomes "advance $r$ by the number of *accepted* tokens," a count that is $\geq 1$ and varies per step, which the per-iteration, token-budgeted scheduler was already built to handle. So speculative decoding reuses the paged cache, the multi-query verify path, and the variable-advance scheduler; what is new is only the proposer and the accept rule. The throughput gain is that when drafts are accepted, several tokens come out of one big-model pass, so the dominant per-token decode cost (bandwidth) is divided across the accepted tokens — claiming the idle-compute slack. That gain is the *mean accepted length per pass*, which depends entirely on workload predictability and drafter quality: extractive or repetitive outputs with the n-gram drafter, or a well-matched draft model, accept many tokens and win large; high-entropy creative generation accepts about one and wins little. So the honest statement is "more accepted tokens per forward pass, with the speedup whatever the acceptance rate yields for the given drafter and workload, reproducible via the throughput benchmark with a speculative config," not a fixed multiplier.

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
