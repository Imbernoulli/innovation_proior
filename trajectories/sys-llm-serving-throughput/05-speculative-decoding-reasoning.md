I've attacked the memory side (paging), the occupancy side (continuous batching), the prefill-interference side (chunked prefill), and the redundant-prefill side (prefix caching). Step back and ask what's left. Prefill is now cheap, shared, and non-blocking. The thing I haven't touched is the *decode* step itself — and decode is where a request spends almost all of its life. Every decode forward pass produces exactly **one** token per request. And I established way back that decode is memory-bandwidth-bound: the step's wall-clock is dominated by streaming the model weights and reading the KV cache, while the actual arithmetic is tiny. So in a decode step I pay the full cost of loading the entire model and cache… to advance each request by a single token. The GPU's compute units are mostly idle during that load. There's enormous slack: the same weight-load could do far more arithmetic "for free" before bandwidth becomes the limit again.

So the bottleneck is now structural to autoregression: token *t+1* needs token *t* as input, so I generate strictly one token per forward pass, and each pass is bandwidth-bound with idle compute. If I could somehow produce *several* tokens per forward pass over the same weight-load, I'd amortize that fixed bandwidth cost over multiple tokens — that's the slack I want to claim.

The obstacle is the sequential dependency: I can't compute token *t+2* until I know token *t+1*, which I don't have yet. So I can't *generate* k tokens in parallel. But — can I *verify* k tokens in parallel? That's a different operation, and the transformer is good at exactly it. A single forward pass over k query positions computes the model's next-token distribution at *every* one of those positions simultaneously (it's what prefill does). So if I had a *guess* of the next k tokens, I could feed all k into one forward pass and, in that single pass, get the model's true distribution at each position — and check, position by position, whether the model would actually have produced my guess.

That reframes the whole thing. Generation is sequential and expensive; *checking a candidate continuation* is parallel and rides one weight-load. So: cheaply propose k candidate next tokens (a **draft**), then run the expensive model once over all k, and **verify** them — accept the longest prefix of the draft that matches what the model would have generated, reject the rest. If the draft is good, I accept several tokens from a single forward pass of the big model; if it's bad, I accept one (the first mismatch position still gives me a correct token from the model's own distribution there) and fall back to ordinary decoding. Either way I never emit a token the model wouldn't have — the verification is against the model's own outputs, so the result is *exactly* what plain decoding would have produced. This is lossless: same output distribution, fewer forward passes.

Let me nail down the accept rule so I'm sure it's lossless and not just "close." I have draft tokens d₁…d_k. I run the target model once over the sequence with those drafts appended, giving the target's distribution p_target at each position. The simplest exact rule for greedy decoding: accept d₁ if it equals argmax of p_target at position 1; if so move to d₂ and accept it iff it equals argmax at position 2 given d₁ was right; stop at the first position where the draft disagrees with the target's argmax. At the stop position I take the target's own token (which I already computed in this pass), so even a fully-wrong draft yields one correct token — never worse than plain decoding. (For sampling there's the analogous rejection-sampling accept rule that preserves the target's sampling distribution exactly; the greedy case is the clean way to see why it's lossless.) The number of accepted tokens per forward pass is 1 + (length of the correct draft prefix). If my drafter is right most of the time, that's several tokens per big-model pass.

Now the real design question: where does a *cheap, good* draft come from? It has to be much cheaper than the target model (or I've gained nothing — the draft cost has to be small against the verify cost) and accurate enough that drafts get accepted often. There's a spectrum of drafters, and I want the engine to support the useful points on it:

The cheapest possible drafter does no neural work at all: just *look it up*. In a lot of real generation, the next few tokens have *already appeared in the context* — the model is quoting the prompt back (extractive QA, "repeat the document", code that reuses earlier identifiers, structured output echoing the input). So an **n-gram / prompt-lookup** drafter: take the last few generated tokens as a pattern, search the existing context for the most recent earlier occurrence of that same pattern, and propose the tokens that followed it last time as the draft. Zero model forward passes, pure string matching over tokens the request already has. When the output echoes the input, acceptance is high and the draft is essentially free; when it doesn't, the draft is wrong and I fall back — but the proposal cost was negligible, so a wrong guess barely costs anything. The matching has to be fast (it runs every decode step), so it wants an efficient longest-suffix-match — a Knuth-Morris-Pratt-style failure function over the (reversed) token stream to find, in linear time, the longest recent n-gram matching the current suffix, then read off the k tokens after it.

The richer drafters use a tiny model to *predict* rather than look up — for the cases where the continuation isn't a literal quote. A small draft model, or a few extra prediction heads bolted onto the target that emit several future tokens in one shot (so the draft itself is one cheap pass), or a lightweight autoregressive head that's trained to mimic the target's next few tokens from its hidden state. These cost a little compute per draft but generalize beyond echoing — they predict plausible continuations the n-gram lookup can't. The engine should accommodate all of them behind one interface: a drafter proposes k tokens; the target verifies; the accepted prefix advances the request.

And here's why this slots cleanly into everything I've built: verification is just a forward pass over k query positions for a request — which is *exactly the multi-query attention the chunked-prefill path already does* against the paged cache. The scheduler's "advance request r by num_new_tokens this step" becomes "advance r by the number of *accepted* draft tokens," which is ≥1 and varies per step — and the per-iteration, token-budgeted scheduler was already built to handle a variable per-step token count per request (that was the whole point of phrasing it that way). So speculative decoding reuses the paged cache, the multi-query verify path, and the variable-advance scheduler; what's new is the *proposer* and the *accept rule*.

Why this raises throughput: when drafts are accepted, several tokens come out of one big-model forward pass, so the per-token bandwidth cost (the dominant decode cost) is divided across the accepted tokens — I claim the idle-compute slack of the bandwidth-bound decode step. The gain is exactly the **mean accepted length per pass**, which depends entirely on how predictable the workload is: extractive / repetitive outputs with the n-gram drafter, or a well-matched draft model, accept many tokens and win big; high-entropy creative generation accepts ~1 and wins little. So this is config- and workload-sensitive — the honest statement is "more accepted tokens per forward pass; the speedup is whatever the acceptance rate yields for the given drafter/workload, reproducible via the throughput benchmark with a speculative config," not a fixed multiplier.

So the change: add a proposer that emits k draft tokens per request (n-gram lookup at the cheap end; a draft model or extra prediction heads at the richer end), verify all k in one target forward pass over the paged cache, accept the longest matching prefix (lossless against the target's own outputs), and advance each request by the accepted count.

The core is the cheapest proposer — the n-gram lookup that finds the longest recent matching suffix and reads off the next k tokens.

```python
# vllm/v1/spec_decode/ngram_proposer.py (excerpt).
# Zero model forward passes: propose the next k tokens by finding the longest
# recent n-gram in the request's own context that matches the current suffix,
# and reading off what followed it last time. Verified by the target model.
def _find_longest_matched_ngram_and_propose_tokens(
        origin_tokens, min_ngram, max_ngram, max_model_len, k):
    total_token = origin_tokens.shape[0]
    if total_token < min_ngram:
        return np.empty((0,), dtype=origin_tokens.dtype)
    k = min(k, max_model_len - total_token)
    if k <= 0:
        return np.empty((0,), dtype=origin_tokens.dtype)

    # Reverse so "longest n-gram matching the suffix" becomes "longest prefix";
    # an LPS (KMP failure function) finds it in linear time.
    tokens = origin_tokens[::-1]
    lps = np.zeros(max_ngram, dtype=np.int32)
    longest_ngram, position, prev_lps, i = 0, 0, 0, 1
    while i < total_token:
        if tokens[prev_lps] == tokens[i]:
            prev_lps += 1
            if prev_lps >= longest_ngram:        # earliest (most recent) match
                longest_ngram, position = prev_lps, i
            if i < max_ngram:
                lps[i] = prev_lps
            if prev_lps == max_ngram:            # cap n-gram length at max_ngram
                prev_lps = lps[max_ngram - 1]
            i += 1
        elif prev_lps != 0:
            prev_lps = lps[prev_lps - 1]         # fall back on mismatch (KMP)
        else:
            i += 1
    if longest_ngram < min_ngram:
        return np.empty((0,), dtype=origin_tokens.dtype)  # nothing to propose
    # ... read the k tokens following the matched n-gram as the draft ...
```
