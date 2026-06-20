The per-iteration scheduler holds occupancy at the ceiling, but I've been treating "advance each request by some tokens this step" as if all token-work is the same. It isn't, and the place it bites is prefill. A fresh request needs its *entire* prompt computed before it can start decoding — and a prompt can be thousands of tokens. Under the scheduler as written, when such a request comes up I try to advance it by its whole prompt length in one step. That single step now runs a huge prefill matmul, and every decoding request in the batch waits behind it. One step that should have taken a decode's worth of time takes a long-prompt-prefill's worth. Everybody's next token is delayed. That's a latency spike, and at fixed-latency SLO a latency spike means I have to back off the load — which costs throughput.

So let me characterize the two kinds of work precisely, because the fix lives in their difference. A decode step over a batch is **memory-bandwidth-bound**: tiny compute (one new query position per request), dominated by streaming the weights and reading the KV cache. A prefill over a long prompt is **compute-bound**: a big matmul over hundreds or thousands of prompt positions, the GPU's tensor cores actually saturated. These two have *complementary* bottlenecks — decode leaves the compute units idle, prefill leaves them busy. Running them in separate steps means: on a decode step the tensor cores idle; on a prefill step the long matmul hogs everything and stalls decodes. Each kind of step wastes exactly the resource the other kind needs.

That complementarity is the lever. If I could put some prefill work and some decode work in the *same* forward pass, the prefill's compute would fill the idle tensor cores that the decodes leave open, and the decodes would ride along on a step they were going to pay for anyway. The GPU step would be busy on both axes instead of bottlenecked on one. But I can't just dump a whole 3000-token prefill alongside the decodes — that's the monopolizing step I'm trying to avoid; it would blow the step's latency back up. I need the prefill contribution to each step to be *bounded*.

So bound it. Don't advance a prefilling request by its whole prompt in one step — split the prompt into **chunks** of at most some token budget, and feed one chunk per step, across several steps, co-batched with whatever decodes are running. A 3000-token prompt with a chunk budget of, say, 512 becomes six prefill chunks spread over six steps; each of those six steps also carries the in-flight decodes. No step contains more than ~512 prefill tokens plus the decode tokens, so no step's latency balloons; meanwhile each of those steps' compute is filled by the prefill chunk instead of idling on a pure-decode pass. I've smeared the big prefill across several already-happening decode steps and used it to fill their slack.

And this falls out of the scheduler's existing shape almost for free — which is why phrasing the scheduler as "advance each request by num_new_tokens, clamped to a per-step token budget" earlier was the right move. Chunked prefill is just: set the clamp. When a prefilling request asks to advance by its whole remaining prompt, cap `num_new_tokens` at a long-prefill threshold (the chunk budget), so it only ingests that many prompt tokens this step and the rest on later steps. The same `token_budget` accounting that packs decodes into a step now also caps how much prefill any one step absorbs. The scheduler doesn't need a new mechanism; it needs the prefill advance to be chunk-bounded.

Now the part that *isn't* free — the attention kernel. When I chunk a prefill, a request's tokens get split across steps: some of its prompt's KV is already in the cache from earlier chunks, and the current chunk's query tokens must attend over *both* the already-cached prefix KV *and* the new chunk's own KV. That's a mixed pattern: within-chunk it's prefill-style attention (a block of query positions attending causally over each other and over the past), and it has to reach back into the paged KV cache for the earlier chunks. So the same step's batch now contains, simultaneously: pure decode requests (one query position, attend over a long cached history) and prefill-chunk requests (many query positions, attend over cached prefix + the new chunk). The attention has to serve both in one launch.

The clean way to express that is a single kernel path that handles a *prefill chunk against the paged cache* and a *decode against the paged cache* under one roof — branch on whether the request's query length this step is >1 (a prefill chunk) or ==1 (a decode). When `max_query_len > 1` there are prefill-chunk requests in the batch, so run the chunk-prefill attention (multi-query, attending over the paged context, `skip_decode=True` so it leaves the pure decodes to the decode path); then run the paged-decode attention for the single-query requests. Both read keys/values through the same block tables the paged cache already maintains — chunked prefill is *built on* paging, it doesn't replace it. The cache layout doesn't change; what changes is that a request's prefill streams into that cache a chunk at a time, and the kernel attends over the accumulated paged KV plus the live chunk.

A couple of consequences I should be clear-eyed about. First, this is a throughput-vs-latency *trade* tuned by the chunk size: smaller chunks → smoother latency (no step is heavy) but more steps and a little more overhead per prompt; larger chunks → fewer steps but heavier individual steps. The right chunk size depends on the model, the hardware, and the traffic's prompt-length mix, so it's a knob, not a universal constant — the gain is real but config-sensitive, and the honest way to state it is "co-batching prefill chunks with decodes raises GPU utilization; the magnitude is measured by the shipped throughput benchmark with the chunked-prefill flag on, for a given model/hardware/workload," not a fixed multiplier. Second, it interacts with the prior rungs exactly as designed: it keeps the per-iteration occupancy high (no monopolizing prefill step pushing decodes out) *and* it improves the quality of each step's compute (idle tensor cores during decode get filled). It's the rung that stops prefill from being a throughput-and-latency tax on decode.

So the change: cap each request's prefill advance at a chunk budget so no step carries a whole long prompt, co-batch those prefill chunks with the running decodes in the same forward pass, and route both through one attention path that handles multi-query prefill-chunk attention and single-query decode attention against the shared paged cache.

The core is the dispatch that runs chunk-prefill and paged-decode attention together over the paged cache.

```python
# vllm/v1/attention/ops/chunked_prefill_paged_decode.py (excerpt).
# One step's batch can hold BOTH prefill-chunk requests (max_query_len > 1,
# many query positions) and decode requests (one query position). Both attend
# over the same paged KV cache via the block_table.
def chunked_prefill_paged_decode(
    query, key, value, output, kv_cache_dtype,
    key_cache, value_cache, block_table,        # the paged cache from rung 1
    query_start_loc, seq_lens, max_seq_len, max_query_len,
    k_scale, v_scale, ...):

    # Prefill-chunk requests present: run multi-query prefill attention over the
    # paged context. skip_decode=True leaves the single-query decodes alone, so
    # this co-runs with -- doesn't clobber -- the decode path below.
    if max_query_len > 1:
        context_attention_fwd(
            q=query, k=key, v=value, o=output,
            k_cache=key_cache, v_cache=value_cache,
            b_loc=block_table,                  # logical->physical, same as paged attn
            b_start_loc=query_start_loc, b_seq_len=seq_lens,
            max_seq_len=max_seq_len, max_input_len=max_query_len,
            k_scale=k_scale, v_scale=v_scale,
            skip_decode=True,                   # decodes handled separately
            ...)

    # Then the paged single-query decode attention for the decode requests,
    # over the same block_table-addressed cache (kernel below).
    # ... kernel_paged_attention_2d over key_cache/value_cache via block_table ...
```
