**Problem (from step 2).** The per-iteration scheduler treats "advance request by N tokens" uniformly, but a fresh request needs its *whole* prompt prefilled before decoding. A long prompt's prefill in one step is a huge compute-bound matmul that monopolizes the forward pass; every decoding request stalls behind it (a latency spike), and at a fixed-latency SLO a spike forces a load back-off that costs throughput. Decode steps (bandwidth-bound) and prefill steps (compute-bound) each waste exactly the resource the other needs.

**Key idea — chunked prefill.** Split a long prefill into chunks of at most a token budget and feed one chunk per step, *co-batched with the in-flight decodes* in the same forward pass. The prefill chunk's compute fills the tensor cores that the decodes leave idle; the decodes ride along on a step that was happening anyway. No step carries more than ~`chunk` prefill tokens plus the decode tokens, so no step's latency balloons.

**Why it works.** The two work types have complementary bottlenecks, so packing a bounded prefill chunk into a decode step makes the step busy on *both* the compute and bandwidth axes instead of bottlenecked on one. It composes with the prior rungs by construction: it's just a clamp on the scheduler's `num_new_tokens` (cap the prefill advance at a `long_prefill_token_threshold` chunk budget), and it reads/writes the same paged KV cache via block tables — chunked prefill is built *on* paging, not a replacement. The chunk size is a throughput-vs-latency knob (smaller = smoother latency, more steps; larger = fewer, heavier steps), so the gain is real but **config-sensitive** — measured by the shipped throughput benchmark with chunked prefill enabled, not a fixed multiplier.

**Change / code.** (1) Scheduler caps each prefilling request's per-step advance at the chunk budget. (2) One attention path handles a step's mixed batch: multi-query prefill-chunk attention (`max_query_len > 1`) over the paged context (`skip_decode=True`), then single-query paged-decode attention — both over the same `block_table`-addressed cache.

```python
# vllm/v1/attention/ops/chunked_prefill_paged_decode.py (excerpt).
def chunked_prefill_paged_decode(
    query, key, value, output, kv_cache_dtype,
    key_cache, value_cache, block_table,          # the paged cache (rung 1)
    query_start_loc, seq_lens, max_seq_len, max_query_len,
    k_scale, v_scale, ...):
    # Prefill-chunk requests present -> multi-query attention over paged context;
    # skip_decode=True so it co-runs with (doesn't clobber) the decode path.
    if max_query_len > 1:
        context_attention_fwd(
            q=query, k=key, v=value, o=output,
            k_cache=key_cache, v_cache=value_cache,
            b_loc=block_table,                    # logical->physical (paged)
            b_start_loc=query_start_loc, b_seq_len=seq_lens,
            max_seq_len=max_seq_len, max_input_len=max_query_len,
            k_scale=k_scale, v_scale=v_scale, skip_decode=True, ...)
    # Then single-query paged-decode attention for the decode requests,
    # over the same block_table-addressed cache.
    ...
```
