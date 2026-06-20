Prefix caching — hash-keyed block reuse of shared prompt prefixes. A shared prefix's KV is computed and
stored once and reused (block tables point at the same ref-counted physical blocks) instead of being
recomputed and re-stored by every request that shares it.

This is a **config-sensitive, workload-dependent** rung: the gain is exactly the amount of prefix the traffic
actually shares. A workload with no common prefixes gets nothing (every block hash misses); a workload where
every request shares a long system prompt / few-shot preamble / RAG document, or forks one prompt into many
samples, gets a large reduction in prefill compute and KV writes for the shared span. So there is no single
multiplier to quote — the effect is reproducible via the shipped prefix-sharing benchmark with caching
enabled.

Measured via `benchmarks/benchmark_prefix_caching.py` (the in-repo prefix-sharing harness) with the enabling
flag `--enable-prefix-caching`. The script replicates each prompt `--repeat-count` times so the shared prefix
recurs; run it once with `--enable-prefix-caching` and once without, on the same fixed model and GPU(s), and
compare throughput / latency — e.g.

    python benchmark_prefix_caching.py --model <fixed-model> \
        --enable-prefix-caching --num-prompts 1 --repeat-count 100 \
        --input-length-range 128:256

On the repeated (prefix-sharing) workload, enabling prefix caching turns the second-and-later occurrences of
the shared prefix into cache hits, skipping their prefill; the magnitude tracks the repeat count and the
shared-prefix length.

Role on the ladder: removes redundant *cross-request* prefill that chunked prefill (rung 3) only spread out,
not eliminated. It is built directly on the paged cache's ref-counted block sharing (rung 1) — prefix caching
is the bookkeeping (the chained content hash + the content→block map) that lets the scheduler recognize and
reuse identical prefixes.

(Provenance: config-sensitive; measured via `benchmark_prefix_caching.py --enable-prefix-caching` — state the
reproduction recipe, not a fabricated number. Code: vllm/v1/core/kv_cache_manager.py, block_pool.py
(get_cached_block / cache_full_blocks), kv_cache_utils.py (hash_block_tokens, the chained block hash).)
