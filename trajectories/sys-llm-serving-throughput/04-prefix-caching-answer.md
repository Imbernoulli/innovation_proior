**Problem (from step 3).** Chunked prefill made prefill cheap *per step* but not less work *in total*. Real traffic shares prefixes across requests — a fixed system prompt, a few-shot preamble, a RAG document, or one prompt forked into several samples — so the same first hundreds of tokens are byte-identical across thousands of requests. Each request still recomputes that shared prefix's KV from scratch and stores its own private copy: duplicated compute and duplicated memory.

**Key idea — prefix caching.** Hash-keyed reuse of shared prompt prefixes at block granularity. Hash each full block of prompt tokens and keep a content-hash → physical-block map. On a prefill, probe each block's hash before computing it: on a hit, point the request's block table at the already-cached physical block (ref-count it, skip its prefill); on a miss, compute and insert. A token's KV depends on its whole prefix, so the block hash is **chained**: `hash(parent_block_hash, block_token_ids, extra_keys)` — block *i*'s key transitively encodes every token before it, so two blocks collide only if their tokens *and* entire context match (KV genuinely identical). `extra_keys` folds in non-token state (LoRA id, multimodal hash) that also changes the KV.

**Why it works.** Paging already lets two block tables point at the same physical block with reference counting (rung 1); prefix caching adds the bookkeeping to *find* identical prefixes. A shared K-token prefix across N requests is then prefilled and stored **once** instead of N times — that prefill compute and KV memory go back to serving more concurrent requests (= throughput). The chained hash is what keeps it correct (no false matches across different histories); reference counting keeps a shared block alive until the last sharer drops it; LRU eviction of ref-count-zero cached blocks (and dropping their map entry) keeps the map pointing only at live, correct blocks. The gain is **workload-dependent** (no shared prefixes → every hash misses → nothing; long shared preamble → large win), so it's reproducible via the prefix-sharing benchmark, not a fixed multiplier.

**Change / code.** Add `hash_block_tokens` (chained over the parent block hash) to key blocks; on prefill, `get_cached_block` probes the content-hash map and reuses a cached physical block on a hit.

```python
# vllm/v1/core/kv_cache_utils.py
def hash_block_tokens(hash_function, parent_block_hash, curr_block_token_ids,
                      extra_keys=None) -> BlockHash:
    if not parent_block_hash:
        parent_block_hash = NONE_HASH                 # first-block sentinel
    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return BlockHash(hash_function(
        (parent_block_hash, curr_block_token_ids_tuple, extra_keys)))  # chained

# vllm/v1/core/block_pool.py
def get_cached_block(self, block_hash, kv_cache_group_ids):
    cached_blocks = []
    for group_id in kv_cache_group_ids:
        block = self.cached_block_hash_to_block.get_one_block(
            make_block_hash_with_group_id(block_hash, group_id))
        if not block:
            return None                                # miss -> compute + insert
        cached_blocks.append(block)
    return cached_blocks                               # hit -> reuse, skip prefill
```
