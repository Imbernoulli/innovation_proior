Chunked prefill made prefill cheap *per step* and stopped it from stalling decodes. But it didn't make prefill *less work in total* — a 2000-token prompt is still 2000 tokens of prefill matmul and 2000 tokens of KV written into the cache, just spread over more steps. So now I want to ask whether I'm doing prefill work I don't need to do at all. And when I look at a real request stream, a lot of it is redundant — not within one request, but *across* requests.

Look at what real traffic shares. A served application almost always prepends the same thing to every request: a system prompt ("You are a helpful assistant…"), a few-shot preamble, a fixed instruction header, a long shared document that many questions are asked against. Different users send different questions, but the first few hundred tokens are byte-for-byte identical across thousands of requests. And there's the sampling case: one prompt forked into several completions (n>1, beam search) — all the forks share the entire prompt. Under what I have now, every one of those requests prefills that shared prefix from scratch: recomputes the same attention keys and values for the same tokens, and writes its own private copy into the paged cache. The compute is duplicated and the memory is duplicated.

The memory duplication I could *almost* already kill — rung 1 gave me the mechanism and I flagged it then. A request addresses its cache through a block table of physical block numbers, and two block tables can list the *same* physical block number. So if two requests share a prefix, their block tables could point at the same physical blocks for that span, storing the shared KV exactly once, with reference counting so the block lives until the last sharer drops it. The capability is sitting right there in the paged design. What's missing is the *bookkeeping* to know that two requests' prefixes are identical, so I can hand the second one the first one's blocks instead of allocating new ones.

So the real question is: how do I cheaply recognize "this request's next block of prompt tokens is identical to a block I already have cached"? I need to look up "have I already computed and cached the KV for *exactly these tokens, in exactly this context*?" — and the answer has to be a fast lookup, not a scan.

A lookup keyed on *content* is a hash map. So: hash the contents of each full block of tokens, and keep a map from block-content-hash → the physical block that holds that block's KV. When a request's prompt is being prefilled, before computing a block I hash its token ids, probe the map, and on a hit I skip the compute entirely and just point the request's block table at the already-cached physical block (bumping its ref count). On a miss I compute it and insert it into the map. The block size from rung 1 is exactly the granularity to hash at — blocks are already the unit of allocation, so "cache a block of KV" and "key a block of KV" line up.

But I have to be careful about *what* I hash, or I'll get false matches that corrupt outputs. Two tokens being identical isn't enough for their KV to be identical — in a transformer, a token's key and value at position *t* depend on *all the tokens before it*, because attention and the causal context shape the representation. Token id 1045 appearing as the 5th token of one prompt has *different* KV from token id 1045 as the 5th token of a *different* prefix, because positions 1–4 differ. So the hash of a block must capture not just the block's own token ids but the *entire prefix leading up to it*. If I hashed only the block's own tokens, I'd match two blocks that have the same local tokens but sit after different histories, and reusing one's KV for the other would be flatly wrong.

The clean way to fold in "the entire prefix" without re-hashing the whole prefix each time is to **chain** the hashes: the hash of block *i* is computed over (the hash of block *i−1*, the token ids in block *i*). So each block's hash transitively encodes every token before it — block *i*'s hash depends on block *i−1*'s hash, which depends on *i−2*'s, all the way back to the first block. Two blocks collide in the map only if they have the same tokens *and* the same parent hash *and* therefore the same entire preceding context. That's exactly the equivalence I need: same content *and* same context ⟺ same KV ⟺ safe to share. The first block has no parent, so it chains off a fixed sentinel hash. I should also fold in any extra keys that change the KV but aren't in the token ids — things like a LoRA adapter id or a multimodal input hash — so a block computed under adapter A doesn't get reused for a request under adapter B.

So the per-block hash is `hash(parent_block_hash, block_token_ids, extra_keys)`, computed left to right along the prompt. Build that, key a content-hash → physical-block map with it, and prefix caching becomes: for each full block of a request's prompt, compute its chained hash, probe the map; hit → reuse the cached physical block (ref-count it onto this request's block table, skip prefill for those tokens); miss → prefill and insert. A shared 500-token prefix across N requests is computed and stored *once*; the other N−1 requests skip those ~31 blocks of prefill and share the blocks.

Two safety details the paging foundation already mostly handles. Reference counting: a shared block must not be freed while any request still points at it, so cached blocks carry a ref count and only return to the free list when it hits zero — the block pool already reference-counts (rung 1). Eviction: cached-but-currently-unused blocks (ref count zero, still holding a hashed prefix) are kept around in case the prefix recurs, but they sit on the free list and can be evicted (LRU) when memory is needed; on eviction I drop their entry from the hash map so I never hand out a stale physical block. The map points only at live, correctly-hashed blocks.

Why this raises throughput at fixed latency: a shared prefix's prefill is pure overhead paid once instead of N times, so for prefix-heavy workloads (shared system prompts, RAG over a fixed document, multi-sample decoding) the prefill compute and the prefill KV writes for the shared span drop by up to the sharing factor — that compute and that memory go back to serving more concurrent requests, which is throughput. The size of the win is entirely a function of how much the workload actually shares: a workload with no common prefixes gets nothing (every block hash misses), one where every request shares a long preamble gets a lot. So this is a config-sensitive, workload-dependent gain — the honest statement is "skips recomputation/storage of shared prefixes; the magnitude is whatever the workload shares, reproducible via the prefix-sharing benchmark with caching enabled," not a fixed multiplier.

So the change: hash each full block of prompt tokens with a hash chained over its parent block's hash (and extra keys), keep a content-hash → physical-block map, and on a prefill, reuse a cached physical block on a hit (ref-count it, skip its prefill) instead of recomputing and re-storing the same prefix.

The core is the chained block hash and the content-keyed lookup.

```python
# vllm/v1/core/kv_cache_utils.py — the chained block hash (excerpt).
def hash_block_tokens(hash_function, parent_block_hash, curr_block_token_ids,
                      extra_keys=None) -> BlockHash:
    """Hash of a full block, chained over the *preceding* blocks so the key
    encodes the entire prefix -- two blocks collide only if their tokens AND
    their whole context match (so their KV is genuinely identical)."""
    if not parent_block_hash:
        parent_block_hash = NONE_HASH                 # sentinel for the first block
    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return BlockHash(
        hash_function((parent_block_hash,             # chain in all prior context
                       curr_block_token_ids_tuple,    # this block's tokens
                       extra_keys)))                  # LoRA id / mm hash / etc.

# vllm/v1/core/block_pool.py — content-keyed reuse (excerpt).
def get_cached_block(self, block_hash, kv_cache_group_ids):
    """On a prefill, probe block_hash in the cache. Hit -> return the already-
    computed physical block (the request's block table points at it, ref-counted;
    its prefill is skipped). Miss -> None, so the block is computed and inserted."""
    cached_blocks = []
    for group_id in kv_cache_group_ids:
        block = self.cached_block_hash_to_block.get_one_block(
            make_block_hash_with_group_id(block_hash, group_id))
        if not block:
            return None                                # cache miss -> compute it
        cached_blocks.append(block)
    return cached_blocks                               # hit -> reuse, skip prefill
```
