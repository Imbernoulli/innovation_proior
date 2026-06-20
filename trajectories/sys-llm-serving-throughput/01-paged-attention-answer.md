**Problem (from prior art).** Throughput at fixed latency is set by batch size, and batch size is set by KV-cache memory. The standard serving scheme reserves a *contiguous* per-request cache sized to the maximum sequence length, so most of the KV region is wasted to internal over-reservation (a short request still holds a max-length slab) and external fragmentation (freed slabs leave non-contiguous holes a new max-length request can't use). The batch can't grow because the cache is full of reserved emptiness.

**Key idea — PagedAttention.** Apply OS virtual-memory paging to the KV cache. Carve the cache into fixed-size **blocks** (e.g. 16 tokens of K/V each), managed by a global free list. A request no longer owns a contiguous chunk; it owns a **block table** mapping its logical token positions to scattered physical block numbers. Growth = pop one free block, append its number to the block table (O(1), no copy, no contiguity). The attention kernel translates logical→physical through the block table as it reads keys/values, exactly as paging hardware does on every memory access.

**Why it works.** Over-reservation collapses: a request holds only ⌈len/block_size⌉ blocks, and the only waste is the unfilled tail of its last block — at most `block_size−1` tokens, bounded by the block size instead of the max sequence length. External fragmentation vanishes: every free block is interchangeable, so any free block satisfies any request. KV waste drops from "most of the cache" to "near zero," the sustainable batch rises to what the *live* tokens need, and since decode is bandwidth-bound (a forward pass over B requests costs ≈ the same as over 1), tokens/sec rises with B. The cost is one indirection per block in the kernel — `physical = block_table[logical]`, then stride into the cache — which is negligible against the key/value reads. (Bonus, exploited by later rungs: two block tables can point at the *same* physical block, so a shared prefix is stored once.)

**Change / code.** (1) Replace contiguous per-request reservation with a block pool + free list; `get_new_blocks` pops physical blocks off the free queue. (2) Rewrite the attention kernel to read the physical block number from the block table and address the cache through it (widen to int64 before the stride multiply or large caches overflow).

```python
# vllm/v1/core/block_pool.py — the KV region is a pool of fixed-size blocks.
def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
    """Get new blocks from the free block pool. No contiguity required:
    any free blocks satisfy any request."""
    if num_blocks > self.get_num_free_blocks():
        raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")
    ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)
    for block in ret:
        self._maybe_evict_cached_block(block)
        assert block.ref_cnt == 0
        block.ref_cnt += 1            # reference-counted: enables block sharing
    return ret
```

```cpp
// csrc/.../attention_kernels.cuh — the paged read (logical -> physical).
const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
     block_idx += NUM_WARPS) {
  const int64_t physical_block_number =          // page-table lookup,
      static_cast<int64_t>(block_table[block_idx]);  // int64 before stride mul
  // key address goes through the physical block number; cache is scattered:
  const cache_t* k_ptr = k_cache
      + physical_block_number * kv_block_stride
      + kv_head_idx * kv_head_stride
      + physical_block_offset * x;
  // ... dot k_ptr with the query; QK^T / softmax / value-weighting unchanged ...
}
```
