The thing capping my throughput isn't the GPU's compute and it isn't the kernels — FasterTransformer already has fast fused kernels and it's still slow under a real request stream. The cap is the batch size, and the batch size is capped by memory. Decode is bandwidth-bound: running a forward pass over 30 requests costs almost the same wall-clock as over 1, because I'm streaming the weights and the cache either way and the per-request arithmetic is tiny. So tokens/sec scales almost linearly with how many requests I can hold concurrently — right up until I run out of KV-cache memory. Everything I want comes down to: fit more requests' caches into the same GPU.

So let me stare hard at *where the KV memory actually goes*, because if a lot of it is wasted, that waste is my throughput. In the standard scheme, when a request arrives I give it a contiguous chunk of cache memory. But the cache grows token by token and I don't know the final length in advance — so to guarantee the contiguous chunk never has to move or grow, I reserve it sized to the *maximum* sequence length the model supports. A request that asks for 2048 tokens but stops after 30 still holds the whole 2048-token reservation for its entire life. That's the first leak: internal over-reservation, dead tail memory inside each request's chunk.

And there's a second leak that's just as bad. Different requests reserve different… no — they all reserve the same max size, but they finish at different times and get freed, leaving holes. When a new request needs a contiguous max-length chunk, it may not fit in any single hole even though the *total* free memory is plenty. That's external fragmentation — free memory I can't use because it isn't contiguous. Between these two, a large fraction of the KV region is memory I've paid for and can't put live tokens in. If I add up over-reservation plus fragmentation, on these systems it's most of the cache. My batch size is small not because the GPU is full of useful data but because it's full of *reserved emptiness*.

Let me try the obvious patches first and watch them fail, so I know I'm not missing something cheap.

Patch one: just shrink the reservation — reserve, say, 256 tokens instead of the max. Now short requests waste less. But the moment a request generates token 257, its contiguous chunk has to grow, and the memory immediately after it is owned by another request. I can't grow in place. I'd have to copy the whole cache to a bigger free region — an expensive move, mid-decode, possibly every few tokens for a long request. And if there's no big-enough free region, I'm stuck: evict a request or crash. So small contiguous reservations trade dead memory for catastrophic copies and out-of-memory mid-generation. Dead end.

Patch two: keep max reservation but raise the configured batch size anyway and hope the requests are short. But "hope they're short" isn't a system; the worst case is exactly when several long requests coincide, and then I overflow and have to evict. I can't safely turn the batch knob up while each request demands a contiguous max-length slab. The knob I want to turn is gated by the *contiguity* requirement itself.

So the contiguity is the enemy. Why did I demand it? Because the attention kernel, to read a request's keys and values, wants them laid out contiguously so it can stride through them. That's the only reason. The cache being one physical contiguous block is a convenience for the kernel, and I'm paying for that convenience with most of my memory.

Now — this exact problem has a name in another field, and I've been circling it. A process needs an address space that can grow, the OS has a fixed pool of physical memory, processes come and go leaving holes, and you must not require each process's memory to be one physical contiguous slab. The OS solution is *paging*: chop physical memory into fixed-size **pages**, give each process a **page table** mapping its contiguous *logical* address space to *scattered physical* pages, and let the hardware translate logical→physical on every access. The process *thinks* its memory is contiguous; physically it's sprinkled across whatever pages were free. Growth is just grabbing one more free page and adding a page-table entry — no copying, no need for a contiguous run. Fragmentation drops to at most one partially-filled page per process (internal fragmentation bounded by the page size), and external fragmentation vanishes entirely because any free page is as good as any other.

That maps onto the KV cache almost one-to-one, and the mapping tells me exactly what to build. Carve the whole KV region into fixed-size **blocks**, each holding the keys/values for a fixed number of tokens — say 16 tokens per block. A request no longer owns a contiguous chunk; it owns a *list* of block numbers — its **block table** — one entry per 16-token span of its sequence, pointing at whatever physical blocks happened to be free. Logically, position *t* in the request lives at logical block ⌊t / 16⌋, offset *t* mod 16. Physically, that logical block is `block_table[t / 16]`, some arbitrary block number in the pool. When the request generates past the end of its last block, I pop one free block off a global free list and append its number to the block table — O(1), no copy, no contiguity needed. When the request finishes, I push all its blocks back onto the free list. Over-reservation is gone: a request holds only ⌈len / 16⌉ blocks, and the only waste is the unfilled tail of its *last* block — at most 15 tokens, bounded by the block size, not by the max sequence length. External fragmentation is gone: every free block is interchangeable.

The catch — the thing that makes this real work and not just a memory-allocator swap — is that the attention kernel can no longer assume a request's KV is contiguous. It now has to do exactly what the OS hardware does: translate logical→physical on the fly. So I have to push the page table *into the attention kernel*. As the kernel walks over a request's past tokens block by block, for each logical block it reads the physical block number out of the block table, then jumps to that physical block in the cache and reads the 16 tokens' keys there. The contiguity that the standard kernel relied on is replaced by one indirection per block: `physical_block = block_table[logical_block]`, then address into the cache at that physical block. That indirection per 16 tokens is cheap — negligible against the actual key/value reads — and in exchange it buys me the ability to scatter the cache and pack the batch.

Let me write down the kernel's inner structure to make sure the indirection actually composes with how attention reads keys. The kernel is launched per (head, sequence). For a given sequence it needs to attend the query over all past key blocks. So: get this sequence's block table — it's a row in the big `[num_seqs, max_num_blocks_per_seq]` block-table tensor, so `block_table = block_tables + seq_idx * max_num_blocks_per_seq`. Then loop over the logical blocks of this sequence (warps split the blocks among themselves for parallelism). For each logical `block_idx`, read the physical block number: `physical_block_number = block_table[block_idx]`. One subtlety I have to get right: the block numbers are stored as int32 to keep the table small, but when I use a block number to index into the cache I multiply it by the block's stride in elements (`kv_block_stride`), and for a large cache that product overflows int32 — so I must widen the block number to int64 *before* multiplying: `static_cast<int64_t>(block_table[block_idx])`. Miss that and large caches silently corrupt. Then the address of a key inside that physical block is `k_cache + physical_block_number * kv_block_stride + kv_head_idx * kv_head_stride + physical_block_offset * x`, where the offset is the token's position *within* the 16-token block. That's the whole translation: logical block → physical block number (the page-table lookup) → physical address (the stride arithmetic). The rest of the kernel — the QK dot products, the softmax, the weighting of values — is unchanged; only *where it fetches keys and values from* now goes through the block table.

There's a bonus the paging view hands me for free, which I'll flag now and exploit fully later: because a request's cache is addressed through a block table of *block numbers*, two requests can have block tables that *point at the same physical block*. If two requests share a prefix — same system prompt, same few-shot preamble, or one prompt forked into several samples — their block tables can list the same physical block numbers for the shared span, storing the identical keys/values exactly once. Contiguous per-request caches couldn't do that; each kept its own copy. Block tables make sharing a pointer-level operation, with reference counting on blocks so a block is only freed when the last sharer drops it. I won't build the sharing machinery this rung — getting the paged layout and the paged kernel right is enough to collapse the over-reservation and fragmentation that's capping my batch — but the door is now open.

So the change this rung is: stop reserving contiguous per-request cache; carve the KV region into fixed-size blocks managed by a free list; give each request a block table mapping its logical token positions to physical blocks; and rewrite the attention kernel to translate logical→physical through that block table as it reads keys and values. The payoff is direct — KV waste drops from "most of the cache" to "at most one partial block per request," the batch size rises to whatever the *live* tokens actually need rather than what the reservations demanded, and since throughput at fixed latency tracks batch size, that headroom is the throughput.

The heart of it is the paged read in the attention kernel — the per-block logical→physical translation that replaces the contiguity assumption.

```cpp
// vllm/csrc/.../attention_kernels.cuh — paged_attention_kernel (excerpt).
// Launched per (head, sequence). The KV cache is one big pool of fixed-size
// physical blocks; this sequence owns a *block table* of physical block numbers,
// one per logical 16-token span. The kernel translates logical->physical here.

// This sequence's page table: a row in [num_seqs, max_num_blocks_per_seq].
const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

// Iterate over the key blocks; warps split the logical blocks among themselves.
for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
     block_idx += NUM_WARPS) {
  // The page-table lookup: logical block_idx -> physical block number.
  // Stored int32 to keep the table small, but widened to int64 BEFORE the
  // stride multiply below, or large caches overflow the index.
  const int64_t physical_block_number =
      static_cast<int64_t>(block_table[block_idx]);

  // Read the keys for this physical block. The address goes through the
  // physical block number * the per-block stride -- the cache is scattered,
  // so contiguity is gone; this indirection is what replaces it.
  for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
    const int physical_block_offset =
        (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;  // pos within the block
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    K_vec k_vecs[NUM_VECS_PER_THREAD];
#pragma unroll
    for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
      const cache_t* k_ptr =
          k_cache + physical_block_number * kv_block_stride   // <- paged address
                  + kv_head_idx * kv_head_stride
                  + physical_block_offset * x;
      // ... load k_vecs[j] from k_ptr and dot with the query as usual ...
    }
  }
  // QK^T, softmax, and the value-weighting that follow are unchanged; only the
  // *location* of the keys/values now flows through the block table.
}
```
