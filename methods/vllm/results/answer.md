# PagedAttention (vLLM)

**Problem.** LLM serving throughput is capped by GPU memory, dominated by the KV cache (e.g. ~800 KB/token, up to 1.6 GB/request for 13B OPT). Existing systems store each request's KV cache as one contiguous tensor pre-allocated to the maximum sequence length, wasting ~60–80% of KV memory to reserved slots, internal fragmentation, and external fragmentation, and making KV-cache sharing across requests impossible.

**Key idea — apply OS paging to the KV cache.** Partition each sequence's KV cache into fixed-size **KV blocks** of $B$ tokens (the "pages"). A request's KV cache is a list of *logical* blocks; a per-request **block table** maps each logical block to a *physical* block that can sit anywhere in the GPU KV region. Physical blocks are allocated **on demand** — a new one only when the last block fills — so nothing is reserved for the maximum length and waste is bounded by under one block per request. **PagedAttention** is an attention kernel that reads these scattered blocks; with $K_j=(k_{(j-1)B+1},\dots,k_{jB})$, $V_j$ similarly,
$$A_{ij}=\frac{\exp(q_i^\top K_j/\sqrt d)}{\sum_{t=1}^{\lceil i/B\rceil}\exp(q_i^\top K_t\mathbf 1/\sqrt d)},\qquad o_i=\sum_{j=1}^{\lceil i/B\rceil}V_jA_{ij}^\top,$$
i.e. standard attention iterated over blocks — same result, arbitrary physical layout.

**Sharing via reference counts + copy-on-write.** Multiple logical blocks (across sequences) may map to one physical block; a per-block reference count tracks sharers, and a block is freed only at count 0. When a sequence writes into a shared block (ref > 1), it copy-on-writes: allocate a fresh block, copy, decrement the original. This gives near-free sharing for parallel sampling (share all full prompt blocks, COW only the diverging last block), beam search (share shifting prefixes, COW one block instead of bulk KV copies), and shared system prefixes (pre-populate once, map read-only into many requests).

**Block size & recovery.** $B=16$ by default — large enough for kernel efficiency, small enough that tail waste and lost sharing stay negligible. Under memory pressure, evict all-or-nothing per sequence (all blocks are accessed together) and gang-evict shared sequence groups; recover by **swapping** to CPU RAM (bounded by GPU KV size) or **recomputation** (cheap: regenerate the dropped KV in one parallel prompt-phase pass over prompt+generated tokens). FCFS with preempt-latest-first. Under tensor parallelism a single centralized block manager broadcasts one logical→physical mapping to all workers.

```python
import torch

class BlockSpaceManager:
    def __init__(self, num_blocks, block_size):
        self.block_size = block_size
        self.free = list(range(num_blocks))
        self.ref_count, self.block_tables = {}, {}

    def _alloc(self):
        pb = self.free.pop(); self.ref_count[pb] = 1; return pb

    def allocate_for_new_token(self, seq_id, num_filled):
        table = self.block_tables.setdefault(seq_id, [])
        if num_filled % self.block_size == 0:          # only when last block is full
            table.append(self._alloc())
        return table[-1]

    def fork(self, parent_id, child_id):               # share parent blocks
        table = list(self.block_tables[parent_id])
        for pb in table: self.ref_count[pb] += 1
        self.block_tables[child_id] = table

    def append_with_cow(self, seq_id, logical_idx):
        table = self.block_tables[seq_id]; pb = table[logical_idx]
        if self.ref_count[pb] > 1:                     # copy-on-write
            new_pb = self._alloc(); copy_block(pb, new_pb)
            self.ref_count[pb] -= 1; table[logical_idx] = new_pb; return new_pb
        return pb

    def free(self, seq_id):
        for pb in self.block_tables.pop(seq_id):
            self.ref_count[pb] -= 1
            if self.ref_count[pb] == 0: self.free.append(pb)


def paged_attention(q_i, block_table, k_cache, v_cache, d):
    scores, values = [], []
    for pb in block_table:                             # blocks at arbitrary addresses
        scores.append(q_i @ k_cache[pb].t() / d**0.5)
        values.append(v_cache[pb])
    A_i = torch.softmax(torch.cat(scores), dim=-1)     # softmax over full history
    return A_i @ torch.cat(values, dim=0)              # o_i = sum_j V_j A_ij^T
```

The serving engine exposes `fork`/`append`/`free` so any decoding algorithm composes from these, and custom CUDA kernels fuse reshape-and-block-write, block-read-and-attention, and batched block-copy. Near-zero KV waste plus cross-request sharing let many more requests batch together, raising serving throughput 2–4× at equal latency versus FasterTransformer and Orca.
