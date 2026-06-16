# PagedAttention (vLLM)

**Problem.** LLM serving throughput is capped by GPU memory, dominated by the KV cache (e.g. ~800 KB/token, up to 1.6 GB/request for 13B OPT). Existing systems store each request's KV cache as one contiguous tensor pre-allocated to the maximum sequence length, wasting ~60–80% of KV memory to reserved slots, internal fragmentation, and external fragmentation, and making KV-cache sharing across requests impossible.

**Key idea — apply OS paging to the KV cache.** Partition each sequence's KV cache into fixed-size **KV blocks** of $B$ tokens (the "pages"). A request's KV cache is a list of *logical* blocks; a per-request **block table** maps each logical block to a *physical* block that can sit anywhere in the GPU KV region. Physical blocks are allocated **on demand** — a new one only when the last block fills — so nothing is reserved for the maximum length, fixed-size blocks eliminate external fragmentation, and internal waste is bounded by under one block per request. **PagedAttention** is an attention kernel that reads these scattered blocks; with $K_j=(k_{(j-1)B+1},\dots,k_{jB})$, $V_j$ similarly,
$$s_{ij}=q_i^\top K_j/\sqrt d,\qquad Z_i=\sum_{t=1}^{\lceil i/B\rceil}\exp(s_{it})\mathbf 1,\qquad A_{ij}=\frac{\exp(s_{ij})}{Z_i},\qquad o_i=\sum_{j=1}^{\lceil i/B\rceil}V_jA_{ij}^\top,$$
i.e. standard attention iterated over blocks, with the final block sliced to its filled slots — same result, arbitrary physical layout.

**Sharing via reference counts + copy-on-write.** Multiple logical blocks (across sequences) may map to one physical block; a per-block reference count tracks sharers, and a block is freed only at count 0. When a sequence writes into a shared block (ref > 1), it copy-on-writes: allocate a fresh block, copy, decrement the original. This gives near-free sharing for parallel sampling (share all full prompt blocks, COW only the diverging last block), beam search (share shifting prefixes, COW one block instead of bulk KV copies), and shared system prefixes (pre-populate once, map read-only into many requests).

**Block size & recovery.** $B=16$ by default — large enough for kernel efficiency, small enough that tail waste and lost sharing stay negligible. Under memory pressure, evict all-or-nothing per sequence (all blocks are accessed together) and gang-evict shared sequence groups; recover by **swapping** to CPU RAM (bounded by GPU KV size) or **recomputation** (cheap: regenerate the dropped KV in one parallel prompt-phase pass over prompt+generated tokens). FCFS with preempt-latest-first. Under tensor parallelism a single centralized block manager broadcasts one logical→physical mapping to all workers.

```python
import torch
from dataclasses import dataclass


@dataclass
class BlockTable:
    blocks: list[int]
    filled: int = 0

class BlockSpaceManager:
    def __init__(self, num_blocks, block_size):
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        self.ref_count = [0] * num_blocks
        self.tables = {}

    def _alloc(self):
        pb = self.free_blocks.pop()
        self.ref_count[pb] = 1
        return pb

    def append(self, seq_id):
        table = self.tables.setdefault(seq_id, BlockTable([]))
        logical_idx = table.filled // self.block_size
        offset = table.filled % self.block_size
        if offset == 0:
            table.blocks.append(self._alloc())
        else:
            pb = table.blocks[logical_idx]
            if self.ref_count[pb] > 1:
                new_pb = self._alloc()
                copy_block(pb, new_pb)
                self.ref_count[pb] -= 1
                table.blocks[logical_idx] = new_pb
        pb = table.blocks[logical_idx]
        table.filled += 1
        return pb, offset

    def fork(self, parent_id, child_id):
        parent = self.tables[parent_id]
        self.tables[child_id] = BlockTable(list(parent.blocks), parent.filled)
        for pb in parent.blocks:
            self.ref_count[pb] += 1

    def free(self, seq_id):
        for pb in self.tables.pop(seq_id).blocks:
            self.ref_count[pb] -= 1
            if self.ref_count[pb] == 0:
                self.free_blocks.append(pb)


def paged_attention(q_i, table, k_cache, v_cache, block_size, d):
    scores, values = [], []
    remaining = table.filled
    for pb in table.blocks:                            # blocks at arbitrary addresses
        n = min(block_size, remaining)
        if n <= 0:
            break
        scores.append(q_i @ k_cache[pb, :n].t() / d**0.5)
        values.append(v_cache[pb, :n])
        remaining -= n
    A_i = torch.softmax(torch.cat(scores), dim=-1)     # softmax over full history
    return A_i @ torch.cat(values, dim=0)              # o_i = sum_j V_j A_ij^T
```

The serving engine exposes `fork`/`append`/`free` so any decoding algorithm composes from these, and custom CUDA kernels fuse reshape-and-block-write, block-read-and-attention, and batched block-copy. Near-zero KV waste plus cross-request sharing let many more requests batch together, raising serving throughput 2–4× at equal latency versus FasterTransformer and Orca.
