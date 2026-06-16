I'm trying to push more requests through one GPU when serving an LLM. Throughput comes from batching — share the weight-loading cost across many requests in flight — and the batch size I can sustain is capped by GPU memory. So I look at where the memory goes, and the answer is the KV cache: every in-flight request has to keep the key and value vectors of all its tokens, because each new token attends back over all previous ones. For a 13B model one token's KV is about 800 KB (two vectors, 5120 hidden, 40 layers, two bytes), and a 2048-token request can need 1.6 GB. On a tens-of-GB GPU that's only a few tens of concurrent requests even if every byte went to KV cache. And the hardware trend is against me — FLOPS are growing much faster than memory capacity — so this is only going to get tighter. The serving system is memory-bound, and the memory in question is the KV cache. If I can store the KV cache more efficiently, I directly buy a bigger batch and more throughput.

So how is the KV cache stored today? The deep-learning stack wants tensors to be contiguous, so each request gets one contiguous slab of memory. But here's the thing: I don't know how long a request will run. The output length isn't known until the model emits an end token. So the system reserves a contiguous slab sized to the *maximum possible* sequence length, up front, for every request.

Let me actually account for the waste in that one decision, because it's worse than it looks. Take a request that could run to 2048 tokens but only generates 30. First, for the whole time it's running, the slots past the current token are *reserved* but empty — they hold nothing until those tokens are generated, and most of them never are, yet they're occupied so other requests can't use them. Second, when the request finishes at 30 tokens, the ~2000 reserved-but-never-used slots were pure *internal fragmentation* — I only learn the true length at the end. Third, because different requests reserve different-sized slabs, the allocator leaves unusable gaps between them — *external fragmentation*. Existing-system profiling says actual live-token utilization can be as low as 20.4%. I'm wasting roughly four-fifths of my most precious resource.

Could I just compact — slide the live KV together to close the gaps? In a latency-sensitive serving loop, no: the KV cache is enormous and moving it around constantly would wreck tail latency. And even if I compacted, the contiguous-slab design has a second, deeper problem.

That second problem is sharing. Look at the decoding algorithms people actually use. Parallel sampling: one prompt, several sampled continuations — every sample shares the *entire prompt's* KV cache. Beam search: $k$ candidates that share long prefixes, and the shared prefixes shift around as the beam evolves. Shared system prompts: thousands of requests that all begin with the same long instruction block. In every case there's KV cache that is *identical* across sequences and could be stored once. But if each sequence owns a private contiguous slab, the shared tokens get *duplicated* in every slab, and worse, beam search ends up doing constant large memory copies as candidates fork and die. The contiguous layout doesn't just waste memory to fragmentation — it forbids sharing.

So the real requirement, stated cleanly: I need to give each request KV storage that (a) grows one token at a time without reserving the maximum up front, and (b) can be shared, at fine granularity, with other sequences that have the same tokens. The contiguous tensor fails both. I need to break the assumption that a request's KV cache lives in one contiguous region.

Where have I seen exactly this problem — a process that needs a logically contiguous, growing address space, but I refuse to commit contiguous physical memory to it, and I want multiple processes to share pieces? That is the operating system's virtual memory and paging. The OS chops physical memory into fixed-size *pages*, gives each process *logical* pages that it sees as contiguous, and keeps a *page table* mapping logical pages to scattered physical pages. Physical pages are allocated on demand as the process touches more address space — nothing is reserved up front. And two processes can map their logical pages to the *same* physical page to share it. That is the precise shape of what I need for the KV cache.

So: partition each sequence's KV cache into fixed-size **KV blocks**, each holding the keys and values for a fixed number $B$ of consecutive tokens — these are my "pages." A request's KV cache becomes a list of *logical* KV blocks, filled left to right as tokens arrive; the last block has some unfilled slots reserved for the next few generations. On the GPU I carve the KV region into *physical* KV blocks and keep a **block table** per request mapping each logical block to a physical block (and recording how many slots are filled). Logical blocks that look contiguous to the model can sit anywhere in physical memory. A new physical block is allocated only when the current last block fills up. Because every physical block has the same size, the allocator no longer leaves variable-size external holes; because I never reserve to the maximum length, the only internal waste per request is at most the unfilled tail of its last block, i.e. less than one block.

But attention needs to read all the keys and values, and now they're scattered across non-contiguous physical blocks. So I need an attention computation that works block by block. That's fine — attention is a sum, and I can group the sum over blocks. Let the keys of block $j$ be $K_j=(k_{(j-1)B+1},\dots,k_{jB})$ and values $V_j$ similarly. For query $q_i$, the attention over block $j$ is the row vector $A_{ij}=(a_{i,(j-1)B+1},\dots,a_{i,jB})$, and

$$s_{ij}=q_i^\top K_j/\sqrt d,\qquad Z_i=\sum_{t=1}^{\lceil i/B\rceil}\exp(s_{it})\mathbf 1,\qquad A_{ij}=\frac{\exp(s_{ij})}{Z_i},\qquad o_i=\sum_{j=1}^{\lceil i/B\rceil}V_jA_{ij}^\top .$$

The kernel walks the block table, fetches each physical block in turn, multiplies $q_i$ against that block's keys to get the block's scores, and accumulates $V_jA_{ij}^\top$ into the output — exactly the standard attention, just iterated over blocks that live at arbitrary physical addresses. For the last block, it slices to the filled token slots and ignores the unused tail. Call this PagedAttention. The denominator normalizes across all $\lceil i/B\rceil$ blocks, so the softmax is over the full history; nothing about the result changes, only where the data sits.

Now the sharing falls out of the block table almost for free, again following the OS. If two sequences share a prompt, I map *both* sequences' logical prompt-blocks to the *same* physical blocks — store the prompt once. Since a physical block can now be pointed at by several logical blocks, I attach a **reference count** to each physical block. Freeing a block decrements the count; the block is actually reclaimed only at count zero. This is page sharing.

The hazard with sharing is writes: if sample A and sample B share a physical block and A appends a new token's KV into it, B would see A's token. The OS answer is **copy-on-write**, and it transfers directly. When a sequence needs to *write* into a shared block (reference count > 1), I allocate a fresh physical block, copy the shared block's contents into it, decrement the original's count, and let the writer use its private copy; the other sharers keep the original. So in parallel sampling, all samples share every full prompt block, and only the *last*, partially-filled block — the one being written into as they diverge — triggers a copy. The prompt, often the bulk of the tokens, is stored once. Beam search is the same mechanism with a richer, shifting tree: candidates share all the blocks of their common ancestry, copy-on-write only the block at the point of divergence, and when a candidate dies its blocks' reference counts drop and any that hit zero are freed — replacing Orca's constant bulk KV copies with at most a one-block copy. A shared system prompt is just a set of physical blocks pre-populated once and mapped (read-only, last block copy-on-write) into every request that uses it, so the prompt-phase compute for the prefix is skipped too.

A couple of design choices the block view forces me to make. What's the block size $B$? It's a tradeoff and I can reason it out from the two failure modes. Too small, and the PagedAttention kernel processes too few tokens per block to use the GPU's parallelism well, and the block table grows long. Too large, and the last-block waste (internal fragmentation) grows and the *probability* that two sequences can share a block drops, because sharing is block-granular — a big block is less likely to be wholly common. A 16-token block is the conservative compromise: enough tokens to keep the kernel efficient, small enough that tail waste and lost sharing stay negligible. So $B=16$ by default.

The other choice is what to do under memory pressure. With on-demand allocation I can still run out of physical blocks when traffic spikes. I need to evict some sequence's blocks and recover them later. Two observations shape the policy. First, all blocks of a sequence are always accessed together (you need the whole history to decode the next token), so the usual "evict the block accessed furthest in the future" heuristic degenerates to all-or-nothing per sequence — evict either all of a sequence's blocks or none. And sequences that share memory (beam candidates of one request) must be evicted together, as a gang, since their blocks are entangled by sharing. Second, for *recovering* an evicted sequence there are two clean options. Swapping: copy its blocks out to CPU RAM and back, classic OS swap to a "disk" that is the CPU; because I never swap out more than fits in the GPU KV region, the CPU swap space is bounded by it. Recomputation: just throw the KV away and recompute it when the sequence is rescheduled — and crucially that recompute is cheap, because the already-generated tokens can be concatenated with the original prompt and run through a single parallel prompt-phase pass rather than token-by-token. Which wins depends on block size and PCIe bandwidth (small blocks make swap's many tiny transfers slow, so recompute wins; large blocks favor swap), so I keep both and let the regime decide. I run an FCFS policy with preempt-latest-first to keep it fair and starvation-free.

One more thing the indirection layer buys me: distributed execution. Under tensor parallelism each GPU worker holds the KV for its own attention heads, but every worker processes the *same* token positions, so they all need blocks for the same logical positions. So I keep a *single* block manager in the centralized scheduler; it computes one logical→physical mapping and broadcasts it with the per-step inputs. Each worker uses the same physical block IDs (storing only its head-slice of each block) and never has to coordinate memory management with the others — the common mapping layer hides all the sharing from the model, which just sees a list of physical block IDs per sequence.

The causal chain: serving throughput is capped by KV-cache memory; the contiguous, max-length per-request layout wastes ~80% of it to reservation and fragmentation and makes sharing impossible; the OS solved this exact growing-contiguous-shareable-address-space problem with paging, so I page the KV cache — fixed-size KV blocks, a per-request block table mapping logical to scattered physical blocks, allocated on demand; a block-wise PagedAttention kernel reads the scattered blocks with no change to the math; reference counts plus copy-on-write give fine-grained sharing for parallel sampling, beam search, and shared prefixes; and an all-or-nothing eviction with swap-or-recompute recovery, plus a single shared block manager under tensor parallelism, complete the system. Here is the manager and the paged kernel.

```python
import torch
from dataclasses import dataclass


@dataclass
class BlockTable:
    blocks: list[int]             # logical block index -> physical block id
    filled: int = 0               # number of valid token slots in this sequence

class BlockSpaceManager:
    def __init__(self, num_blocks, block_size):
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks)) # physical block ids
        self.ref_count = [0] * num_blocks          # physical block -> refs
        self.tables = {}                           # seq id -> BlockTable

    def _alloc(self):
        pb = self.free_blocks.pop()              # on-demand: no max reservation
        self.ref_count[pb] = 1
        return pb

    def append(self, seq_id):
        table = self.tables.setdefault(seq_id, BlockTable([]))
        logical_idx = table.filled // self.block_size
        offset = table.filled % self.block_size

        if offset == 0:                           # previous logical blocks are full
            table.blocks.append(self._alloc())
        else:
            pb = table.blocks[logical_idx]
            if self.ref_count[pb] > 1:            # copy-on-write for a shared tail
                new_pb = self._alloc()
                copy_block(pb, new_pb)            # fused block-copy kernel
                self.ref_count[pb] -= 1
                table.blocks[logical_idx] = new_pb

        pb = table.blocks[logical_idx]
        table.filled += 1
        return pb, offset                         # slot for the next token's KV

    def fork(self, parent_id, child_id):
        # child shares all parent blocks (e.g. parallel sample / beam child)
        parent = self.tables[parent_id]
        self.tables[child_id] = BlockTable(list(parent.blocks), parent.filled)
        for pb in parent.blocks:
            self.ref_count[pb] += 1

    def free(self, seq_id):
        for pb in self.tables.pop(seq_id).blocks:
            self.ref_count[pb] -= 1
            if self.ref_count[pb] == 0:            # reclaim only at zero refs
                self.free_blocks.append(pb)


def paged_attention(q_i, table, k_cache, v_cache, block_size, d):
    # walk the block table; mask off the unfilled tail slots
    scores, values = [], []
    remaining = table.filled
    for pb in table.blocks:
        n = min(block_size, remaining)
        if n <= 0:
            break
        K_j = k_cache[pb, :n]                      # n x d at arbitrary address
        V_j = v_cache[pb, :n]
        scores.append(q_i @ K_j.t() / d**0.5)      # q_i . k for this block
        values.append(V_j)
        remaining -= n
    A_i = torch.softmax(torch.cat(scores), dim=-1) # softmax over the full history
    return A_i @ torch.cat(values, dim=0)          # o_i = sum_j V_j A_ij^T
```
