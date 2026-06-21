We want to push more requests through a single GPU when serving a large language model, and the only lever that matters is the batch size we can sustain, because batching is what amortizes the cost of streaming the model weights across many in-flight requests. The batch we can fit is capped by GPU memory, and when we look at where the memory actually goes, the answer is the key-value cache: every in-flight request must keep the key and value vectors of all its tokens, because each new token attends back over the entire history. For a 13B OPT model one token's KV is about $800$ KB (two vectors $\times$ 5120 hidden $\times$ 40 layers $\times$ 2 bytes), and a 2048-token request can need $1.6$ GB. On a tens-of-GB GPU that is only a few tens of concurrent requests even if every byte went to KV cache, and the hardware trend makes it worse — FLOPS are growing far faster than memory capacity, so serving is increasingly memory-bound and the memory in question is the KV cache. Storing it more efficiently buys a bigger batch directly.

The trouble is how the KV cache is stored today. The deep-learning stack wants contiguous tensors, but the output length of a request is unknown until the model emits an end token, so existing systems reserve, up front, one contiguous slab sized to the maximum possible sequence length, for every request. Accounting the waste in that single decision is sobering. Take a request that could run to 2048 tokens but generates only 30: for its whole lifetime the slots past the current token are *reserved* but empty, occupied so no one else can use them; when it finishes, the roughly 2000 reserved-but-never-used slots were pure *internal fragmentation* that we only learn about at the end; and because different requests reserve different-sized slabs, the allocator leaves unusable gaps between them — *external fragmentation*. Profiling of such systems shows live-token utilization as low as $20.4\%$, so roughly four-fifths of the most precious resource is wasted. Compaction — sliding the live KV together — is no escape in a latency-sensitive loop, since the cache is enormous and moving it constantly would wreck tail latency. And the contiguous slab has a second, deeper failure: it forbids *sharing*. Parallel sampling draws several continuations from one prompt and every sample shares the entire prompt's KV; beam search keeps $k$ candidates over long, shifting common prefixes; shared system prompts have thousands of requests all beginning with the same long instruction block. In each case identical KV could be stored once, but private contiguous slabs duplicate it in every slab, and beam search degenerates into constant bulk KV copies as candidates fork and die. Stated cleanly, what we need is KV storage that grows one token at a time without reserving the maximum, and that can be shared at fine granularity with other sequences holding the same tokens. The contiguous tensor fails both, so the assumption that a request's KV must live in one contiguous region has to go.

This is exactly the problem the operating system solved with virtual memory and paging: a process needs a logically contiguous, growing address space, we refuse to commit contiguous physical memory to it, and we want multiple processes to share pieces. So I propose **PagedAttention**, the mechanism behind vLLM, which pages the KV cache. We partition each sequence's KV cache into fixed-size **KV blocks**, each holding the keys and values for $B$ consecutive tokens — these are the "pages." A request's KV cache becomes a list of *logical* KV blocks, filled left to right as tokens arrive, with the last block's unfilled slots reserved for the next few generations. The GPU KV region is carved into *physical* KV blocks, and a per-request **block table** maps each logical block to a physical block (and records how many slots are filled). Logical blocks look contiguous to the model but the physical blocks they point to can sit anywhere. A new physical block is allocated only when the current last block fills up; nothing is reserved to the maximum length. Because every physical block is the same size, the allocator never leaves variable-size external holes, and the only internal waste per request is the unfilled tail of its last block — strictly less than one block.

Attention now has to read keys and values that are scattered across non-contiguous physical blocks, but attention is a sum, so we group the sum over blocks with no change to the result. Letting block $j$ hold $K_j=(k_{(j-1)B+1},\dots,k_{jB})$ and $V_j$ similarly, the per-block scores, normalizer, attention weights, and output for query $q_i$ are
$$s_{ij}=q_i^\top K_j/\sqrt d,\qquad Z_i=\sum_{t=1}^{\lceil i/B\rceil}\exp(s_{it})\mathbf 1,\qquad A_{ij}=\frac{\exp(s_{ij})}{Z_i},\qquad o_i=\sum_{j=1}^{\lceil i/B\rceil}V_jA_{ij}^\top .$$
The kernel walks the block table, fetches each physical block in turn, multiplies $q_i$ against that block's keys to get the block's scores, and accumulates $V_jA_{ij}^\top$ into the output; for the last block it slices to the filled slots and ignores the unused tail. The denominator $Z_i$ normalizes across all $\lceil i/B\rceil$ blocks, so the softmax is still over the full history — nothing about the result changes, only where the data sits.

The sharing then falls out of the block table almost for free, again copying the OS. If two sequences share a prompt, we map both sequences' logical prompt-blocks to the *same* physical blocks and store the prompt once. Since a physical block can be pointed at by several logical blocks, each physical block carries a **reference count**; freeing decrements it and a block is actually reclaimed only at count zero. The hazard with sharing is writes — if samples A and B share a block and A appends a token's KV into it, B would see A's token — and the OS answer, **copy-on-write**, transfers directly: when a sequence needs to write into a shared block (reference count $> 1$), we allocate a fresh physical block, copy the shared block's contents into it, decrement the original's count, and let the writer use its private copy while the other sharers keep the original. In parallel sampling all samples share every full prompt block and only the last, partially-filled block — the one being written as they diverge — triggers a copy, so the prompt, usually the bulk of the tokens, is stored once. Beam search is the same mechanism over a shifting tree: candidates share the blocks of their common ancestry, copy-on-write the single block at the point of divergence, and when a candidate dies its blocks' reference counts drop and any that reach zero are freed — replacing Orca's constant bulk KV copies with at most a one-block copy. A shared system prompt is just physical blocks pre-populated once and mapped read-only (last block copy-on-write) into every request that uses it, which also skips the prompt-phase compute for that prefix.

Two design choices the block view forces are worth spelling out. The block size $B$ is a genuine tradeoff between two failure modes. Too small, and PagedAttention processes too few tokens per block to use the GPU's parallelism well, and the block table grows long. Too large, and the last-block tail waste grows while the *probability* that two sequences can share a block drops, because sharing is block-granular and a big block is less likely to be wholly common. A 16-token block is the conservative compromise — enough tokens to keep the kernel efficient, small enough that tail waste and lost sharing stay negligible — so $B=16$ by default. The second choice is recovery under memory pressure, because on-demand allocation can still exhaust physical blocks on a traffic spike. Two observations shape the policy: all blocks of a sequence are always accessed together (decoding the next token needs the whole history), so eviction degenerates to all-or-nothing per sequence, and sequences entangled by sharing (beam candidates of one request) must be gang-evicted together. For recovering an evicted sequence there are two clean options — *swapping* its blocks out to CPU RAM and back, where the CPU swap space stays bounded by the GPU KV region because we never swap out more than fits in it, or *recomputation*, throwing the KV away and regenerating it when the sequence is rescheduled, which is cheap because the already-generated tokens are concatenated with the original prompt and run through a single parallel prompt-phase pass rather than token-by-token. Which wins depends on block size and PCIe bandwidth — small blocks make swap's many tiny transfers slow so recompute wins, large blocks favor swap — so we keep both and let the regime decide, under an FCFS policy with preempt-latest-first to stay fair and starvation-free. Finally, the indirection layer buys distributed execution: under tensor parallelism each worker holds the KV for its own heads but all workers process the same token positions, so a single centralized block manager computes one logical-to-physical mapping and broadcasts it with the per-step inputs; every worker uses the same physical block IDs (storing only its head-slice of each block) and never coordinates memory management with the others. The model just sees a list of physical block IDs per sequence, and the mapping layer hides all the sharing.

The chain is therefore: serving throughput is capped by KV-cache memory; the contiguous, max-length per-request layout wastes about $80\%$ of it to reservation and fragmentation and makes sharing impossible; the OS solved this exact growing-contiguous-shareable-address-space problem with paging, so we page the KV cache into fixed-size blocks with a per-request block table allocated on demand; a block-wise PagedAttention kernel reads the scattered blocks with no change to the math; reference counts plus copy-on-write give fine-grained sharing for parallel sampling, beam search, and shared prefixes; and all-or-nothing eviction with swap-or-recompute recovery, plus a single shared block manager under tensor parallelism, complete the system. The serving engine exposes `fork`/`append`/`free` so any decoding algorithm composes from these, custom CUDA kernels fuse reshape-and-block-write, block-read-and-attention, and batched block-copy, and near-zero KV waste plus cross-request sharing let many more requests batch together, raising serving throughput $2$–$4\times$ at equal latency versus FasterTransformer and Orca.

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
