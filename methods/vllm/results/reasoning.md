I'm trying to push more requests through one GPU when serving an LLM. Throughput comes from batching — share the weight-loading cost across many requests in flight — and the batch size I can sustain is capped by GPU memory. So I look at where the memory goes, and the answer is the KV cache: every in-flight request has to keep the key and value vectors of all its tokens, because each new token attends back over all previous ones. For a 13B model one token's KV is about 800 KB; let me confirm that figure rather than trust it — two vectors (K and V), 5120 hidden, 40 layers, two bytes in FP16 gives $2\times5120\times40\times2 = 819{,}200$ bytes, which is exactly 800 KB. A 2048-token request then needs $800\text{ KB}\times2048 = 1.56$ GB. On a tens-of-GB GPU that's only a few tens of concurrent requests even if every byte went to KV cache. And the hardware trend is against me — FLOPS are growing much faster than memory capacity — so this is only going to get tighter. The serving system is memory-bound, and the memory in question is the KV cache. If I can store the KV cache more efficiently, I directly buy a bigger batch and more throughput.

So how is the KV cache stored today? The deep-learning stack wants tensors to be contiguous, so each request gets one contiguous slab of memory. But here's the thing: I don't know how long a request will run. The output length isn't known until the model emits an end token. So the system reserves a contiguous slab sized to the *maximum possible* sequence length, up front, for every request.

Let me actually account for the waste in that one decision, because it's worse than it looks. Take a request that could run to 2048 tokens but only generates 30. First, for the whole time it's running, the slots past the current token are *reserved* but empty — they hold nothing until those tokens are generated, and most of them never are, yet they're occupied so other requests can't use them. Second, when the request finishes at 30 tokens, the ~2000 reserved-but-never-used slots were pure *internal fragmentation* — I only learn the true length at the end. Third, because different requests reserve different-sized slabs, the allocator leaves unusable gaps between them — *external fragmentation*. Existing-system profiling says actual live-token utilization can be as low as 20.4%. I'm wasting roughly four-fifths of my most precious resource.

Could I just compact — slide the live KV together to close the gaps? In a latency-sensitive serving loop, no: the KV cache is enormous and moving it around constantly would wreck tail latency. And even if I compacted, the contiguous-slab design has a second, deeper problem.

That second problem is sharing. Look at the decoding algorithms people actually use. Parallel sampling: one prompt, several sampled continuations — every sample shares the *entire prompt's* KV cache. Beam search: $k$ candidates that share long prefixes, and the shared prefixes shift around as the beam evolves. Shared system prompts: thousands of requests that all begin with the same long instruction block. In every case there's KV cache that is *identical* across sequences and could be stored once. But if each sequence owns a private contiguous slab, the shared tokens get *duplicated* in every slab, and worse, beam search ends up doing constant large memory copies as candidates fork and die. The contiguous layout doesn't just waste memory to fragmentation — it forbids sharing.

So the real requirement, stated cleanly: I need to give each request KV storage that (a) grows one token at a time without reserving the maximum up front, and (b) can be shared, at fine granularity, with other sequences that have the same tokens. The contiguous tensor fails both. The common thread is that I want to keep the *logical* picture the model sees — one contiguous, growing run of tokens per sequence — while refusing to commit one contiguous run of *physical* memory to it. That's a familiar shape: a process wants a contiguous, growing address space, the system declines to back it with contiguous physical RAM, and on top of that several processes want to share pieces. Operating systems handle exactly that with virtual memory and paging: chop physical memory into fixed-size *pages*, give each process *logical* pages it sees as contiguous, keep a *page table* mapping logical to scattered physical pages, allocate physical pages on demand as the process touches more address space, and let two processes map their logical pages onto the *same* physical page to share. Every clause of that lines up with a clause of my requirement, so let me try to carry the construction over to the KV cache and see whether it actually delivers (a) and (b), or whether something about attention breaks it.

So: partition each sequence's KV cache into fixed-size **KV blocks**, each holding the keys and values for a fixed number $B$ of consecutive tokens — these are my "pages." A request's KV cache becomes a list of *logical* KV blocks, filled left to right as tokens arrive; the last block has some unfilled slots reserved for the next few generations. On the GPU I carve the KV region into *physical* KV blocks and keep a **block table** per request mapping each logical block to a physical block (and recording how many slots are filled). Logical blocks that look contiguous to the model can sit anywhere in physical memory. A new physical block is allocated only when the current last block fills up. Because every physical block has the same size, the allocator no longer leaves variable-size external holes — external fragmentation is gone by construction. For internal waste, the only unfilled slots are in a request's last block, so let me bound it concretely with $B=16$: a 30-token request takes $\lceil30/16\rceil=2$ blocks $=32$ slots, wasting 2; a 100-token request takes 7 blocks $=112$ slots, wasting 12; and when the length is a multiple of 16 (512, 2048) the waste is exactly 0. So per-request internal waste is strictly less than one block — at most 15 tokens regardless of how long the request is — versus the ~2000 reserved tokens of the contiguous design. That's the (a) half of the requirement met, and I can see the numbers.

But attention needs to read all the keys and values, and now they're scattered across non-contiguous physical blocks. This is the step that could break the whole idea, so I have to check the math rather than wave at it. Attention for query $q_i$ is a softmax-weighted sum over all keys/values $j\le i$. Grouping the sum over blocks should be fine *if* the softmax normalizer is taken over the full history and not per-block. Let the keys of block $j$ be $K_j=(k_{(j-1)B+1},\dots,k_{jB})$ and values $V_j$ similarly. For query $q_i$, the attention over block $j$ is the row vector $A_{ij}=(a_{i,(j-1)B+1},\dots,a_{i,jB})$, and

$$s_{ij}=q_i^\top K_j/\sqrt d,\qquad Z_i=\sum_{t=1}^{\lceil i/B\rceil}\exp(s_{it})\mathbf 1,\qquad A_{ij}=\frac{\exp(s_{ij})}{Z_i},\qquad o_i=\sum_{j=1}^{\lceil i/B\rceil}V_jA_{ij}^\top .$$

The denominator $Z_i$ sums $\exp$ of the scores across *all* blocks before dividing, so the weights are a single softmax over the whole length, not a per-block softmax — that's the part I don't want to get wrong. Let me just compute both ways on a small case to be sure I haven't fooled myself. Take $d=4$, a history of 5 keys/values drawn at random, and $B=2$ so the blocks are $\{k_1,k_2\},\{k_3,k_4\},\{k_5\}$ — note the last block is partial, which is the case I'm most worried about. Flat attention: $s=q^\top K/\sqrt d$ over all 5, one softmax, $o=\sum_j a_j v_j$. Block attention: scores per block, $Z$ = the sum of $\exp$ over all three blocks, weights $\exp(s_{ij})/Z$, accumulate $V_j A_{ij}^\top$. Running it, the flat output is $(1.215387,-0.347874,-0.055508,0.073718)$ and the block output is the same to all printed digits, with a max absolute difference of $1.4\times10^{-17}$ — floating-point zero. So the block grouping is genuinely the identity on the result; the only thing it changes is where the data sits and that the last block is sliced to its filled slots. Call this kernel PagedAttention.

The kernel, then, walks the block table, fetches each physical block in turn, multiplies $q_i$ against that block's keys to get the block's scores, and accumulates $V_jA_{ij}^\top$ into the output, taking the softmax across the concatenation so the normalizer is global. For the last block it slices to the filled token slots and ignores the unused tail.

Now for the (b) half — sharing. The block table gives me the same lever the OS has: if two sequences share a prompt, map *both* sequences' logical prompt-blocks to the *same* physical blocks, storing the prompt once. A physical block can now be pointed at by several logical blocks, so I attach a **reference count** to each physical block; freeing a block decrements the count and it's actually reclaimed only at count zero. The hazard is writes: if sample A and sample B share a physical block and A appends a new token's KV into it, B would see A's token. The OS answer to shared-page writes is **copy-on-write**, and it transfers directly: when a sequence needs to *write* into a shared block (reference count > 1), allocate a fresh physical block, copy the shared block's contents into it, decrement the original's count, and let the writer use its private copy while the other sharers keep the original.

I should check that this actually does what I claimed for parallel sampling — that the *prompt* (the bulk of the tokens) is stored once and only the diverging tail gets copied — because if copy-on-write fires on the full prompt blocks too, I've bought nothing. So let me trace the manager code directly. With $B=2$, run a 5-token prompt on seq 0: it allocates 3 physical blocks (the third half-full at `filled=5`), each with ref count 1. Fork seq 1 from seq 0: now all three shared blocks have ref count 2. Then generate 3 tokens on each sequence and watch which `append` calls trigger `copy_block`. The trace fires exactly one copy — on the partially-filled last prompt block — and afterward seq 0 holds blocks `[19,18,16,15]` and seq 1 holds `[19,18,17,14]`: they still *share* blocks 19 and 18 (the two full prompt blocks), and diverge only from the third block onward. So copy-on-write touched a single block, not the prompt, and the full prompt blocks stayed shared between both samples. That's the behavior I wanted, verified on the actual code path rather than asserted. Beam search is the same mechanism with a richer, shifting tree: candidates share all the blocks of their common ancestry, copy-on-write only the block at the point of divergence, and when a candidate dies its blocks' reference counts drop and any that hit zero are freed — replacing Orca's constant bulk KV copies with at most a one-block copy. A shared system prompt is just a set of physical blocks pre-populated once and mapped (read-only, last block copy-on-write) into every request that uses it, so the prompt-phase compute for the prefix is skipped too.

A couple of design choices the block view forces me to make. What's the block size $B$? It's a tradeoff between two failure modes. Too small, and the PagedAttention kernel processes too few tokens per block to use the GPU's parallelism well, and the block table grows long. Too large, and two things degrade: the last-block internal waste grows (it's bounded by $B-1$ tokens per request, so $B=16$ caps it at 15 and $B=128$ would cap it at 127), and the *probability* that two sequences can share a block drops, because sharing is block-granular — a block is shared only if its whole $B$-token span is common, and a longer span is less likely to be wholly common. A 16-token block sits in the middle: enough tokens to keep the kernel efficient, small enough that the tail waste I computed above stays a single-digit percentage and block-level sharing stays likely. So $B=16$ by default.

The other choice is what to do under memory pressure. With on-demand allocation I can still run out of physical blocks when traffic spikes. I need to evict some sequence's blocks and recover them later. Two observations shape the policy. First, all blocks of a sequence are always accessed together (you need the whole history to decode the next token), so the usual "evict the block accessed furthest in the future" heuristic degenerates to all-or-nothing per sequence — evict either all of a sequence's blocks or none. And sequences that share memory (beam candidates of one request) must be evicted together, as a gang, since their blocks are entangled by sharing. Second, for *recovering* an evicted sequence there are two clean options. Swapping: copy its blocks out to CPU RAM and back, classic OS swap to a "disk" that is the CPU; because I never swap out more than fits in the GPU KV region, the CPU swap space is bounded by it. Recomputation: just throw the KV away and recompute it when the sequence is rescheduled — and crucially that recompute is cheap, because the already-generated tokens can be concatenated with the original prompt and run through a single parallel prompt-phase pass rather than token-by-token. Which wins depends on block size and PCIe bandwidth (small blocks make swap's many tiny transfers slow, so recompute wins; large blocks favor swap), so I keep both and let the regime decide. I run an FCFS policy with preempt-latest-first to keep it fair and starvation-free.

One more thing the indirection layer buys me: distributed execution. Under tensor parallelism each GPU worker holds the KV for its own attention heads, but every worker processes the *same* token positions, so they all need blocks for the same logical positions. So I keep a *single* block manager in the centralized scheduler; it computes one logical→physical mapping and broadcasts it with the per-step inputs. Each worker uses the same physical block IDs (storing only its head-slice of each block) and never has to coordinate memory management with the others — the common mapping layer hides all the sharing from the model, which just sees a list of physical block IDs per sequence.

The causal chain: serving throughput is capped by KV-cache memory; the contiguous, max-length per-request layout wastes ~80% of it to reservation and fragmentation and makes sharing impossible; the OS's paging solves the same growing-contiguous-shareable-address-space problem, and carrying it over checks out — fixed-size KV blocks with a per-request block table mapping logical to scattered physical blocks, allocated on demand, drive internal waste below one block per request and kill external fragmentation; a block-wise PagedAttention kernel reads the scattered blocks and computes the identical result (verified to $10^{-17}$ against flat attention); reference counts plus copy-on-write give fine-grained sharing for parallel sampling, beam search, and shared prefixes (the code trace shows one shared sequence forking off a single COW block while the prompt stays shared); and an all-or-nothing eviction with swap-or-recompute recovery, plus a single shared block manager under tensor parallelism, complete the system. Here is the manager and the paged kernel.

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
