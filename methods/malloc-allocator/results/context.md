# Context: a general-purpose dynamic memory allocator

## Research question

A C program asks the runtime for memory at arbitrary moments and in arbitrary sizes through
two calls: `p = malloc(n)` hands back a pointer to `n` usable bytes, and `free(p)` returns a
previously handed-out block. The allocator owns one contiguous region of address space — the
*heap* — whose high end it can push outward by asking the operating system for more (the Unix
`sbrk(k)` call moves the program break up by `k` bytes; `mmap` can obtain a separate region).
Within that region the allocator must, with **no knowledge of future requests**, decide where
to put each block and how to reclaim freed ones.

The hard constraint that shapes everything: once a block's address is returned to the
application, **it can never be moved**. The application holds raw pointers into it; relocating
a live block would invalidate them. So unlike a garbage-collected or compacting heap, this
allocator can never defragment by sliding blocks together. Whatever layout decisions it makes
are permanent until the block is freed.

Two quantities measure a solution and they pull against each other:

- **Throughput** — allocation and free operations completed per unit time. The ideal is O(1)
  per call; the danger is a per-call cost that grows with the number of blocks on the heap.
- **Utilization** — of all the bytes the allocator has pulled from the OS (its peak
  high-water mark), what fraction was ever simultaneously *live payload*. The complement is
  waste, and waste comes from fragmentation.

A good allocator must be fast *and* frugal at the same time, on workloads it cannot predict.

## Background

**The malloc/free model and where waste comes from.** Each block the allocator manages must
carry a little bookkeeping so the allocator can find its size and tell allocated from free.
That bookkeeping plus alignment rounding is one source of waste; the structure of requests is
the other. Two kinds of fragmentation are distinguished:

- *Internal fragmentation* — the block handed out is larger than the payload requested,
  because of per-block bookkeeping, alignment to (typically) 8-byte boundaries, or a minimum
  block size. It is easy to measure: it is determined the instant the block is placed.
- *External fragmentation* — the heap as a whole holds enough free bytes to satisfy a
  request, but no single contiguous free block is large enough. This depends on the entire
  past history of requests and on the allocator's placement decisions, and because blocks
  cannot be moved, it cannot be undone after the fact.

A well-known illustration: allocate several small blocks, free every other one, then request
a block twice the size of one of the holes. The freed bytes total more than enough, but
they're scattered in pieces too small to use — the request fails or forces the heap to grow.

**The three policy knobs.** Any allocator of this family makes three kinds of decision, and
each trades throughput against utilization:

- *Placement* — given a request, which free block to carve it from. **First-fit** takes the
  first free block big enough (fast, but leaves splinters near the list head). **Next-fit**
  resumes scanning from where the last search stopped (preserves locality, often worse
  fragmentation). **Best-fit** takes the smallest free block that fits (least external
  fragmentation on real workloads, but naively requires scanning all free blocks).
- *Splitting* — when a chosen free block is larger than needed, whether to split off the
  remainder as a new free block or hand over the whole thing (trading external splinters for
  internal waste).
- *Coalescing* — when a block is freed, whether to merge it with adjacent free blocks. Done
  *immediately* on each `free`, or *deferred* until needed.

**The classical analysis (Knuth, TAOCP Vol. 1 §2.5).** The original treatment of dynamic
storage allocation studies exactly first-fit and best-fit on a list of free blocks, and
introduces two enabling ideas that the field still rests on:

- *The boundary-tag method.* If a block records its size and allocated/free status not only in
  a header at its front but also in a *footer* replicating that word at its tail, then from any
  block the word immediately before its header is the footer of the physically preceding block.
  That makes the previous block's size and status readable at a fixed offset — so when a block
  is freed, both its physical neighbors can be inspected and merged in constant time, without
  any search. The size/status word can be packed into a single machine word because 8-byte
  alignment leaves the low 3 bits of every size unused, free to hold flag bits.
- *The buddy system.* Restrict block sizes to powers of two; a block of size 2^k splits into
  two "buddies" of size 2^(k-1) whose addresses differ only in bit k, so coalescing is a
  cheap address-mask test. Fast and simple, at the cost of heavy internal fragmentation
  (every request rounds up to a power of two).

**Empirical guidance on policy.** Large surveys of allocator behavior on real program traces
(Wilson, Johnstone, Neely, and Boles, "Dynamic Storage Allocation: A Survey and Critical
Review," 1995) report that best-fit policies — in various exact and approximate forms — tend
to produce the least fragmentation across real workloads, more than first-fit or next-fit.

**Free-block organizations.** How free blocks are tracked determines how fast placement is:
- *Implicit list* — every block (free or allocated) carries a size header; the allocator
  walks the heap block-by-block by adding sizes. Trivial, but a search steps over allocated
  blocks too, so it is O(total blocks).
- *Explicit list* — thread only the free blocks onto a linked list, storing the `next`/`prev`
  pointers inside the free block's own (currently unused) payload, so allocated blocks pay no
  extra space. Search is now O(free blocks).
- *Sorted / tree organizations* — keep free blocks ordered by size (e.g. a balanced tree
  keyed on length) so best-fit is a fast lookup, at the cost of more pointer overhead and
  rebalancing per operation.

## Baselines

- **Trivial bump allocator.** `malloc` returns the next sequential address and bumps a
  pointer; `free` is a no-op. The fastest possible allocator: it never reclaims, so any
  long-running program exhausts memory. It frames the extreme of throughput with zero
  utilization.

- **Implicit-list, first-fit, with boundary-tag coalescing.** Headers on every block;
  `malloc` linearly scans from the heap start for the first free block that fits, splitting
  the remainder; `free` flips the allocated bit and, using boundary tags, merges with the
  physically adjacent free neighbors in O(1). Correct and compact, and it shows boundary-tag
  coalescing in its simplest form.

- **Explicit free list, first-fit, LIFO insertion.** Only free blocks are linked, via
  pointers stored in their payloads, so allocation scans O(free blocks) instead of all
  blocks — faster when memory is mostly in use. Newly freed blocks are spliced to the
  list head (LIFO: constant-time, but studies suggest worse fragmentation than inserting in
  address order).

- **Buddy system.** Power-of-two blocks with O(1) address-mask coalescing — very fast.
  Every request rounds up to a power of two.

## Evaluation settings

The natural yardstick is a *trace-driven* harness: a sequence of `malloc`/`free` requests
(synthetic patterns and recorded allocation traces from real programs — compilers,
interpreters, GUI toolkits, string-processing and network-heavy programs) replayed against
the allocator. Two metrics are read off each trace:

- **Throughput** — operations served per second (equivalently, average instructions per
  `malloc`/`free`).
- **Peak memory utilization** — the maximum aggregate live payload reached over the trace,
  divided by the heap high-water mark (the largest heap size the allocator ever requested from
  the OS); i.e. how little of the high-water heap was wasted.

The two are reported together precisely because they trade off, and an allocator is judged on
the pair across a spread of workloads (the "minimize anomalies" goal: do well everywhere, not
just on one pattern). The allocator runs over a thin memory-system shim that models `sbrk`
(extend the break) and bounds the heap, and a consistency checker walks the heap to verify
every block's header matches its footer, free blocks are coalesced, and any auxiliary
free-block structure contains exactly the free blocks.

## Code framework

The primitives that already exist: a memory shim exposing `mem_sbrk(k)` (extend the heap,
return the old break or -1), `mem_heap_lo()`/`mem_heap_hi()`, and 8-byte alignment. On top of
that the allocator exposes the standard interface and a heap checker. What the bookkeeping
inside a block looks like, how free blocks are tracked, and the placement/coalescing logic are
exactly the slots the design must fill.

```c
/* Word/alignment constants */
#define WSIZE      4          /* header/footer word (bytes) */
#define DSIZE      8          /* double word; alignment */
#define CHUNKSIZE  (1<<12)    /* default amount to grow the heap by */
#define MAX(x,y)   ((x) > (y) ? (x) : (y))

/* TODO: how do we pack a block's size + flags into a word, and read them back?
 *       how do we get from a payload pointer to its header / footer / neighbors?
 *       (this is the per-block bookkeeping layout we must choose) */

/* TODO: how are free blocks organized so placement is fast?
 *       (the free-block data structure goes here) */

int  mm_init(void);                 /* lay out the initial empty heap */
void *mm_malloc(size_t size);       /* place a request, growing the heap if needed */
void  mm_free(void *bp);            /* reclaim a block */
void *mm_realloc(void *bp, size_t size);

static void *extend_heap(size_t words) {
    /* grow the break by an aligned amount, make the new space one free block */
    /* TODO: knit the new free block into whatever free-block structure we choose,
     *       merging with the old top if it was free */
    return NULL; /* TODO */
}

static void *coalesce(void *bp) {
    /* a block bp just became free — merge it with adjacent free memory */
    /* TODO: inspect physical neighbors and merge in O(1) — needs the layout above */
    return bp; /* TODO */
}

static void *find_fit(size_t asize) {
    /* choose a free block of at least asize bytes (the placement policy) */
    return NULL; /* TODO */
}

static void place(void *bp, size_t asize) {
    /* mark bp allocated; split off the remainder if it is big enough */
    /* TODO: splitting policy + free-structure maintenance */
}
```
