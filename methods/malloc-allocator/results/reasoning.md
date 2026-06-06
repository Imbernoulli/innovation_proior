OK, let me think this through from scratch. I have one contiguous slab of address space — the heap — and a way to push its high end outward by asking the OS for more (`sbrk(k)` moves the program break up by `k` bytes). A program is going to hammer me with two calls in some order I can't see coming: `malloc(n)`, give me `n` usable bytes; `free(p)`, here's a block back. I have to decide, on the fly, where each requested block lives inside the slab and how to reclaim the freed ones. That's it. No app knowledge — I don't know the sizes, the order, the lifetimes, nothing.

The single fact that constrains everything: once I hand the program a pointer, I can never move that block. The program holds the raw address and will dereference it; if I slide the block somewhere else to tidy up, I've corrupted it. So I have no compaction, ever. Whatever I do has to be right the first time, because I can't shuffle blocks together later to recover wasted space.

So what am I even optimizing? Two things, and I can already feel them fighting. One is **throughput** — how many malloc/free calls I can serve per second. I'd love each to be O(1), constant work no matter how big the heap gets; the nightmare is work that grows with the number of blocks. The other is **utilization** — of all the bytes I've ever had to pull from the OS (my peak high-water mark), what fraction was actually live program payload at the same time. Everything that isn't is waste, and waste is fragmentation.

Let me pin down the waste, because it's two different beasts. If I round a request up — to align it, or to fit my bookkeeping, or to a minimum block size — the block is bigger than the payload and those extra bytes are dead for the block's whole life. That's *internal* fragmentation, and it's decided the moment I place the block, easy to see. The nasty one is *external* fragmentation: the heap holds plenty of free bytes in total, but they're scattered in chunks none of which is big enough for the request in front of me. Picture allocating a row of small blocks, freeing every other one, then asking for a block twice the size of one hole. The free bytes add up to more than enough — but they're in pieces, and I can't move blocks to glue them together. So that request either fails or forces me to grow the heap even though, in aggregate, I had the room. External fragmentation is a function of the entire history of requests *and* of my own placement choices, and it's irreversible. That tells me placement and reclamation aren't bookkeeping details — they're the whole game.

Let me get the most trivial allocator on the table so I know the boundary. `malloc`: return the current top, bump it by `n`. `free`: do nothing. This is the fastest allocator that can possibly exist — both calls are a handful of instructions. And it's worthless: it never reuses anything, so any program that runs a while just marches the break to the moon and dies. Utilization is essentially zero. Fine — that's the throughput extreme with no frugality. Real work is: keep something near that speed while actually reclaiming and reusing freed space.

To reuse freed space I have to *find* it. So the first real question is: when a request comes in, how do I locate a free block big enough? I need to know, walking around the heap, where blocks begin and end and which are free. Cheapest idea: put a little header word at the front of every block holding its size, plus a bit saying allocated-or-free. Then I can stand at the start of the heap and walk block to block — read a size, jump that many bytes, read the next size — visiting every block in address order. Call it the **implicit free list**: it's "implicit" because I'm not maintaining a separate list at all, the sizes themselves chain the blocks together. To satisfy a `malloc(n)`, I walk from the start and take the first free block that fits, splitting off the leftover as a smaller free block.

I can already make the header cheap. I'm going to align everything to 8 bytes anyway (alignment is non-negotiable — the hardware wants doubles and pointers aligned, and handing back a misaligned payload is a correctness bug). If every size is a multiple of 8, the low 3 bits of any size word are always zero — dead bits I can steal. So I pack size-or'ed-with-the-allocated-bit into one word: `PACK(size, alloc) = size | alloc`, and read them back as `GET_SIZE = word & ~0x7` and `GET_ALLOC = word & 0x1`. One word of header, no waste. Good.

Now `free`. The program hands me a pointer; I flip the allocated bit off. But if I stop there, external fragmentation will eat me alive — I'll accumulate a checkerboard of tiny free blocks that can never serve a big request, even with megabytes free in total. So on free I must **coalesce**: if a physical neighbor is also free, merge into one bigger block. That's the only weapon I have against external fragmentation, given I can't move blocks.

Let me try to actually do the coalescing and watch it break. Block `bp` was just freed, size `s`. My neighbor *after* me in memory starts at `bp + s` — and there sits its header, which I can read: is it free? If so, easy, I absorb it: `s += size(next)`, rewrite my header. Now the neighbor *before* me. I want to merge backward too, but... where is its header? It's somewhere below me at `(start of prev block)`, and I have no idea how far back that is, because I'd need to know the previous block's *size* to step back over it — and its size lives in *its* header, which is exactly the thing I can't reach without already knowing its size. Circular. I'm walking forward fine but I can't see backward. So immediate, in-place coalescing with the previous block is impossible with only a forward header.

I need the previous block's size to be readable from a fixed offset below me. So: what if every block also stores its size at its *tail*, a footer word duplicating the header? Then the word immediately below my header — at `bp - DSIZE`, i.e. just under where my header sits — is the *footer of the block physically before me*. I read that footer, it gives me the previous block's size, and I step back exactly that far to land on the previous block's header. Backward coalescing in O(1), no search. This is Knuth's **boundary tag**: tag the block at both boundaries so adjacency can be tested and crossed in constant time from either side. The footer costs one extra word per block, but it buys constant-time two-sided coalescing, which is the thing that keeps external fragmentation in check. Worth it.

Let me set up the pointer arithmetic carefully, because off-by-a-word here corrupts the heap silently. Let `bp` be the payload pointer I hand the program. The header sits one word before it: `HDRP(bp) = bp - WSIZE`. The footer sits at the very end of the block: `FTRP(bp) = bp + GET_SIZE(HDRP(bp)) - DSIZE` — that's the block start plus its size, minus the footer word and minus the header word, i.e. back up two words from the end-of-block address. The next block's payload: `NEXT_BLKP(bp) = bp + GET_SIZE(bp - WSIZE)` — add my own size (read from my header) to step over me. The previous block: `PREV_BLKP(bp) = bp - GET_SIZE(bp - DSIZE)` — read the *previous block's footer*, which is the word at `bp - DSIZE`, get its size, and subtract to land on its payload. That `bp - DSIZE` is the boundary tag doing its job.

Now coalesce is a clean four-case analysis on whether each neighbor is allocated. Let `prev_alloc = GET_ALLOC(FTRP(PREV_BLKP(bp)))` and `next_alloc = GET_ALLOC(HDRP(NEXT_BLKP(bp)))`, my size `s = GET_SIZE(HDRP(bp))`.

Case 1, both neighbors allocated: nothing to merge, leave `bp` as is.

Case 2, prev allocated, next free: I swallow the next block. `s += GET_SIZE(HDRP(NEXT_BLKP(bp)))`, then write the free header at my own header and the free footer at... `FTRP(bp)` — and here's a subtlety I have to get right: once I've *increased* `s` in my header, `FTRP(bp)` recomputes using the new bigger size, so it now points at the *next* block's old footer, exactly the new combined block's tail. So writing header then footer with the new `s` lands the footer in the right place. `bp` is unchanged (I'm still the front of the merged block).

Case 3, prev free, next allocated: now the front of the merged block is the *previous* block, not me. `s += GET_SIZE(HDRP(PREV_BLKP(bp)))`. I write my footer (`FTRP(bp)` — still at my tail, the merged block's tail) with `s`, and I write the *previous block's header* (`HDRP(PREV_BLKP(bp))`) with `s`, then set `bp = PREV_BLKP(bp)` so the merged block's payload pointer is the previous block's. Order matters: I must read `PREV_BLKP(bp)` before I clobber sizes, which I do.

Case 4, both free: swallow both. `s += GET_SIZE(HDRP(PREV_BLKP(bp))) + GET_SIZE(FTRP(NEXT_BLKP(bp)))`. The merged block runs from the previous block's header to the next block's footer, so I write the previous block's header and the next block's footer with `s`, and set `bp = PREV_BLKP(bp)`. Done.

One more thing for this to be safe: the walk must not run off either end of the heap. So I bracket the heap with sentinels. At the bottom, a **prologue** block — a tiny allocated block (header+footer, both marked allocated, size 8) that the coalescer will see as a permanently-allocated left neighbor, so case 3/4 never steps below it. At the top, an **epilogue** — a zero-size allocated header — so a forward walk and the next-neighbor check always hit an "allocated, size 0" terminator. With those, the four cases never read outside the heap. When I grow the heap with `extend_heap`, I lay the new free block where the old epilogue was and write a fresh epilogue just past it, then immediately `coalesce` — so if the old top block was free, the new space merges right in.

So: implicit list + boundary-tag coalescing is correct and tight on space. Let me stare at its throughput, because that's where it's going to hurt. `free` is O(1) — flip bits, do the constant-work coalesce. But `malloc` walks the implicit list from the start looking for a fit, and the implicit list contains *every* block, allocated and free alike. So when the heap is large and mostly full — which is the normal steady state of a long-running program — a single `malloc` steps over thousands of allocated blocks to reach a free one. That's O(total blocks) per allocation. Unacceptable. The allocated blocks are pure overhead in that scan; I'm paying to skip them every single time.

The fix writes itself once I name the problem: don't walk allocated blocks. Keep a list of *only the free blocks*. Now `malloc` scans O(free blocks), not O(all blocks) — and when the heap is mostly full there are *few* free blocks, exactly the case the implicit list was worst at. This is the **explicit free list**. But a separate linked list needs `next`/`prev` pointers per free block — where do I put them without bloating every allocated block? Here's the lovely part: a *free* block's payload is, by definition, not in use. So I store the `next` and `prev` pointers right inside the free block's payload area. Allocated blocks pay nothing extra; free blocks reuse space they weren't using anyway. (This does set a minimum block size — a block must be big enough to hold header, footer, and two pointers when free — but that's a small fixed floor.)

So now `malloc` searches the explicit list and takes a fit; `free` flips the bit, coalesces via the boundary tags as before, and splices the resulting free block into the list. Coalescing now has to also unlink any neighbor it absorbs from the free list (that neighbor was free, so it was on the list) and link the merged block back in. When I insert a freshly freed block, where in the list? Simplest is **LIFO** — push it at the head, O(1). There's an alternative, address-ordered insertion, keep the list sorted by address, which the fragmentation folklore says packs a little tighter — but it costs a search on every free to find the insertion point. LIFO is constant-time and the fragmentation hit is modest, so I'll take LIFO and keep `free` O(1).

This is already a real allocator. But let me push on placement, because I haven't honestly confronted *which* free block to pick, and that's where utilization is won or lost. On real program traces, the empirical record is clear and a little surprising: **best-fit** — always carve the request from the *smallest* free block that still fits — produces the least external fragmentation, beating first-fit and next-fit across realistic workloads. Intuition: first-fit grabs whichever block comes up first, often a large one, and chops a big block into a request-plus-splinter, sprinkling the heap with awkward leftovers; best-fit instead spends the tightest block, tending to leave either nothing or a leftover that's itself a useful size, and it *preserves* the large blocks for the large requests that will need them.

But best-fit, done literally, means scanning *every* free block on every `malloc` to find the global smallest-that-fits. On a long explicit list that's slow — I've traded my throughput back away. So the tension is sharp: best-fit is what I want on space, an exhaustive scan is what I can't afford on time. I want best-fit's *placement* without best-fit's *search*.

Let me look hard at why the search is expensive. It's expensive because one flat list mixes every size together, so to find the smallest-that-fits I have no choice but to look at all of them. What if the list weren't flat? Suppose I keep *several* free lists, one per *size class* — the first range covers the minimum legal free block, and each later range doubles the previous upper bound. Now to serve `malloc(n)`: compute the smallest class whose range can hold the adjusted size, and go straight to that list. Every legal block in smaller classes is too small, and every block in this class is in the tightest range that might fit. So I still guard each candidate with `asize <= block_size`, but the *first* block I find there that passes the guard is, to within the width of the class, the smallest useful block I can get cheaply. First-fit *within the right class* is an **approximation of best-fit over the whole heap** — and I got it without scanning the whole heap, just the one short list for that class. If that class has no fitting block, I walk up to the next larger class and take from there, splitting off the remainder. And if every class fails, I `sbrk` for more and carve from the fresh space.

This is the move. **Segregated free lists.** The throughput is back to near-constant: indexing into the class array is O(1), and each per-class list is short because it only holds one size range; the search cost is roughly the number of classes I might have to climb, which is logarithmic in the represented sizes when the ranges double. And the placement quality is best-fit-like, because the starting class is the smallest range that can hold the request, and the in-list guard prevents an undersized block from slipping through. I can even see the limit: if I made the classes infinitely fine — one class per exact size — then first-fit-in-class would be *exact* best-fit. The segregation *is* the approximation of best-fit, dialed by how fine I cut the classes. That's the unification: I set out to approximate best-fit cheaply, and the data structure that does it is a bucketed free list. Coarser buckets, faster and looser; finer buckets, slower and tighter — the throughput/utilization dial is literally the bucket granularity.

So `find_fit(asize)`: find the smallest class whose size range can hold `asize`, first-fit within it, and if nothing, advance to larger classes until something fits or I run out. `free`: flip the bit, coalesce via boundary tags, and reinsert the *merged* block into the class matching its *new, larger* size — so coalescing and segregation cooperate, a merged block migrates up to a bigger class where the next big request will find it. I have to splice the list maintenance into the four coalescing cases without changing the physical math. I cache `prev = PREV_BLKP(bp)` and `next = NEXT_BLKP(bp)` before rewriting tags. If both neighbors are allocated, there is no absorbed list node; I just insert `bp` into its class. If only `next` is free, I remove `next` from its old class, add its size to mine, write my header and the new footer at `FTRP(bp)` after the header has the larger size, then insert `bp`. If only `prev` is free, I remove `prev`, add its size to mine, write the merged footer at my old tail `FTRP(bp)`, write the merged header at `HDRP(prev)`, set `bp = prev`, and insert that pointer. If both are free, I remove both absorbed neighbors, add both sizes, write the merged header at `HDRP(prev)` and the merged footer at `FTRP(next)`, set `bp = prev`, and insert the three-block result. The signs are all "add sizes"; the only pointer reassignment is in the cases where the previous block becomes the front.

Now the **splitting** policy, which I waved at. When I place an `asize` request into a free block of size `csize`, the leftover is `csize - asize`. If I split, that leftover becomes a free block and goes back on a list. But if the leftover is smaller than my minimum block size — too small to hold a header, footer, and two free-list pointers — I can't make it a legal free block; it would be an unusable splinter. So the rule: split only if `csize - asize` is at least the minimum block size; otherwise hand over the whole `csize` block and eat the difference as internal fragmentation. That's the splitting trade-off made concrete — split aggressively and I create tiny external splinters, split conservatively and I bloat blocks internally; the minimum-block-size threshold is where I draw the line.

Growing the heap should happen only after the segregated search has failed. If I call `sbrk` before checking the existing free blocks, I increase the high-water heap even though a reusable block might already exist, which directly hurts utilization. So the order is fixed: search the classes first; only on a miss extend by `max(asize, CHUNKSIZE)`. The extension itself has to preserve the boundary-tag invariant. The old epilogue header becomes the header of the new free block, the new block gets a matching footer, and a fresh zero-size allocated epilogue is written after it. Then I call `coalesce` immediately, because if the old top block was already free, the newly obtained bytes should become one larger free block rather than two adjacent free blocks.

And **coalescing timing**: I've been coalescing immediately, inside `free`. The alternative is to *defer* — leave freed blocks at their current size hoping a same-size request comes back soon (saves a coalesce now and a split later), and only coalesce lazily when a search fails or fragmentation crosses a threshold. Deferred coalescing can lift throughput on programs that churn a few fixed sizes. But it needs a policy for *when* to give up and coalesce, and without app knowledge a wrong guess strands memory. Immediate coalescing is the honest, robust default, and the boundary tags make it cheap and always-available, so I'll coalesce on every free and leave deferral as a tuning knob.

Let me assemble it. The heap starts with an alignment pad, a prologue (allocated 8-byte block, header+footer) and an epilogue (zero-size allocated header) as the coalescing sentinels, while the per-class list heads are initialized to empty. `mm_malloc(size)` adjusts the request up to include header+footer and round to 8 (`asize = max(MINBLK, ALIGN(size + 2*WSIZE))`), then `find_fit` over the classes, `place` (split if the remainder reaches minimum size), and if nothing fits, `extend_heap` and place there. `mm_free(bp)` clears the allocated bit in header and footer and calls `coalesce`, which runs the four-case boundary-tag merge and maintains the segregated lists.

```c
/* ---- block bookkeeping: one packed word, boundary tags ---- */
#define WSIZE      4                 /* header/footer word */
#define DSIZE      8                 /* alignment / double word */
#define CHUNKSIZE  (1<<12)           /* default heap growth */
#define NCLASSES   16                /* number of segregated size classes */
#define MINBLK     (2*WSIZE + 2*sizeof(void *)) /* hdr+ftr+next+prev */
#define MAX(x,y)   ((x) > (y) ? (x) : (y))
#define ALIGN(size) (((size) + (DSIZE-1)) & ~0x7)

#define PACK(size, alloc)  ((size) | (alloc))            /* low 3 bits free: 8-aligned */
#define GET(p)             (*(unsigned int *)(p))
#define PUT(p, val)        (*(unsigned int *)(p) = (val))
#define GET_SIZE(p)        (GET(p) & ~0x7)
#define GET_ALLOC(p)       (GET(p) & 0x1)

#define HDRP(bp)      ((char *)(bp) - WSIZE)                         /* header word */
#define FTRP(bp)      ((char *)(bp) + GET_SIZE(HDRP(bp)) - DSIZE)    /* footer (boundary tag) */
#define NEXT_BLKP(bp) ((char *)(bp) + GET_SIZE((char *)(bp) - WSIZE))/* step over self */
#define PREV_BLKP(bp) ((char *)(bp) - GET_SIZE((char *)(bp) - DSIZE))/* read prev footer, step back */

/* free-block payload holds the explicit-list links: next/prev pointers */
#define NEXT_FREE(bp) (*(char **)(bp))
#define PREV_FREE(bp) (*(char **)((char *)(bp) + sizeof(char *)))

static char *heap_listp;             /* points just past the prologue */
static char *free_lists[NCLASSES];   /* one LIFO list head per size class */

/* which class does a block of `size` bytes belong to: ranges double from MINBLK */
static int class_of(size_t size) {
    int c = 0;
    size_t limit = MINBLK;
    while (size > limit && c < NCLASSES-1) { limit <<= 1; c++; }
    return c;
}

/* LIFO insert / splice-out within the segregated lists */
static void insert_free(char *bp) {
    int c = class_of(GET_SIZE(HDRP(bp)));
    NEXT_FREE(bp) = free_lists[c];
    PREV_FREE(bp) = NULL;
    if (free_lists[c]) PREV_FREE(free_lists[c]) = bp;
    free_lists[c] = bp;
}
static void remove_free(char *bp) {
    int c = class_of(GET_SIZE(HDRP(bp)));
    if (PREV_FREE(bp)) NEXT_FREE(PREV_FREE(bp)) = NEXT_FREE(bp);
    else               free_lists[c] = NEXT_FREE(bp);
    if (NEXT_FREE(bp)) PREV_FREE(NEXT_FREE(bp)) = PREV_FREE(bp);
}

/* four-case boundary-tag coalescing, maintaining the seglists */
static void *coalesce(char *bp) {
    char *prev = PREV_BLKP(bp);
    char *next = NEXT_BLKP(bp);
    size_t prev_alloc = GET_ALLOC(FTRP(prev));
    size_t next_alloc = GET_ALLOC(HDRP(next));
    size_t size = GET_SIZE(HDRP(bp));

    if (prev_alloc && next_alloc) {                 /* case 1: nothing to merge */
        insert_free(bp);
    } else if (prev_alloc && !next_alloc) {         /* case 2: swallow next */
        remove_free(next);
        size += GET_SIZE(HDRP(next));
        PUT(HDRP(bp), PACK(size, 0));
        PUT(FTRP(bp), PACK(size, 0));               /* FTRP now uses new size -> right place */
        insert_free(bp);
    } else if (!prev_alloc && next_alloc) {         /* case 3: swallow prev */
        remove_free(prev);
        size += GET_SIZE(HDRP(prev));
        PUT(FTRP(bp), PACK(size, 0));
        PUT(HDRP(prev), PACK(size, 0));
        bp = prev;
        insert_free(bp);
    } else {                                        /* case 4: swallow both */
        remove_free(prev);
        remove_free(next);
        size += GET_SIZE(HDRP(prev)) + GET_SIZE(HDRP(next));
        PUT(HDRP(prev), PACK(size, 0));
        PUT(FTRP(next), PACK(size, 0));
        bp = prev;
        insert_free(bp);
    }
    return bp;
}

static void *extend_heap(size_t words) {
    char *bp;
    size_t size = (words % 2) ? (words+1)*WSIZE : words*WSIZE;   /* keep 8-aligned */
    if ((bp = mem_sbrk(size)) == (void *)-1) return NULL;
    PUT(HDRP(bp), PACK(size, 0));            /* new free block where old epilogue was */
    PUT(FTRP(bp), PACK(size, 0));
    PUT(HDRP(NEXT_BLKP(bp)), PACK(0, 1));    /* fresh epilogue */
    return coalesce(bp);                     /* merge with old top if it was free */
}

int mm_init(void) {
    for (int c = 0; c < NCLASSES; c++) free_lists[c] = NULL;
    if ((heap_listp = mem_sbrk(4*WSIZE)) == (void *)-1) return -1;
    PUT(heap_listp, 0);                          /* alignment padding */
    PUT(heap_listp + 1*WSIZE, PACK(DSIZE, 1));   /* prologue header */
    PUT(heap_listp + 2*WSIZE, PACK(DSIZE, 1));   /* prologue footer */
    PUT(heap_listp + 3*WSIZE, PACK(0, 1));       /* epilogue header */
    heap_listp += 2*WSIZE;
    if (extend_heap(CHUNKSIZE/WSIZE) == NULL) return -1;
    return 0;
}

/* segregated find: smallest fitting class, first-fit within, climb on miss */
static void *find_fit(size_t asize) {
    for (int c = class_of(asize); c < NCLASSES; c++)
        for (char *bp = free_lists[c]; bp; bp = NEXT_FREE(bp))
            if (asize <= GET_SIZE(HDRP(bp)))      /* first fit in class ~ best fit overall */
                return bp;
    return NULL;
}

/* place asize into free block bp; split only if remainder is a legal block */
static void place(char *bp, size_t asize) {
    size_t csize = GET_SIZE(HDRP(bp));
    remove_free(bp);
    if (csize - asize >= MINBLK) {                /* split */
        PUT(HDRP(bp), PACK(asize, 1));
        PUT(FTRP(bp), PACK(asize, 1));
        char *rem = NEXT_BLKP(bp);
        PUT(HDRP(rem), PACK(csize - asize, 0));
        PUT(FTRP(rem), PACK(csize - asize, 0));
        insert_free(rem);
    } else {                                      /* keep whole: internal frag */
        PUT(HDRP(bp), PACK(csize, 1));
        PUT(FTRP(bp), PACK(csize, 1));
    }
}

void *mm_malloc(size_t size) {
    if (heap_listp == 0 && mm_init() == -1) return NULL;
    if (size == 0) return NULL;
    size_t asize = MAX(MINBLK, ALIGN(size + 2*WSIZE));             /* hdr+ftr, round to 8 */
    char *bp;
    if ((bp = find_fit(asize))) { place(bp, asize); return bp; }
    size_t ext = MAX(asize, CHUNKSIZE);                           /* grow only after no fit */
    if ((bp = extend_heap(ext / WSIZE)) == NULL) return NULL;
    place(bp, asize);
    return bp;
}

void mm_free(void *bp) {
    size_t size = GET_SIZE(HDRP(bp));
    PUT(HDRP(bp), PACK(size, 0));     /* clear alloc bit, both tags */
    PUT(FTRP(bp), PACK(size, 0));
    coalesce(bp);                     /* immediate boundary-tag coalescing + reinsert */
}
```

The whole chain, in one breath: I can't move blocks, so I fight external fragmentation only by *placing well* and *merging on free*; a single header word per block (with the allocated bit stolen from the low 3 alignment bits) lets me walk the heap, and a duplicated footer — the boundary tag — lets me cross to the previous block in O(1) so I can coalesce both neighbors the instant a block is freed; scanning every block to find a free one is O(all blocks), so I thread only the free blocks on an explicit list using their own dead payload for the links; one flat list still forces a full scan to approximate the best-fit placement that real workloads reward, so I shard the free blocks into per-size-class lists, and first-fit *within the right class* becomes best-fit-like placement without the global scan — finer classes meaning closer to exact best-fit, which is the dial between throughput and utilization; split a chosen block only when the remainder is a legal block, grow the heap only after the segregated search misses, and coalesce immediately because the boundary tags make it cheap.
