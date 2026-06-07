# Synthesis — malloc-allocator (segregated free lists + boundary tags)

## Pain point
General-purpose C programs call malloc/free with no knowledge of future requests. Allocator
must place arbitrary-size requests into one contiguous heap region (grown via sbrk), reclaim
on free, never relocate live blocks (pointers held by app). Two goals in tension:
- **Throughput**: ops/sec of malloc+free. Want near-O(1).
- **Utilization (peak)**: payload / heap-high-water. Hurt by fragmentation:
  - internal: block bigger than payload (header/footer/alignment/min-block).
  - external: enough total free memory but no single free block large enough.
No relocation → can't fix external frag by compaction → must place well + coalesce.

## Knowable pre-method facts (context Background)
- Wilson et al. 1995 survey: best-fit (various approximations) produces least fragmentation
  on real loads vs first-fit/next-fit. (Lea cites this.)
- Knuth TAOCP vol.1 §2.5: original first-fit/best-fit analysis; the buddy system; the
  **boundary-tag** trick — store size at BOTH ends of a block so the physically-preceding
  block's footer is readable at a fixed offset → O(1) coalescing of physical neighbors.
- Next-fit / nearest-fit: helps locality but worsens fragmentation (Lea).
- 50%-rule / fragmentation folklore from Knuth.

## Derivation chain (the spine of reasoning.md)
1. malloc/free interface, one heap, no relocation → goals throughput vs utilization, both
   trade off through the placement/splitting/coalescing policies.
2. Need to find free blocks. **Implicit list**: every block has a header (size+alloc bit);
   walk block-to-block by adding size. find_fit = O(total blocks). Simple, but slow when
   heap is mostly full (steps over allocated blocks too).
3. free must coalesce or external frag explodes. Coalescing with NEXT block is easy (header
   reachable). Coalescing with PREV block needs its size — but we only have a forward header.
   → **boundary tag (Knuth)**: replicate size+alloc in a FOOTER at the block's end. Then
   PREV_BLKP(bp) = bp - size_in_prev_footer. O(1) coalescing both directions, 4 cases.
4. Implicit list alloc still O(all blocks). **Explicit free list**: thread only FREE blocks
   on a doubly linked list, storing next/prev in the free block's payload area (free, so
   reusable; no extra space for allocated blocks). alloc now O(free blocks). free splices
   in/out + coalesce. LIFO insertion = O(1) but slightly worse frag than address-ordered.
5. First-fit on one explicit list still scans a long list; and a single list mixes all sizes
   so best-fit would require full scan. **Segregated free lists**: array of lists, one per
   size class (small exact classes 8,16,24,...; larger power-of-two ranges). malloc(n):
   compute class, search that list (first-fit within class ≈ best-fit overall because all
   members are ~same size), else next larger class, split, else sbrk. free: coalesce, then
   reinsert into class of the resulting size. Near-O(1)/log; first-fit-in-class approximates
   best-fit without scanning the whole heap. Extreme (one class per size) = exact best-fit.
6. Splitting policy: if remainder ≥ min block size, split; else give whole block (internal
   frag) — trades internal frag vs leaving tiny unusable splinters.
7. Wilderness/top chunk (Lea): the block bordering the highest sbrk address is the only one
   that can be grown; treat it as "bigger than all" so it's used only when nothing else fits
   → avoids preventable sbrk and the fragmentation of carving the top first.
8. Coalescing policy: immediate (coalesce in free) vs deferred (caching). Immediate is the
   baseline; boundary tags make coalescing possible at any time.

## Boundary-tag coalescing — the 4 cases (verify signs)
bp just freed, size s. prev_alloc = footer of PREV_BLKP, next_alloc = header of NEXT_BLKP.
- Case 1 (prev a, next a): nothing to merge.
- Case 2 (prev a, next free): s += size(next); write header(bp)/footer(bp)=s. bp unchanged.
- Case 3 (prev free, next a): s += size(prev); write footer(bp)/header(prev)=s; bp=PREV.
- Case 4 (both free): s += size(prev)+size(next); header(prev)/footer(next)=s; bp=PREV.
All correct in CS:APP mm.c (read it). In explicit/seglist version, each case also
remove_from_seglist the absorbed neighbor(s) and add_to_seglist the merged block.

## Macros (grounded in CS:APP mm.c)
WSIZE=4, DSIZE=8, CHUNKSIZE=1<<12. PACK(size,alloc)=size|alloc (size is 8-aligned so low 3
bits free). GET/PUT word. GET_SIZE=GET&~0x7, GET_ALLOC=GET&0x1. HDRP=bp-WSIZE,
FTRP=bp+GET_SIZE(HDRP)-DSIZE. NEXT_BLKP=bp+GET_SIZE(bp-WSIZE), PREV_BLKP=bp-GET_SIZE(bp-DSIZE).
Heap init: alignment pad, prologue (8/alloc) header+footer, epilogue (0/alloc) header.
Prologue/epilogue are sentinels so coalescing never walks off the heap. asize: <=DSIZE →
2*DSIZE; else DSIZE*ceil((size+DSIZE)/DSIZE) (room for header+footer, 8-aligned).
place: split if csize-asize >= 2*DSIZE (min block).

## Segregated structure (grounded in ivannikitovic mm.c + CMU lecture)
Array of free-list heads (bucket per size class). find_bucket(words): smallest k with
words*WSIZE <= 2^k. find_fit: start at class for asize, first-fit within list; if none,
advance to larger classes. Free block layout: header, next ptr, prev ptr, ..., footer.
add_to_bucket: LIFO at head. remove_from_bucket: splice doubly-linked. coalesce removes
neighbors from their lists, merges, re-adds merged block. place splits, removing the chosen
block and adding the remainder.

## Code landing
answer.md = clean 64-bit-safe-ish CS:APP-style segregated-fit allocator: macros + mm_init +
extend_heap + boundary-tag coalesce (with seglist maintenance) + segregated find_fit +
place(split) + LIFO insert/remove + malloc/free. Faithful to CS:APP mm.c skeleton extended
with seglist per ivannikitovic / CMU lecture 17.

## Sources
- Primary: Doug Lea "A Memory Allocator" (gee.cs.oswego.edu/dl/html/malloc.html); CS:APP
  mm.c (csapp.cs.cmu.edu/3e/ics3/code/vm/malloc/mm.c); Knuth TAOCP v1 §2.5 (boundary tags,
  first/best-fit, buddy).
- Background/explainer: CMU 15-213 Lecture 17 "Dynamic Memory Allocation" (seglist, policies).
- Code: CS:APP mm.c (implicit+boundary tag), ivannikitovic seglist mm.c (segregated fit).
