# A segregated-fit dynamic memory allocator (segregated free lists + boundary tags)

## Problem

Implement `malloc`/`free` over a single contiguous heap (growable at its top via `sbrk`),
with no knowledge of the request stream and **no relocation** of live blocks. Maximize two
quantities that trade off: **throughput** (near-O(1) per call) and **utilization** (low
internal + external fragmentation). Because blocks can't be moved, external fragmentation can
only be controlled by *good placement* and *coalescing on free*.

## Key ideas

- **Boundary tags (Knuth).** Each block stores its size and allocated bit in a one-word
  *header* and a duplicate *footer*. The allocated bit hides in the low 3 bits of the size,
  which are always zero under 8-byte alignment, so the tag is a single word. The footer of
  the physically preceding block sits at a fixed offset below the current header, making the
  previous block's size/status readable in O(1) — so a freed block can be merged with *both*
  physical neighbors in constant time, no search.

- **Explicit free lists.** Link only the free blocks, storing `next`/`prev` pointers inside
  each free block's own (unused) payload — allocated blocks carry no link overhead. Search is
  O(free blocks) instead of O(all blocks).

- **Segregated free lists (the placement engine).** Keep an array of free lists, one per
  geometric size range starting at the minimum legal free block size. For `malloc(n)`, index
  directly to the smallest class whose range can hold the adjusted request, first-fit within
  it with an explicit `asize <= block_size` guard, and climb to larger classes on a miss.
  That approximates best-fit over the whole heap without scanning every free block.

- **Splitting policy.** After placing `asize` in a block of size `csize`, split off the
  remainder only if it is at least the minimum legal free-block size (`hdr+ftr+next+prev`,
  using the platform pointer width); otherwise hand over the whole block to avoid splinters.

- **Heap extension on miss.** Search existing segregated lists first. Only if no fit exists,
  extend the heap by `max(asize, CHUNKSIZE)`, write the new free block where the old epilogue
  was, install a fresh epilogue, and immediately coalesce with the old top block if it was free.

- **Immediate coalescing.** Coalesce inside `free` (cheap, thanks to boundary tags). Deferred
  coalescing is a possible tuning knob but needs a workload-dependent policy.

## Algorithm

```
malloc(size):
  asize = max(MINBLK, round_up_to_8(size + header + footer))
  bp = find_fit(asize)                # smallest fitting class, first-fit within, climb on miss
  if bp == NULL:
      bp = extend_heap(max(asize, CHUNKSIZE))   # grow the top, coalesce with old top
  place(bp, asize)                    # split if remainder >= MINBLK; maintain seglists
  return bp

free(bp):
  clear allocated bit in header and footer
  coalesce(bp)                        # 4-case boundary-tag merge; reinsert merged block by class

coalesce(bp):                         # prev_alloc / next_alloc from boundary tags
  case both allocated      -> insert bp
  case next free           -> remove next; merge; insert
  case prev free           -> remove prev; merge; bp = prev; insert
  case both free           -> remove prev, next; merge all three; bp = prev; insert
```

## Code

A CS:APP-style segregated-fit allocator with boundary-tag coalescing. It runs over a
`mem_sbrk`/`mem_heap_lo`/`mem_heap_hi` memory shim and obeys 8-byte alignment.

```c
#include <string.h>
#include "mm.h"
#include "memlib.h"

/* ---- constants and packed-word / boundary-tag macros ---- */
#define WSIZE      4                 /* header/footer word (bytes) */
#define DSIZE      8                 /* double word; alignment */
#define CHUNKSIZE  (1<<12)           /* default heap growth (bytes) */
#define NCLASSES   16                /* number of segregated size classes */
#define MINBLK     (2*WSIZE + 2*sizeof(void *)) /* hdr + ftr + next + prev */
#define MAX(x,y)   ((x) > (y) ? (x) : (y))
#define ALIGN(size) (((size) + (DSIZE-1)) & ~0x7)

#define PACK(size, alloc)  ((size) | (alloc))         /* alloc bit in low 3 (8-aligned) bits */
#define GET(p)             (*(unsigned int *)(p))
#define PUT(p, val)        (*(unsigned int *)(p) = (val))
#define GET_SIZE(p)        (GET(p) & ~0x7)
#define GET_ALLOC(p)       (GET(p) & 0x1)

#define HDRP(bp)      ((char *)(bp) - WSIZE)
#define FTRP(bp)      ((char *)(bp) + GET_SIZE(HDRP(bp)) - DSIZE)    /* boundary tag */
#define NEXT_BLKP(bp) ((char *)(bp) + GET_SIZE((char *)(bp) - WSIZE))
#define PREV_BLKP(bp) ((char *)(bp) - GET_SIZE((char *)(bp) - DSIZE))/* via prev footer */

/* free-block payload holds the explicit-list links */
#define NEXT_FREE(bp) (*(char **)(bp))
#define PREV_FREE(bp) (*(char **)((char *)(bp) + sizeof(char *)))

static char *heap_listp;             /* just past the prologue */
static char *free_lists[NCLASSES];   /* LIFO list head per size class */

static int class_of(size_t size) {
    int c = 0;
    size_t limit = MINBLK;
    while (size > limit && c < NCLASSES-1) { limit <<= 1; c++; }
    return c;
}

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

static void *coalesce(char *bp) {
    char *prev = PREV_BLKP(bp);
    char *next = NEXT_BLKP(bp);
    size_t prev_alloc = GET_ALLOC(FTRP(prev));
    size_t next_alloc = GET_ALLOC(HDRP(next));
    size_t size = GET_SIZE(HDRP(bp));

    if (prev_alloc && next_alloc) {                 /* case 1 */
        insert_free(bp);
    } else if (prev_alloc && !next_alloc) {         /* case 2: next free */
        remove_free(next);
        size += GET_SIZE(HDRP(next));
        PUT(HDRP(bp), PACK(size, 0));
        PUT(FTRP(bp), PACK(size, 0));
        insert_free(bp);
    } else if (!prev_alloc && next_alloc) {         /* case 3: prev free */
        remove_free(prev);
        size += GET_SIZE(HDRP(prev));
        PUT(FTRP(bp), PACK(size, 0));
        PUT(HDRP(prev), PACK(size, 0));
        bp = prev;
        insert_free(bp);
    } else {                                        /* case 4: both free */
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
    size_t size = (words % 2) ? (words+1)*WSIZE : words*WSIZE;
    if ((bp = mem_sbrk(size)) == (void *)-1) return NULL;
    PUT(HDRP(bp), PACK(size, 0));            /* new free block where epilogue was */
    PUT(FTRP(bp), PACK(size, 0));
    PUT(HDRP(NEXT_BLKP(bp)), PACK(0, 1));    /* fresh epilogue header */
    return coalesce(bp);
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

static void *find_fit(size_t asize) {           /* segregated first-fit ~ best-fit */
    for (int c = class_of(asize); c < NCLASSES; c++)
        for (char *bp = free_lists[c]; bp; bp = NEXT_FREE(bp))
            if (asize <= GET_SIZE(HDRP(bp)))
                return bp;
    return NULL;
}

static void place(char *bp, size_t asize) {
    size_t csize = GET_SIZE(HDRP(bp));
    remove_free(bp);
    if (csize - asize >= MINBLK) {              /* split */
        PUT(HDRP(bp), PACK(asize, 1));
        PUT(FTRP(bp), PACK(asize, 1));
        char *rem = NEXT_BLKP(bp);
        PUT(HDRP(rem), PACK(csize - asize, 0));
        PUT(FTRP(rem), PACK(csize - asize, 0));
        insert_free(rem);
    } else {                                    /* keep whole */
        PUT(HDRP(bp), PACK(csize, 1));
        PUT(FTRP(bp), PACK(csize, 1));
    }
}

void *mm_malloc(size_t size) {
    if (heap_listp == 0 && mm_init() == -1) return NULL;
    if (size == 0) return NULL;
    size_t asize = MAX(MINBLK, ALIGN(size + 2*WSIZE));
    char *bp;
    if ((bp = find_fit(asize))) { place(bp, asize); return bp; }
    size_t ext = MAX(asize, CHUNKSIZE);
    if ((bp = extend_heap(ext / WSIZE)) == NULL) return NULL;
    place(bp, asize);
    return bp;
}

void mm_free(void *bp) {
    if (bp == 0) return;
    size_t size = GET_SIZE(HDRP(bp));
    PUT(HDRP(bp), PACK(size, 0));
    PUT(FTRP(bp), PACK(size, 0));
    coalesce(bp);
}

void *mm_realloc(void *ptr, size_t size) {
    if (ptr == NULL) return mm_malloc(size);
    if (size == 0) { mm_free(ptr); return NULL; }
    void *newptr = mm_malloc(size);
    if (!newptr) return NULL;
    size_t old_payload = GET_SIZE(HDRP(ptr)) - 2*WSIZE;
    size_t copy_size = size < old_payload ? size : old_payload;
    memcpy(newptr, ptr, copy_size);
    mm_free(ptr);
    return newptr;
}
```
