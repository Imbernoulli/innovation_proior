The method I am presenting is a segregated-fit dynamic memory allocator with boundary-tag coalescing. It implements the classic malloc/free contract over a single contiguous heap that can be grown at its high end by invoking the operating system, for example through sbrk. The allocator receives requests for arbitrary amounts of memory at unpredictable times and must return aligned pointers that remain valid and immovable for the lifetime of each allocation. Because the program holds raw addresses, the allocator can never compact the heap by sliding live blocks together; the only tools available for controlling waste are careful placement of new allocations and aggressive merging of adjacent free blocks whenever one is returned.

The two competing objectives are throughput and utilization. Throughput means that malloc and free should complete in roughly constant time, regardless of how many blocks are already on the heap. Utilization means that the peak amount of memory obtained from the operating system should not be much larger than the maximum live payload the program actually needs at any moment. The gap between these two is fragmentation, and it comes in two forms. Internal fragmentation occurs when a block handed out is larger than the requested payload because of bookkeeping overhead, alignment rounding, or a minimum block size. External fragmentation occurs when the heap contains enough free bytes in total but no single contiguous free region is large enough to satisfy the next request. External fragmentation is the harder problem because it depends on the entire history of requests and on the allocator's placement decisions, and it cannot be repaired after the fact by moving blocks.

The design therefore starts with bookkeeping that makes adjacency cheap to inspect. Every block carries a single packed word at its front, the header, encoding both the block size and an allocated/free flag. Because every block is aligned to eight bytes, the low three bits of every size are always zero, so the allocated bit can hide inside those bits with no extra storage. To coalesce with the physically previous block, however, the allocator needs that previous block's size, and the header is not at a fixed offset below the current block. The solution is a boundary tag: the same packed size-and-status word is duplicated in a footer at the tail of every block. The footer of the previous block sits at a fixed offset immediately below the current block's header, so from any block the allocator can read the previous block's size in constant time and step back over it. This makes both forward and backward coalescing O(1).

With boundary tags in place, freeing a block triggers immediate coalescing. If both physical neighbors are allocated, the newly freed block is simply inserted into the free-block structure. If the next neighbor is free, it is removed from the free structure and absorbed. If the previous neighbor is free, it is removed and the merged block starts at the previous block's address. If both neighbors are free, all three are merged into one larger block. The order of pointer arithmetic matters: the previous and next pointers are cached before any size fields are overwritten, and the footer is written after the header has been updated with the new size so that the footer macro lands in the correct location. The heap is bracketed by a small allocated prologue at the bottom and a zero-size allocated epilogue at the top so that these neighbor inspections never read past the heap boundaries.

Coalescing alone is not enough; the allocator also needs a fast way to find a free block that fits a malloc request. A simple implicit list that walks every block, allocated or free, would make malloc O(total blocks), which is unacceptable when the heap is large and mostly full. The fix is an explicit free list that links only the free blocks, storing next and previous pointers inside each free block's currently unused payload area. Allocated blocks therefore pay no link overhead. Even with an explicit list, a single flat list mixes all sizes together, so a best-fit search that scans the entire list for the smallest fitting block is still expensive.

The key placement idea is to approximate best-fit without a global scan by using segregated free lists. The allocator maintains an array of free lists, one for each geometric size range starting at the minimum legal free block size. To serve a request, it computes the smallest size class whose range can hold the adjusted request and searches only that list, taking the first block that is large enough. If that list yields nothing, it climbs to the next larger class. Because the starting class is the tightest range that can fit the request, first-fit within the right class behaves like best-fit to within the width of the class. Finer size classes move closer to exact best-fit, while coarser classes reduce maintenance work; the class granularity is the direct dial between utilization and throughput.

When a free block is chosen for allocation, the allocator splits it only if the remainder is large enough to be a legal free block itself, meaning it can hold a header, a footer, and the two free-list pointers. If the remainder is smaller than that minimum, the entire block is handed over and the excess becomes internal fragmentation. This is preferable to creating splinters that cannot be reused. The heap is extended by calling sbrk only after the segregated search has failed; the extension size is at least the request size and at least a fixed chunk size to amortize system calls. The new space becomes a free block where the old epilogue sat, a fresh epilogue is written beyond it, and the new block is immediately coalesced with the old top block if that top block was already free.

The resulting allocator is the segregated-fit dynamic memory allocator with boundary-tag coalescing: header and footer tags give constant-time two-sided merging, explicit free lists remove allocated blocks from the search path, and geometric size classes give best-fit-like placement without scanning the entire heap. Immediate coalescing on free keeps external fragmentation in check, while lazy heap growth preserves the high-water mark until existing free space is genuinely exhausted.

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

In summary, the segregated-fit dynamic memory allocator with boundary-tag coalescing resolves the tension between speed and space by keeping allocated bits and sizes in packed boundary tags, linking only free blocks in per-size-class lists, and choosing blocks through a class-bucketed first-fit search that approximates best-fit. It extends the heap only when existing free space cannot satisfy a request, splits blocks only when the leftover is reusable, and merges adjacent free neighbors immediately on every free. The code above realizes exactly this design over a `mem_sbrk`-backed heap shim: `mm_init` lays down the alignment pad, the allocated prologue and epilogue sentinels, and empty per-class list heads; `mm_malloc` computes the aligned request size and routes it through `find_fit` and `place`, extending the heap only on a miss; and `mm_free` clears the boundary tags and hands the block to `coalesce`, which runs the four-case merge and reinserts the result into its size class.
