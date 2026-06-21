The method I am presenting is a segregated-fit dynamic memory allocator with boundary-tag coalescing. It implements the classic malloc/free contract over a single contiguous heap that can be grown at its high end by invoking the operating system, for example through sbrk. The allocator receives requests for arbitrary amounts of memory at unpredictable times and must return aligned pointers that remain valid and immovable for the lifetime of each allocation. Because the program holds raw addresses, the allocator can never compact the heap by sliding live blocks together; the only tools available for controlling waste are careful placement of new allocations and aggressive merging of adjacent free blocks whenever one is returned.

The two competing objectives are throughput and utilization. Throughput means that malloc and free should complete in roughly constant time, regardless of how many blocks are already on the heap. Utilization means that the peak amount of memory obtained from the operating system should not be much larger than the maximum live payload the program actually needs at any moment. The gap between these two is fragmentation, and it comes in two forms. Internal fragmentation occurs when a block handed out is larger than the requested payload because of bookkeeping overhead, alignment rounding, or a minimum block size. External fragmentation occurs when the heap contains enough free bytes in total but no single contiguous free region is large enough to satisfy the next request. External fragmentation is the harder problem because it depends on the entire history of requests and on the allocator's placement decisions, and it cannot be repaired after the fact by moving blocks.

The design therefore starts with bookkeeping that makes adjacency cheap to inspect. Every block carries a single packed word at its front, the header, encoding both the block size and an allocated/free flag. Because every block is aligned to eight bytes, the low three bits of every size are always zero, so the allocated bit can hide inside those bits with no extra storage. To coalesce with the physically previous block, however, the allocator needs that previous block's size, and the header is not at a fixed offset below the current block. The solution is a boundary tag: the same packed size-and-status word is duplicated in a footer at the tail of every block. The footer of the previous block sits at a fixed offset immediately below the current block's header, so from any block the allocator can read the previous block's size in constant time and step back over it. This makes both forward and backward coalescing O(1).

With boundary tags in place, freeing a block triggers immediate coalescing. If both physical neighbors are allocated, the newly freed block is simply inserted into the free-block structure. If the next neighbor is free, it is removed from the free structure and absorbed. If the previous neighbor is free, it is removed and the merged block starts at the previous block's address. If both neighbors are free, all three are merged into one larger block. The order of pointer arithmetic matters: the previous and next pointers are cached before any size fields are overwritten, and the footer is written after the header has been updated with the new size so that the footer macro lands in the correct location. The heap is bracketed by a small allocated prologue at the bottom and a zero-size allocated epilogue at the top so that these neighbor inspections never read past the heap boundaries.

Coalescing alone is not enough; the allocator also needs a fast way to find a free block that fits a malloc request. A simple implicit list that walks every block, allocated or free, would make malloc O(total blocks), which is unacceptable when the heap is large and mostly full. The fix is an explicit free list that links only the free blocks, storing next and previous pointers inside each free block's currently unused payload area. Allocated blocks therefore pay no link overhead. Even with an explicit list, a single flat list mixes all sizes together, so a best-fit search that scans the entire list for the smallest fitting block is still expensive.

The key placement idea is to approximate best-fit without a global scan by using segregated free lists. The allocator maintains an array of free lists, one for each geometric size range starting at the minimum legal free block size. To serve a request, it computes the smallest size class whose range can hold the adjusted request and searches only that list, taking the first block that is large enough. If that list yields nothing, it climbs to the next larger class. Because the starting class is the tightest range that can fit the request, first-fit within the right class behaves like best-fit to within the width of the class. Finer size classes move closer to exact best-fit, while coarser classes reduce maintenance work; the class granularity is the direct dial between utilization and throughput.

When a free block is chosen for allocation, the allocator splits it only if the remainder is large enough to be a legal free block itself, meaning it can hold a header, a footer, and the two free-list pointers. If the remainder is smaller than that minimum, the entire block is handed over and the excess becomes internal fragmentation. This is preferable to creating splinters that cannot be reused. The heap is extended by calling sbrk only after the segregated search has failed; the extension size is at least the request size and at least a fixed chunk size to amortize system calls. The new space becomes a free block where the old epilogue sat, a fresh epilogue is written beyond it, and the new block is immediately coalesced with the old top block if that top block was already free.

The resulting allocator is the segregated-fit dynamic memory allocator with boundary-tag coalescing: header and footer tags give constant-time two-sided merging, explicit free lists remove allocated blocks from the search path, and geometric size classes give best-fit-like placement without scanning the entire heap. Immediate coalescing on free keeps external fragmentation in check, while lazy heap growth preserves the high-water mark until existing free space is genuinely exhausted.

```python
"""
Small Python simulation of the segregated-fit allocator core.
It verifies placement, splitting, coalescing, and non-overlap on a synthetic trace.
"""

from __future__ import annotations

NCLASSES = 16
WSIZE = 4
DSIZE = 8
CHUNKSIZE = 1 << 12
OVERHEAD = 2 * WSIZE
MINBLK = OVERHEAD + 2 * 16  # header+footer + two pointer slots (16 bytes each on 64-bit)


def align(size: int) -> int:
    return (size + (DSIZE - 1)) & ~(DSIZE - 1)


def class_of(size: int) -> int:
    c, limit = 0, MINBLK
    while size > limit and c < NCLASSES - 1:
        limit <<= 1
        c += 1
    return c


class Block:
    def __init__(self, start: int, size: int, allocated: bool = True):
        self.start = start
        self.size = size
        self.allocated = allocated

    def __repr__(self):
        return f"Block({self.start}, {self.size}, {'alloc' if self.allocated else 'free'})"


class SegregatedAllocator:
    def __init__(self):
        self.heap_start = 0
        self.heap_end = 0
        self.blocks: list[Block] = []
        self.free_lists: list[list[Block]] = [[] for _ in range(NCLASSES)]

    def _insert_free(self, block: Block):
        c = class_of(block.size)
        block.allocated = False
        self.free_lists[c].insert(0, block)

    def _remove_free(self, block: Block):
        c = class_of(block.size)
        self.free_lists[c].remove(block)

    def _find_fit(self, asize: int) -> Block | None:
        for c in range(class_of(asize), NCLASSES):
            for block in self.free_lists[c]:
                if block.size >= asize:
                    return block
        return None

    def _extend_heap(self, asize: int) -> Block:
        ext = max(asize, CHUNKSIZE)
        block = Block(self.heap_end, ext, False)
        self.blocks.append(block)
        self.heap_end += ext
        return self._coalesce(block)

    def _place(self, block: Block, asize: int):
        self._remove_free(block)
        remainder = block.size - asize
        if remainder >= MINBLK:
            block.allocated = True
            block.size = asize
            rem = Block(block.start + asize, remainder, False)
            self.blocks.append(rem)
            self.blocks.sort(key=lambda b: b.start)
            self._coalesce(rem)
        else:
            block.allocated = True

    def malloc(self, size: int) -> Block:
        if size <= 0:
            raise ValueError("size must be positive")
        asize = max(MINBLK, align(size + OVERHEAD))
        block = self._find_fit(asize)
        if block is None:
            block = self._extend_heap(asize)
        self._place(block, asize)
        return block

    def free(self, block: Block):
        if not block.allocated:
            raise ValueError("double free")
        self._coalesce(block)

    def _coalesce(self, block: Block):
        i = self.blocks.index(block)
        merged = block

        # Merge with left neighbor if it is free.
        if i > 0 and not self.blocks[i - 1].allocated:
            left = self.blocks[i - 1]
            self._remove_free(left)
            left.size += block.size
            self.blocks.pop(i)
            merged = left
            i = self.blocks.index(merged)

        # Merge with right neighbor if it is free.
        if i + 1 < len(self.blocks) and not self.blocks[i + 1].allocated:
            right = self.blocks[i + 1]
            self._remove_free(right)
            merged.size += right.size
            self.blocks.pop(i + 1)

        self._insert_free(merged)
        return merged

    def check_invariants(self):
        # Blocks must partition [heap_start, heap_end] exactly.
        assert self.blocks, "heap must contain at least one block"
        assert self.blocks[0].start == self.heap_start
        for a, b in zip(self.blocks, self.blocks[1:]):
            assert a.start + a.size == b.start, "blocks must be contiguous"
        assert self.blocks[-1].start + self.blocks[-1].size == self.heap_end
        # No two adjacent free blocks (they would have been coalesced).
        for a, b in zip(self.blocks, self.blocks[1:]):
            assert a.allocated or b.allocated, "adjacent free blocks must be coalesced"
        # Free lists must contain exactly the free blocks.
        free_from_lists = sum(len(lst) for lst in self.free_lists)
        free_from_blocks = sum(1 for b in self.blocks if not b.allocated)
        assert free_from_lists == free_from_blocks, "free-list accounting mismatch"

    def utilization(self) -> float:
        live = sum(b.size - OVERHEAD for b in self.blocks if b.allocated)
        return live / self.heap_end if self.heap_end else 0.0


if __name__ == "__main__":
    a = SegregatedAllocator()
    # Synthetic trace: create holes, then reuse them, then grow.
    b1 = a.malloc(64)
    b2 = a.malloc(128)
    b3 = a.malloc(256)
    a.check_invariants()
    a.free(b2)
    a.check_invariants()
    # This request should fit into the coalesced hole left by b2.
    b4 = a.malloc(100)
    a.check_invariants()
    b5 = a.malloc(8192)
    a.check_invariants()
    a.free(b1)
    a.free(b3)
    a.free(b4)
    a.free(b5)
    a.check_invariants()
    print(f"heap high-water mark: {a.heap_end} bytes")
    print(f"final utilization: {a.utilization():.2%}")
    print("all invariants passed")
```

In summary, the segregated-fit dynamic memory allocator with boundary-tag coalescing resolves the tension between speed and space by keeping allocated bits and sizes in packed boundary tags, linking only free blocks in per-size-class lists, and choosing blocks through a class-bucketed first-fit search that approximates best-fit. It extends the heap only when existing free space cannot satisfy a request, splits blocks only when the leftover is reusable, and merges adjacent free neighbors immediately on every free. The Python simulation above exercises these mechanisms on a small trace and checks that the heap remains contiguous, coalesced, and internally consistent throughout.
