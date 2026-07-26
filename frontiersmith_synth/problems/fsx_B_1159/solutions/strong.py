# TIER: strong
# INSIGHT: free-block choice is really a TLB-locality decision, not a
# fragmentation-only one, AND the coalesce/defer choice is a *timing*
# decision that should be scripted against what the trace does next -- both
# of which the solver can see in full, since the whole trace is given
# upfront on stdin.
#   1. Placement: among free berths of the exact requested size, prefer one
#      whose SECTOR the crew's roster currently remembers (warm), picking the
#      most-recently-used such sector; only fall back to fragmentation-driven
#      (lowest-address) choice when no warm option exists. This directly
#      trades the textbook best-fit/lowest-address rule for a cache-locality
#      one.
#   2. Coalescing timing: defer joining a freed berth with its sibling if the
#      SAME size will be requested again later in the trace (so the exact
#      warm block is still there to reuse); only join immediately once that
#      size is provably never needed again, trimming fragmentation without
#      destroying a hot address range that is about to be reused.
import sys
from bisect import bisect_right
from collections import OrderedDict


def main():
    it = sys.stdin.buffer.read().split()
    p = 0
    heap = int(it[p]); page = int(it[p + 1]); tlb = int(it[p + 2]); n = int(it[p + 3])
    p += 4

    events = []
    for _ in range(n):
        typ = it[p].decode(); p += 1
        if typ == 'A':
            events.append(('A', int(it[p]), int(it[p + 1]))); p += 2
        elif typ == 'F':
            events.append(('F', int(it[p]))); p += 1
        else:
            events.append(('T', int(it[p]), int(it[p + 1]))); p += 2

    # future-need index, per size: sorted list of positions of 'A' events
    # requesting that exact size.
    alloc_positions = {}
    for idx, ev in enumerate(events):
        if ev[0] == 'A':
            alloc_positions.setdefault(ev[2], []).append(idx)

    def needed_again(size, after_idx):
        lst = alloc_positions.get(size)
        if not lst:
            return False
        j = bisect_right(lst, after_idx)
        return j < len(lst)

    free_blocks = {0: heap}
    alloc_map = {}
    tlb_od = OrderedDict()
    out = []

    def find_ancestor(addr, size):
        for a, s in free_blocks.items():
            if s >= size and a <= addr < a + s and (addr - a) % size == 0:
                return a, s
        return None

    def recency_rank():
        return {pg: i for i, pg in enumerate(tlb_od.keys())}

    def choose_alloc(size):
        exact = [a for a, s in free_blocks.items() if s == size]
        if exact:
            warm = [a for a in exact if (a // page) in tlb_od]
            if warm:
                rank = recency_rank()
                return max(warm, key=lambda a: rank[a // page])
            return min(exact)
        bigger = [(s, a) for a, s in free_blocks.items() if s > size]
        if not bigger:
            bigger = [(s, a) for a, s in free_blocks.items() if s >= size]
        bigger.sort()
        min_s = bigger[0][0]
        same = [a for s, a in bigger if s == min_s]
        warm = [a for a in same if (a // page) in tlb_od]
        if warm:
            rank = recency_rank()
            return max(warm, key=lambda a: rank[a // page])
        return min(same)

    def do_alloc(id_, size):
        addr = choose_alloc(size)
        a, s = find_ancestor(addr, size)
        del free_blocks[a]
        while s > size:
            s //= 2
            left, right = a, a + s
            if addr < right:
                keep, other = left, right
            else:
                keep, other = right, left
            free_blocks[other] = s
            a = keep
        alloc_map[id_] = (addr, size)
        out.append("A %d %d" % (id_, addr))

    def do_free(id_, idx):
        addr, size = alloc_map.pop(id_)
        flag = 0 if needed_again(size, idx) else 1
        out.append("F %d %d" % (id_, flag))
        free_blocks[addr] = size
        if flag == 1:
            a, s = addr, size
            while s < heap:
                buddy = a ^ s
                if free_blocks.get(buddy) == s:
                    del free_blocks[a]; del free_blocks[buddy]
                    a = min(a, buddy); s *= 2
                    free_blocks[a] = s
                else:
                    break

    def do_touch(id_, offset):
        addr, size = alloc_map[id_]
        pg = (addr + offset) // page
        if pg in tlb_od:
            tlb_od.move_to_end(pg)
        else:
            if len(tlb_od) >= tlb:
                tlb_od.popitem(last=False)
            tlb_od[pg] = True

    for idx, ev in enumerate(events):
        if ev[0] == 'A':
            do_alloc(ev[1], ev[2])
        elif ev[0] == 'F':
            do_free(ev[1], idx)
        else:
            do_touch(ev[1], ev[2])

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
