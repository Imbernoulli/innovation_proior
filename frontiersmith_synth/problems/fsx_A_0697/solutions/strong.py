# TIER: strong
"""Reserved-slot exchange argument.

Insight: a footprint-only best-fit sweep has no notion of "this crate matters
for locality" -- it reuses whichever hole is free right now, so two crates
that are checked constantly but never coexist in time can still end up on
opposite ends of the shelf if some unrelated crate happens to occupy the
freed hole first. But *any* set of crates whose stays are pairwise disjoint
in time can, by definition, share one fixed shelf slot with zero feasibility
risk -- there is never a moment two of them are both present.

So: read the check trace, find the frequently-checked ("hot") crates, and
compute the LARGEST pairwise-disjoint subset of them via the classic
interval-scheduling exchange argument (sort by departure time, greedily keep
a crate whenever it starts after the last kept one leaves -- this provably
maximizes the count of mutually compatible crates). Pin that whole subset to
one reserved shelf slot at offset 0, sized to the largest crate in it. Every
check on any of them then touches only that tiny fixed slot's aisles,
independent of how many hot crates there are or when they arrive. Every
other crate is packed with an ordinary best-fit sweep in the shelf space
*above* the reserved slot -- a small, bounded footprint tax in exchange for
collapsing the hot crates' aisle footprint to a constant.
"""
import sys


def main():
    data = sys.stdin.read().split()
    pos = 0

    def next_int():
        nonlocal pos
        v = int(data[pos])
        pos += 1
        return v

    N = next_int()
    M = next_int()
    next_int()  # PAGE
    next_int()  # LAMBDA

    sizes = [0] * N
    births = [0] * N
    deaths = [0] * N
    for i in range(N):
        sizes[i] = next_int()
        births[i] = next_int()
        deaths[i] = next_int()

    touch_count = [0] * N
    for _ in range(M):
        next_int()  # t
        c = next_int() - 1
        touch_count[c] += 1

    max_touch = max(touch_count) if touch_count else 0
    hot_idx = []
    if max_touch > 0:
        threshold = max(1, (max_touch + 1) // 2)
        candidates = [i for i in range(N) if touch_count[i] >= threshold]
        # Exchange-argument interval scheduling: sort by departure time,
        # greedily keep whatever is compatible with what's already kept.
        # This provably gives a maximum pairwise-disjoint subset.
        candidates.sort(key=lambda i: deaths[i])
        last_death = -1
        for i in candidates:
            if births[i] >= last_death:
                hot_idx.append(i)
                last_death = deaths[i]

    reserved_width = max((sizes[i] for i in hot_idx), default=0)
    is_reserved = [False] * N
    for i in hot_idx:
        is_reserved[i] = True

    events = []
    for i in range(N):
        events.append((births[i], 1, i))
        events.append((deaths[i], 0, i))
    events.sort()

    free = []  # holes live entirely at addresses >= reserved_width
    addr = [0] * N
    peak = reserved_width

    def bestfit(size):
        best = -1
        for k in range(len(free)):
            s, l = free[k]
            if l < size:
                continue
            if best == -1 or l < free[best][1] or (l == free[best][1] and s < free[best][0]):
                best = k
        return best

    def alloc(size):
        nonlocal peak
        k = bestfit(size)
        if k == -1:
            start = peak
            peak += size
            return start
        s, l = free[k]
        start = s
        if l == size:
            free.pop(k)
        else:
            free[k][0] = s + size
            free[k][1] = l - size
        return start

    def dealloc(start, size):
        free.append([start, size])
        free.sort()
        merged = []
        for iv in free:
            if merged and merged[-1][0] + merged[-1][1] == iv[0]:
                merged[-1][1] += iv[1]
            else:
                merged.append(iv)
        free[:] = merged

    for _, typ, i in events:
        if is_reserved[i]:
            if typ == 1:
                addr[i] = 0
            continue
        if typ == 0:
            dealloc(addr[i], sizes[i])
        else:
            addr[i] = alloc(sizes[i])

    sys.stdout.write("\n".join(str(a) for a in addr) + "\n")


if __name__ == "__main__":
    main()
