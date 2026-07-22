# TIER: greedy
"""The obvious first approach: size-aware LRU. Always admit on a miss (bypass only
when the object physically cannot ever fit); when room is needed, evict the
least-recently-used resident objects first, oldest first, until enough capacity is
freed. No lookahead at all -- purely reactive to recency, exactly the textbook
policy a size-aware cache normally reaches for. It never represents "refuse to
cache" as anything but a last resort."""
import sys
from collections import OrderedDict


def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    N = int(next(it)); K = int(next(it)); C = int(next(it))
    sizes = [0] * (K + 1)
    for i in range(1, K + 1):
        sizes[i] = int(next(it))
    seq = [int(next(it)) for _ in range(N)]

    resident = OrderedDict()  # id -> size, ordered LRU -> MRU
    used = 0
    out = []

    for oid in seq:
        s = sizes[oid]
        if oid in resident:
            out.append("H")
            resident.move_to_end(oid)
            continue
        if s > C:
            out.append("B")
            continue
        if used + s <= C:
            resident[oid] = s
            used += s
            out.append("A 0")
            continue
        # evict oldest residents until enough room is freed
        evicted = []
        freed = 0
        need = used + s - C
        while freed < need and resident:
            old_id, old_size = next(iter(resident.items()))
            del resident[old_id]
            used -= old_size
            freed += old_size
            evicted.append(old_id)
        resident[oid] = s
        used += s
        out.append("A %d %s" % (len(evicted), " ".join(str(e) for e in evicted)))

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
