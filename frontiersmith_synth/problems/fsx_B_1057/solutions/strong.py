# TIER: strong
"""Clairvoyant appraiser: one whole-trace pass computing, for every visit, the exact
NEXT occurrence index of that object (bisect over precomputed per-id occurrence
lists) -- the "next-use distance" the innovation hook calls for. On a miss:

  - if the object is never requested again, BYPASS unconditionally (admission the
    textbook LRU/GDSF-style policies do not even represent: caching it can only
    ever cost eviction room for zero future benefit).
  - otherwise, only admit by evicting residents whose OWN next occurrence (computed
    fresh, at this same instant, via the same lookahead) is no sooner than this
    object's -- an exchange argument: never evict something you provably need again
    before you'd need the newcomer. If freeing enough room requires evicting a
    strictly more valuable resident, BYPASS instead of admitting.

This is a generalization of Belady/MIN to variable object sizes with an explicit
bypass action; it is a heuristic (offline generalized caching with sizes is
NP-hard), not a proof of optimality, but it exploits genuine lookahead structure
that a purely-reactive recency policy cannot see.
"""
import sys
from bisect import bisect_right
from collections import OrderedDict


def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    N = int(next(it)); K = int(next(it)); C = int(next(it))
    sizes = [0] * (K + 1)
    for i in range(1, K + 1):
        sizes[i] = int(next(it))
    seq = [int(next(it)) for _ in range(N)]

    INF = N + 1
    positions = [[] for _ in range(K + 1)]
    for i, oid in enumerate(seq):
        positions[oid].append(i)

    def next_after(oid, i):
        lst = positions[oid]
        j = bisect_right(lst, i)
        return lst[j] if j < len(lst) else INF

    resident = OrderedDict()  # id -> size
    used = 0
    out = []

    for i, oid in enumerate(seq):
        s = sizes[oid]
        if oid in resident:
            out.append("H")
            continue

        if s > C:
            out.append("B")
            continue

        nu = next_after(oid, i)
        if nu == INF:
            out.append("B")
            continue

        if used + s <= C:
            resident[oid] = s
            used += s
            out.append("A 0")
            continue

        need = used + s - C
        cands = []
        for rid, rsize in resident.items():
            nr = next_after(rid, i)
            if nr >= nu:
                cands.append((nr, rsize, rid))
        cands.sort(key=lambda c: (-c[0], -c[1], c[2]))

        freed = 0
        chosen = []
        for nr, rsize, rid in cands:
            if freed >= need:
                break
            chosen.append(rid)
            freed += rsize

        if freed >= need:
            for rid in chosen:
                rsize = resident.pop(rid)
                used -= rsize
            resident[oid] = s
            used += s
            out.append("A %d %s" % (len(chosen), " ".join(str(r) for r in chosen)))
        else:
            out.append("B")

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
