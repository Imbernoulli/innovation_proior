# TIER: greedy
"""The obvious 'textbook' fix over the trivial order: a myopic cache-affinity
list scheduler.  At every step, among the errands whose dependencies are
already satisfied, run whichever one overlaps the current cart contents the
most (simulating the cart as it goes).  This reacts to locality but never
discovers the hidden grid geometry, so on large/tight-cart instances it still
thrashes: it optimizes one step ahead, not the tile the errand belongs to."""
import sys
from collections import OrderedDict


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    K = int(next(it))

    op_reads = {}
    op_write = {}
    op_ids = []
    for _ in range(N):
        oid = int(next(it))
        k = int(next(it))
        reads = [int(next(it)) for _ in range(k)]
        w = int(next(it))
        op_reads[oid] = reads
        op_write[oid] = w
        op_ids.append(oid)

    adj = {o: [] for o in op_ids}
    indeg = {o: 0 for o in op_ids}
    for _ in range(M):
        u = int(next(it))
        v = int(next(it))
        adj[u].append(v)
        indeg[v] += 1

    ready = set(o for o in op_ids if indeg[o] == 0)
    od = OrderedDict()  # simulated cart, LRU at front
    order = []

    def overlap(oid):
        s = set(op_reads[oid])
        s.add(op_write[oid])
        return len(s & od.keys())

    while ready:
        best = None
        best_score = -1
        for oid in ready:
            sc = overlap(oid)
            if sc > best_score or (sc == best_score and (best is None or oid < best)):
                best_score = sc
                best = oid
        oid = best
        ready.remove(oid)
        order.append(oid)

        for a in op_reads[oid] + [op_write[oid]]:
            if a in od:
                od.move_to_end(a)
            else:
                od[a] = True
                if len(od) > K:
                    od.popitem(last=False)

        for w in adj[oid]:
            indeg[w] -= 1
            if indeg[w] == 0:
                ready.add(w)

    sys.stdout.write(" ".join(map(str, order)))


if __name__ == "__main__":
    main()
