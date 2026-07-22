# TIER: strong
"""Insight: relabelling erased every numeric hint of which errands belong to
the same reading-list collection, but it could not erase the fact that two
consultation errands drawing from the SAME collection's pool must literally
share a read address somewhere. Union two errands whenever they share a read
slot; the connected components ARE the hidden collections (address pools of
different collections are disjoint by construction, so this recovery is
exact, not approximate). A one-off "clearance" errand shares no address with
anyone -- it inherits its collection from the single consultation errand it
unlocks (found via the dependence edge), which the union-find step has
already placed correctly.

Once collections are known, the schedule stops being "whatever is ready with
best overlap right now" (the myopic trap: clearance errands for many
collections finish in a scattered, arbitrary order, so a collection's
consultations become ready at unpredictable, spread-out moments, and a purely
reactive scheduler ping-pongs between collections losing the pool from cache
every time). Instead: commit to ONE target collection, race through ALL of
its currently-ready clearance errands first (that is what makes its
consultations ready), then run all of its ready consultations back-to-back
(cache-affinity order among them), and only THEN move to the next
collection. This is a genuine dependence-respecting tiling: clearance
errands for the CURRENT collection are prioritized purely to enable a clean
consultation batch, not because they offer any cache benefit themselves."""
import sys
from collections import defaultdict, Counter, OrderedDict


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

    # --- 1. recover collections: union errands that share a read address ---
    parent = {o: o for o in op_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    addr_to_ops = defaultdict(list)
    for oid in op_ids:
        for a in op_reads[oid]:
            addr_to_ops[a].append(oid)
    for _addr, ops_here in addr_to_ops.items():
        if len(ops_here) > 1:
            first = ops_here[0]
            for other in ops_here[1:]:
                union(first, other)

    cluster_of = {o: find(o) for o in op_ids}
    comp_size = Counter(cluster_of.values())
    # a singleton component (no address shared with any other errand) is a
    # one-off clearance errand; it inherits the collection of what it unlocks
    is_clearance = {o: comp_size[cluster_of[o]] == 1 for o in op_ids}
    for oid in op_ids:
        if is_clearance[oid] and adj[oid]:
            cluster_of[oid] = cluster_of[adj[oid][0]]

    # --- 2. dependence-respecting tiling: commit to one collection at a time ---
    ready = set(o for o in op_ids if indeg[o] == 0)
    od = OrderedDict()
    order = []

    def overlap(oid):
        s = set(op_reads[oid])
        s.add(op_write[oid])
        return len(s & od.keys())

    def touch(oid):
        for a in op_reads[oid] + [op_write[oid]]:
            if a in od:
                od.move_to_end(a)
            else:
                od[a] = True
                if len(od) > K:
                    od.popitem(last=False)

    while ready:
        target = min(cluster_of[o] for o in ready)
        progressed = True
        while progressed:
            progressed = False
            clearances_here = [o for o in ready if cluster_of[o] == target and is_clearance[o]]
            if clearances_here:
                oid = min(clearances_here)
            else:
                members_here = [o for o in ready if cluster_of[o] == target and not is_clearance[o]]
                oid = max(members_here, key=lambda o: (overlap(o), -o)) if members_here else None
            if oid is None:
                break
            ready.remove(oid)
            order.append(oid)
            touch(oid)
            for w in adj[oid]:
                indeg[w] -= 1
                if indeg[w] == 0:
                    ready.add(w)
            progressed = True

    sys.stdout.write(" ".join(map(str, order)))


if __name__ == "__main__":
    main()
