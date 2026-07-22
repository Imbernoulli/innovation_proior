# TIER: strong
# Two-part insight, matching the two traps the generator plants:
#
# 1) DECOMPOSE the buffers into weakly-connected components of the precedence DAG. A
#    component that is fully allocated-and-freed in isolation (nothing from another
#    component interleaved into its address range) provably returns the allocator's
#    wilderness top to exactly where it stood before the component started -- so processing
#    components ONE AT A TIME (in any order) always achieves peak = max over components of
#    that component's own local peak, which weakly dominates every interleaving. This is an
#    invariant of the coalescing/"wilderness-chunk" allocator, not a search: any order works.
#
# 2) WITHIN each component (a sliding-window chain), free-as-early-as-legal is NOT optimal:
#    it lets first-fit wedge a smaller, soon-allocated buffer into part of a hole, leaving an
#    unusable remainder that is too small for a same-chain buffer that needs the FULL hole
#    shortly after -- forcing an avoidable extension of the arena. The true lever is exactly
#    WHEN (within its legal window) to free each buffer, which changes what shape of hole is
#    on offer to first-fit at each moment -- an emergent placement-state effect, not a
#    live-byte-count effect (every schedule here frees each buffer at essentially the same
#    relative liveness cost). We search this per-component free-timing space directly with
#    the real allocator simulation as the objective (never the live-byte surrogate), via
#    seeded hill-climbing over integer "extra delay" values.
import sys, random


def simulate_peak(sizes, ops):
    free_list = []
    top = 0
    peak = 0
    block = {}
    for v in ops:
        i = abs(v)
        if v > 0:
            sz = sizes[i]
            placed = False
            for k in range(len(free_list)):
                fs, fsz = free_list[k]
                if fsz >= sz:
                    block[i] = fs
                    if fsz == sz:
                        free_list.pop(k)
                    else:
                        free_list[k] = (fs + sz, fsz - sz)
                    if fs + sz > peak:
                        peak = fs + sz
                    placed = True
                    break
            if not placed:
                block[i] = top
                top += sz
                if top > peak:
                    peak = top
        else:
            s = block[i]
            sz = sizes[i]
            idx = 0
            while idx < len(free_list) and free_list[idx][0] < s:
                idx += 1
            free_list.insert(idx, (s, sz))
            if idx + 1 < len(free_list):
                ns, nsz = free_list[idx]
                rs, rsz = free_list[idx + 1]
                if ns + nsz == rs:
                    free_list[idx] = (ns, nsz + rsz)
                    free_list.pop(idx + 1)
            if idx - 1 >= 0:
                ls, lsz = free_list[idx - 1]
                ns, nsz = free_list[idx]
                if ls + lsz == ns:
                    free_list[idx - 1] = (ls, lsz + nsz)
                    free_list.pop(idx)
                    idx -= 1
            ns, nsz = free_list[idx]
            if ns + nsz == top:
                top = ns
                free_list.pop(idx)
    return peak


def solve_component(idxs, sizes, children_of):
    """idxs: buffer indices of one weakly-connected component, in ascending (valid alloc)
    order. Returns the best-found op sequence (global buffer ids) for JUST this component,
    found by hill-climbing over per-buffer free-delay values, scored with the real
    allocator simulation restricted to this component (valid since a self-contained
    component always starts the allocator fresh, per the invariant above)."""
    m = len(idxs)
    local_sizes = [0] + [sizes[b] for b in idxs]
    local_id = {b: k + 1 for k, b in enumerate(idxs)}
    idxset = set(idxs)
    local_children = [[] for _ in range(m + 1)]
    for b in idxs:
        i = local_id[b]
        for c in children_of[b]:
            if c in idxset:
                local_children[i].append(local_id[c])
    parents_local = [[] for _ in range(m + 1)]
    for i in range(1, m + 1):
        for c in local_children[i]:
            parents_local[c].append(i)

    # earliest legal alloc-STEP at which buffer i may be freed (the step at which its last
    # required child has been allocated; a childless buffer is legal at its own step).
    remaining = [len(local_children[i]) for i in range(m + 1)]
    earliest = [0] * (m + 1)
    for i in range(1, m + 1):
        for p in parents_local[i]:
            remaining[p] -= 1
            if remaining[p] == 0:
                earliest[p] = i
        if remaining[i] == 0 and earliest[i] == 0:
            earliest[i] = i

    def build(delays):
        by_step = {}
        for i in range(1, m + 1):
            t = min(m, earliest[i] + delays[i])
            by_step.setdefault(t, []).append(i)
        ops = []
        done = set()
        for k in range(1, m + 1):
            ops.append(k)
            for j in sorted(by_step.get(k, [])):
                ops.append(-j)
                done.add(j)
        for i in range(1, m + 1):
            if i not in done:
                ops.append(-i)
        return ops

    seed = (2971 * m + 104729 * sum(local_sizes) + 17) & 0x7fffffff
    rng = random.Random(seed)

    zero_delays = [0] * (m + 1)
    best_delays = zero_delays[:]
    best_peak = simulate_peak(local_sizes, build(zero_delays))

    # Multiple hill-climbing restarts (the free-ASAP start plus several randomized starts):
    # a single climb from delay=0 can stall in a local optimum that hides a genuinely better
    # hole-alignment a few random jumps away, especially for longer chains.
    restarts = 1 + min(18, max(6, m))
    iters = min(4000, max(800, m * 100))
    for r in range(restarts):
        cur = zero_delays[:] if r == 0 else [rng.randint(0, m) for _ in range(m + 1)]
        cur[0] = 0
        cur_peak = simulate_peak(local_sizes, build(cur))
        if cur_peak < best_peak:
            best_peak = cur_peak
            best_delays = cur[:]
        for _ in range(iters):
            i = rng.randint(1, m)
            old = cur[i]
            cur[i] = rng.randint(0, m)
            pk = simulate_peak(local_sizes, build(cur))
            if pk <= cur_peak:
                cur_peak = pk
                if pk < best_peak:
                    best_peak = pk
                    best_delays = cur[:]
            else:
                cur[i] = old

    best_ops_local = build(best_delays)
    return [idxs[abs(v) - 1] * (1 if v > 0 else -1) for v in best_ops_local]


def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    N = int(next(it))
    sizes = [0] + [int(next(it)) for _ in range(N)]
    M = int(next(it))
    children_of = [[] for _ in range(N + 1)]
    edges = []
    for _ in range(M):
        p = int(next(it))
        c = int(next(it))
        children_of[p].append(c)
        edges.append((p, c))

    parent = list(range(N + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for p, c in edges:
        union(p, c)

    comp_map = {}
    for i in range(1, N + 1):
        comp_map.setdefault(find(i), []).append(i)
    comps = list(comp_map.values())
    for c in comps:
        c.sort()
    comps.sort(key=lambda c: c[0])

    final_ops = []
    for idxs in comps:
        final_ops.extend(solve_component(idxs, sizes, children_of))

    sys.stdout.write('\n'.join(str(x) for x in final_ops) + '\n')


if __name__ == '__main__':
    main()
