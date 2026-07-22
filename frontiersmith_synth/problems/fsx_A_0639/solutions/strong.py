# TIER: strong
# Insight: simulate forward what each early insertion forecloses.  For every
# ordered pair (u, v), if u's footprint would intersect v's access corridor,
# then u must NOT be installed before v -- v must precede u, or v is lost the
# instant u seals its channel.  This gives a "must-precede" digraph.  A set
# of parts is jointly installable iff the digraph restricted to that set is
# acyclic (any topological order of it is a legal, fully-successful install
# order).  Cycles must be broken by sacrificing the cheapest part in the
# cycle (a value-weighted feedback-vertex-set heuristic), not an arbitrary
# one -- that is exactly what protects a high-value part whose channel would
# otherwise be sealed by a cheap "enabler" part rushed in first.
import sys


def rect_cells(x, y, w, h):
    return frozenset((x + dx, y + dy) for dx in range(w) for dy in range(h))


def find_cycle(nodes, succ):
    color = {u: 0 for u in nodes}
    cyc = [None]

    def dfs(u, stack):
        color[u] = 1
        stack.append(u)
        for v in sorted(succ.get(u, ())):
            if color.get(v) == 0:
                dfs(v, stack)
                if cyc[0] is not None:
                    return
            elif color.get(v) == 1:
                idx = stack.index(v)
                cyc[0] = stack[idx:]
                return
        stack.pop()
        color[u] = 2

    for u in sorted(nodes):
        if color[u] == 0:
            dfs(u, [])
            if cyc[0] is not None:
                return cyc[0]
    return None


def main():
    d = sys.stdin.read().split()
    p = 0
    n = int(d[p]); p += 1
    W = int(d[p]); p += 1
    H = int(d[p]); p += 1
    values = []
    footprints = []
    corridors = []
    for _ in range(n):
        val = int(d[p]); p += 1
        fx = int(d[p]); p += 1; fy = int(d[p]); p += 1
        fw = int(d[p]); p += 1; fh = int(d[p]); p += 1
        cx = int(d[p]); p += 1; cy = int(d[p]); p += 1
        cw = int(d[p]); p += 1; ch = int(d[p]); p += 1
        values.append(val)
        footprints.append(rect_cells(fx, fy, fw, fh))
        corridors.append(rect_cells(cx, cy, cw, ch))

    # succ[v] = set of u such that v must precede u (edge v -> u)
    succ = {i: set() for i in range(n)}
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            if footprints[u] & corridors[v]:
                succ[v].add(u)

    kept = set(range(n))
    sacrificed = []
    while True:
        sub = {u: {w for w in succ.get(u, ()) if w in kept} for u in kept}
        cyc = find_cycle(kept, sub)
        if cyc is None:
            break
        worst = min(cyc, key=lambda x: (values[x], x))
        kept.discard(worst)
        sacrificed.append(worst)

    indeg = {u: 0 for u in kept}
    for u in kept:
        for v in succ.get(u, ()):
            if v in kept:
                indeg[v] += 1
    avail = sorted(u for u in kept if indeg[u] == 0)
    order = []
    while avail:
        avail.sort()
        u = avail.pop(0)
        order.append(u)
        for v in sorted(succ.get(u, ())):
            if v in kept:
                indeg[v] -= 1
                if indeg[v] == 0:
                    avail.append(v)

    final = order + sorted(sacrificed)
    print(" ".join(str(i) for i in final))


if __name__ == "__main__":
    main()
