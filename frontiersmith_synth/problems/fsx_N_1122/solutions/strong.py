# TIER: strong
"""
The insight: a node that needs several incoming pulses to combine (each one
alone is below its threshold, but their sum clears it) can only be crossed
"for free" if ALL of those pulses land in the very same simulation step --
because a node only emits once, right after it activates, and any partial
credit that misses its partner decays away for good.

So instead of firing every source at time 0 (the "simultaneous seed set"),
walk forward from each source along its relay chain to find which
convergence node it feeds, measure the chain length (= propagation delay),
and DELAY the shorter chains' sources so every chain feeding the same
convergence node finishes at the exact same step. That reinforcing wave
clears the join with zero extra events; only sources are ever fired.
Whatever the alignment search cannot resolve (multi-owner conflicts,
degenerate topology) is patched directly as a safety net, but on this
family that almost never triggers.
"""
import sys


def read_instance(toks):
    p = 0

    def nxt():
        nonlocal p
        v = toks[p]
        p += 1
        return v

    n = int(nxt()); m = int(nxt())
    decay_num = int(nxt()); decay_den = int(nxt())
    ext_boost = int(nxt()); horizon = int(nxt())
    theta = [int(nxt()) for _ in range(n)]
    children = [[] for _ in range(n)]
    parents = [[] for _ in range(n)]
    indeg = [0] * n
    for _ in range(m):
        u = int(nxt()); v = int(nxt()); w = int(nxt())
        children[u].append((v, w))
        parents[v].append((u, w))
        indeg[v] += 1
    return n, decay_num, decay_den, ext_boost, horizon, theta, children, parents, indeg


def simulate(n, theta, children, decay_num, decay_den, ext_boost, horizon, events):
    acc = [0] * n
    active = [False] * n
    just_activated = [False] * n
    for t in range(horizon):
        for node in events.get(t, []):
            if not active[node]:
                acc[node] += ext_boost
        contrib = [0] * n
        for u in range(n):
            if just_activated[u]:
                for (v, w) in children[u]:
                    if not active[v]:
                        contrib[v] += w
        for v in range(n):
            if contrib[v] and not active[v]:
                acc[v] += contrib[v]
        newly = []
        for v in range(n):
            if not active[v] and acc[v] >= theta[v]:
                newly.append(v)
        just_activated = [False] * n
        for v in newly:
            active[v] = True
            just_activated[v] = True
        for v in range(n):
            if not active[v]:
                acc[v] = (acc[v] * decay_num) // decay_den
    return active


def main():
    toks = sys.stdin.read().split()
    n, decay_num, decay_den, ext_boost, horizon, theta, children, parents, indeg = read_instance(toks)

    roots = [v for v in range(n) if indeg[v] == 0]

    # For each root, walk forward through single-parent relay nodes until a
    # convergence node (indeg >= 2) or a dead end is reached.
    # record: convergence_node -> list of (root, offset, weight)
    convergence = {}
    root_has_target = set()
    for r in roots:
        cur = r
        hops = 0
        while True:
            outs = children[cur]
            if len(outs) != 1:
                break  # dead end (or unexpected branching) -- no join to align
            v, w = outs[0]
            if indeg[v] >= 2:
                convergence.setdefault(v, []).append((r, hops, w))
                root_has_target.add(r)
                break
            cur = v
            hops += 1

    fire_time = {}
    patch_nodes = set()

    for j, entries in convergence.items():
        if len(entries) < 2:
            continue
        anchor_offset = max(off for (_r, off, _w) in entries)
        ok = True
        proposed = {}
        for (r, off, _w) in entries:
            t0 = anchor_offset - off
            if t0 < 0:
                ok = False
                break
            if r in fire_time and fire_time[r] != t0:
                # this root is already pinned by a different join -- can't
                # also satisfy this one by retiming it.
                ok = False
                continue
            proposed[r] = t0
        total_w = sum(w for (_r, _off, w) in entries)
        if total_w < theta[j]:
            ok = False
        if ok:
            fire_time.update(proposed)
        else:
            patch_nodes.add(j)
            for (r, _off, _w) in entries:
                proposed.setdefault(r, 0)
            fire_time.update({r: t for r, t in proposed.items() if r not in fire_time})

    for r in roots:
        if r not in fire_time:
            fire_time[r] = 0

    events = {}
    for r in roots:
        events.setdefault(fire_time[r], []).append(r)

    active = simulate(n, theta, children, decay_num, decay_den, ext_boost, horizon, events)

    schedule = [(r, fire_time[r]) for r in roots]
    patch_time = horizon - 1
    for v in range(n):
        if not active[v]:
            events.setdefault(patch_time, []).append(v)
            schedule.append((v, patch_time))
            active[v] = True

    out = [str(len(schedule))]
    for (v, t) in schedule:
        out.append(f"{v} {t}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
