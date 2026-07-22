#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE forbidden-window de Bruijn instance to stdout.

Instance = (k, L, forbidden set). Nodes are length-(L-1) strings over digits 0..k-1;
each length-L window w is a directed edge w[:-1] -> w[1:]. We delete a forbidden set of
windows/edges. Two instance flavours:
  - "easy": the deleted edges form a union of cycles, so removing them preserves every
    node's in-degree==out-degree (an Eulerian circuit still exists directly).
  - "trap": the deleted edges are an unstructured random subset, chosen (by rejection
    sampling, tuned deterministically from testId) so that (a) the remaining allowed graph
    is still strongly connected (a solution exists) but (b) a real per-node degree
    imbalance remains, forcing non-trivial shortest-path repair before an Eulerian tour
    exists.
Seeding: a local random.Random(testId * 1_000_003 + salt) only -- fully deterministic.
"""
import sys, random, string
from collections import deque

DIGITS = string.digits


def node_list(k, nm1):
    if nm1 == 0:
        return [""]
    out = [""]
    for _ in range(nm1):
        out = [p + d for p in out for d in DIGITS[:k]]
    return sorted(out)


def full_edges(k, L):
    return sorted(w for w in (p + d for p in node_list(k, L - 1) for d in DIGITS[:k]))


def build_out_adj(allowed, L):
    adj = {}
    for w in allowed:
        u, v = w[:-1], w[1:]
        adj.setdefault(u, []).append((v, w[-1]))
    for u in adj:
        adj[u].sort()
    return adj


def build_in_adj(allowed, L):
    adj = {}
    for w in allowed:
        u, v = w[:-1], w[1:]
        adj.setdefault(v, []).append(u)
    return adj


def active_nodes(allowed):
    s = set()
    for w in allowed:
        s.add(w[:-1]); s.add(w[1:])
    return sorted(s)


def strongly_connected(allowed):
    act = active_nodes(allowed)
    if not act:
        return False
    out_adj = build_out_adj(allowed, len(next(iter(allowed))) if allowed else 1)
    in_adj = build_in_adj(allowed, len(next(iter(allowed))) if allowed else 1)
    root = act[0]

    def bfs(adj_fwd, is_out):
        seen = {root}
        q = deque([root])
        while q:
            u = q.popleft()
            nbrs = adj_fwd.get(u, [])
            for item in nbrs:
                v = item[0] if is_out else item
                if v not in seen:
                    seen.add(v); q.append(v)
        return seen

    f = bfs(out_adj, True)
    b = bfs(in_adj, False)
    act_set = set(act)
    return f == act_set and b == act_set


def imbalance(allowed):
    """returns dict node -> (indeg - outdeg)"""
    ind, outd = {}, {}
    for w in allowed:
        u, v = w[:-1], w[1:]
        outd[u] = outd.get(u, 0) + 1
        ind[v] = ind.get(v, 0) + 1
    d = {}
    for n in set(list(ind.keys()) + list(outd.keys())):
        d[n] = ind.get(n, 0) - outd.get(n, 0)
    return d


def bfs_dist(src, out_adj):
    dist = {src: 0}
    q = deque([src])
    while q:
        u = q.popleft()
        for v, _c in out_adj.get(u, []):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def greedy_augment_cost(allowed):
    """Estimate total extra edges needed to balance in/out degree, via greedy
    nearest-neighbour matching of surplus (need extra out) to deficit (need extra in)
    nodes using BFS shortest-path distance. Deterministic given `allowed` (sorted input,
    no randomness). Returns total augmentation length (int)."""
    d = imbalance(allowed)
    plus = sorted([n for n, v in d.items() if v > 0 for _ in range(v)])   # need extra OUT
    minus = sorted([n for n, v in d.items() if v < 0 for _ in range(-v)])  # need extra IN
    if not plus:
        return 0
    out_adj = build_out_adj(allowed, 0)
    total = 0
    used_minus = [False] * len(minus)
    dist_cache = {}
    for p in plus:
        if p not in dist_cache:
            dist_cache[p] = bfs_dist(p, out_adj)
        dmap = dist_cache[p]
        best_j, best_d = -1, None
        for j, m in enumerate(minus):
            if used_minus[j]:
                continue
            dd = dmap.get(m)
            if dd is None:
                continue
            if best_d is None or dd < best_d or (dd == best_d and m < minus[best_j]):
                best_d, best_j = dd, j
        if best_j == -1:
            return -1  # cannot balance -> not strongly connected enough; caller retries
        used_minus[best_j] = True
        total += best_d
    return total


def gen_easy(k, L, rng, n_cycles):
    full = full_edges(k, L)
    fullset = set(full)
    for attempt in range(60):
        remaining = set(full)
        F = set()
        ok = True
        for _c in range(n_cycles):
            start_candidates = sorted(remaining)
            if not start_candidates:
                ok = False; break
            cur = rng.choice(start_candidates)
            path_nodes = [cur[: L - 1]]
            path_edges = []
            pos = {cur[: L - 1]: 0}
            found_cycle = False
            for _step in range(4 * len(node_list(k, L - 1)) + 4):
                node = path_nodes[-1]
                outs = sorted(d for d in DIGITS[:k] if (node + d) in remaining)
                if not outs:
                    break
                c = rng.choice(outs)
                w = node + c
                nxt = w[1:]
                path_edges.append(w)
                if nxt in pos:
                    cyc = path_edges[pos[nxt]:]
                    F |= set(cyc)
                    remaining -= set(cyc)
                    found_cycle = True
                    break
                pos[nxt] = len(path_nodes)
                path_nodes.append(nxt)
            if not found_cycle:
                ok = False; break
        if not ok:
            continue
        allowed = fullset - F
        if allowed and strongly_connected(allowed):
            return sorted(F)
    return []  # fallback: no forbidden windows at all (still a valid, if unchallenging, instance)


def gen_trap(k, L, rng, frac_lo, frac_hi, min_aug_frac):
    full = full_edges(k, L)
    fullset = set(full)
    n = len(full)
    best_F, best_aug = None, -1
    for attempt in range(250):
        frac = frac_lo + (frac_hi - frac_lo) * rng.random()
        m = max(1, min(n - 1, int(round(frac * n))))
        F = set(rng.sample(full, m))
        allowed = fullset - F
        if not allowed or not strongly_connected(allowed):
            continue
        aug = greedy_augment_cost(allowed)
        if aug < 0:
            continue
        if aug > best_aug:
            best_aug, best_F = aug, F
        if aug >= min_aug_frac * len(allowed):
            return sorted(F)
    return sorted(best_F) if best_F is not None else []


# testId -> (k, L, flavour, params)
PLAN = {
    1: (2, 2, "easy", dict(n_cycles=0)),
    2: (2, 3, "easy", dict(n_cycles=1)),
    3: (3, 2, "easy", dict(n_cycles=2)),
    4: (2, 4, "easy", dict(n_cycles=2)),
    5: (3, 3, "trap", dict(frac_lo=0.18, frac_hi=0.30, min_aug_frac=0.12)),
    6: (2, 5, "trap", dict(frac_lo=0.15, frac_hi=0.25, min_aug_frac=0.10)),
    7: (4, 2, "trap", dict(frac_lo=0.20, frac_hi=0.35, min_aug_frac=0.12)),
    8: (3, 4, "trap", dict(frac_lo=0.18, frac_hi=0.28, min_aug_frac=0.10)),
    9: (4, 3, "trap", dict(frac_lo=0.18, frac_hi=0.28, min_aug_frac=0.10)),
    10: (4, 5, "trap", dict(frac_lo=0.14, frac_hi=0.22, min_aug_frac=0.08)),
}


def main():
    tid = int(sys.argv[1])
    if tid not in PLAN:
        tid = ((tid - 1) % 10) + 1
    k, L, flavour, params = PLAN[tid]
    rng = random.Random(tid * 1_000_003 + 17)
    if flavour == "easy":
        if params["n_cycles"] == 0:
            F = []
        else:
            F = gen_easy(k, L, rng, params["n_cycles"])
    else:
        F = gen_trap(k, L, rng, params["frac_lo"], params["frac_hi"], params["min_aug_frac"])
    out = [f"{k} {L} {len(F)}"]
    out.extend(F)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
