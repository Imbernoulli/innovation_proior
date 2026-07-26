# TIER: strong
"""The insight: split the pipes into BRIDGE edges (whose removal disconnects
the topology -- found by a standard O(n+m) DFS bridge-finding pass on the
topology alone, ignoring caps) and INTERIOR edges (everything else, which sit
inside 2-edge-connected clusters). This exploits the structural lever hidden
in the single weight vector:

  * short-time targets (t -> 0) are dominated by total edge weight / the
    degree sequence, which is overwhelmingly determined by the many interior
    edges -> fit ONE shared interior scale against the early part of the
    grid.
  * long-time targets (t -> infinity) are dominated by the spectral gap,
    which -- because bridges are the network's sparsest cut -- is set almost
    entirely by the few bridge weights -> exhaustively search the (small)
    integer combinations of just the bridge weights against the late part of
    the grid.

Because the two blocks barely interact with each other's regime, this
2-block search (a handful of numbers total, rather than one uniform
knob over all edges) reaches a far better joint fit than any single global
scale can, then a couple of alternating refinement passes clean up residual
cross-talk."""
import sys
import math
import numpy as np


def read_instance():
    toks = sys.stdin.read().split()
    p = 0
    n = int(toks[p]); p += 1
    m = int(toks[p]); p += 1
    T = int(toks[p]); p += 1
    edges = []
    for _ in range(m):
        u = int(toks[p]); v = int(toks[p + 1]); cap = int(toks[p + 2]); p += 3
        edges.append((u, v, cap))
    ts = [float(toks[p + j]) for j in range(T)]; p += T
    targets = [float(toks[p + j]) for j in range(T)]; p += T
    return n, edges, ts, targets


def find_bridges(n, edges):
    adj = [[] for _ in range(n)]
    for idx, (u, v, cap) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
    disc = [-1] * n
    low = [0] * n
    timer = [0]
    bridges = set()
    visited = [False] * n

    # iterative DFS to avoid recursion-depth issues on larger n
    for s in range(n):
        if visited[s]:
            continue
        stack = [(s, -1, iter(adj[s]))]
        visited[s] = True
        disc[s] = low[s] = timer[0]
        timer[0] += 1
        while stack:
            u, parent_edge, it = stack[-1]
            advanced = False
            for v, idx in it:
                if idx == parent_edge:
                    continue
                if visited[v]:
                    low[u] = min(low[u], disc[v])
                else:
                    visited[v] = True
                    disc[v] = low[v] = timer[0]
                    timer[0] += 1
                    stack.append((v, idx, iter(adj[v])))
                    advanced = True
                    break
            if not advanced:
                stack.pop()
                if stack:
                    pu, pidx, _ = stack[-1]
                    low[pu] = min(low[pu], low[u])
                    if low[u] > disc[pu]:
                        bridges.add(parent_edge)
    return bridges


def trace_values(n, edges, weights, ts):
    L = np.zeros((n, n))
    for (u, v, cap), w in zip(edges, weights):
        L[u, u] += w
        L[v, v] += w
        L[u, v] -= w
        L[v, u] -= w
    eigs = np.linalg.eigvalsh(L)
    return [float(np.sum(np.exp(-t * eigs))) for t in ts]


def log_err_sub(n, edges, weights, ts, targets, idxs):
    tv = trace_values(n, edges, weights, ts)
    return max(abs(math.log(max(tv[i], 1e-12)) - math.log(max(targets[i], 1e-12)))
               for i in idxs)


def full_err(n, edges, weights, ts, targets):
    return log_err_sub(n, edges, weights, ts, targets, range(len(ts)))


def main():
    n, edges, ts, targets = read_instance()
    m = len(edges)
    T = len(ts)

    bridges = find_bridges(n, edges)
    interior_idx = [i for i in range(m) if i not in bridges]
    bridge_idx = [i for i in range(m) if i in bridges]

    early = list(range(0, max(1, T // 2)))
    late = list(range(T // 2, T))

    def build(beta_int, cut_weights):
        w = [0] * m
        for i in interior_idx:
            cap = edges[i][2]
            w[i] = max(1, min(cap, round(beta_int * cap)))
        for i, cw in zip(bridge_idx, cut_weights):
            cap = edges[i][2]
            w[i] = max(1, min(cap, cw))
        return w

    default_cut = [max(1, edges[i][2] // 2) for i in bridge_idx]

    # stage 1: fit a single interior scale against the SHORT-TIME targets only
    best_beta, best_err = 0.5, float("inf")
    for bi in range(0, 101, 2):
        beta = bi / 100.0
        w = build(beta, default_cut)
        e = log_err_sub(n, edges, w, ts, targets, early)
        if e < best_err:
            best_err, best_beta = e, beta

    # stage 2: exhaustively search the (few) bridge weights against LONG-TIME
    # targets, with the interior scale fixed
    caps = [edges[i][2] for i in bridge_idx]
    best_cut, best_err2 = default_cut, float("inf")

    def rec(pos, cur):
        nonlocal best_cut, best_err2
        if pos == len(caps):
            w = build(best_beta, cur)
            e = log_err_sub(n, edges, w, ts, targets, late)
            if e < best_err2:
                best_err2, best_cut = e, list(cur)
            return
        for val in range(1, caps[pos] + 1):
            cur.append(val)
            rec(pos + 1, cur)
            cur.pop()

    rec(0, [])

    # stage 3: refine the interior scale against the FULL grid, cut fixed
    best_beta2, best_err3 = best_beta, float("inf")
    for bi in range(0, 101):
        beta = bi / 100.0
        w = build(beta, best_cut)
        e = full_err(n, edges, w, ts, targets)
        if e < best_err3:
            best_err3, best_beta2 = e, beta

    # stage 4: one more bridge-weight pass against the FULL grid, interior fixed
    best_cut2, best_err4 = best_cut, float("inf")

    def rec2(pos, cur):
        nonlocal best_cut2, best_err4
        if pos == len(caps):
            w = build(best_beta2, cur)
            e = full_err(n, edges, w, ts, targets)
            if e < best_err4:
                best_err4, best_cut2 = e, list(cur)
            return
        for val in range(1, caps[pos] + 1):
            cur.append(val)
            rec2(pos + 1, cur)
            cur.pop()

    rec2(0, [])

    w_final = build(best_beta2, best_cut2)
    sys.stdout.write(" ".join(str(x) for x in w_final) + "\n")


if __name__ == "__main__":
    main()
