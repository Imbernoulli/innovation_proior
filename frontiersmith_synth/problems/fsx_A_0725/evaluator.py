#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0725 -- "The Frontier Ring: Quarantine on a Contact Chain"
(family: diffusion-frontier-intervention; format B, quality-metric, objective=minimize).

THEME.  A contact network is built from a CHAIN of dense clusters (workplaces / villages)
plus one large, densely-connected DECOY cluster hanging off the far end of the chain.
Every cluster has a local "hub" node connected (via a star) to every other node in its
cluster, so once a cluster's hub is infected the whole cluster is exposed almost at once.
Consecutive clusters in the chain are linked by only a handful of "gate" edges (a narrow
bridge); a few seeded long-range "shortcut" edges also connect distant clusters directly
(a small-world contact network, not a pure chain). An outbreak (deterministic SIR: a node
is infectious for exactly D rounds after infection, then recovers and can no longer be
infected or transmit) starts at the hub of the FIRST cluster and spreads along edges.

Each round the program may IMMUNIZE up to `rate_cap` currently-susceptible nodes (a node
already infected or recovered cannot be immunized -- the attempt is simply wasted), and the
program has a fixed TOTAL budget `total_budget` (strictly less than rate_cap*T) that must be
REALLOCATED across the T rounds: it cannot spend the maximum every round, so it must decide
WHEN as well as WHERE to intervene as the outbreak's frontier moves through the chain.

INNOVATION HOOK.  The obvious plan is to rank nodes ONCE by (static) degree in the whole
graph and immunize the top-ranked unimmunized nodes every round in that fixed order. This
is a trap: the graph's single highest-degree hub sits in the DECOY cluster, which the
outbreak reaches late (if ever) via the far end of the chain, and the first cluster's own
hub is itself the outbreak's origin -- already infected before any intervention can act on
it. A static top-degree list therefore burns budget on nodes that are irrelevant (the decoy)
or already lost (the origin hub) while the real transmission frontier -- the handful of gate
nodes the outbreak is ABOUT to reach next -- goes unprotected and the chain falls cluster by
cluster. The insight that pays is to track the CURRENT frontier (susceptible nodes adjacent
to an infectious node) round by round and immunize the ring just ahead of it, cutting
transmission paths before they are traversed rather than protecting the already-exposed core.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "N": int, "edges": [[u,v], ...], "seeds": [node ids],
             "T": int, "D": int, "rate_cap": int, "total_budget": int}
  stdout: ONE JSON object:
            {"schedule": [[ids immunized at round 0], [round 1], ..., [round T-1]]}

  A schedule is VALID iff it is a list of exactly T lists; every entry of every round-list
  is an integer node id in [0,N); no round-list contains a duplicate id; every round-list
  has length <= rate_cap; and the TOTAL number of entries across all rounds is
  <= total_budget. Any violation (wrong shape, non-integer, out of range, in-round
  duplicate, per-round or total budget overrun, a crash, a timeout, or non-JSON output)
  scores that instance 0.0. Immunizing a node that turns out to already be infected or
  recovered by the time its round is reached is legal but simply has no effect (wasted
  budget), matching real quarantine logistics.

SIMULATION (deterministic, run entirely inside the evaluator; no wall-time / RNG at
scoring time). States: S(usceptible) -> I(nfectious) -> R(ecovered), or S -> V(accinated).
Per round t = 0..T-1, synchronously:
  1. Recovery: any I node infected at round t0 with t0 + D <= t becomes R.
  2. Immunization: for each id in schedule[t], if it is currently S it becomes V.
  3. Transmission: every S node adjacent to at least one currently-I node becomes I
     (infect_round = t); newly infected nodes remain infectious for D rounds starting now.
The seed nodes start the simulation already I with infect_round = 0. The objective is the
TOTAL number of nodes ever infected (|{v : infect_round[v] != -1}|) -- lower is better.

SCORING.  Per instance the evaluator computes:
    noint = total ever infected if the program does nothing at all (empty schedule) --
            a weak reference, mapped to ~0.1,
    ub    = number of seed nodes -- the value if the outbreak were contained the instant
            it started (an idealised, generally unreachable bound given the budget),
    cand  = total ever infected achieved by the candidate's schedule,
and normalises with an affine anchor:
    r = clamp( 0.1 + 0.9 * (noint - cand) / (noint - ub), 0, 1 )
Doing nothing scores ~0.1; reaching the (essentially unreachable) instant-containment bound
scores 1.0; the mean of r over all instances is the final Ratio.

ISOLATION.  The candidate runs OS-sandboxed in a fresh subprocess via `isorun.run_candidate`
and only ever sees the public instance; the seeds/edges are already public (the candidate
needs them to plan), but the noint/ub references and all validity/simulation logic live only
in this (parent) evaluator process, so a frame-walking candidate learns nothing extra.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt_int


# ----------------------------- graph construction ---------------------------
def _build_cluster_edges(ni, size, p_pct):
    """Local edges (0-indexed within the cluster). Node 0 is the hub: connected to
    every other node (star), plus extra random mesh edges among the rest at p_pct%."""
    edges = set()
    for i in range(1, size):
        edges.add((0, i))
    for i in range(1, size):
        for j in range(i + 1, size):
            if ni(1, 100) <= p_pct:
                edges.add((i, j))
    return edges


def _gate(off, size, side, k):
    """k-th gate node on the 'hi' (top, excludes hub) or 'lo' (bottom, excludes hub)
    side of a cluster occupying global indices [off, off+size)."""
    if side == "hi":
        return off + size - 1 - k
    return off + 1 + k


def _build_graph(seed, n_chain, sizes, decoy_size, decoy_p, chain_p, bridge_redundancy,
                  n_shortcuts):
    ni = _rng(seed)
    clusters = []      # (offset, size) per cluster, chain first then decoy last
    edges = []
    offset = 0
    for sz in sizes:
        for (a, b) in _build_cluster_edges(ni, sz, chain_p):
            edges.append((a + offset, b + offset))
        clusters.append((offset, sz))
        offset += sz
    # narrow bridges between consecutive chain clusters
    for i in range(n_chain - 1):
        off_a, sz_a = clusters[i]
        off_b, sz_b = clusters[i + 1]
        for k in range(bridge_redundancy):
            edges.append((_gate(off_a, sz_a, "hi", k), _gate(off_b, sz_b, "lo", k)))
    # decoy cluster: bigger & denser, attached only at the FAR end of the chain
    for (a, b) in _build_cluster_edges(ni, decoy_size, decoy_p):
        edges.append((a + offset, b + offset))
    decoy_off = offset
    clusters.append((decoy_off, decoy_size))
    off_last, sz_last = clusters[n_chain - 1]
    for k in range(bridge_redundancy):
        edges.append((_gate(off_last, sz_last, "hi", k + bridge_redundancy),
                      _gate(decoy_off, decoy_size, "lo", k)))
    N = decoy_off + decoy_size
    # a handful of seeded long-range shortcuts (small-world contact links)
    for _ in range(n_shortcuts):
        ci = ni(0, len(clusters) - 1)
        cj = ni(0, len(clusters) - 1)
        if ci == cj:
            continue
        off_i, sz_i = clusters[ci]
        off_j, sz_j = clusters[cj]
        a = off_i + ni(0, sz_i - 1)
        b = off_j + ni(0, sz_j - 1)
        if a != b:
            edges.append((a, b))
    # dedupe
    edges = sorted(set((min(a, b), max(a, b)) for (a, b) in edges if a != b))
    seeds = [clusters[0][0], clusters[0][0] + min(3, sizes[0] - 1)]
    return {"N": N, "edges": edges, "seeds": sorted(set(seeds))}


def _build_instances():
    """Deterministic instance family: (seed, n_chain, sizes, decoy_size, decoy_p, chain_p,
    T, D, rate_cap, total_budget, bridge_redundancy, n_shortcuts)."""
    specs = [
        (4001, 5, [20, 22, 21, 23, 22], 30, 55, 25, 16, 2, 3, 16, 4, 5),
        (4002, 5, [19, 21, 20, 22, 21], 28, 55, 27, 16, 2, 3, 16, 4, 6),
        (4003, 5, [21, 23, 22, 24, 23], 32, 52, 24, 17, 2, 3, 17, 4, 7),
        (4004, 4, [22, 24, 23, 25], 32, 55, 26, 14, 2, 3, 14, 4, 5),
        (4005, 6, [18, 20, 19, 21, 20, 19], 28, 55, 24, 18, 2, 3, 18, 4, 6),
        (4006, 5, [20, 22, 21, 23, 22], 30, 55, 25, 16, 3, 3, 16, 4, 5),
        (4007, 5, [20, 22, 21, 23, 22], 30, 55, 25, 16, 2, 4, 20, 4, 6),
        (4008, 5, [19, 21, 20, 22, 21], 28, 58, 25, 16, 2, 3, 15, 4, 7),
        # harder / larger held-out instances (generalization)
        (4101, 6, [24, 26, 25, 27, 26, 25], 36, 55, 24, 20, 2, 4, 20, 4, 8),
        (4102, 6, [25, 27, 26, 28, 27, 26], 38, 55, 23, 20, 2, 4, 22, 4, 9),
    ]
    out = []
    for (seed, n_chain, sizes, decoy_size, decoy_p, chain_p, T, D, rate_cap, total_budget,
         bridge_redundancy, n_shortcuts) in specs:
        g = _build_graph(seed, n_chain, sizes, decoy_size, decoy_p, chain_p,
                          bridge_redundancy, n_shortcuts)
        inst = {
            "name": f"outbreak{seed}", "N": g["N"], "edges": g["edges"], "seeds": g["seeds"],
            "T": T, "D": D, "rate_cap": rate_cap, "total_budget": total_budget,
        }
        out.append(inst)
    return out


# ----------------------------- simulation -----------------------------------
def _adjacency(inst):
    adj = [[] for _ in range(inst["N"])]
    for a, b in inst["edges"]:
        adj[a].append(b)
        adj[b].append(a)
    return adj


def _simulate(inst, adj, schedule):
    N, T, D = inst["N"], inst["T"], inst["D"]
    state = [0] * N          # 0=S 1=I 2=R 3=V
    infect_round = [-1] * N
    for s in inst["seeds"]:
        state[s] = 1
        infect_round[s] = 0
    for t in range(T):
        for i in range(N):
            if state[i] == 1 and infect_round[i] + D <= t:
                state[i] = 2
        for nid in schedule[t]:
            if state[nid] == 0:
                state[nid] = 3
        infset = set(i for i in range(N) if state[i] == 1)
        newly = []
        for u in range(N):
            if state[u] == 0:
                for v in adj[u]:
                    if v in infset:
                        newly.append(u)
                        break
        for u in newly:
            state[u] = 1
            infect_round[u] = t
    return sum(1 for i in range(N) if infect_round[i] != -1)


def _validate_schedule(inst, answer):
    """Return the validated schedule (list of T lists of ints) or None if infeasible."""
    if not isinstance(answer, dict):
        return None
    sched = answer.get("schedule")
    if not isinstance(sched, list) or len(sched) != inst["T"]:
        return None
    N = inst["N"]
    rate_cap = inst["rate_cap"]
    total_budget = inst["total_budget"]
    out = []
    total = 0
    for round_list in sched:
        if not isinstance(round_list, list) or len(round_list) > rate_cap:
            return None
        seen_this_round = set()
        cleaned = []
        for x in round_list:
            if isinstance(x, bool) or not isinstance(x, int):
                return None
            if x < 0 or x >= N:
                return None
            if x in seen_this_round:
                return None
            seen_this_round.add(x)
            cleaned.append(x)
        out.append(cleaned)
        total += len(cleaned)
        if total > total_budget:
            return None
    return out


# ----------------------------- scoring driver --------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        adj = _adjacency(inst)
        noint = _simulate(inst, adj, [[] for _ in range(inst["T"])])
        ub = len(inst["seeds"])
        denom = max(noint - ub, 1e-9)
        public = {"name": inst["name"], "N": inst["N"],
                  "edges": [list(e) for e in inst["edges"]],
                  "seeds": list(inst["seeds"]), "T": inst["T"], "D": inst["D"],
                  "rate_cap": inst["rate_cap"], "total_budget": inst["total_budget"]}
        ans, st = isorun.run_candidate(cand, public, timeout=10)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            sched = _validate_schedule(inst, ans)
        except Exception:
            sched = None
        if sched is None:
            vec.append(0.0)
            continue
        try:
            cand_inf = _simulate(inst, adj, sched)
        except Exception:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (noint - cand_inf) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        r = max(0.0, min(1.0, r))
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
