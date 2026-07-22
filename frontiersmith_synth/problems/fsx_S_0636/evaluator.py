#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_S_0636 -- "Spill-Aware Register Coloring on a Budget"
(family: spill-aware-conflict-coloring; format B, quality-metric).

THEME. An interference graph G=(V,E) has n nodes (temporaries / values that are
simultaneously live at some program point) and an edge between any two nodes that
must NOT share a physical register (they interfere). Only k physical registers
(colors 0..k-1) are available. Each node v carries a `weight[v] >= 1`: the runtime
cost paid if v is left uncolored and must be SPILLED to memory instead. A coloring
assigns each node either a color in [0,k) or leaves it spilled (-1); it is PROPER
iff no edge joins two nodes holding the SAME color (spilled nodes never conflict).
The allocator's job is to choose which nodes to spill (if any are forced) so as to
MINIMIZE the total spill weight of the uncolored nodes.

This composes three mechanisms into one objective:
  - saturation-degree-order: the natural construction order for a coloring pass is
    to always color next whichever uncolored node currently has the most DISTINCT
    colors already forced among its colored neighbors (highest "saturation") --
    the classic DSATUR register-allocation heuristic.
  - chromatic-lower-bound-probe: any k+1 mutually-interfering nodes (a clique) can
    never all get distinct colors from only k of them, so at least one must spill;
    more generally, for ANY clique C found in the graph, at least |C|-k of its
    members must spill, and the cheapest such forced subset (in isolation) is a
    valid LOWER BOUND on the true minimum spill weight. The evaluator grows greedy
    cliques from every seed node to compute this probe bound.
  - kempe-chain-move: once an initial coloring is built, a node that ended up
    spilled can sometimes still be rescued: if a neighbor holding color c can
    itself be shifted to a different color (possibly by flipping colors along a
    connected chain of nodes alternating between two colors -- a Kempe chain,
    which is always safe: flipping BOTH colors on a maximal alternating component
    never creates a new conflict), color c becomes free at that neighbor without
    breaking anyone else, and the spilled node can take it.

TRAP. A plain saturation-degree pass is completely BLIND to weight: it colors
whichever nodes reach maximum saturation first, with no regard for how expensive
it would be to leave them spilled. The generator plants (a) tight cliques of size
k+1 where one member is far more valuable than the rest, and (b) a cheap, densely
interconnected "core" cluster of exactly k mutually-interfering low-weight nodes
that a blind pass fills first, exhausting all k colors before it ever reaches one
or more high-weight "hub" nodes that interfere with the entire core -- forcing the
expensive hub nodes to spill when a cheap core member could have been sacrificed
instead (or, since hub nodes never interfere with each other, ALL of them could
share one color freed by evicting a single cheap core member).

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "n": int, "k": int,
             "weights": [int]*n,          # spill weight per node, >= 1
             "edges": [[u,v], ...]}       # undirected interference edges, u<v
  stdout: ONE JSON object:
            {"colors": [c_0, ..., c_{n-1}]}   # c_i in {-1, 0, 1, ..., k-1}
          -1 means node i is spilled. A coloring is VALID iff `colors` has exactly
          n integer entries each in [-1, k-1] and no edge (u,v) has
          colors[u] == colors[v] != -1. Wrong length/type/range, a conflicting
          edge, a crash, a timeout, or non-JSON -> that instance scores 0.0.

SCORING (deterministic; no wall-time). Per instance:
    S_weak = spill weight of a fixed WEAK baseline: color nodes in raw index
             order 0..n-1, each taking the smallest color not yet used by an
             already-colored neighbor, or spilling if none is free. Entirely
             weight-blind.
    LB     = the chromatic-lower-bound probe: grow a greedy clique from every
             node, and for each clique C with |C| > k take
             sum of the (|C|-k) SMALLEST weights in C; LB is the max over all
             probed cliques (0 if none exceeds k). LB is a valid but generally
             LOOSE lower bound on the true minimum spill weight.
    S_cand = spill weight of the candidate's (validated) coloring.
  normalized:
    r = clamp( 0.1 + 0.85 * (S_weak - S_cand) / max(1e-9, S_weak - LB), 0, 1 )
  Reproducing the weak baseline scores ~0.1; a smaller spill than the baseline
  scores higher (up to 0.95 if it matches the lower-bound probe exactly, which is
  usually unreachable, leaving headroom); a larger spill scores lower, floored at
  0. The final score is the mean of r over all instances (easy sanity cases,
  generic medium-density cases, and several trap cases of both kinds above).

ISOLATION. The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance. Every reference
(weak baseline, lower-bound probe) and all validation happen in THIS parent
process.

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


# ----------------------------- instance builders ----------------------------
def _random_instance(seed, name, n, k, edge_permille, wlo, whi):
    ni = _rng(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if ni(1, 1000) <= edge_permille:
                edges.append([u, v])
    weights = [ni(wlo, whi) for _ in range(n)]
    return {"name": name, "n": n, "k": k, "weights": weights, "edges": edges}


def _pure_clique_instance(seed, name, k, wlo_hi=(1, 4), whigh=300):
    ni = _rng(seed)
    n = k + 1  # clique of size k+1: exactly one forced spill
    weights = [ni(wlo_hi[0], wlo_hi[1]) for _ in range(k)]
    weights.append(whigh)  # index k = H, the expensive node, LAST index on purpose
    edges = [[i, j] for i in range(n) for j in range(i + 1, n)]
    return {"name": name, "n": n, "k": k, "weights": weights, "edges": edges}


def _hub_instance(seed, name, k, num_camo, num_victims, v_wrange, c_wrange):
    ni = _rng(seed)
    core = list(range(k))
    camo = list(range(k, k + num_camo))
    victims = list(range(k + num_camo, k + num_camo + num_victims))
    n = k + num_camo + num_victims
    weights = [0] * n
    for i in core:
        weights[i] = ni(c_wrange[0], c_wrange[1])
    for i in camo:
        weights[i] = ni(c_wrange[0], c_wrange[1])
    for i in victims:
        weights[i] = ni(v_wrange[0], v_wrange[1])
    edges = []
    for i in range(len(core)):
        for j in range(i + 1, len(core)):
            edges.append([core[i], core[j]])
    for c in camo:
        picks = set()
        tries = 0
        while len(picks) < min(2, len(core)) and tries < 50:
            picks.add(ni(0, len(core) - 1))
            tries += 1
        for ci in picks:
            u, v = min(c, core[ci]), max(c, core[ci])
            edges.append([u, v])
    for v in victims:            # full hub: victim interferes with the ENTIRE core
        for ci in core:
            edges.append([ci, v])
    return {"name": name, "n": n, "k": k, "weights": weights, "edges": edges}


def _build_instances():
    out = []
    # easy sanity cases: sparse, generous k -> little/no spill needed
    out.append(_random_instance(1101, "easy_sparse_a", 12, 5, 180, 1, 10))
    out.append(_random_instance(1102, "easy_sparse_b", 14, 6, 150, 1, 10))
    # generic medium-density cases: order quality matters, no planted trap
    out.append(_random_instance(1201, "generic_mid_a", 18, 4, 300, 1, 50))
    out.append(_random_instance(1202, "generic_mid_b", 20, 4, 330, 1, 60))
    out.append(_random_instance(1203, "generic_mid_c", 24, 5, 320, 1, 80))  # harder / held-out
    # trap A: tight (k+1)-clique with one very expensive member at the last index
    out.append(_pure_clique_instance(1301, "trap_clique_a", 5, (1, 4), 300))
    out.append(_pure_clique_instance(1302, "trap_clique_b", 6, (1, 4), 400))
    # trap B: cheap dense k-core "fills first", expensive hub node(s) get stranded
    out.append(_hub_instance(1401, "trap_hub_a", 4, 3, 2, (150, 250), (1, 5)))
    out.append(_hub_instance(1402, "trap_hub_b", 5, 4, 3, (150, 300), (1, 5)))
    out.append(_hub_instance(1403, "trap_hub_c", 6, 5, 3, (200, 350), (1, 5)))  # harder / held-out
    return out


# ----------------------------- references / scoring -------------------------
def _adj_sets(n, edges):
    adj = [set() for _ in range(n)]
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _weak_baseline_spill(n, k, weights, adj):
    colors = [-1] * n
    for u in range(n):
        used = set(colors[w] for w in adj[u] if colors[w] != -1)
        c = None
        for cand in range(k):
            if cand not in used:
                c = cand
                break
        colors[u] = c if c is not None else -1
    return sum(weights[i] for i in range(n) if colors[i] == -1)


def _clique_lower_bound(n, adj, weights, k):
    best = 0
    for s in range(n):
        clique = [s]
        candidates = set(adj[s])
        while candidates:
            pick = None
            pick_score = -1
            for v in sorted(candidates):
                sc = len(candidates & adj[v])
                if sc > pick_score:
                    pick_score = sc
                    pick = v
            clique.append(pick)
            candidates &= adj[pick]
        if len(clique) > k:
            ws = sorted(weights[v] for v in clique)
            val = sum(ws[: len(clique) - k])
            if val > best:
                best = val
    return best


def _validate(inst, answer):
    if not isinstance(answer, dict):
        return None
    colors = answer.get("colors")
    n = inst["n"]; k = inst["k"]
    if not isinstance(colors, list) or len(colors) != n:
        return None
    out = []
    for c in colors:
        if isinstance(c, bool) or not isinstance(c, int):
            return None
        if c < -1 or c >= k:
            return None
        out.append(c)
    for (u, v) in inst["edges"]:
        if out[u] == out[v] and out[u] != -1:
            return None
    return out


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        n = inst["n"]; k = inst["k"]; weights = inst["weights"]; edges = inst["edges"]
        adj = _adj_sets(n, edges)
        s_weak = _weak_baseline_spill(n, k, weights, adj)
        lb = _clique_lower_bound(n, adj, weights, k)
        denom = s_weak - lb
        if denom < 1e-9:
            denom = 1e-9
        public = {"name": inst["name"], "n": n, "k": k,
                  "weights": list(weights), "edges": [list(e) for e in edges]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            colors = _validate(inst, ans)
        except Exception:
            colors = None
        if colors is None:
            vec.append(0.0)
            continue
        s_cand = sum(weights[i] for i in range(n) if colors[i] == -1)
        r = 0.1 + 0.85 * (s_weak - s_cand) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
