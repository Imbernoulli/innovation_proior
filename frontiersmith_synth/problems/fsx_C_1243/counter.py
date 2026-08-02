import sys
from collections import deque

# Format D checker -- Qubit-Routing plan verifier + op-count scorer.
#
# Input <in>:  n m e ; e edge lines "u v" (coupling graph, physical qubits
#              0..n-1) ; m gate lines "t a b" (t in {0,1} gate type, a,b
#              logical qubit ids, program order = input order, 1-indexed).
# Output <out>: n ; a permutation perm[0..n-1] (initial mapping: logical i
#              starts on physical perm[i]) ; T ; T operation lines, each
#              "S p q" (SWAP adjacent physical qubits p,q) or "G i" (execute
#              gate i, 1-indexed, referencing the INPUT's own (t,a,b) -- the
#              solver cannot forge a gate's type/qubits, only choose whether
#              and when to run it).
#
# A gate index may be OMITTED from the "G" stream only if it is part of a
# planted cancelling pair (see `cancel_partner`, computed from the gate list
# alone): two gates with the same (type, unordered qubit pair) cancel to the
# identity if nothing between them (in program order) touches either qubit,
# regardless of any physical SWAPs performed meanwhile. Both members of a
# pair must be omitted together.
#
# Objective (minimize): F = 3*num_swaps + 1*num_gates_executed (a SWAP costs
# 3 elementary two-qubit operations; a native 2-qubit gate costs 1).
# Baseline B = an internal "detour via physical qubit 0" trivial construction
# (identity mapping, no cancellation, route every gate via qubit 0).
# Ratio = min(1, 0.1 * B / F).

MAXT = 20000


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def bfs_path(n, adj, src, dst):
    if src == dst:
        return [src]
    parent = [-2] * n
    parent[src] = -1
    dq = deque([src])
    while dq:
        u = dq.popleft()
        if u == dst:
            break
        for v in adj[u]:
            if parent[v] == -2:
                parent[v] = u
                dq.append(v)
    if parent[dst] == -2:
        return None  # disconnected (shouldn't happen -- graph is connected)
    path = [dst]
    cur = dst
    while parent[cur] != -1:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path


def compute_cancel_partner(gates):
    """Deterministic single pass: gate i cancels with the most recent open
    gate j of the same (type, unordered pair) iff neither of its two qubits
    was touched (by a still-standing gate) since j. Returns dict idx->idx
    (1-indexed), symmetric, each idx appears at most once."""
    m = len(gates)
    last_touch = {}       # qubit -> last surviving gate index touching it
    open_pending = {}      # (pair, type) -> open gate index
    partner = {}
    for i in range(1, m + 1):
        t, a, b = gates[i - 1]
        pair = (a, b) if a < b else (b, a)
        key = (pair, t)
        j = open_pending.get(key)
        if j is not None and last_touch.get(a, -1) <= j and last_touch.get(b, -1) <= j:
            partner[i] = j
            partner[j] = i
            del open_pending[key]
            # j,i together net to identity -- do NOT mark a,b as "touched"
        else:
            open_pending[key] = i
            last_touch[a] = i
            last_touch[b] = i
    return partner


def route_cost_true_shortest(n, adj, pos, at, a, b):
    """Mutates pos/at; returns list of swap (p,q) pairs performed to bring
    logical a,b adjacent via the true shortest path."""
    p, q = pos[a], pos[b]
    if q in adj[p]:
        return []
    path = bfs_path(n, adj, p, q)
    swaps = []
    for k in range(len(path) - 2):
        u, v = path[k], path[k + 1]
        lu, lv = at[u], at[v]
        at[u], at[v] = lv, lu
        pos[lu], pos[lv] = v, u
        swaps.append((u, v))
    return swaps


def graph_center(n, adj):
    """Vertex minimizing eccentricity (max distance to any other vertex),
    tie-broken by smallest id -- a deterministic, structure-blind 'hub'."""
    best_v, best_ecc = 0, None
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        dq = deque([s])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    dq.append(v)
        ecc = max(dist)
        if best_ecc is None or ecc < best_ecc:
            best_ecc = ecc
            best_v = s
    return best_v


def route_cost_via_anchor(n, adj, pos, at, a, b, anchor=0):
    """Trivial/baseline routing: always detour the moving qubit through the
    fixed anchor physical qubit `anchor`."""
    swaps = []
    p = pos[a]
    if p != anchor:
        path1 = bfs_path(n, adj, p, anchor)
        for k in range(len(path1) - 1):
            u, v = path1[k], path1[k + 1]
            lu, lv = at[u], at[v]
            at[u], at[v] = lv, lu
            pos[lu], pos[lv] = v, u
            swaps.append((u, v))
    q = pos[b]
    if q in adj[anchor]:
        return swaps
    path2 = bfs_path(n, adj, anchor, q)
    for k in range(len(path2) - 2):
        u, v = path2[k], path2[k + 1]
        lu, lv = at[u], at[v]
        at[u], at[v] = lv, lu
        pos[lu], pos[lv] = v, u
        swaps.append((u, v))
    return swaps


def compute_baseline(n, edges, gates):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    pos = list(range(n))   # identity mapping
    at = list(range(n))
    anchor = graph_center(n, adj)
    total = 0
    for (t, a, b) in gates:
        swaps = route_cost_via_anchor(n, adj, pos, at, a, b, anchor=anchor)
        total += 3 * len(swaps) + 1
    return total


def main():
    try:
        inp = open(sys.argv[1]).read().split()
    except Exception:
        fail("cannot read input")
    try:
        out = open(sys.argv[2]).read().split()
    except Exception:
        fail("cannot read output")

    it = iter(inp)

    def nxt_int():
        return int(next(it))

    try:
        n = nxt_int(); m = nxt_int(); e = nxt_int()
    except Exception:
        fail("bad header")
    if not (2 <= n <= 500 and 1 <= m <= 5000 and 0 <= e <= 5000):
        fail("bad dims")

    try:
        edges = []
        adj = [set() for _ in range(n)]
        for _ in range(e):
            u = nxt_int(); v = nxt_int()
            if not (0 <= u < n and 0 <= v < n) or u == v:
                fail("bad edge")
            edges.append((u, v))
            adj[u].add(v)
            adj[v].add(u)
        gates = []
        for _ in range(m):
            t = nxt_int(); a = nxt_int(); b = nxt_int()
            if t not in (0, 1) or not (0 <= a < n and 0 <= b < n) or a == b:
                fail("bad gate")
            gates.append((t, a, b))
    except Exception:
        fail("bad instance body")

    try:
        next(it)
        fail("trailing input tokens")
    except StopIteration:
        pass

    # connectivity sanity (own generator guard)
    visited = [False] * n
    dq = deque([0]); visited[0] = True; cnt = 1
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True; cnt += 1; dq.append(v)
    if cnt != n:
        fail("coupling graph disconnected (bad instance)")

    partner = compute_cancel_partner(gates)
    B = compute_baseline(n, edges, gates)

    # ---- parse participant artifact ----
    if not out:
        fail("empty output")
    pos_tok = 0

    def take():
        nonlocal pos_tok
        if pos_tok >= len(out):
            raise IndexError
        v = out[pos_tok]
        pos_tok += 1
        return v

    try:
        n_out = int(take())
    except Exception:
        fail("bad n echo")
    if n_out != n:
        fail("n mismatch")

    try:
        mapping = [int(take()) for _ in range(n)]
    except Exception:
        fail("bad mapping tokens")
    if sorted(mapping) != list(range(n)):
        fail("initial mapping is not a permutation of 0..n-1")

    try:
        T = int(take())
    except Exception:
        fail("bad op count")
    if T < 0 or T > MAXT:
        fail("op count out of range")

    pos = mapping[:]           # pos[logical] = physical
    at = [0] * n                # at[physical] = logical
    for logical, phys in enumerate(pos):
        at[phys] = logical

    num_swaps = 0
    num_gates = 0
    last_g = 0
    executed = set()

    try:
        for _ in range(T):
            typ = take()
            if typ == "S":
                p = int(take()); q = int(take())
                if not (0 <= p < n and 0 <= q < n) or p == q:
                    fail("SWAP: bad qubit id")
                if q not in adj[p]:
                    fail("SWAP: physical qubits not coupling-graph-adjacent")
                lp, lq = at[p], at[q]
                at[p], at[q] = lq, lp
                pos[lp], pos[lq] = q, p
                num_swaps += 1
            elif typ == "G":
                i = int(take())
                if not (1 <= i <= m):
                    fail("GATE: index out of range")
                if i <= last_g:
                    fail("GATE: indices must be strictly increasing")
                last_g = i
                t, a, b = gates[i - 1]
                pa, pb = pos[a], pos[b]
                if pb not in adj[pa]:
                    fail("GATE %d: logical qubits %d,%d not adjacent when executed" % (i, a, b))
                num_gates += 1
                executed.add(i)
            else:
                fail("unknown op tag %r" % typ)
    except IndexError:
        fail("truncated / wrong token count")
    except (ValueError, TypeError):
        fail("non-integer parameter")

    if pos_tok != len(out):
        fail("trailing artifact tokens")

    skipped = [i for i in range(1, m + 1) if i not in executed]
    for i in skipped:
        j = partner.get(i)
        if j is None:
            fail("gate %d skipped but is not part of any cancelling pair" % i)
        if j in executed:
            fail("gate %d skipped but its cancel partner %d was executed" % (i, j))

    F = 3 * num_swaps + 1 * num_gates
    # F == 0 is only reachable if every single gate legitimately cancelled
    # away (fully verified above) and zero swaps were needed -- a genuinely
    # optimal, feasible artifact; cap at 1.0 rather than divide by zero.
    ratio = min(1.0, 0.1 * B / max(1e-9, F)) if F > 0 else 1.0
    print("F=%d B=%d swaps=%d gates_exec=%d cancelled_pairs=%d Ratio: %.6f" %
          (F, B, num_swaps, num_gates, len(skipped) // 2, ratio))


if __name__ == "__main__":
    main()
