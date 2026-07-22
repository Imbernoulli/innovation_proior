# TIER: strong
# The insight: recast toolpathing as a MINIMUM-COST EULERIZATION of the cut
# graph. A connected graph's edges can always be decomposed into exactly
# max(1, odd_deg/2) open trails (Euler's theorem) -- that count is a fixed
# property of the graph, not of the visiting strategy, and it is usually far
# below "one pierce per part" once shared/T-junction edges are accounted for.
# We realize the decomposition by pairing up odd-degree vertices with cheap
# "phantom" (airtime-only) edges, running a single Eulerian circuit over the
# augmented multigraph (Hierholzer), and then slicing that circuit at the
# phantom edges: each slice is one continuous pierce-to-retract trail, and
# the phantom edge crossed between two consecutive trails becomes exactly
# the real airtime move the checker charges for that hand-off. We also pick
# which phantom edge is left "unclosed" (the most expensive one) so it is
# never actually charged as a transition.
import sys, math

def dist(xs, ys, a, b):
    return math.hypot(xs[a] - xs[b], ys[a] - ys[b])

def hierholzer(adj, used, start):
    ptr = {v: 0 for v in adj}
    stack_v = [start]
    stack_e = []
    circuit_v = []
    circuit_e = []
    while stack_v:
        v = stack_v[-1]
        advanced = False
        while ptr[v] < len(adj[v]):
            nbr, eid = adj[v][ptr[v]]
            ptr[v] += 1
            if not used[eid]:
                used[eid] = True
                stack_v.append(nbr)
                stack_e.append(eid)
                advanced = True
                break
        if not advanced:
            circuit_v.append(stack_v.pop())
            if stack_e:
                circuit_e.append(stack_e.pop())
    circuit_v.reverse()
    circuit_e.reverse()
    return circuit_v, circuit_e

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    def rdi(): return int(next(it))
    n = rdi(); m = rdi(); P = rdi()
    xs = [0.0] * (n + 1); ys = [0.0] * (n + 1)
    for v in range(1, n + 1):
        xs[v] = float(rdi()); ys[v] = float(rdi())
    eu = [0] * (m + 1); ev = [0] * (m + 1)
    adj = {v: [] for v in range(1, n + 1)}
    for e in range(1, m + 1):
        u = rdi(); v = rdi()
        eu[e] = u; ev[e] = v
        adj[u].append((v, e))
        adj[v].append((u, e))
    K = rdi()
    for _ in range(K):
        rdi(); rdi(); rdi(); rdi()

    # ---- connected components over real edges ----
    comp_id = [0] * (n + 1)
    comps = []
    for s in range(1, n + 1):
        if comp_id[s] or not adj[s]:
            continue
        cid = len(comps) + 1
        stack = [s]
        comp_id[s] = cid
        verts = [s]
        while stack:
            v = stack.pop()
            for nb, _e in adj[v]:
                if not comp_id[nb]:
                    comp_id[nb] = cid
                    verts.append(nb)
                    stack.append(nb)
        comps.append(verts)

    phantom_id = m
    all_trails = []

    for verts in comps:
        deg = {v: len(adj[v]) for v in verts}
        odd = [v for v in verts if deg[v] % 2 == 1]

        # greedy nearest-neighbor matching among odd-degree vertices
        remaining = list(odd)
        pairs = []
        remaining_set = set(remaining)
        # simple O(k^2) nearest neighbor matching, deterministic order
        remaining.sort()
        while remaining:
            a = remaining.pop(0)
            if a not in remaining_set:
                continue
            remaining_set.discard(a)
            best = None; bd = None
            for b in remaining:
                if b not in remaining_set:
                    continue
                d = dist(xs, ys, a, b)
                if bd is None or d < bd:
                    bd = d; best = b
            if best is not None:
                remaining_set.discard(best)
                pairs.append((a, best, bd))
        remaining = [v for v in odd if v in remaining_set]  # unmatched leftover (should be empty)

        aug_adj = {v: list(adj[v]) for v in verts}
        phantom_ids_here = []
        for (a, b, d) in pairs:
            phantom_id += 1
            eid = phantom_id
            eu.append(a); ev.append(b)  # extend arrays (indices align by append order beyond m)
            aug_adj[a].append((b, eid))
            aug_adj[b].append((a, eid))
            phantom_ids_here.append(eid)
        phantom_set = set(phantom_ids_here)

        used = {}
        for v in verts:
            for _nb, e in aug_adj[v]:
                used[e] = False
        start = verts[0]
        circuit_v, circuit_e = hierholzer(aug_adj, used, start)

        Mc = len(circuit_e)
        if Mc == 0:
            continue

        phantom_positions = [idx for idx, e in enumerate(circuit_e) if e in phantom_set]
        if not phantom_positions:
            # already Eulerian: whole component is one trail
            all_trails.append(circuit_v)
            continue

        # skip (leave uncharged) the longest phantom edge
        def plen(idx):
            e = circuit_e[idx]
            a = circuit_v[idx]; b = circuit_v[idx + 1]
            return dist(xs, ys, a, b)
        skip = max(phantom_positions, key=plen)

        Mtot = Mc
        rot_e = circuit_e[skip + 1:] + circuit_e[:skip + 1]
        rot_v = circuit_v[skip + 1:] + circuit_v[1:skip + 2]

        cur_trail = [rot_v[0]]
        for t in range(Mtot):
            e = rot_e[t]
            if e in phantom_set:
                if len(cur_trail) > 1:
                    all_trails.append(cur_trail)
                cur_trail = [rot_v[t + 1]]
            else:
                cur_trail.append(rot_v[t + 1])
        if len(cur_trail) > 1:
            all_trails.append(cur_trail)

    out = [str(len(all_trails))]
    for path in all_trails:
        out.append(f"{len(path)-1} " + " ".join(map(str, path)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
