# TIER: strong
# Insight: the fixed look-ahead rule can only ever compare/reorder cues that
# are BOTH sitting in its depth-w buffer at the same time -- and the buffer
# only ever holds cues whose sheet positions are close together (it admits
# strictly in sheet order, w-deep). So cue pairs within w of each other are a
# cheap, purely-local PROXY for "reordering-reachable" pairs -- not a hard
# guarantee (a cue that keeps losing to nearer rivals can be delayed past its
# original w-deep entry window, so it is not literally the exact reachable
# set) -- but it is exactly the signal a raw-adjacency or frequency count
# throws away. Build a co-occurrence graph over cue pairs within w of each
# other on that basis, cluster it under the groove-capacity budget (an
# approximate reconstruction of the hidden "sides"), then chain
# the clusters into a groove order by how strongly they interact. Cells that
# are cued close together end up on nearby, contiguous grooves; the FIXED
# window rule then sweeps each such cluster in track order almost for free,
# even though the raw cue order inside a cluster is scrambled. This exploits
# the window, not just a better linear-arrangement of the raw cue sequence.
import sys
from collections import defaultdict


def main():
    data = sys.stdin.read().split()
    p = 0
    N = int(data[p]); T = int(data[p + 1]); cap = int(data[p + 2]); w = int(data[p + 3]); M = int(data[p + 4])
    p += 5
    Q = [int(data[p + k]) for k in range(M)]

    # ---- 1) co-occurrence graph over reachable (within-window) pairs ----
    edge = defaultdict(int)
    for i in range(M):
        qi = Q[i]
        lim = min(i + w, M)
        for j in range(i + 1, lim):
            qj = Q[j]
            if qi == qj:
                continue
            a, b = (qi, qj) if qi < qj else (qj, qi)
            edge[(a, b)] += 1

    # ---- 2) capacity-budgeted union-find clustering ----
    parent = list(range(N))
    size = [1] * N

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    max_cluster = w + cap
    edges_sorted = sorted(edge.items(), key=lambda kv: (-kv[1], kv[0]))
    for (a, b), wt in edges_sorted:
        ra, rb = find(a), find(b)
        if ra != rb and size[ra] + size[rb] <= max_cluster:
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]

    groups = defaultdict(list)
    for cell in range(N):
        groups[find(cell)].append(cell)
    cluster_ids = sorted(groups.keys())
    cluster_of = {}
    for idx, root in enumerate(cluster_ids):
        for cell in groups[root]:
            cluster_of[cell] = idx
    clusters = [sorted(groups[root]) for root in cluster_ids]
    C = len(clusters)

    # ---- 3) inter-cluster affinity, then a nearest-neighbour chain order ----
    cedge = defaultdict(int)
    for (a, b), wt in edge.items():
        ca, cb = cluster_of[a], cluster_of[b]
        if ca != cb:
            key = (ca, cb) if ca < cb else (cb, ca)
            cedge[key] += wt

    adj = defaultdict(dict)
    for (ca, cb), wt in cedge.items():
        adj[ca][cb] = wt
        adj[cb][ca] = wt

    remaining = set(range(C))
    # start from the largest cluster (most impactful to place well)
    start = max(remaining, key=lambda c: (len(clusters[c]), -c))
    chain = [start]
    remaining.discard(start)
    while remaining:
        last = chain[-1]
        cand = None
        best_wt = -1
        for nb, wt in adj.get(last, {}).items():
            if nb in remaining and wt > best_wt:
                best_wt = wt
                cand = nb
        if cand is None:
            cand = min(remaining)  # no signal left: stable fallback
        chain.append(cand)
        remaining.discard(cand)

    # ---- 3b) WITHIN each cluster, also chain-order the cells by their own
    # co-occurrence weight (same nearest-neighbour-chain idea, one level
    # down): cells that are cued right next to each other land on adjacent
    # grooves inside the cluster's range too, not just "some contiguous
    # range in arbitrary order".
    local_adj = defaultdict(dict)
    for (a, b), wt in edge.items():
        if cluster_of[a] == cluster_of[b]:
            local_adj[a][b] = wt
            local_adj[b][a] = wt

    def chain_cluster(members):
        if len(members) <= 2:
            return list(members)
        rem = set(members)
        deg_sum = {m: sum(local_adj.get(m, {}).values()) for m in members}
        cur = max(rem, key=lambda m: (deg_sum[m], -m))
        order = [cur]
        rem.discard(cur)
        while rem:
            best, best_wt = None, -1
            for nb, wt in local_adj.get(cur, {}).items():
                if nb in rem and wt > best_wt:
                    best_wt, best = wt, nb
            if best is None:
                best = min(rem)
            order.append(best)
            rem.discard(best)
            cur = best
        return order

    # ---- 4) pack clusters, in chain order, DENSELY onto grooves (no forced
    # track break between clusters -> uses exactly ceil(N/cap) <= T grooves,
    # so it can never overflow the groove budget no matter how the graph
    # clustered). Cells of one cluster stay contiguous; at most `cap-1` cells
    # of a neighbouring cluster share the boundary groove.
    def build_trk(chain_order):
        flat = []
        for cidx in chain_order:
            flat.extend(chain_cluster(clusters[cidx]))
        t = [0] * N
        for k, cell in enumerate(flat):
            t[cell] = min(T - 1, k // cap)
        return t

    # ---- 5) local-search refinement on the REAL objective. The chain order
    # from step 3 is a heuristic proxy; here we directly simulate the fixed
    # window rule (same mechanics as the checker) and hill-climb by swapping
    # adjacent clusters in the chain whenever it actually lowers total travel
    # -- a genuine refinement against the true metric, not just more of the
    # same heuristic.
    def simulate(trk):
        pos = min(w, M)
        buf = list(range(pos))
        arm = 0
        cost = 0
        while buf:
            best_i, best_key = 0, None
            for i, qi in enumerate(buf):
                key = (abs(trk[Q[qi]] - arm), qi)
                if best_key is None or key < best_key:
                    best_key, best_i = key, i
            qi = buf.pop(best_i)
            cost += abs(trk[Q[qi]] - arm)
            arm = trk[Q[qi]]
            if pos < M:
                buf.append(pos)
                pos += 1
        return cost

    best_chain = chain
    best_trk = build_trk(best_chain)
    best_F = simulate(best_trk)
    for _ in range(3):
        improved = False
        for i in range(len(best_chain) - 1):
            cand_chain = best_chain[:]
            cand_chain[i], cand_chain[i + 1] = cand_chain[i + 1], cand_chain[i]
            cand_trk = build_trk(cand_chain)
            cand_F = simulate(cand_trk)
            if cand_F < best_F:
                best_chain, best_trk, best_F = cand_chain, cand_trk, cand_F
                improved = True
        if not improved:
            break

    sys.stdout.write(" ".join(str(x) for x in best_trk) + "\n")


if __name__ == "__main__":
    main()
