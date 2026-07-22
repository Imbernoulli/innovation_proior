import sys
import random

# ---------------------------------------------------------------------------
# threshold-firebreak-blocking :: "Cut firebreaks to stop a spreading cascade"
#
# A weighted directed influence graph. A fixed set of SOURCE nodes starts
# active. A non-source node v activates once the sum of amp[u]*w(u,v) over
# its already-active in-neighbours u reaches its threshold theta[v] (a
# classic linear-threshold cascade, but each node u's OUTGOING edges are
# scaled by its own amplification factor amp[u] once u is active -- a
# "superspreader" effect). The solver removes up to K non-source nodes
# (firebreaks) BEFORE the cascade starts, to minimize the final count of
# activated non-source nodes.
#
# Planted structure per test:
#   sources -> a redundant, densely-interconnected MESH (many alternate
#     paths -- a small budget spent inside the mesh barely dents it)
#   mesh -> per branch: a redundant FUNNEL (several mesh parents feed each
#     funnel node, and ALL funnel nodes of a branch feed the SAME single
#     AMPLIFIER node) -> the amplifier fans out (with its large amp
#     multiplier) to many low-threshold LEAF followers that activate only
#     through it.
#   The amplifier node is therefore the unique effective cut point of its
#   branch: no small near-source cut and no funnel-edge cut disconnects it
#   (both are redundant / oversized for the budget), but removing that one
#   deep node kills its entire leaf flood.
#   A pool of NOISE nodes adds background clutter unrelated to any branch.
# ---------------------------------------------------------------------------

N_SCHED     = [30, 55, 90, 140, 200, 260, 320, 380, 440, 500]
S_SCHED     = [2, 2, 2, 3, 3, 3, 3, 3, 4, 4]
MESH_FRAC   = [0.30, 0.28, 0.26, 0.24, 0.23, 0.22, 0.21, 0.20, 0.20, 0.19]
NBRANCH     = [1, 2, 3, 3, 4, 5, 8, 4, 9, 10]
K_SCHED     = [4, 6, 6, 7, 7, 8, 7, 10, 7, 8]
NOISE_FRAC  = [0.15, 0.16, 0.16, 0.17, 0.17, 0.17, 0.17, 0.20, 0.17, 0.17]

BASE_SEED = 264949131


PS = 3            # private pendant satellites per near-source "hub" node
HUB_EXTRA = 2     # extra out-weight edges per hub (for the degree ranking)
LEAF_W_LO, LEAF_W_HI = 0.05, 0.08   # raw (unamplified) weight on amp->leaf edges
AMP_LO, AMP_HI = 15.0, 30.0         # amplification factor range


def build(tid):
    rnd = random.Random(BASE_SEED + tid)
    N = N_SCHED[tid - 1]
    S = S_SCHED[tid - 1]
    mesh_size = max(6, round(MESH_FRAC[tid - 1] * N))
    nbranch = NBRANCH[tid - 1]
    K = K_SCHED[tid - 1]
    sat_total = min(K, mesh_size) * PS   # extra nodes the hub satellites add
    noise_target = max(4, round(NOISE_FRAC[tid - 1] * N))

    # Mesh / branch / noise sizing targets the nominal N exactly (unchanged
    # by the satellite feature below) so branch sizes -- and hence the
    # amplifiers' true share of the cascade -- stay exactly as tuned. The
    # hub satellites are added ON TOP afterwards; the actual printed N is
    # whatever the allocator ends up using (nominal N + sat_total).
    deep_budget = N - S - mesh_size - noise_target
    min_branch = 8
    if deep_budget < nbranch * min_branch:
        deep_budget = nbranch * min_branch
    noise_count = N - S - mesh_size - deep_budget
    if noise_count < 0:
        shrink = min(mesh_size - 5, -noise_count)
        mesh_size -= max(0, shrink)
        noise_count = N - S - mesh_size - deep_budget
        noise_count = max(0, noise_count)
        deep_budget = N - S - mesh_size - noise_count

    weights = [rnd.uniform(0.6, 1.7) for _ in range(nbranch)]
    wsum = sum(weights)
    branch_total = []
    used = 0
    for i, wt in enumerate(weights):
        if i < nbranch - 1:
            cnt = max(min_branch, round(deep_budget * wt / wsum))
        else:
            cnt = deep_budget - used
        cnt = max(min_branch, cnt)
        used += cnt
        branch_total.append(cnt)
    drift = deep_budget - sum(branch_total)
    noise_count = max(0, noise_count + drift)
    total_check = S + mesh_size + sum(branch_total) + noise_count
    noise_count += (N - total_check)
    if noise_count < 0:
        # extremely defensive; schedule above never triggers this
        noise_count = 0

    nid = 0

    def alloc(cnt):
        nonlocal nid
        cnt = max(0, cnt)
        ids = list(range(nid + 1, nid + 1 + cnt))
        nid += cnt
        return ids

    sources = alloc(S)
    mesh = alloc(mesh_size)

    theta = {}
    amp = {}
    edges = []

    for s in sources:
        theta[s] = 0.0
        amp[s] = 1.0

    # Every mesh node gets its OWN direct edge from a (randomly chosen)
    # source, weight comfortably above threshold. Sources can never be
    # removed, so the only way to stop a mesh node activating is to remove
    # that specific node -- and mesh_size is always >> K, so no budget can
    # come close to cutting the source's full neighbourhood. A handful of
    # extra mesh-to-mesh edges add out-degree variety (for the greedy
    # heuristic to rank by) without changing this robustness.
    for m in mesh:
        src_parent = rnd.choice(sources)
        w = rnd.uniform(0.9, 1.3)
        edges.append((src_parent, m, w))
        theta[m] = 0.75
        amp[m] = 1.0
    extra_mesh_edges = max(1, mesh_size // 3)
    for _ in range(extra_mesh_edges):
        u = rnd.choice(mesh)
        v = rnd.choice(mesh)
        if u != v:
            w = rnd.uniform(0.3, 0.9)
            edges.append((u, v, w))

    # Precompute each branch's funnel/leaf split up front (before any nodes
    # for it are allocated) so the hub decoys below can be sized to match
    # the SAME out-degree distribution as a real amplifier -- otherwise raw
    # out-DEGREE (edge count, independent of weight) alone would single out
    # amplifiers just as easily as raw out-weight did before this fix.
    funnel_counts = []
    leaf_counts = []
    for bt in branch_total:
        fc = max(3, min(5, bt // 4))
        lc = max(4, bt - fc - 1)
        funnel_counts.append(fc)
        leaf_counts.append(lc)

    # Exactly K of the mesh nodes ("hubs") get a private pendant satellite
    # cluster (PS nodes, fed ONLY by that hub) plus a small out-weight
    # boost AND a degree-matched set of inert decoy edges (tiny weight, to
    # random existing mesh/noise-eligible nodes -- functionally irrelevant
    # to the cascade) so hubs' raw out-DEGREE looks just like an
    # amplifier's. A raw-degree or raw-weight ranking therefore finds a mix
    # of real hubs and cannot reliably separate them from amplifiers; only
    # noticing the `amp` field and simulating its effect does. This is the
    # genuine, but strictly bounded (O(K)), payoff a near-source firebreak
    # earns: it can shave off these K local pendant clusters, but it
    # structurally cannot reach the deep amplifier branches.
    hub_count = min(K, mesh_size)
    hubs = mesh[:hub_count]
    decoy_pool_sizes = leaf_counts if leaf_counts else [10]
    for hi, h in enumerate(hubs):
        for _ in range(HUB_EXTRA):
            v = rnd.choice(mesh)
            if v != h:
                edges.append((h, v, rnd.uniform(0.4, 0.6)))
        sats = alloc(PS)
        for sa in sats:
            w = rnd.uniform(0.8, 1.1)
            edges.append((h, sa, w))
            theta[sa] = 0.7
            amp[sa] = 1.0
        decoy_deg = decoy_pool_sizes[hi % len(decoy_pool_sizes)]
        decoy_deg = max(0, decoy_deg + rnd.randint(-2, 2))
        for _ in range(decoy_deg):
            v = rnd.choice(mesh)
            if v != h:
                edges.append((h, v, rnd.uniform(0.01, 0.03)))

    for bi in range(nbranch):
        funnel_count = funnel_counts[bi]
        leaf_count = leaf_counts[bi]
        ids = alloc(funnel_count + 1 + leaf_count)
        funnels = ids[:funnel_count]
        amp_node = ids[funnel_count]
        leaves = ids[funnel_count + 1:]

        for f in funnels:
            parents = rnd.sample(mesh, k=min(2, len(mesh)))
            for p in parents:
                w = rnd.uniform(0.85, 1.15)
                edges.append((p, f, w))
            theta[f] = 0.8
            amp[f] = 1.0

        for f in funnels:
            w = rnd.uniform(0.9, 1.1)
            edges.append((f, amp_node, w))
        theta[amp_node] = 0.85
        amp_factor = rnd.uniform(AMP_LO, AMP_HI)
        amp[amp_node] = amp_factor

        # Leaf edges are deliberately LOW raw weight: an amplifier's true
        # reach comes from amp_factor multiplying these small weights, not
        # from a large stored weight or a large degree. This is the crux of
        # the hook -- a node's raw out-degree / out-weight (what any
        # degree-style heuristic sees) stays small even for an amplifier
        # feeding dozens of leaves, so nothing short of noticing the amp
        # field AND simulating its downstream effect reveals which nodes
        # are the real threats.
        for lf in leaves:
            w = rnd.uniform(LEAF_W_LO, LEAF_W_HI)
            edges.append((amp_node, lf, w))
            theta[lf] = amp_factor * w * 0.75
            amp[lf] = 1.0

    noise = alloc(noise_count)
    for nnode in noise:
        k = rnd.randint(1, 3)
        pool = mesh + [x for x in noise if x < nnode]
        parents = rnd.sample(pool, k=min(k, len(pool)))
        if not parents:
            parents = rnd.sample(mesh, k=1)
        for p in parents:
            w = rnd.uniform(0.5, 1.0)
            edges.append((p, nnode, w))
        theta[nnode] = 1.2
        amp[nnode] = 1.0

    # nid is the true final node count (nominal N plus the hub-satellite
    # nodes added on top); use it as the printed N.
    return nid, S, K, sources, theta, amp, edges


def main():
    tid = int(sys.argv[1])
    N, S, K, sources, theta, amp, edges = build(tid)
    M = len(edges)
    out = []
    out.append("%d %d %d %d" % (N, M, K, S))
    out.append(" ".join(str(x) for x in sources))
    for i in range(1, N + 1):
        out.append("%.6f %.6f" % (theta[i], amp[i]))
    for (u, v, w) in edges:
        out.append("%d %d %.6f" % (u, v, w))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
