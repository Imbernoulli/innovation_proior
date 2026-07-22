import sys, random

# ---- interbank clearing-injection instance generator ----------------------
# Bank 0 is the shared SINK (owes nothing to anyone, absorbs all final chain
# payments; e_sink = 0, p_bar_sink = 0, can never default).
#
# Each instance is built from three kinds of clusters:
#   * fan-chain hub cluster(s): a hub owes a SMALL amount `w` to each of K
#     independent branches. Each branch is a linear chain of D pass-through
#     banks (zero external assets, each owing exactly `w` -- what it was just
#     promised -- to the next link), finally paying the sink. Because every
#     non-hub node's own liability equals exactly what it is promised, the
#     same fraction f = (hub's paid fraction) recurs unchanged at every link
#     of every chain: total shortfall AT EACH OF THE D+1 LEVELS (hub included)
#     equals the hub's own gap. So closing the hub's gap (a small, LOCAL
#     amount) resolves (D+1) x gap of total system-wide shortfall -- a genuine
#     chain-depth multiplier. Crucially every non-hub node's OWN nominal
#     shortfall is just `w` (small), so no naive "biggest shortfall" scan
#     ever mistakes a chain node for the important target.
#   * decoy bank(s): isolated (no incoming edges), owes a moderately large
#     amount W (bigger than any single hub gap or chain hop, but far smaller
#     than the cluster's total systemic value) to the sink directly. Rescuing
#     a decoy only ever helps the decoy itself, dollar for dollar -- no
#     multiplier -- yet its raw shortfall LOOKS scariest of all to a
#     shortfall-ranked scan.
#   * calm banks: p_bar = 0, cannot ever default; pure padding for scale.
#
# The rescue budget C is sized to just cover the hub gap(s), with a small
# margin -- never anywhere near a decoy's own shortfall.


def build(K_fanout, D_depth, w, gap_frac):
    """One fan-chain hub cluster. Local id 0 = hub; ids 1..K*D = chain nodes
    (branch-major order); 'SINK' is a sentinel resolved by the caller.
    Returns (num_local_nodes, edges, e_local, gap, hub_pbar)."""
    num_local = 1 + K_fanout * D_depth
    e_local = [0] * num_local
    edges = []
    pbar_local = [0] * num_local
    nid = lambda b, d: 1 + b * D_depth + d  # branch b (0..K-1), depth d (0..D-1)
    for b in range(K_fanout):
        edges.append((0, nid(b, 0), w))
        pbar_local[0] += w
        for d in range(D_depth - 1):
            edges.append((nid(b, d), nid(b, d + 1), w))
            pbar_local[nid(b, d)] += w
        edges.append((nid(b, D_depth - 1), "SINK", w))
        pbar_local[nid(b, D_depth - 1)] += w
    hub_pbar = pbar_local[0]
    gap = max(2, int(hub_pbar * gap_frac))
    gap = min(gap, hub_pbar - 1)
    e_local[0] = hub_pbar - gap
    return num_local, edges, e_local, gap, hub_pbar


def main():
    t = int(sys.argv[1])
    rng = random.Random(20260710 + 97 * t)

    # ---- difficulty / scale ladder ----------------------------------------
    n_hubs = 1 if t <= 6 else 2
    if n_hubs == 1:
        clusters = [(4 + t // 2, 3 + t // 2)]        # (K_fanout, D_depth)
    else:
        clusters = [(4 + (t - 6), 3 + (t - 6)), (3 + (t - 7), 3 + (t - 7) // 2)]
    clusters = [(min(K, 12), min(D, 9)) for (K, D) in clusters]
    gap_frac = 0.18
    n_decoy = 1 + (1 if t >= 4 else 0) + (1 if t >= 8 else 0)
    n_calm = 4 * t

    banks_e = [0]          # bank 0 = SINK
    edges_global = []

    def new_bank(e):
        banks_e.append(e)
        return len(banks_e) - 1

    total_gap = 0
    total_cluster_potential = 0   # sum of (D+1)*gap per cluster -- the total
                                   # systemic shortfall each cluster can incur
    max_leaf = 0                  # largest single non-hub node liability (w)
    max_gap = 0
    for ci, (K_fanout, D_depth) in enumerate(clusters):
        w = 8 + (ci * 3) + (t % 3)
        K_local, edges, e_local, gap, hub_pbar = build(K_fanout, D_depth, w, gap_frac)
        total_gap += gap
        total_cluster_potential += (D_depth + 1) * gap
        max_leaf = max(max_leaf, w)
        max_gap = max(max_gap, gap)
        local_to_global = {}
        for li in range(K_local):
            local_to_global[li] = new_bank(e_local[li])
        for (u, v, wt) in edges:
            gu = local_to_global[u]
            gv = 0 if v == "SINK" else local_to_global[v]
            edges_global.append((gu, gv, wt))

    # ---- decoy banks: isolated, e=0, own shortfall bigger than any single
    # hub gap / chain hop (so a naive shortfall scan always prefers them),
    # but their COMBINED total is calibrated relative to the cluster's real
    # systemic value so that (a) the whole rescue budget C can be wasted on
    # decoys alone without ever spilling into the real hub/chain nodes, and
    # (b) decoys still leave enough of B dominated by the multiplier signal.
    DECOY_FRAC = 0.65     # combined decoy shortfall as a fraction of cluster potential
    C_FRAC = 0.30         # budget as a fraction of total B (cluster + decoy)
    decoy_total_target = max(int(DECOY_FRAC * total_cluster_potential), 4 * (max_gap + max_leaf))
    decoy_floor = int(1.6 * max(max_gap, max_leaf)) + 5
    per_decoy = max(decoy_floor, decoy_total_target // max(1, n_decoy))
    decoy_Ws = []
    for k in range(n_decoy):
        W = per_decoy + k * 7 + rng.randint(0, 6)
        decoy_Ws.append(W)
        b = new_bank(0)
        edges_global.append((b, 0, W))

    B_est = total_cluster_potential + sum(decoy_Ws)
    C = max(int(1.15 * total_gap) + 1, int(C_FRAC * B_est))
    # keep decoys collectively big enough that greedy (which always drains
    # decoys before ever touching a hub/chain node) can never spill past them
    if C > int(0.85 * sum(decoy_Ws)):
        C = int(0.85 * sum(decoy_Ws))
        C = max(C, int(1.15 * total_gap) + 1)

    # ---- calm padding banks: safe, p_bar = 0, isolated ----
    for k in range(n_calm):
        e_calm = rng.randint(5, 5 + 3 * t)
        new_bank(e_calm)

    N = len(banks_e)

    out = []
    out.append(f"{N} {C}")
    out.append(" ".join(str(x) for x in banks_e))
    out.append(str(len(edges_global)))
    for (u, v, wt) in edges_global:
        out.append(f"{u+1} {v+1} {wt}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
