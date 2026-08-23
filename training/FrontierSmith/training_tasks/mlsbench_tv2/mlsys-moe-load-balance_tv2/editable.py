# EDITABLE SECTION
# Variant objective: the straggler IS the objective. Judge every stage by
# the load of the single hottest GPU (and hottest node) it leaves behind,
# not by average fullness. The scaffold uses longest-processing-time
# greedy packing, batched across layers, as the max-load-first baseline;
# peak-directed repair passes are the intended headroom.
# ================================================================

def balanced_packing(weight: torch.Tensor, num_packs: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted items into num_packs balanced packs.

    Args:
        weight: [B, n] — weight of each item across B batches
        num_packs: number of packs

    Returns:
        pack_index: [B, n] — which pack (0..num_packs-1) each item goes to
        rank_in_pack: [B, n] — position (0..items_per_pack-1) within the pack

    Constraint: each pack must contain exactly n // num_packs items.

    Scaffold: batched LPT (longest-processing-time) greedy. Items are
    visited heaviest-first; each goes to the currently lightest pack that
    still has a free slot — the classic minimax packing heuristic. The
    item loop is Python, but every step inside it is a tensor op across
    ALL batches at once, so cost scales with n rather than B*n. Headroom:
    swap-based repair that unloads the argmax pack after the greedy pass.
    """
    B, n = weight.shape
    assert n % num_packs == 0
    items_per_pack = n // num_packs
    device = weight.device

    if items_per_pack == 1:
        idx = torch.arange(n, dtype=torch.int64, device=device).expand(B, -1)
        return idx, torch.zeros_like(idx)

    w = weight.float()
    order = w.sort(-1, descending=True).indices          # [B, n]
    loads = torch.zeros(B, num_packs, device=device)
    counts = torch.zeros(B, num_packs, dtype=torch.int64, device=device)
    pack_index = torch.empty(B, n, dtype=torch.int64, device=device)
    rank_in_pack = torch.empty(B, n, dtype=torch.int64, device=device)
    rows = torch.arange(B, dtype=torch.int64, device=device)
    for j in range(n):
        item = order[:, j]                                # [B]
        item_w = w.gather(1, item.unsqueeze(1)).squeeze(1)
        open_pack = counts < items_per_pack
        visible = torch.where(open_pack, loads, torch.full_like(loads, float("inf")))
        dest = visible.argmin(dim=-1)                     # lightest open pack
        pack_index[rows, item] = dest
        rank_in_pack[rows, item] = counts[rows, dest]
        loads[rows, dest] += item_w
        counts[rows, dest] += 1
    return pack_index, rank_in_pack


def replicate_experts(
    weight: torch.Tensor, num_phy: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate num_log logical experts into num_phy physical slots.

    Args:
        weight: [B, num_log] — load per logical expert
        num_phy: total physical expert slots (>= num_log)

    Returns:
        phy2log: [B, num_phy] — logical expert ID for each physical slot
        rank: [B, num_phy] — replica rank (0 = original, 1+ = copies)
        logcnt: [B, num_log] — number of replicas per logical expert

    Scaffold: peak-chasing greedy. Each spare slot is granted to the
    expert whose per-replica load is currently the largest — i.e. always
    attack the present maximum. This is exactly the minimax move at
    replica granularity; whether it remains the right move once the
    GPU-level packing is taken into account is the variant's question.
    """
    B, num_log = weight.shape
    device = weight.device
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(B, 1)
    rank = torch.zeros(B, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(B, num_log, dtype=torch.int64, device=device)
    rows = torch.arange(B, dtype=torch.int64, device=device)
    for slot in range(num_log, num_phy):
        peak = (weight / logcnt.float()).argmax(dim=-1)   # current worst expert
        phy2log[:, slot] = peak
        rank[:, slot] = logcnt[rows, peak]
        logcnt[rows, peak] += 1
    return phy2log, rank, logcnt


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Main entry point: hierarchical minimax expert placement.

    Stage 1 sets the node-level peak (group-to-node packing), Stage 2
    attacks the per-replica peak (replication), Stage 3 sets the GPU-level
    peak (replica-to-GPU packing). The straggler audit this variant asks
    for amounts to instrumenting which of the three leaves the residual
    maximum behind.

    Args:
        weight: [L, E] — token load per expert per layer
        num_replicas: total physical expert slots (multiple of num_gpus)
        num_groups: number of expert groups
        num_nodes: number of server nodes
        num_gpus: total GPUs (multiple of num_nodes)

    Returns:
        phy2log: [L, num_replicas] — logical expert for each physical slot
        log2phy: [L, E, max_rep] — physical IDs per expert (-1 = unused)
        logcnt: [L, E] — replica count per expert
    """
    L, E = weight.shape
    weight = weight.float().cpu()
    group_size = E // num_groups
    gpus_per_node = num_gpus // num_nodes
    slots_per_gpu = num_replicas // num_gpus
    groups_per_node = num_groups // num_nodes
    experts_per_node = E // num_nodes
    slots_per_node = num_replicas // num_nodes

    def invert(perm):
        out = torch.empty_like(perm)
        out.scatter_(1, perm,
                     torch.arange(perm.size(1), dtype=torch.int64).expand(perm.shape))
        return out

    # Stage 1 — group-to-node packing: the node-level maximum is decided here.
    group_load = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    g_pack, g_rank = balanced_packing(group_load, num_nodes)
    log2mlog = (((g_pack * groups_per_node + g_rank) * group_size).unsqueeze(-1)
                + torch.arange(group_size)).flatten(-2)
    mlog2log = invert(log2mlog)

    # Stage 2 — in-node replication aimed at the current per-replica peak.
    node_w = weight.gather(-1, mlog2log).view(-1, experts_per_node)
    p2m, p_rank, m_cnt = replicate_experts(node_w, slots_per_node)

    # Stage 3 — replica-to-GPU packing: the GPU-level maximum is decided here.
    rep_w = (node_w / m_cnt.float()).gather(-1, p2m)
    q_pack, q_rank = balanced_packing(rep_w, gpus_per_node)
    p2pp = q_pack * slots_per_gpu + q_rank
    pp2p = invert(p2pp)

    pp2m = p2m.gather(-1, pp2p)
    pp2m = (pp2m.view(L, num_nodes, -1)
            + torch.arange(0, E, experts_per_node).view(1, -1, 1)).flatten(-2)
    phy2log = mlog2log.gather(-1, pp2m)
    phyrank = p_rank.gather(-1, pp2p).view(L, -1)
    logcnt = m_cnt.view(L, -1).gather(-1, log2mlog)

    max_rep = logcnt.max().item()
    log2phy = torch.full((L, E, max_rep), -1, dtype=torch.int64)
    log2phy.view(L, -1).scatter_(
        -1, phy2log * max_rep + phyrank,
        torch.arange(num_replicas).expand(L, -1),
    )
    return phy2log, log2phy, logcnt
