# EDITABLE SECTION
# Variant objective: one untuned rule for every traffic regime. Nothing
# in this block may branch on which deployment profile is running; any
# adaptivity must be computed from the workload tensor itself. The
# scaffold pairs a zigzag (snake) packing with traffic-proportional
# replication — both closed-form, both knob-free.
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

    Scaffold: zigzag (snake) assignment. Items sorted heaviest-first are
    dealt to packs in order 0,1,...,P-1,P-1,...,1,0, so each pack receives
    one heavy-leaning and one light-leaning item per double round. The
    pattern is distribution-agnostic — no threshold decides what counts
    as "hot" — which is exactly the knob-freeness this variant demands.
    Its weakness under extreme tails (the top item can dominate a whole
    double round) is the robustness gap left to close.
    """
    B, n = weight.shape
    assert n % num_packs == 0
    items_per_pack = n // num_packs
    device = weight.device

    if items_per_pack == 1:
        idx = torch.arange(n, dtype=torch.int64, device=device).expand(B, -1)
        return idx, torch.zeros_like(idx)

    sorted_idx = weight.float().sort(-1, descending=True).indices     # [B, n]
    pos = torch.arange(n, dtype=torch.int64, device=device)
    cycle = pos % (2 * num_packs)
    snake_pack = torch.where(cycle < num_packs, cycle, 2 * num_packs - 1 - cycle)
    snake_rank = pos // num_packs
    pack_index = torch.empty(B, n, dtype=torch.int64, device=device)
    rank_in_pack = torch.empty(B, n, dtype=torch.int64, device=device)
    pack_index.scatter_(1, sorted_idx, snake_pack.expand(B, n).contiguous())
    rank_in_pack.scatter_(1, sorted_idx, snake_rank.expand(B, n).contiguous())
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

    Scaffold: traffic-proportional replication by largest remainder.
    Spare slots are apportioned to experts in proportion to their share of
    total load (floor of the ideal quota, leftovers to the largest
    fractional parts). The quota adapts continuously to the observed
    distribution — flat traffic spreads replicas, heavy tails concentrate
    them — with no tunable threshold anywhere. Fully batched: quota and
    remainder ordering are closed-form tensor expressions, and the slot
    layout is decoded from the replica counts by one flattened
    repeat_interleave over all rows at once.
    """
    B, num_log = weight.shape
    device = weight.device
    spare = num_phy - num_log
    counts = torch.ones(B, num_log, dtype=torch.int64, device=device)
    if spare > 0:
        w = weight.float()
        quota = w / w.sum(-1, keepdim=True).clamp(min=1e-8) * spare
        floor_extra = quota.floor().to(torch.int64)
        leftovers = spare - floor_extra.sum(-1)                        # [B]
        frac_order = (quota - quota.floor()).argsort(-1, descending=True)
        grant = (torch.arange(num_log, dtype=torch.int64, device=device).expand(B, -1)
                 < leftovers.unsqueeze(1)).to(torch.int64)
        bonus = torch.zeros_like(floor_extra)
        bonus.scatter_(1, frac_order, grant)
        counts = counts + floor_extra + bonus

    ids = torch.arange(num_log, dtype=torch.int64, device=device).expand(B, -1)
    phy2log = torch.repeat_interleave(ids.reshape(-1),
                                      counts.reshape(-1)).view(B, num_phy)
    starts = counts.cumsum(-1) - counts
    slots = torch.arange(num_phy, dtype=torch.int64, device=device).expand(B, -1)
    rank = slots - starts.gather(1, phy2log)
    return phy2log, rank, counts


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Main entry point: knob-free hierarchical placement.

    All three stages run the same code on every profile; whatever
    adaptivity exists is carried by the workload statistics flowing
    through them. Per-config uniformity of the resulting scores — not the
    best single-config number — is what this variant optimizes for.

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
    phy_per_gpu = num_replicas // num_gpus
    groups_per_node = num_groups // num_nodes
    experts_per_node = E // num_nodes
    replicas_per_node = num_replicas // num_nodes

    def _inverse(perm):
        out = torch.empty_like(perm)
        out.scatter_(1, perm,
                     torch.arange(perm.size(1), dtype=torch.int64).expand(perm.shape))
        return out

    # Stage 1: snake-pack group traffic across nodes (same rule, any regime).
    gw = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    npack, nrank = balanced_packing(gw, num_nodes)
    log2mlog = (((npack * groups_per_node + nrank) * group_size).unsqueeze(-1)
                + torch.arange(group_size)).flatten(-2)
    mlog2log = _inverse(log2mlog)

    # Stage 2: proportional replication within each node.
    wnode = weight.gather(-1, mlog2log).view(-1, experts_per_node)
    pmap, prank, pcnt = replicate_experts(wnode, replicas_per_node)

    # Stage 3: snake-pack per-replica traffic onto the node's GPUs.
    wrep = (wnode / pcnt.float()).gather(-1, pmap)
    gpack, grank = balanced_packing(wrep, gpus_per_node)
    slot_of = gpack * phy_per_gpu + grank
    slot_inv = _inverse(slot_of)

    placed = pmap.gather(-1, slot_inv)
    placed = (placed.view(L, num_nodes, -1)
              + torch.arange(0, E, experts_per_node).view(1, -1, 1)).flatten(-2)
    phy2log = mlog2log.gather(-1, placed)
    phyrank = prank.gather(-1, slot_inv).view(L, -1)
    logcnt = pcnt.view(L, -1).gather(-1, log2mlog)

    max_rep = logcnt.max().item()
    log2phy = torch.full((L, E, max_rep), -1, dtype=torch.int64)
    log2phy.view(L, -1).scatter_(
        -1, phy2log * max_rep + phyrank,
        torch.arange(num_replicas).expand(L, -1),
    )
    return phy2log, log2phy, logcnt
