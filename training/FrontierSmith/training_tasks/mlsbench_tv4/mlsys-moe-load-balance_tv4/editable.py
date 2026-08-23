# EDITABLE SECTION
# Variant objective: the hierarchy leads. Node-level flatness and
# replica locality are the primary objectives; per-GPU polish is
# whatever Stage 3 can recover afterwards. The scaffold's packing is a
# round-based lightest-node matching (strongest at the group-to-node
# stage this variant centers); replication is a one-shot top-k grant
# that never lets a replica leave its node.
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

    Scaffold: round-based matching. Items are taken heaviest-first in
    rounds of ``num_packs``; within a round, packs are re-sorted by
    accumulated load and the heaviest remaining item is matched to the
    lightest pack, second-heaviest to second-lightest, and so on. With
    two groups per node (the stress topology) this is exactly one
    heavy-with-light pairing round — the decision this variant says the
    whole score hangs on. The Python loop runs items_per_pack times,
    with all batches processed per iteration as tensors.
    """
    B, n = weight.shape
    assert n % num_packs == 0
    items_per_pack = n // num_packs
    device = weight.device

    if items_per_pack == 1:
        idx = torch.arange(n, dtype=torch.int64, device=device).expand(B, -1)
        return idx, torch.zeros_like(idx)

    w = weight.float()
    order = w.sort(-1, descending=True).indices                       # [B, n]
    loads = torch.zeros(B, num_packs, device=device)
    pack_index = torch.empty(B, n, dtype=torch.int64, device=device)
    rank_in_pack = torch.empty(B, n, dtype=torch.int64, device=device)
    for r in range(items_per_pack):
        chunk = order[:, r * num_packs:(r + 1) * num_packs]           # [B, P]
        dest = loads.argsort(dim=-1)                                  # lightest first
        pack_index.scatter_(1, chunk, dest)
        rank_in_pack.scatter_(1, chunk, torch.full_like(chunk, r))
        loads.scatter_add_(1, dest, w.gather(1, chunk))
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

    Scaffold: one-shot top-k grant. The ``spare`` heaviest experts of the
    node each receive exactly one extra replica (cycling if spares exceed
    the expert count). Because this runs per node AFTER groups were
    assigned, every copy stays inside its expert's node — locality holds
    at its ceiling by construction, which is the variant's first
    commitment. The slot layout is grouped by expert id and materialised
    without any per-row work: a 1 is scattered at each expert's first
    slot and a running sum turns those markers into expert ids, with
    ranks falling out of a cumulative-count subtraction.
    """
    B, num_log = weight.shape
    device = weight.device
    spare = num_phy - num_log
    counts = torch.ones(B, num_log, dtype=torch.int64, device=device)
    if spare > 0:
        full_rounds, tail = divmod(spare, num_log)
        counts += full_rounds
        if tail:
            hot = weight.float().topk(tail, dim=-1).indices
            counts.scatter_add_(1, hot, torch.ones_like(hot))

    starts = counts.cumsum(-1) - counts                    # first slot per expert
    marker = torch.zeros(B, num_phy, dtype=torch.int64, device=device)
    marker.scatter_(1, starts, torch.ones_like(starts))
    phy2log = marker.cumsum(-1) - 1                        # run-length decode
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
    Main entry point: hierarchy-first placement.

    Reading order mirrors the priority order of this variant: the
    group-to-node stage is the primary algorithm (node balance), the
    replication stage is locality-preserving by construction, and the
    GPU stage merely polishes whatever headroom the first two left.

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

    def inv_perm(perm):
        out = torch.empty_like(perm)
        out.scatter_(1, perm,
                     torch.arange(perm.size(1), dtype=torch.int64).expand(perm.shape))
        return out

    # PRIMARY stage — pair groups onto nodes so node totals are flat.
    gload = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    npos, nrk = balanced_packing(gload, num_nodes)
    log2mlog = (((npos * groups_per_node + nrk) * group_size).unsqueeze(-1)
                + torch.arange(group_size)).flatten(-2)
    mlog2log = inv_perm(log2mlog)

    # Locality stage — replicas granted strictly within each node.
    local_w = weight.gather(-1, mlog2log).view(-1, experts_per_node)
    r2e, rrk, rcnt = replicate_experts(local_w, slots_per_node)

    # Polish stage — spread the node's replicas over its own GPUs.
    load_per_rep = (local_w / rcnt.float()).gather(-1, r2e)
    gpos, grk = balanced_packing(load_per_rep, gpus_per_node)
    to_slot = gpos * slots_per_gpu + grk
    from_slot = inv_perm(to_slot)

    slot_expert = r2e.gather(-1, from_slot)
    slot_expert = (slot_expert.view(L, num_nodes, -1)
                   + torch.arange(0, E, experts_per_node).view(1, -1, 1)).flatten(-2)
    phy2log = mlog2log.gather(-1, slot_expert)
    phyrank = rrk.gather(-1, from_slot).view(L, -1)
    logcnt = rcnt.view(L, -1).gather(-1, log2mlog)

    max_rep = logcnt.max().item()
    log2phy = torch.full((L, E, max_rep), -1, dtype=torch.int64)
    log2phy.view(L, -1).scatter_(
        -1, phy2log * max_rep + phyrank,
        torch.arange(num_replicas).expand(L, -1),
    )
    return phy2log, log2phy, logcnt
