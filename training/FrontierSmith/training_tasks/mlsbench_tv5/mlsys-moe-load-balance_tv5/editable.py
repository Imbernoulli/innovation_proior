# EDITABLE SECTION
# Variant objective: live inside a strict latency envelope. Every
# operation below is a fixed, data-independent tensor expression or a
# single sort/top-k — no per-item loops, no per-row loops. The scaffold
# deliberately starts at the extreme fast end (index-striped packing,
# one-sort replication); buying balance back WITHOUT leaving the
# envelope is the whole game.
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

    Scaffold: index striping — item i goes to pack i % num_packs at rank
    i // num_packs. Zero data-dependent work: no sort, no comparison, no
    loop; the entire assignment is two arange expressions broadcast over
    the batch. This is the floor of the runtime axis. The first
    millisecond of budget spent should buy a weight sort feeding this
    same stripe pattern; the question is what each further increment
    (snake ordering, one repair sweep) buys per unit of measured time.
    """
    B, n = weight.shape
    assert n % num_packs == 0
    device = weight.device

    pos = torch.arange(n, dtype=torch.int64, device=device)
    pack_index = (pos % num_packs).expand(B, n).contiguous()
    rank_in_pack = (pos // num_packs).expand(B, n).contiguous()
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

    Scaffold: one sort, then pure arithmetic. Slots 0..num_log-1 hold the
    identity layout; the spare slots are appended as +1 copies of the
    heaviest experts, taken as a prefix of a single descending argsort
    (wrapping in whole rounds if spares exceed the expert count, which
    never loops in practice). Cost: one sort plus concatenations —
    per-slot greedy loops are exactly what the latency envelope forbids.
    """
    B, num_log = weight.shape
    device = weight.device
    spare = num_phy - num_log
    hot_order = weight.float().argsort(dim=-1, descending=True)       # [B, num_log]
    base = torch.arange(num_log, dtype=torch.int64, device=device).expand(B, -1)

    id_cols = [base]
    rank_cols = [torch.zeros(B, num_log, dtype=torch.int64, device=device)]
    counts = torch.ones(B, num_log, dtype=torch.int64, device=device)
    left, wave = spare, 1
    while left > 0:
        take = min(left, num_log)
        grant = hot_order[:, :take]
        id_cols.append(grant)
        rank_cols.append(torch.full((B, take), wave, dtype=torch.int64, device=device))
        counts.scatter_add_(1, grant, torch.ones_like(grant))
        left -= take
        wave += 1

    phy2log = torch.cat(id_cols, dim=1)
    rank = torch.cat(rank_cols, dim=1)
    return phy2log, rank, counts


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Main entry point: envelope-bounded hierarchical placement.

    The three-stage hierarchy is kept (it is what protects locality), but
    every stage is restricted to the fast vocabulary: fixed index
    arithmetic, at most one sort or top-k per stage, batched over all
    layers and nodes in one shot. Timing is part of the contract — the
    harness medians 20 runs — so any added cleverness must show up in the
    balance columns by more than it costs in the runtime column.

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
    per_gpu = num_replicas // num_gpus
    groups_per_node = num_groups // num_nodes
    experts_per_node = E // num_nodes
    per_node = num_replicas // num_nodes

    def _inv(perm):
        out = torch.empty_like(perm)
        out.scatter_(1, perm,
                     torch.arange(perm.size(1), dtype=torch.int64).expand(perm.shape))
        return out

    # Stage 1: stripe groups over nodes (constant-time assignment).
    per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    s1_pack, s1_rank = balanced_packing(per_group, num_nodes)
    log2mlog = (((s1_pack * groups_per_node + s1_rank) * group_size).unsqueeze(-1)
                + torch.arange(group_size)).flatten(-2)
    mlog2log = _inv(log2mlog)

    # Stage 2: one-sort replication inside each node.
    w_in_node = weight.gather(-1, mlog2log).view(-1, experts_per_node)
    slot2exp, slot_rank, exp_cnt = replicate_experts(w_in_node, per_node)

    # Stage 3: stripe the node's slots over its GPUs.
    w_slot = (w_in_node / exp_cnt.float()).gather(-1, slot2exp)
    s3_pack, s3_rank = balanced_packing(w_slot, gpus_per_node)
    fwd = s3_pack * per_gpu + s3_rank
    bwd = _inv(fwd)

    final = slot2exp.gather(-1, bwd)
    final = (final.view(L, num_nodes, -1)
             + torch.arange(0, E, experts_per_node).view(1, -1, 1)).flatten(-2)
    phy2log = mlog2log.gather(-1, final)
    phyrank = slot_rank.gather(-1, bwd).view(L, -1)
    logcnt = exp_cnt.view(L, -1).gather(-1, log2mlog)

    max_rep = logcnt.max().item()
    log2phy = torch.full((L, E, max_rep), -1, dtype=torch.int64)
    log2phy.view(L, -1).scatter_(
        -1, phy2log * max_rep + phyrank,
        torch.arange(num_replicas).expand(L, -1),
    )
    return phy2log, log2phy, logcnt
