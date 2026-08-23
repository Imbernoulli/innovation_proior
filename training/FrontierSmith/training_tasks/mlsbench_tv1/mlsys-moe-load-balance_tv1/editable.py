# EDITABLE SECTION
# Variant objective: expert placement under a SCARCE replication budget and
# long-tail traffic, judged by the weakest configuration. Ration the few
# spare replicas (see `budget_pressure`), keep each expert's replicas
# node-local, and keep the algorithm batched tensor math whose runtime
# stays flat as the topology grows.
# ================================================================

def budget_pressure(weight: torch.Tensor, num_phy: int) -> torch.Tensor:
    """Scarcity statistic: fraction of traffic the spare budget CANNOT relieve.

    Args:
        weight: [B, n] — per-expert load
        num_phy: total physical slots (>= n)

    Returns:
        [B] tensor in [0, 1]. Near 0: the surplus could replicate every
        expert carrying meaningful load (comfortable regime). Near 1:
        almost all traffic sits on experts the budget cannot reach
        (rationing regime). The placeholder computes it cheaply but does
        not yet act on it — steering replication and packing with this
        statistic is the intended lever of the variant.
    """
    B, n = weight.shape
    spare = max(int(num_phy) - n, 0)
    if spare == 0:
        return torch.ones(B)
    top = weight.float().topk(min(spare, n), dim=-1).values.sum(-1)
    total = weight.float().sum(-1).clamp(min=1e-8)
    return 1.0 - top / total


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

    Placeholder: fully batched sorted ROUND-ROBIN — the j-th heaviest item
    goes to pack j % num_packs. Valid, and fast at any topology size (no
    per-item Python loop), but it never looks at accumulated pack load, so
    under long-tail skew the pack holding the hottest item stays hot.
    Better balance at the same runtime (zigzag ordering, load-aware repair
    passes, pressure-weighted assignment) is the headroom.
    """
    B, n = weight.shape
    assert n % num_packs == 0
    items_per_pack = n // num_packs
    device = weight.device

    if items_per_pack == 1:
        idx = torch.arange(n, dtype=torch.int64, device=device).expand(B, -1)
        return idx, torch.zeros_like(idx)

    sorted_idx = weight.float().sort(-1, descending=True).indices  # [B, n]
    pos = torch.arange(n, dtype=torch.int64, device=device)
    pack_of_pos = (pos % num_packs).expand(B, n).contiguous()
    rank_of_pos = (pos // num_packs).expand(B, n).contiguous()
    pack_index = torch.empty(B, n, dtype=torch.int64, device=device)
    rank_in_pack = torch.empty(B, n, dtype=torch.int64, device=device)
    pack_index.scatter_(1, sorted_idx, pack_of_pos)
    rank_in_pack.scatter_(1, sorted_idx, rank_of_pos)
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

    Placeholder: sequential greedy — each spare slot goes to the expert
    with the highest current per-replica load. Reasonable when spare slots
    are plentiful; under rationing it never asks whether relieving a
    second-tier expert would shrink the eventual per-GPU maximum more than
    another copy of the top expert (see `budget_pressure`).
    """
    B, num_log = weight.shape
    device = weight.device
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(B, 1)
    rank = torch.zeros(B, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(B, num_log, dtype=torch.int64, device=device)
    idx_b = torch.arange(B, dtype=torch.int64, device=device)
    for i in range(num_log, num_phy):
        eff = weight / logcnt.float()
        top = eff.argmax(dim=-1)
        phy2log[:, i] = top
        rank[:, i] = logcnt[idx_b, top]
        logcnt[idx_b, top] += 1
    return phy2log, rank, logcnt


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Main entry point: hierarchical placement under a scarce replica budget.

    Stage 1: Pack expert groups across nodes (inter-node balancing)
    Stage 2: Ration replicas to popular experts within each node
    Stage 3: Pack physical replicas to GPUs (intra-node balancing)

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

    The hierarchy is preserved (replicas of an expert never leave its
    node, protecting locality); the quality of the result rests entirely
    on the three stage primitives above, which is where the variant's
    rationing logic belongs.
    """
    L, E = weight.shape
    weight = weight.float().cpu()
    group_size = E // num_groups
    gpus_per_node = num_gpus // num_nodes
    phy_per_gpu = num_replicas // num_gpus
    groups_per_node = num_groups // num_nodes
    experts_per_node = E // num_nodes
    replicas_per_node = num_replicas // num_nodes

    def inv(perm):
        out = torch.empty_like(perm)
        out.scatter_(1, perm, torch.arange(perm.size(1), dtype=torch.int64).expand(perm.shape))
        return out

    # Stage 1
    tpg = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    gpi, grk = balanced_packing(tpg, num_nodes)
    log2mlog = (((gpi * groups_per_node + grk) * group_size).unsqueeze(-1)
                + torch.arange(group_size)).flatten(-2)
    mlog2log = inv(log2mlog)

    # Stage 2
    tpm = weight.gather(-1, mlog2log).view(-1, experts_per_node)
    p2m, prk, mcnt = replicate_experts(tpm, replicas_per_node)

    # Stage 3
    tpp = (tpm / mcnt.float()).gather(-1, p2m)
    pi, ri = balanced_packing(tpp, gpus_per_node)
    p2pp = pi * phy_per_gpu + ri
    pp2p = inv(p2pp)

    pp2m = p2m.gather(-1, pp2p)
    pp2m = (pp2m.view(L, num_nodes, -1)
            + torch.arange(0, E, experts_per_node).view(1, -1, 1)).flatten(-2)
    pp2log = mlog2log.gather(-1, pp2m)
    pprank = prk.gather(-1, pp2p).view(L, -1)
    logcnt = mcnt.view(L, -1).gather(-1, log2mlog)

    mx = logcnt.max().item()
    log2phy = torch.full((L, E, mx), -1, dtype=torch.int64)
    log2phy.view(L, -1).scatter_(
        -1, pp2log * mx + pprank,
        torch.arange(num_replicas).expand(L, -1),
    )
    return pp2log, log2phy, logcnt
