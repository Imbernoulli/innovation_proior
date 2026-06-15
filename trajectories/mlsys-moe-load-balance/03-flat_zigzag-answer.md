**Problem.** The vectorized hierarchical zigzag fixed runtime (now ~1 ms) and held perfect locality,
but its balance is capped by **Stage 1**: greedy and snake score identical balance_node because the
group-to-node assignment, not the packing rule, is the bottleneck. With `groups_per_node = 1`
(deepseek-v3) Stage 1 has no freedom; with `groups_per_node = 2` and extreme skew (stress-skew) it has
almost none, so balance is stuck at 0.659 and 0.222 and balance_node at 0.702 and 0.336. The hierarchy
buys perfect locality by spending balance.

**Key idea.** Drop the node hierarchy and do a single **global** pass — the established global policy,
here taken as the explicit branch. Replicate over *all* logical experts at once, then snake-pack *all*
physical replicas directly across *all* GPUs, ignoring group and node structure. The packer now sees
the whole problem: a heavy replica can be balanced against a light slot anywhere in the cluster, not
just within its inherited node, so the node-level floor on the achievable max-GPU load disappears.

**Why.** The zigzag feedback localized the cap to the hierarchy (identical balance_node across two
different packers ⇒ the cap is upstream of packing, in Stage 1). The cheapest test is to remove the
hierarchy and give the packer the full GPU pool. The zigzag packing rule is kept verbatim — the runtime
lesson stands — and run globally it is still cheap. The trade is explicit: locality is no longer
guaranteed, because an expert's replicas can scatter across nodes, so `locality` falls off 1.0 (toward
`1/num_nodes` for the heavily-replicated hot experts). It is a net win only where the balance gain on
the strangled configs outweighs the bounded locality loss; low-replica experts stay node-local, so
locality holds in the low-0.90s on the real configs and drops hardest on the 16-node stress config. The
replication water-filling and the packed-slot/logical-slot bookkeeping are unchanged; `num_groups` and
`num_nodes` are accepted but not consulted on this path.

**Hyperparameters / contract.** No tunables; same validity contract (exactly `num_replicas //
num_gpus` per GPU via the snake's one-item-per-pack-per-round structure, ≥ 1 replica per expert,
`logcnt` sums to `num_replicas`). The literal scaffold edit (`custom_eplb.py`, lines 62–209):

```python
def balanced_packing(weight: torch.Tensor, num_packs: int) -> Tuple[torch.Tensor, torch.Tensor]:
    B, n = weight.shape
    assert n % num_packs == 0

    if n // num_packs == 1:
        idx = torch.arange(n, dtype=torch.int64, device=weight.device).expand(B, -1)
        return idx, torch.zeros_like(idx)

    sorted_idx = weight.float().sort(-1, descending=True).indices

    positions = torch.arange(n, device=weight.device)
    block_id = positions // num_packs
    pos_in_block = positions % num_packs
    is_even = block_id % 2 == 0
    pack_assign = torch.where(is_even, pos_in_block, num_packs - 1 - pos_in_block)
    rank_assign = block_id

    pack_expanded = pack_assign.unsqueeze(0).expand(B, -1)
    rank_expanded = rank_assign.unsqueeze(0).expand(B, -1)
    pack_index = torch.zeros(B, n, dtype=torch.int64, device=weight.device)
    rank_in_pack = torch.zeros(B, n, dtype=torch.int64, device=weight.device)
    pack_index.scatter_(-1, sorted_idx, pack_expanded)
    rank_in_pack.scatter_(-1, sorted_idx, rank_expanded)

    return pack_index.cpu(), rank_in_pack.cpu()


def replicate_experts(
    weight: torch.Tensor, num_phy: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    # Flat (non-hierarchical) approach: skip group-to-node, go directly to global
    L, E = weight.shape
    weight = weight.float().cpu()
    phy_per_gpu = num_replicas // num_gpus

    # Step 1: Replicate experts globally
    phy2log, phyrank, logcnt = replicate_experts(weight, num_replicas)

    # Step 2: Pack all replicas to GPUs directly using zigzag
    tokens_per_phy = (weight / logcnt.float()).gather(-1, phy2log)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy, num_gpus)

    def inv(perm):
        out = torch.empty_like(perm)
        out.scatter_(1, perm, torch.arange(perm.size(1), dtype=torch.int64).expand(perm.shape))
        return out

    phy2pphy = pack_index * phy_per_gpu + rank_in_pack
    pphy2phy = inv(phy2pphy)

    final_phy2log = phy2log.gather(-1, pphy2phy)
    final_rank = phyrank.gather(-1, pphy2phy)

    mx = logcnt.max().item()
    log2phy = torch.full((L, E, mx), -1, dtype=torch.int64)
    log2phy.view(L, -1).scatter_(
        -1, final_phy2log * mx + final_rank,
        torch.arange(num_replicas).expand(L, -1),
    )
    return final_phy2log, log2phy, logcnt
```
