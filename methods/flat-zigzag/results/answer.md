# Flat Zigzag Expert Placement

Flat zigzag is the explicit global branch of DeepSeek's Expert Parallelism Load Balancer with the sequential greedy packing loop replaced by a vectorized serpentine packing heuristic. It keeps EPLB's redundant-expert apportionment and output maps, but skips the group-to-node hierarchy: replicate experts globally, then pack all physical replicas across all GPUs.

## Method

1. **Replicate hot experts.** Start every logical expert with one copy. For each redundant slot, add a copy to the expert with the largest current effective load `weight / logcnt`. Under the even-split model for traffic among copies, this is the discrete water-filling/apportionment step for minimizing the maximum per-replica load.

2. **Pack physical replicas.** Compute each physical slot's load as `(weight / logcnt).gather(-1, phy2log)`. The original EPLB packing primitive is modified LPT: sort heaviest first and repeatedly choose the least-loaded pack with remaining capacity. The classical LPT bound is `4/3 - 1/(3m)` for identical-machine makespan; the vectorized zigzag replacement is a heuristic, not that theorem.

3. **Use the serpentine index pattern.** After sorting, let `pos` be the sorted position and `P = num_packs`. Set `block_id = pos // P`, `pos_in_block = pos % P`, `pack = pos_in_block` on even blocks, and `pack = P - 1 - pos_in_block` on odd blocks. The rank inside a pack is `block_id`. This gives exactly one item per pack per block and scatters the assignment back through `sorted_idx` to original item order.

4. **Relabel into GPU order.** Convert `(pack, rank_in_pack)` to `phy2pphy[p] = pack_index[p] * phy_per_gpu + rank_in_pack[p]`, invert it to `pphy2phy[q] = p`, gather `phy2log` and replica ranks through that inverse, then scatter packed slot `q` to `log2phy[e, r]` with flattened index `e * max_rep + r`.

## Code

```python
from typing import Tuple
import torch


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

## Relation To Hierarchical EPLB

Hierarchical EPLB first packs expert groups to nodes, then replicates and packs within each node. The official entry point uses that hierarchy when `num_groups % num_nodes == 0`; otherwise it falls back to `rebalance_experts_hierarchical(weight, num_replicas, 1, 1, num_gpus)`, which makes the group-to-node relabeling the identity and runs replication and packing once over the whole model. Flat zigzag makes that global path explicit and changes only the packing primitive from running-load greedy to the rank-only serpentine assignment.
