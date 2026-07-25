Serving a large Mixture-of-Experts model under expert parallelism means every routed expert lives on its own GPU, and a token's top-K choices are shipped out through a synchronous all-to-all dispatch before the expert FFNs and gathered back by a combine. Because the combine waits for the slowest participant, per-layer latency is set by the most-loaded GPU, not the average. Real serving traffic is anything but uniform: a small set of hot experts attracts most tokens, and which experts are hot drifts as the input distribution changes. The hardware is also hierarchical, with GPUs grouped into nodes where fast NVLink inside a node and scarce InfiniBand between nodes mean that scattering a single expert's replicas across many nodes quietly explodes inter-node traffic. The placement problem, then, is to rebalance expert replicas across GPUs and nodes periodically from observed load estimates, trading off per-GPU balance, per-node balance, and node-locality, and to do it cheaply enough to run online every few minutes.

The standard levers do not solve this serving-time problem. Capacity factors and auxiliary load-balance losses were designed for training; at inference, dropping tokens is unacceptable and the auxiliary loss does not decide where physical expert weights live. Auxiliary-loss-free training balance keeps the training router even but offers no serving-time placement mechanism. Device- and node-limited routing bounds the communication footprint per token, but it is a routing constraint, not a placement algorithm, and it implicitly assumes a co-located expert layout that still has to be produced. Classical greedy makespan scheduling gets close for assigning indivisible items to identical machines, but its guarantees assume the item sizes are fixed; if one expert is hotter than the per-GPU fair share, no whole-expert assignment can balance the peak, and a flat global pack can scatter experts across nodes and defeat the locality that routing assumes.

The method is EPLB, the Expert Parallelism Load Balancer. It is a greedy, hierarchical bin-packing algorithm built around three ideas: largest-first least-loaded packing, replication of hot experts to make them divisible, and a nested layout that keeps every expert's replicas on a single node. The global variant used when there is no group/node structure to preserve is exactly the same procedure with the group and node counts collapsed to one.

EPLB works in three stages. First, it sums the load of each expert group and packs the groups onto nodes using a balanced largest-first least-loaded rule, so each node receives exactly the same number of groups and the per-node load is balanced. Keeping whole groups on one node guarantees that no expert's replicas will ever cross a node boundary, preserving the locality that node-limited routing counts on. Second, within each node, EPLB replicates the logical experts into that node's physical slots. Because a single over-hot expert cannot be balanced as a whole item, replicas make its effective load tunable: with a fixed budget of extra slots, each additional replica is given to the expert whose current per-replica load is largest, which is the only move that can lower the current ceiling. Third, the per-node replicas are packed onto that node's GPUs, again by largest-first least-loaded assignment restricted so that every GPU receives exactly the same number of slots. The per-replica load of each slot, computed as the expert's total load divided by its replica count, is what drives the final GPU-level packing.

The greedy choices are justified by the identical-machine makespan theory: arbitrary list scheduling is within a factor of two minus one over the number of machines of optimal, and sorting largest-first tightens the bound to four-thirds minus one over three machines. Replication is necessary because no whole-expert placement can do better than the largest single expert's load, and greedily targeting the largest per-replica load attacks the bottleneck directly. The hierarchy is necessary because a flat global pack can improve raw GPU balance while scattering replicas across nodes and inflating inter-node bandwidth; nesting replication and GPU packing inside each node gives locality for free.

```python
from typing import Tuple
import torch


def balanced_packing(weight: torch.Tensor, num_packs: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack n weighted items into num_packs packs of exactly n/num_packs items each,
    balancing per-pack weight (LPT greedy). Returns (pack_index, rank_in_pack)."""
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:                       # one item per pack: identity assignment
        pack_index = torch.arange(weight.size(-1), dtype=torch.int64,
                                  device=weight.device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    indices = weight.float().sort(-1, descending=True).indices.cpu()   # LPT: largest first
    pack_index = torch.full_like(weight, fill_value=-1, dtype=torch.int64, device='cpu')
    rank_in_pack = torch.full_like(pack_index, fill_value=-1)
    for i in range(num_layers):
        pack_weights = [0] * num_packs
        pack_items = [0] * num_packs
        for group in indices[i]:
            pack = min((p for p in range(num_packs) if pack_items[p] < groups_per_pack),
                       key=pack_weights.__getitem__)                   # least-loaded feasible pack
            assert pack_items[pack] < groups_per_pack
            pack_index[i, group] = pack
            rank_in_pack[i, group] = pack_items[pack]
            pack_weights[pack] += weight[i, group]
            pack_items[pack] += 1
    return pack_index, rank_in_pack


def replicate_experts(weight: torch.Tensor, num_phy: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replicate num_log experts into num_phy slots, greedily giving each extra slot to the
    expert with the largest current per-replica load w_i/count_i. Returns (phy2log, rank, logcnt)."""
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    device = weight.device
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)   # everyone starts with 1
    arangen = torch.arange(n, dtype=torch.int64, device=device)
    for i in range(num_log, num_phy):
        redundant_indices = (weight / logcnt).max(dim=-1).indices       # argmax per-replica load
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]
        logcnt[arangen, redundant_indices] += 1
    return phy2log, rank, logcnt


def rebalance_experts_hierarchical(weight: torch.Tensor, num_physical_experts: int,
                                   num_groups: int, num_nodes: int, num_gpus: int):
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0
    phy_experts_per_gpu = num_physical_experts // num_gpus

    def inverse(perm: torch.Tensor) -> torch.Tensor:
        inv = torch.empty_like(perm)
        inv.scatter_(1, perm, torch.arange(perm.size(1), dtype=torch.int64,
                                           device=perm.device).expand(perm.shape))
        return inv

    # Stage 1: pack groups onto nodes (per-node balance + locality)
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(tokens_per_group, num_nodes)
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) * group_size).unsqueeze(-1)
                + torch.arange(group_size, dtype=torch.int64,
                               device=group_pack_index.device)).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Stage 2: replicate hot experts within each node
    tokens_per_mlog = weight.gather(-1, mlog2log).view(-1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(tokens_per_mlog, num_physical_experts // num_nodes)

    # Stage 3: pack physical replicas onto GPUs within each node
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy, num_gpus // num_nodes)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(-1, pphy2phy)
    pphy2mlog = (pphy2mlog.view(num_layers, num_nodes, -1) +
                 torch.arange(0, num_logical_experts, num_logical_experts // num_nodes,
                              device=group_pack_index.device).view(1, -1, 1)).flatten(-2)
    pphy2log = mlog2log.gather(-1, pphy2mlog)
    pphyrank = phyrank.gather(-1, pphy2phy).view(num_layers, -1)
    logcnt = mlogcnt.view(num_layers, -1).gather(-1, log2mlog)
    return pphy2log, pphyrank, logcnt


def rebalance_experts(weight: torch.Tensor, num_replicas: int, num_groups: int,
                      num_nodes: int, num_gpus: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Entry point. Returns (phy2log [L, num_replicas], log2phy [L, E, max_rep], logcnt [L, E])."""
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()
    if num_groups % num_nodes == 0:                       # hierarchical policy (locality-aware)
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:                                                 # global policy = hierarchy with 1 group/node
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus)
    maxlogcnt = logcnt.max().item()
    log2phy = torch.full((num_layers, num_logical_experts, maxlogcnt),
                         -1, dtype=torch.int64, device=logcnt.device)
    log2phy.view(num_layers, -1).scatter_(
        -1, phy2log * maxlogcnt + phyrank,
        torch.arange(num_replicas, dtype=torch.int64, device=log2phy.device).expand(num_layers, -1))
    return phy2log, log2phy, logcnt


__all__ = ['rebalance_experts']
```
