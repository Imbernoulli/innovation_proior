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
    balancing per-pack weight by largest-first least-loaded assignment."""
    B, n = weight.shape
    assert n % num_packs == 0
    items_per_pack = n // num_packs

    if items_per_pack == 1:
        idx = torch.arange(n, dtype=torch.int64, device=weight.device).expand(B, -1)
        return idx, torch.zeros_like(idx)

    sorted_idx = weight.float().sort(-1, descending=True).indices.cpu()
    pack_index = torch.full((B, n), -1, dtype=torch.int64)
    rank_in_pack = torch.full((B, n), -1, dtype=torch.int64)
    for b in range(B):
        loads = [0.0] * num_packs
        counts = [0] * num_packs
        for j in range(n):
            item = sorted_idx[b, j].item()
            best = min(
                (p for p in range(num_packs) if counts[p] < items_per_pack),
                key=lambda p: loads[p],
            )
            pack_index[b, item] = best
            rank_in_pack[b, item] = counts[best]
            loads[best] += weight[b, item].item()
            counts[best] += 1
    return pack_index, rank_in_pack


def replicate_experts(
    weight: torch.Tensor, num_phy: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replicate num_log experts into num_phy slots, greedily giving each extra slot to the
    expert with the largest current per-replica load w_i / count_i."""
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
    """Return (phy2log [L, num_replicas], log2phy [L, E, max_rep], logcnt [L, E])."""
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

    # Stage 1: pack groups onto nodes.
    tpg = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    gpi, grk = balanced_packing(tpg, num_nodes)
    log2mlog = (((gpi * groups_per_node + grk) * group_size).unsqueeze(-1)
                + torch.arange(group_size)).flatten(-2)
    mlog2log = inv(log2mlog)

    # Stage 2: replicate hot experts within each node.
    tpm = weight.gather(-1, mlog2log).view(-1, experts_per_node)
    p2m, prk, mcnt = replicate_experts(tpm, replicas_per_node)

    # Stage 3: pack replicas onto GPUs within each node.
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
```
