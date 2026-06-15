# EPLB (Expert Parallelism Load Balancer), distilled

EPLB is the serving-time placement algorithm for an expert-parallel MoE model. Given an
estimate of each expert's token load and the cluster shape, it decides how many physical copies
(replicas) of each logical expert to create and which GPU each copy lands on, so that the
busiest GPU — which bottlenecks the synchronous all-to-all — is as light as possible. In the
hierarchical policy, it also keeps each expert's replicas co-located on a single node to bound
inter-node traffic. It is a greedy, hierarchical bin-packing built from two classical ideas plus
one new move: greedy makespan scheduling (LPT), and **replication** to make an over-hot expert
divisible.

## Problem it solves

Serving an MoE model under expert parallelism: each routed expert is an FFN on a GPU; a token's
top-K routing triggers an all-to-all dispatch/combine. The per-layer latency is set by the
**maximum** GPU load, and real traffic is heavily skewed and drifting, so a uniform layout
wastes the cluster. GPUs are grouped into nodes with fast intra-node (NVLink) and slow inter-node
(InfiniBand) links, so a placement must (1) balance the max GPU load, (2) balance the max node
load, and (3) keep an expert's replicas on as few nodes as possible. The plan re-runs online
from observed loads, so it must be cheap.

## Key idea

Three nested greedy stages, each a balanced bin-pack, with a replication step in the middle:

1. **Pack groups onto nodes (LPT).** Sum each expert group's load; greedily assign groups to
   nodes, each node getting exactly `num_groups/num_nodes` groups, balancing per-node load.
   Keeping whole groups on one node preserves the locality the group/node-limited routing
   assumes, so an expert's replicas never cross a node.
2. **Replicate hot experts within each node.** A single expert hotter than the per-GPU fair
   share cannot be balanced by any whole-item assignment, so make it divisible: give it `r`
   physical replicas, and if requests are spread across those copies, a replica carries about
   `w/r`. With a fixed budget of `num_phy − num_log` extra slots, hand them out greedily — each
   step, add a replica to the expert with the largest current **per-replica** load
   `w_i / count_i`, the move that attacks the current ceiling rather than a non-bottleneck
   replica.
3. **Pack replicas onto GPUs within each node (LPT).** Greedily assign each node's physical
   replicas, by per-replica load, to that node's GPUs, each GPU getting exactly `num_phy/num_gpus`
   slots.

The **global** policy (no group structure, used for large-EP decoding) is the same procedure
with `num_groups = num_nodes = 1`.

## Why greedy, and why these choices

- **LPT (sort descending, then least-loaded-feasible pack), not arbitrary order.** Greedy
  makespan scheduling is near-optimal for an NP-hard problem: list scheduling gives makespan
  `≤ (2 − 1/m)·OPT`, and sorting items largest-first (Longest Processing Time, Graham 1969)
  tightens it to `≤ (4/3 − 1/(3m))·OPT`. The list-scheduling proof charges the final machine's
  start time to average work. LPT improves the additive term because the critical last item is
  the smallest item in its prefix: if it is larger than `OPT/3`, all jobs in that prefix are too
  large for any optimum machine to hold three of them, and LPT is already optimal on that
  at-most-two-jobs prefix; otherwise the additive term is at most `(1 - 1/m)OPT/3`.
- **Equal item count per pack.** The hardware allocates a fixed number of physical slots per GPU
  (and groups per node), so the pack restricts the greedy choice to packs that are not yet full.
- **Replicate `argmax_i w_i/count_i`.** That is the per-replica load currently setting the
  ceiling; adding a replica elsewhere leaves that ceiling untouched, so this is the right
  myopic step toward equalizing per-replica load.
- **Hierarchy over a flat global pack.** A global pack may improve raw GPU balance while
  scattering an expert's replicas across nodes, inflating the inter-node all-to-all the routing
  was designed to bound; nesting replication and GPU-packing *inside* each node keeps replicas
  node-local and yields locality for free.
- **CPU + Python loops.** The greedy "least-loaded feasible pack" decision depends on the
  running loads of all prior placements, so it is inherently sequential; it is kept exact and
  simple at the cost of speed.

## Final algorithm

Three return maps, per layer `L` over `E` logical experts and `num_replicas` physical slots:
`phy2log` `[L, num_replicas]` (logical expert per slot), `log2phy` `[L, E, max_rep]` (slots per
expert, `-1` padded), `logcnt` `[L, E]` (replica count per expert). Constraints: `E % num_groups
== 0`, `num_groups % num_nodes == 0`, `num_gpus % num_nodes == 0`, `num_replicas % num_gpus ==
0`; each GPU hosts exactly `num_replicas // num_gpus` slots; every logical expert keeps ≥ 1
replica; `logcnt` sums to `num_replicas` per layer.

## Working code

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

## Approximation guarantee it borrows from

The ordering comes from the classical identical-machine results: arbitrary list scheduling is
within `2 − 1/m` of optimal, and unconstrained LPT improves that to `4/3 − 1/(3m)`. EPLB adds a
hard equal-slot constraint (`n/m` items per pack), so those ratios motivate the largest-first
least-loaded choice but are not a proof for the constrained placement. Replication removes the
hard floor that any single item above the fair share would otherwise impose on the max, and the
hierarchical branch trades some pure GPU-balance freedom for node locality (an expert's replicas
stay on one node).
