I have a Mixture-of-Experts model to serve, and the thing that immediately matters is not the average expert load. The layer waits for the busiest GPU. If one GPU holds a cluster of hot experts, the dispatch, expert compute, and combine path all wait for that GPU while the others sit idle. The load vector I have is already estimated per layer and per logical expert. I am not trying to predict it here. I am trying to turn it into a physical placement that keeps the maximum GPU load small, while still respecting the fixed number of expert slots each GPU can hold.

The first wall is indivisibility. If one logical expert receives more tokens than a fair GPU share, no rearrangement of single copies can hide that load. Wherever that expert sits, its GPU is heavy. So I need to split that expert's traffic across physical replicas. If expert `e` has load `w_e` and I give it `c_e` copies, then the effective load of one physical copy is `w_e / c_e`, assuming the dispatch chooses among copies evenly. That makes replication the first decision: choose integer counts `c_e >= 1`, with the counts summing to the physical slot budget.

After that, I still have to place the physical copies. Every GPU has exactly `num_replicas // num_gpus` slots, so this is not open-ended bin packing. It is an equal-cardinality partition: split weighted physical replicas into `num_gpus` packs, exactly the same number of items in each pack, and make the largest pack sum as small as possible.

The obvious packing rule is the scheduling rule I already trust: sort items heaviest first, then place each item on the least-loaded pack that still has room. I need to be precise about the theorem attached to that intuition. For ordinary identical-machine makespan, Longest Processing Time has the `4/3 - 1/(3m)` worst-case factor on `m` machines. My serving routine has a hard per-pack cardinality cap, so the code is only LPT-like: it keeps the heaviest-first order and restricts choices to packs with remaining slots. The classical constant is the right mental anchor for why heaviest-first greedy is strong, but I do not carry it over to every cardinality-constrained variant, and I certainly do not carry it over to a replacement that stops looking at running loads. What the greedy gives me operationally is clear: it places the big items while there is still freedom, then lets small items fill the gaps.

Written directly, that primitive is simple.

```python
def balanced_packing(weight, num_packs):
    B, n = weight.shape
    assert n % num_packs == 0
    items_per_pack = n // num_packs
    sorted_idx = weight.float().sort(-1, descending=True).indices
    pack_index = torch.full((B, n), -1, dtype=torch.int64)
    rank_in_pack = torch.full((B, n), -1, dtype=torch.int64)
    for b in range(B):
        loads = [0.0] * num_packs
        counts = [0] * num_packs
        for j in range(n):
            item = sorted_idx[b, j].item()
            best = min((p for p in range(num_packs) if counts[p] < items_per_pack),
                       key=lambda p: loads[p])
            pack_index[b, item] = best
            rank_in_pack[b, item] = counts[best]
            loads[best] += weight[b, item].item()
            counts[best] += 1
    return pack_index, rank_in_pack
```

This has the right shape mathematically and the wrong shape computationally. The assignment of item `j` depends on the pack loads after items `0..j-1`. That running state is the whole point of the greedy, so there is no clean tensor expression for the exact rule. I can batch the input tensor, but I cannot batch away the dependency that chooses the least-loaded nonfull pack after every item. The Python loop over items, with a scan over packs inside it, is the cost center.

So I ask what I actually need from the packing step. I need equal pack sizes and a good high-low interleaving after sorting by load. I do not necessarily need the exact least-loaded choice. If the assignment depends only on the sorted rank of an item, then all the work becomes index arithmetic. The danger is plain round-robin. With `P` packs, round 0 sends sorted positions `0,1,...,P-1` to packs `0,1,...,P-1`; round 1 sends positions `P,P+1,...,2P-1` to the same pack order. Pack 0 then receives the heaviest item of every round, while pack `P-1` receives the lightest item of every round. That bakes in imbalance.

The fix is to reverse direction every round. In the first block of `P` sorted items I deal left to right. In the second block I deal right to left. Then pack 0 receives the heaviest item of block 0 but the lightest item of block 1; pack `P-1` receives the lightest item of block 0 but the heaviest item of block 1. The same pairing continues across blocks. This is not the LPT greedy anymore, because it never observes the running pack sums. It is a rank-only serpentine heuristic whose purpose is to keep the heaviest-of-round advantage from accumulating on the same pack.

Now the index math is exact. Let `pos` be a sorted position. The block, or round, is `block_id = pos // num_packs`. The offset inside the block is `pos_in_block = pos % num_packs`. On even blocks the pack is `pos_in_block`; on odd blocks it is `num_packs - 1 - pos_in_block`. Since every block contributes exactly one item to every pack, the slot rank inside the pack is just `block_id`. Because `n` is divisible by `num_packs`, there are exactly `n // num_packs` blocks, so every pack has exactly that many items.

The only remaining detail is orientation. `sorted_idx[b, pos]` maps a sorted position back to the original item id. The caller wants arrays indexed by original item id, so I scatter the rank-based assignments through `sorted_idx`.

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
```

The `n // num_packs == 1` case is worth keeping separate. If there is one item per pack, any load-balancing choice is just a permutation with rank zero. The implementation returns the identity mapping there, which is exactly what the downstream linear slot calculation expects.

Now I need the replica counts. Under the even-split model for traffic among copies, the count-only objective is `min max_e w_e / c_e` subject to integer `c_e >= 1` and a fixed total count. The greedy rule is to start every expert with one copy and give each additional copy to the expert whose current `w_e / c_e` is largest. I can see why that is the right apportionment rule by imagining the decreasing quotient streams `w_e/1, w_e/2, w_e/3, ...`. Giving an extra copy to expert `e` consumes the current quotient `w_e/c_e` and exposes the next one. To push the final maximum below a threshold `T`, I must have consumed every quotient above `T`. Consuming the largest currently exposed quotient at every step is therefore the discrete water-filling rule for this bottleneck objective: after a fixed number of extra copies, no other allocation can have a smaller maximum exposed quotient without also consuming a quotient that this allocation leaves exposed.

The code follows that directly. The first `num_log` physical slots are the mandatory original copies. The later slots are overwritten one by one with the selected logical expert id. `rank` stores which copy number this slot is, with rank zero already used by the original.

```python
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
```

This loop remains sequential over the redundant slots because each new copy changes `logcnt`, but it is batched over layers and it is not the old per-item pack-selection bottleneck. The costly part was the repeated least-loaded-bin search; the replication loop is the intended apportionment step.

Now I have to decide how much hierarchy to preserve. If I am optimizing a topology-aware placement, I can first pack expert groups to nodes, then replicate within each node, then pack replicas to GPUs inside that node. That keeps a routed group, and hence all replicas of its experts, on one node. It is the locality-preserving route. The global policy is the official fallback when the group-to-node hierarchy is disabled: it calls the same hierarchical routine with `num_groups=1` and `num_nodes=1`, so the group-to-node relabeling becomes the identity and the node-local subproblem becomes the whole model. I can write that collapsed path directly by replicating across all logical experts at once and packing all physical replicas across all GPUs at once. This gives up the guarantee that one expert's replicas stay on a single node, but it also removes the group-to-node stage and gives the packing step the whole GPU pool.

The global assembly is short. I move the load tensor to CPU float, because all subsequent index tensors are CPU in this implementation. I compute the number of physical slots per GPU. I replicate globally to `num_replicas`. Then the load of each physical slot is `weight / logcnt` gathered through `phy2log`, because `phy2log` says which logical expert each physical slot represents. Finally I run the rank-only packing over all physical slots and all GPUs.

```python
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
```

The packing output is still indexed by the original physical-slot order. `pack_index[p]` is the GPU chosen for physical slot `p`, and `rank_in_pack[p]` is that slot's position inside the GPU. The GPU-ordered linear slot id is therefore `pack_index * phy_per_gpu + rank_in_pack`. I call that `phy2pphy`: original physical slot to packed physical slot. What I need for the returned `phy2log` is the inverse: for each packed slot, which original physical slot landed there? A permutation inverse is a scatter. If `perm[p] = q`, write `out[q] = p`.

```python
    def inv(perm):
        out = torch.empty_like(perm)
        out.scatter_(1, perm, torch.arange(perm.size(1), dtype=torch.int64).expand(perm.shape))
        return out

    phy2pphy = pack_index * phy_per_gpu + rank_in_pack
    pphy2phy = inv(phy2pphy)

    final_phy2log = phy2log.gather(-1, pphy2phy)
    final_rank = phyrank.gather(-1, pphy2phy)
```

Now `final_phy2log[q]` is the logical expert at packed slot `q`, and `final_rank[q]` is which copy of that logical expert it is. Reading `final_phy2log` in contiguous chunks of `phy_per_gpu` gives the per-GPU expert slots. Equal cardinality is already guaranteed by the packing construction: every pack receives exactly one item at each block id, so the linear ids cover exactly one block of `phy_per_gpu` positions per GPU.

The last return value is the reverse lookup `log2phy`. Its third dimension must be wide enough for the most-replicated expert, so I take `mx = logcnt.max().item()`. For a packed slot `q`, with logical expert `e = final_phy2log[q]` and copy rank `r = final_rank[q]`, the flattened destination inside `[E, mx]` is `e * mx + r`. Scattering `q` to that destination gives `log2phy[e, r] = q`; positions for nonexistent ranks remain `-1`.

```python
    mx = logcnt.max().item()
    log2phy = torch.full((L, E, mx), -1, dtype=torch.int64)
    log2phy.view(L, -1).scatter_(
        -1, final_phy2log * mx + final_rank,
        torch.arange(num_replicas).expand(L, -1),
    )
    return final_phy2log, log2phy, logcnt
```

All legality checks are now consequences of the construction, assuming the usual valid call where `num_replicas >= E` and `num_replicas` is divisible by `num_gpus`. Every logical expert starts with one copy, and replication only adds copies, so counts are positive. The loop adds exactly `num_replicas - E` extra copies, so `logcnt.sum(-1) == num_replicas`. The packing creates exactly one item per pack per block, so each GPU receives exactly `phy_per_gpu` slots. The inverse permutation only reorders physical slots, and the `log2phy` scatter uses `(logical id, copy rank)` pairs already present in the packed layout.

Putting the pieces together, the final routine is the explicit global load-balancing branch with one substitution in the packing primitive: keep the same redundant-expert apportionment and the same packed-slot/logical-slot bookkeeping, but replace the sequential least-loaded-with-room packing loop by the rank-only serpentine assignment.

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
