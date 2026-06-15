# Zigzag (snake) balanced packing for hierarchical MoE expert placement, distilled

Zigzag replaces the sequential greedy bin-packing inside a three-stage hierarchical MoE
expert-placement algorithm with a vectorized "snake" (boustrophedon) assignment. Items sorted
by load are dealt to packs in alternating direction round by round — 0,1,…,P−1 then
P−1,…,1,0 — which pairs the heavy end of one round with the light end of the next. When the
sorted load sequence keeps the packs in that alternating emptiest-bin order, this fixed
positional pattern is Longest-Processing-Time greedy written in closed form; on the smoothed
expert and per-replica load sequences used by the hierarchy, it is the cheap positional
approximation to that behavior. It keeps node locality by leaving the hierarchy intact and
replaces the per-item Python packing loop with one sort, simple index arithmetic, and two
`scatter_`s.

## Problem it solves

During MoE inference under expert parallelism, the skewed and drifting per-expert token load is
periodically rebalanced by recomputing an expert replication-and-placement plan. The plan must
balance load across GPUs and across nodes, keep each expert's replicas co-located on one node
(inter-node bandwidth is the bottleneck), and — because it runs online on the critical path — be
cheap. The established hierarchical algorithm meets the first three but its bin-packing core is a
nested Python loop, costing ~540 ms per rebalance.

## Key idea

The bin-packing subproblem is balanced (equal-cardinality) multiway number partitioning: split n
weighted items into P packs of exactly n/P items, minimizing the spread of pack sums. The greedy
heuristic (LPT-style: sort descending, assign each item to the emptiest pack with free capacity) is
inherently sequential — item j's pack depends on the running loads after items 1..j−1 — so it
cannot be vectorized.

The move is to find an assignment that depends only on each item's **sorted position**, not on
running state. Plain round-robin (rank r → pack r mod P) is positional and vectorizable but
overloads pack 0, because every round of P items runs in the same direction and pours its
heaviest element into the lowest-index pack; the imbalances stack. Reversing every other round —
the **snake** — addresses this: even rounds sweep packs 0→P−1, odd rounds sweep P−1→0, so each
pack alternately receives the heavy end and the light end of successive rounds, and the spreads
tend to cancel within each pack.

Why it matches greedy only under the right sorted-sequence condition: after the first forward
round, pack P−1 is the emptiest, so LPT gives the next item to pack P−1. The whole odd round
continues in reverse only if each just-filled low-load pack climbs above the next candidate:
after pack P−1 receives rank P, pack P−2 must be the emptiest; after pack P−2 receives rank
P+1, pack P−3 must be the emptiest; and so on. Locally affine descending block pairs satisfy
this exactly, because `w[p] + w[2P−1−p]` is constant across p, so a forward-plus-backward pair
equalizes the pack sums and the canonical greedy tie-break can start the next forward round.
Smooth expert-load sequences approximate that alternating emptiest-bin order, especially in
Stage 3 after replication has shaved the peaks. On arbitrary unsorted weights, or sorted weights
with cliffs or strong curvature, greedy can diverge because it reads the running loads and the
snake does not.

Cardinality comes for free: with n divisible by P there are n/P rounds, each dealing one item to
each pack, so every pack ends with exactly n/P items — no capacity filter needed. The
within-pack rank is just the round number.

## Where it is applied

The three-stage hierarchy is unchanged; only `balanced_packing` is replaced:
- **Stage 1** — snake-pack routing groups onto nodes (balances node load, keeps a group node-local).
- **Stage 2** — replicate hot experts within each node: a genuinely sequential argmax loop
  (each extra replica goes to the expert with the largest per-replica load `weight/logcnt`,
  which changes as copies are added), but it iterates only over the *redundant* count and is not
  the bottleneck, so it stays a loop.
- **Stage 3** — snake-pack the per-replica loads onto the GPUs within each node.

Applying the snake to *both* packing stages is what delivers the speedup: Stage 3 packs more items than
Stage 1, so leaving it as a loop would leave most of the Python cost in place.

## Complexity

Greedy `balanced_packing`: L·n·P interpreted iterations (outer over layers, inner over items,
each an O(P) min-scan). Snake `balanced_packing`: one batched sort plus O(1) tensor kernels
(arange / where / two scatters) whose count is independent of n. The hierarchy still determines
locality; the change is that the packing assignments are constructed in bulk instead of by a
host-language item loop.

## Working code

```python
from typing import Tuple
import torch


def balanced_packing(weight: torch.Tensor, num_packs: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Partition each row's n items into num_packs packs of exactly n // num_packs items,
    balancing pack sums, via the vectorized snake (boustrophedon) pattern. No per-item loop."""
    B, n = weight.shape
    assert n % num_packs == 0

    if n // num_packs == 1:                           # one item per pack -> identity
        idx = torch.arange(n, dtype=torch.int64, device=weight.device).expand(B, -1)
        return idx, torch.zeros_like(idx)

    sorted_idx = weight.float().sort(-1, descending=True).indices   # heaviest first (LPT order)

    positions    = torch.arange(n, device=weight.device)
    block_id     = positions // num_packs             # round index
    pos_in_block = positions % num_packs              # offset within the round
    is_even      = block_id % 2 == 0
    # even rounds sweep 0..P-1, odd rounds sweep P-1..0  -> heavy/light pairing per pack
    pack_assign  = torch.where(is_even, pos_in_block, num_packs - 1 - pos_in_block)
    rank_assign  = block_id                           # within-pack slot = round number

    # scatter per-sorted-position assignments back to original item order
    pack_expanded = pack_assign.unsqueeze(0).expand(B, -1)
    rank_expanded = rank_assign.unsqueeze(0).expand(B, -1)
    pack_index    = torch.zeros(B, n, dtype=torch.int64, device=weight.device)
    rank_in_pack  = torch.zeros(B, n, dtype=torch.int64, device=weight.device)
    pack_index.scatter_(-1, sorted_idx, pack_expanded)
    rank_in_pack.scatter_(-1, sorted_idx, rank_expanded)

    return pack_index.cpu(), rank_in_pack.cpu()


def replicate_experts(weight: torch.Tensor, num_phy: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Grow num_log experts to num_phy replicas, minimizing the max per-replica load. Each
    extra replica goes to the expert with the largest current per-replica load (weight/count).
    Sequential in the replica draws, but only num_phy - num_log iterations."""
    B, num_log = weight.shape
    device = weight.device
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(B, 1)
    rank = torch.zeros(B, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(B, num_log, dtype=torch.int64, device=device)
    idx_b = torch.arange(B, dtype=torch.int64, device=device)
    for i in range(num_log, num_phy):
        eff = weight / logcnt.float()                 # per-replica load
        top = eff.argmax(dim=-1)                      # most-overloaded expert
        phy2log[:, i] = top
        rank[:, i] = logcnt[idx_b, top]
        logcnt[idx_b, top] += 1
    return phy2log, rank, logcnt


def rebalance_experts(weight: torch.Tensor, num_replicas: int, num_groups: int,
                      num_nodes: int, num_gpus: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Three-stage hierarchical placement with snake packing in Stages 1 and 3.

    Args:
        weight:       [L, E] token load per expert per layer
        num_replicas: total physical expert slots (multiple of num_gpus)
        num_groups:   number of routing groups (divisor of E)
        num_nodes:    number of server nodes
        num_gpus:     total GPUs (multiple of num_nodes)
    Returns:
        phy2log: [L, num_replicas]; log2phy: [L, E, max_rep] (-1 = unused); logcnt: [L, E]
    """
    L, E = weight.shape
    weight = weight.float().cpu()
    group_size = E // num_groups
    gpus_per_node = num_gpus // num_nodes
    phy_per_gpu = num_replicas // num_gpus
    groups_per_node = num_groups // num_nodes
    experts_per_node = E // num_nodes
    replicas_per_node = num_replicas // num_nodes

    def inv(perm):                                     # per-row inverse permutation
        out = torch.empty_like(perm)
        out.scatter_(1, perm, torch.arange(perm.size(1), dtype=torch.int64).expand(perm.shape))
        return out

    # Stage 1: snake-pack groups onto nodes
    tpg = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    gpi, grk = balanced_packing(tpg, num_nodes)
    log2mlog = (((gpi * groups_per_node + grk) * group_size).unsqueeze(-1)
                + torch.arange(group_size)).flatten(-2)
    mlog2log = inv(log2mlog)

    # Stage 2: replicate hot experts within each node
    tpm = weight.gather(-1, mlog2log).view(-1, experts_per_node)
    p2m, prk, mcnt = replicate_experts(tpm, replicas_per_node)

    # Stage 3: snake-pack the per-replica loads onto GPUs within each node
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
        -1, pp2log * mx + pprank, torch.arange(num_replicas).expand(L, -1))
    return pp2log, log2phy, logcnt
```
