**Problem.** Serving an MoE under expert parallelism, the per-layer latency is set by the most-loaded
GPU; live load is skewed and drifting. The placement must balance the max GPU load *and* the max node
load *and* keep each expert's replicas concentrated on one node (inter-node bandwidth is the
bottleneck), recomputed cheaply online. The starting fill is the established three-stage hierarchical
greedy bin-packing — the original DeepSeek EPLB reference algorithm.

**Key idea.** Three nested greedy stages, each a balanced (equal-cardinality) bin-pack, with a
replication step in the middle:
1. **Pack groups onto nodes (LPT-style).** Sum each group's load; greedily assign groups to nodes,
   each node getting exactly `num_groups / num_nodes`. Keeping whole groups on one node preserves the
   locality the node-limited routing assumes, so an expert's replicas never cross a node.
2. **Replicate hot experts within each node.** A single expert hotter than the per-GPU fair share
   cannot be balanced by any whole-item assignment, so make it divisible: with `num_phy − num_log`
   extra slots, each step adds a replica to the expert with the largest current per-replica load
   `weight / logcnt` — the move that attacks the current ceiling.
3. **Pack replicas onto GPUs within each node (LPT-style).** Greedily assign each node's replicas, by
   per-replica load, to that node's GPUs, each GPU getting exactly `num_replicas / num_gpus` slots.

**Why these choices.** Greedy makespan scheduling is near-optimal for an NP-hard problem — list
scheduling `≤ (2 − 1/m)·OPT`, largest-first LPT `≤ (4/3 − 1/(3m))·OPT` — so largest-first
least-loaded is the right packing instinct; the equal-slot hardware constraint restricts the greedy
choice to non-full packs (the bound motivates but does not prove the constrained variant).
Replication removes the hard floor a single over-hot expert would impose on the peak. The hierarchy,
rather than a flat global pack, keeps replicas node-local and yields locality for free. The packing
loops are inherently sequential (each choice reads the running loads), so they run on the CPU in plain
Python — exact and simple, but slow.

**Hyperparameters / contract.** No tunables. Each GPU hosts exactly `num_replicas // num_gpus`
physical experts; every logical expert keeps ≥ 1 replica; `logcnt.sum(-1) == num_replicas` per layer.
The literal scaffold edit (the default fill of `custom_eplb.py`, lines 62–209):

```python
def balanced_packing(weight: torch.Tensor, num_packs: int) -> Tuple[torch.Tensor, torch.Tensor]:
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
```
