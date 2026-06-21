The vectorized hierarchical zigzag did exactly what I asked: runtime collapsed from the 100–256 ms range
to 1.6/0.85/1.2/1.6 ms, locality held at exactly 1.000 on all four configs, and balance came in where I
predicted — qwen3-moe 0.919, deepseek-v2 0.906, deepseek-v3 0.659, stress-skew pinned at 0.222. The task
score jumped from 0.255 to 0.375, almost all of it the removed runtime tax. The runtime wound is closed,
and the score is now bottlenecked by *balance*: mediocre on deepseek-v3 (0.659) and catastrophic on
stress-skew (0.222), dragging the geometric mean down hard while the other two configs sit near 0.92. So
the question for this rung is not speed; it is what is capping the balance. And the numbers point
precisely: greedy and zigzag, two *different* packers, score *identical* balance_node — 0.702 on
deepseek-v3, 0.336 on stress-skew. That is the tell. The bottleneck is not the packing rule at all; it is
upstream, in **Stage 1**, the group-to-node assignment. With $\text{groups\_per\_node} = 1$ on
deepseek-v3, Stage 1 has *no freedom whatsoever* — one group per node, the assignment forced, and whatever
node-level imbalance the eight group loads happen to have is locked in before replication or GPU packing
begin. On stress-skew it is worse: two wildly-skewed groups crammed two-to-a-node across 16 nodes. The
hierarchy is *buying* the perfect locality by *spending* balance — by insisting every group, and every
expert's replicas, stay whole on one node, it hands Stage 1 a rigid, low-freedom partition and then asks
the later stages to balance within a structure that is already lopsided.

I propose **flat (global) zigzag packing**: drop the node hierarchy entirely and do a single global pass.
This is not a new algorithm — it is the established global policy, the official fallback the hierarchical
entry point uses when the group-to-node divisibility is absent, the same procedure with the group and node
counts collapsed to one giant pool. Here I choose it *deliberately*, as the explicit branch even though
the hierarchy is available, to test the hypothesis that the hierarchy is what caps balance. Replicate over
*all* logical experts at once, then snake-pack *all* the physical replicas directly across *all* the GPUs
in the cluster, ignoring group and node structure. The reason it should balance better is that the packer
now sees the whole problem at once. In the hierarchy, Stage 3 packs only one node's
$\text{replicas\_per\_node}$ items onto that node's $\text{gpus\_per\_node}$ GPUs — a small, isolated
subproblem with whatever lopsided load the node inherited from Stage 1. Flat, the packer sees all
$\text{num\_replicas}$ items and all $\text{num\_gpus}$ packs together, so a heavy replica on what would
have been an overloaded node can be balanced against a light slot *anywhere* in the cluster. The single
largest constraint on the achievable max-GPU load — that each GPU's load is bounded below by the average
of its node's inherited load — simply disappears. The zigzag packing rule itself I keep verbatim, because
the runtime lesson stands: it is the vectorized snake, one sort and index arithmetic, no per-item loop,
and run globally over all replicas and all GPUs it is still cheap.

The construction is simpler than the hierarchy, and the maps still come out right. Replicate first: start
every logical expert with one copy and hand each of the $\text{num\_replicas} - E$ extra slots to the
expert with the largest current per-replica load $\text{weight}/\text{logcnt}$ — the same water-filling
rule as the hierarchical Stage 2, now over all $E$ experts at once rather than per node. It is still the
genuinely-sequential argmax loop, but only over the redundant count, so it is not the cost center. Then
each physical slot's load is $\text{weight}/\text{logcnt}$ gathered through `phy2log` (which logical expert
each slot represents), and I snake-pack those $\text{num\_replicas}$ per-replica loads across
$\text{num\_gpus}$ packs in one shot. The packing returns, per slot, its GPU and its rank within that GPU;
the GPU-ordered linear slot id is $\text{pack\_index}\cdot\text{phy\_per\_gpu} + \text{rank\_in\_pack}$, I
invert that permutation to get which original slot landed at each packed position, gather `phy2log` and the
replica ranks through the inverse, and scatter the packed slot ids into `log2phy[e, r]` sized by the max
replica count. Equal cardinality is automatic — the snake puts exactly one item per pack per round, so
each GPU gets exactly $\text{phy\_per\_gpu}$ slots — and every expert keeps $\ge 1$ copy because
replication only adds. The `num_groups` and `num_nodes` arguments are still accepted by the entry point,
but the flat path simply does not consult them; the group-to-node relabeling Stage 1 used is gone.

The cost is the metric I am about to pay. `locality` counts, per expert, how many distinct nodes hold its
replicas and credits $1/\text{nodes\_per\_expert}$, traffic-weighted. The hierarchy scored a perfect 1.000
everywhere because every expert was confined to one node by construction; the flat pack has no such
confinement and scatters an expert's replicas across whatever GPUs the snake assigns, which can sit on
different nodes. A hot expert with many replicas, spread across the cluster for balance, will touch
several nodes, so its locality drops toward $1/\text{num\_nodes}$. This is a genuine trade, and whether it
nets out depends on the arithmetic of the four equally-weighted terms. I expect it to win because the
hierarchy's balance deficits are *large and concentrated* — 0.659 and 0.222 on two configs — while the
locality I give up is *bounded*: replication factors are modest (the budget is roughly 1.25–2×), so most
experts have one or two replicas, and a one- or two-replica expert can only touch one or two nodes, keeping
its locality high; only the heavily-replicated hot experts scatter, and they are a minority of the
traffic-weighted mass. So I expect locality to stay in the low-to-mid 0.90s on the real configs — a few
points lost, not collapsed — while balance climbs from the 0.66–0.92 range up toward the high 0.97s. On
stress-skew the trade is steeper: 16 nodes puts the floor at $1/16 = 0.0625$, the replication budget is
tightest (1.5×), and the skew is extreme, so the hot experts get the most replicas and scatter widest —
locality there should fall hardest, into the low 0.7s, but balance should climb from the floored 0.222 up
toward the low 0.9s, a far larger swing on a config where balance was the catastrophe.

I want to be honest about the seam, because it is where this rung could be beaten later. By dropping the
hierarchy I have *decoupled* the two objectives the hierarchy fused: the hierarchy got locality for free
and paid in balance; the flat pack gets balance and pays in locality. Neither does the thing the task
actually rewards, which is both at once — a method that kept the node structure (locality near 1.0) while
finding more node-level balancing freedom than the rigid one-group-per-node Stage 1 allows would dominate
both rungs. I do not have that method in hand; the flat pack is the move the zigzag feedback points to,
because the feedback said the cap was the hierarchy and the cheapest way to test that is to remove it. So
the expectations: runtime stays in single-digit milliseconds (still the vectorized snake, Stage 1 work
removed), perhaps marginally slower on the smaller configs since one large pool can cost a touch more than
several tiny per-node packs, but nowhere near greedy's hundreds; balance rises substantially on the two
strangled configs with balance_node rising in lockstep off 0.702 and 0.336, because that node-level
imbalance was a Stage 1 artifact and there is no Stage 1 now; and locality drops off 1.000, into the
low-to-mid 0.90s on the real configs and the low 0.7s on stress-skew, which is the price. The balance gains
are large and sit on the configs dragging the geometric mean, the locality losses are bounded, so the task
score should climb past zigzag's 0.375 into the high 0.30s.

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
