## Research question

A Mixture-of-Experts (MoE) language model routes each token to a small number of expert
sub-networks chosen from a large pool (DeepSeek-V3, for example, has 256 routed experts per
layer; Qwen3-MoE has 128). At inference time the experts are spread across many GPUs under
expert parallelism, and the per-expert token load is highly skewed and *non-stationary*: a
handful of "hot" experts attract most of the traffic, and which experts are hot drifts with
the input distribution. Because every GPU must finish its share before the layer can proceed,
the slowest (most loaded) GPU sets the latency — so load imbalance directly wastes hardware.

The system therefore has to recompute, *periodically and online*, a placement plan: how many
physical replicas to give each logical expert, and which GPU each replica sits on, so that the
per-GPU token load is as even as possible. Three things make this hard at once. (1) The plan
must balance load at two granularities — across GPUs and across the server *nodes* that group
them — because both the per-GPU compute and the per-node aggregate matter. (2) It must respect
*locality*: keeping all replicas of a given expert (and the experts of a routing group) on a
single node, because the intra-node interconnect (NVLink, ~160 GB/s) is several times faster
than the inter-node fabric (InfiniBand, ~50 GB/s), so a placement that scatters an expert's
replicas across nodes pays a heavy all-to-all communication cost that pure load metrics do not
see. (3) It must be *cheap to compute*, because rebalancing runs repeatedly on the serving
critical path as the load shifts; a placement algorithm that takes hundreds of milliseconds is
itself a latency tax. The precise goal: an expert replication-and-placement algorithm that
holds per-GPU and per-node balance and inter-node locality at the quality of the established
hierarchical method, while making the placement computation cheap enough to run frequently.

## Background

**MoE inference and expert parallelism.** In an MoE layer a gating network selects, per token,
the top-k of E routed experts; only the selected experts run. Under expert parallelism each
GPU owns a subset of the experts, and tokens are dispatched (all-to-all) to wherever their
chosen experts live, then the outputs are combined back. The cost of a layer is dominated by
the busiest GPU, so the placement of experts onto GPUs is a first-order performance knob.

**Skewed, drifting expert load.** Expert utilization is far from uniform: a Zipf-like tail
means a few experts handle a disproportionate share of tokens, and the identity of the hot
experts changes as the workload changes. This is the empirical fact that makes a *static*,
uniform "one expert per GPU" layout leave some GPUs idle while others saturate. The standard
response, established for DeepSeek-V3 deployment, is a *redundant-experts* strategy: allocate
extra physical slots (e.g. 32 redundant experts in prefilling, so each GPU holds its experts
plus an extra one) and place additional *replicas* of the hottest experts, so their traffic is
split across copies. Because the load drifts, the redundancy plan is recomputed from online
load statistics (a moving average of recent per-expert token counts) and adjusted periodically.

**Group-limited / node-limited routing and locality.** DeepSeek-V3 partitions its 256 routed
experts into 8 groups of 32 and constrains each token to be routed to experts in at most M=4
nodes (node-limited routing), specifically to bound the inter-node all-to-all. This is what
makes *locality* a hard requirement rather than a nicety: the routing already promises that a
token touches few nodes, and a placement that keeps each group's experts (and each expert's
replicas) confined to one node is what lets that promise translate into low communication. The
intra-node-vs-inter-node bandwidth gap is the physical reason.

**Balanced multiway number partitioning.** The combinatorial core of placement is an old
problem: given n weighted items, partition them into P parts so the part sums are as equal as
possible. With the "minimize the largest sum" objective this is identical-machines makespan
scheduling / multiway number partitioning, which is NP-hard. The classic practical heuristics
are *greedy list scheduling* — process items in some order, drop each into the part with the
currently smallest sum, which guarantees the largest sum is within a factor (2 − 1/P) of
optimal — and its sorted refinement *LPT* (Longest Processing Time, Graham 1969): sort items
in descending order first, then greedily fill the emptiest part, improving the unconstrained
worst-case factor to 4/3 − 1/(3P) (7/6 for P=2). Both place the largest items first and let
later, smaller items fill the gaps. The placement problem here is a *cardinality-constrained*
(balanced) variant: every part must hold exactly n/P items, because every GPU has a fixed
number of expert slots — so the greedy must only ever consider parts that still have free
capacity, and the classical LPT bound is useful background rather than the exact theorem being
invoked.

**Diagnostic fact about the reference rebalancer.** The established hierarchical placement
algorithm is correct and balances well, but its bin-packing core is implemented as nested
Python loops — an outer loop over layers and an inner loop over items, each item doing a linear
scan over parts to find the emptiest one with capacity. Profiled on medium MoE deployments this
takes on the order of 540 ms per rebalance, with an imbalance factor (the ratio of average to
maximum per-GPU load) around 0.66. The per-element Python iteration, not the arithmetic, is the
cost: the arithmetic is small enough that an implementation dominated by batched tensor
operations should not spend most of its time in the host-language loop overhead.

## Baselines

**Hierarchical greedy bin-packing placement (DeepSeek-AI 2024, the EPLB reference).** The
established three-stage algorithm. Stage 1 sums each routing group's load and packs the groups
onto nodes with greedy balanced bin-packing, so node loads are even and a group's experts stay
on one node. Stage 2 replicates experts *within* each node: starting from one replica each, it
repeatedly hands an extra replica to whichever expert currently has the largest *per-replica*
load (load / current replica count), which is the argmax of `weight / logcnt`, until the
node's slot budget is filled — this is the redundant-experts idea, driving down the peak
per-replica load. Stage 3 packs the resulting physical replicas onto the GPUs within each node,
again by greedy balanced bin-packing on the per-replica loads, so each GPU gets exactly
`num_replicas / num_gpus` replicas and the GPU loads are even. The greedy packing routine,
called in Stages 1 and 3, is LPT-style sorted greedy with a cardinality cap: sort the items by
weight descending, then for each item in turn assign it to the least-loaded part that still has
free slots. Concretely:

```python
def balanced_packing(weight, num_packs):          # weight: [B, n]
    B, n = weight.shape
    items_per_pack = n // num_packs               # exactly this many per pack
    sorted_idx = weight.float().sort(-1, descending=True).indices.cpu()
    pack_index   = torch.full((B, n), -1, dtype=torch.int64)
    rank_in_pack = torch.full((B, n), -1, dtype=torch.int64)
    for b in range(B):                            # loop over layers
        loads  = [0.0] * num_packs
        counts = [0]   * num_packs
        for j in range(n):                        # loop over items, in desc-weight order
            item = sorted_idx[b, j].item()
            best = min((p for p in range(num_packs) if counts[p] < items_per_pack),
                       key=lambda p: loads[p])     # emptiest pack with free capacity
            pack_index[b, item]   = best
            rank_in_pack[b, item] = counts[best]
            loads[best]  += weight[b, item].item()
            counts[best] += 1
    return pack_index, rank_in_pack
```

The hierarchy is what earns locality (groups and replicas stay node-confined) and the sorted
emptiest-bin packing gives good balance while the cap enforces fixed hardware slot counts.
**Gap:** the `balanced_packing` core is a per-element Python loop — `B × n` iterations, each scanning `num_packs` parts — so its
cost grows with the tensor size and it runs in hundreds of milliseconds, even though every
quantity it manipulates is a small integer index. The running-load dependence (item j's pack
depends on the loads after items 1..j−1) is what forces the sequential scan, and that is exactly
what stalls on a GPU, which wants batched, data-parallel work.

**Flat (non-hierarchical) placement.** A simpler alternative skips Stage 1 entirely: replicate
experts globally and pack all replicas directly onto all GPUs in one shot, ignoring the node
structure. This removes the group-to-node stage and can balance per-GPU load well, but because
it never confines an expert's replicas to a node, it scatters replicas across the whole cluster.
**Gap:** in a multi-node deployment this gives up locality — an expert's replicas can land on
several different nodes, so serving a token to that expert may cross the slow inter-node fabric,
inflating the all-to-all cost that the node-limited routing was designed to bound. Per-GPU
balance alone does not capture this; the locality metric does.

## Evaluation settings

The natural yardsticks are real MoE deployment shapes plus a stress configuration. Each is a
choice of E experts, G routing groups, N nodes, D GPUs, and a replica budget R (a multiple of D
that fixes the per-GPU slot count R/D), with a synthetic Zipf-skewed per-expert load trace
(parameterized by a Zipf exponent and a skew factor) standing in for online statistics gathered
over datasets such as ShareGPT and GSM8K:

| Config | E | G | N | D | R | zipf · skew |
|---|---|---|---|---|---|---|
| deepseek-v3 | 256 | 8 | 8 | 64 | 320 | 0.7 · 0.85 |
| qwen3-moe | 128 | 8 | 4 | 32 | 160 | 0.5 · 0.70 |
| deepseek-v2 | 160 | 8 | 4 | 32 | 192 | 0.6 · 0.75 |
| stress-skew | 256 | 32 | 16 | 128 | 384 | 1.0 · 0.95 |

The constraints `E % G == 0`, `G % N == 0`, `D % N == 0`, `R % D == 0` always hold; `stress-skew`
is the hardest (16 nodes, a tighter 1.5× replica budget instead of 2×, only `groups_per_node = 2`
so the group-to-node stage is non-trivial, and the most extreme long tail). Four metrics are
reported per config: **balance** = mean / max per-GPU load (higher better, ≤ 1); **balance_node**
= mean / max per-node load (higher better, ≤ 1); **locality** = traffic-weighted inverse of the
number of distinct nodes holding each expert's replicas, averaged over experts and layers (1.0 if
every expert's replicas sit on a single node, 1/N if scattered uniformly); and **runtime_ms**, the
median wall time of the placement algorithm over timed iterations. The combined score weights the
four equally per config and takes the geometric mean across configs, so a method cannot trade one
away: a flat scheme that scatters replicas to maximize per-GPU balance loses locality, and a
scheme that co-locates but ignores skew loses balance.

## Code framework

The placement algorithm plugs into a fixed serving harness: a periodic rebalancer is handed a
per-layer, per-expert load tensor (the moving average of recent token counts) and the deployment
shape, and must return the maps the engine uses to dispatch tokens. The harness, the data
pipeline that produces `weight`, the hierarchy parameters, and the output map format are all
given; what is *not* given is the internal policy for turning a weighted item list into a
balanced, capacity-limited partition. That packing routine is the single open slot.

```python
from typing import Tuple
import torch


def balanced_packing(weight: torch.Tensor, num_packs: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Partition the n weighted items of each row of `weight` [B, n] into `num_packs`
    packs of EXACTLY n // num_packs items each, making the pack sums as balanced as
    possible. Returns (pack_index [B, n], rank_in_pack [B, n]).

    Fill in the placement policy here.
    """
    B, n = weight.shape
    assert n % num_packs == 0
    if n // num_packs == 1:                       # one item per pack: identity
        idx = torch.arange(n, dtype=torch.int64, device=weight.device).expand(B, -1)
        return idx, torch.zeros_like(idx)

    # TODO: choose a pack and a rank for every item while keeping exactly
    #       n // num_packs items in every pack.
    pass


def replicate_experts(weight: torch.Tensor, num_phy: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Grow `num_log` logical experts to `num_phy` physical replicas, minimizing the
    maximum per-replica load. Start with one replica each; repeatedly give an extra
    replica to the expert with the largest current per-replica load (weight / count).
    Returns (phy2log [B, num_phy], rank [B, num_phy], logcnt [B, num_log])."""
    B, num_log = weight.shape
    device = weight.device
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(B, 1)
    rank = torch.zeros(B, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(B, num_log, dtype=torch.int64, device=device)
    idx_b = torch.arange(B, dtype=torch.int64, device=device)
    for i in range(num_log, num_phy):             # one extra replica per redundant slot
        eff = weight / logcnt.float()             # per-replica load
        top = eff.argmax(dim=-1)                  # most-overloaded expert
        phy2log[:, i] = top
        rank[:, i] = logcnt[idx_b, top]
        logcnt[idx_b, top] += 1
    return phy2log, rank, logcnt


def rebalance_experts(weight: torch.Tensor, num_replicas: int, num_groups: int,
                      num_nodes: int, num_gpus: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Three-stage hierarchical placement:
      Stage 1: pack routing groups onto nodes (balance node load, keep a group on one node).
      Stage 2: replicate hot experts within each node.
      Stage 3: pack the physical replicas onto the GPUs within each node.
    Returns the physical<->logical maps the serving engine dispatches with."""
    L, E = weight.shape
    weight = weight.float().cpu()
    group_size = E // num_groups
    gpus_per_node = num_gpus // num_nodes
    phy_per_gpu = num_replicas // num_gpus
    groups_per_node = num_groups // num_nodes
    experts_per_node = E // num_nodes
    replicas_per_node = num_replicas // num_nodes

    def inv(perm):                                # inverse of a permutation, per row
        out = torch.empty_like(perm)
        out.scatter_(1, perm, torch.arange(perm.size(1), dtype=torch.int64).expand(perm.shape))
        return out

    # Stage 1: groups -> nodes
    tpg = weight.unflatten(-1, (num_groups, group_size)).sum(-1)   # token load per group
    gpi, grk = balanced_packing(tpg, num_nodes)
    log2mlog = (((gpi * groups_per_node + grk) * group_size).unsqueeze(-1)
                + torch.arange(group_size)).flatten(-2)
    mlog2log = inv(log2mlog)

    # Stage 2: replicate within nodes
    tpm = weight.gather(-1, mlog2log).view(-1, experts_per_node)
    p2m, prk, mcnt = replicate_experts(tpm, replicas_per_node)

    # Stage 3: replicas -> GPUs within nodes
    tpp = (tpm / mcnt.float()).gather(-1, p2m)    # per-replica load
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

The hierarchy, the replication rule, the map-composition bookkeeping, and the output format are
all in place; the one empty body is `balanced_packing`'s assignment of packs and ranks under the
fixed slot constraint.
