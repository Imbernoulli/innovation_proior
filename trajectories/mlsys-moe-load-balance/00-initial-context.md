## Research question

A Mixture-of-Experts model is served under **expert parallelism**: every routed expert is a
separate feed-forward network whose weights live on one GPU, and a token's top-K routing sends its
hidden state to the GPUs holding its chosen experts via an all-to-all *dispatch* before the expert
FFNs and an all-to-all *combine* after. Because the combine cannot start until *every* GPU has
finished its experts' work, the per-layer latency is set by the **most-loaded** GPU, not the average.
Live traffic is heavily skewed — a few experts soak up most of the tokens — and the hot set drifts as
the input distribution changes, so a static one-expert-per-GPU layout wastes most of the cluster.

The single thing being designed is the **expert placement algorithm**: given an online estimate of
per-expert token load and the cluster shape, decide how many physical replicas each logical expert
gets and which GPU each replica lands on, so that (1) the max GPU load is small, (2) the max *node*
load is small, and (3) each expert's replicas stay concentrated on as few nodes as possible — because
GPUs are grouped into nodes with fast intra-node NVLink and slow, scarce inter-node InfiniBand, and a
placement that scatters an expert across nodes detonates exactly the inter-node all-to-all the
node-limited routing was built to bound. The plan re-runs online as loads change, so it must also be
**cheap**. Everything else about the serving stack is fixed.

## Prior art before the first rung

The first rung is the established three-stage hierarchical placement. The lineage it reacts to:

- **GShard / Switch capacity + auxiliary loss (Lepikhin et al. 2020; Fedus et al. 2021).** Cap each
  expert's tokens per batch and drop the overflow; add a differentiable load-balance auxiliary loss
  pushing the router toward uniform usage. *Gap:* both levers act at *training* time on the *router*;
  dropping tokens is unacceptable for inference, and neither decides where physical expert weights
  live at serving time nor handles a single expert too hot to fit on one GPU.
- **Device-/node-limited routing (DeepSeek-V2, DeepSeek-AI 2024).** Cap the number of devices/nodes a
  token's experts may span, establishing the **group** structure the placement must respect. *Gap:* a
  routing constraint, not a placement — it bounds per-token communication footprint but never says
  which GPU holds which expert; it *assumes* a co-located layout something else must produce.
- **Greedy makespan scheduling — LPT / list scheduling (Graham 1966, 1969).** Assign weighted items
  to identical machines to minimize the max load: list scheduling is within `2 − 1/m` of optimal,
  largest-first (LPT) within `4/3 − 1/(3m)`. *Gap:* it balances a *fixed* set of indivisible items, so
  a single item above the per-unit fair share floors the peak no matter how the rest are arranged; and
  plain LPT does not enforce the equal item count per machine the hardware here requires.

## The fixed substrate

The placement plugs into an MoE serving stack that already exists: a router producing per-token top-K
affinities, an all-to-all dispatch/combine around the expert FFNs, and an online-statistics layer that
accumulates a per-expert token-load estimate `weight` of shape `[L, E]` (layers × logical experts).
The cluster shape is given as `num_replicas` (total physical slots, a multiple of `num_gpus`),
`num_groups` (routing groups, dividing `E`), `num_nodes`, and `num_gpus` (a multiple of `num_nodes`).
The substrate is plain tensor primitives (`torch`, `numpy`): summing / sorting / gathering / scattering
weights into integer index maps. The evaluation harness, the workload generator, and the placement
validator are frozen.

## The editable interface

Exactly one region of `custom_eplb.py` is editable (lines 62–209): three functions the serving runtime
calls. The contract is fixed; every rung on the ladder is a different fill of it.

- `balanced_packing(weight, num_packs)` — partition each row's `n` weighted items into `num_packs`
  packs of **exactly** `n // num_packs` items each, balancing per-pack sums; return `(pack_index,
  rank_in_pack)`.
- `replicate_experts(weight, num_phy)` — grow `num_log` logical experts to `num_phy` physical slots;
  return `(phy2log, rank, logcnt)`.
- `rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus)` — the entry point; return
  `phy2log [L, num_replicas]` (logical expert per physical slot), `log2phy [L, E, max_rep]` (physical
  slots per logical expert, `-1` padded), `logcnt [L, E]` (replica count per expert).

Every valid plan must satisfy: `E % num_groups == 0`, `num_groups % num_nodes == 0`, `num_gpus %
num_nodes == 0`, `num_replicas % num_gpus == 0`; each GPU hosts exactly `num_replicas // num_gpus`
physical experts; every logical expert keeps ≥ 1 replica; `logcnt.sum(-1) == num_replicas` per layer.

The starting point is the scaffold default: the **hierarchical greedy bin-packing** — Stage 1 packs
groups onto nodes, Stage 2 replicates hot experts within each node, Stage 3 packs replicas onto GPUs
within each node — with `balanced_packing` written as the textbook sequential greedy (sort descending,
drop each item on the least-loaded non-full pack). Each later rung replaces exactly these definitions.

```python
# EDITABLE region of custom_eplb.py (lines 62-209) — default fill: hierarchical greedy bin-packing

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

    # Stage 1: pack groups onto nodes
    tpg = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    gpi, grk = balanced_packing(tpg, num_nodes)
    log2mlog = (((gpi * groups_per_node + grk) * group_size).unsqueeze(-1)
                + torch.arange(group_size)).flatten(-2)
    mlog2log = inv(log2mlog)

    # Stage 2: replicate hot experts within each node
    tpm = weight.gather(-1, mlog2log).view(-1, experts_per_node)
    p2m, prk, mcnt = replicate_experts(tpm, replicas_per_node)

    # Stage 3: pack physical replicas onto GPUs within each node
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

## Evaluation settings

Four MoE deployments derived from real architectures plus one stress configuration, each specified by
the number of logical experts `E`, groups `G`, nodes `N`, GPUs `D`, and physical-slot budget `R`:

| Config | E | G | N | D | R | zipf · skew |
|---|---|---|---|---|---|---|
| `deepseek-v3`  | 256 | 8  | 8  | 64  | 320 | 0.7 · 0.85 |
| `qwen3-moe`    | 128 | 8  | 4  | 32  | 160 | 0.5 · 0.70 |
| `deepseek-v2`  | 160 | 8  | 4  | 32  | 192 | 0.6 · 0.75 |
| `stress-skew`  | 256 | 32 | 16 | 128 | 384 | 1.0 · 0.95 |

`stress-skew` is a synthetic stress test: the 16-node hierarchy is the largest in the suite, the
replication budget is tighter (1.5× rather than 2×), `groups_per_node = 2` makes Stage 1 group-to-node
packing non-trivial, and the workload follows a long-tail Zipf distribution. Per-expert load is
synthesized from a skewed Zipf traffic model mixed with a uniform base; seed 42.

Four metrics per config: **balance** = `mean_gpu_load / max_gpu_load` (higher better, capped at 1.0);
**balance_node** = the same ratio at node granularity; **locality** = traffic-weighted node locality of
replicas (`1 / nodes_per_expert` averaged over experts weighted by traffic — replicas all on one node
score 1.0, uniformly scattered score `1/N`); **runtime_ms** = median wall time over 20 timed iterations
(lower better). The per-config score weights the four equally; the task score is the geometric mean
across the four configs. All three balance/locality terms are required: a flat scheme that scatters
replicas to maximize per-GPU balance loses locality; a method that co-locates without addressing skew
loses balance.
