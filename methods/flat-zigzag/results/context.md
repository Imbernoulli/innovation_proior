## Research question

A Mixture-of-Experts transformer routes each token to a small subset of many feed-forward
experts. Under expert parallelism, those experts are sharded across GPUs: tokens are dispatched
to the GPUs that hold their chosen experts, processed there, and gathered back. The serving
latency of the MoE layer is governed by the busiest GPU, not by the average GPU, so skewed
expert popularity turns into a direct wall-clock bottleneck.

The inputs to the placement problem are per-layer estimates of logical-expert load, such as
moving averages of recent token counts. Load prediction itself is out of scope. The decision is
where to put physical expert slots: every logical expert must keep at least one copy, heavily used
experts may receive redundant copies, and the resulting physical slots must be assigned to GPUs.
The placement has to satisfy hard serving constraints: the total replica budget is fixed, every GPU
holds exactly `num_replicas // num_gpus` expert slots, the output must expose both physical-to-
logical and logical-to-physical maps, and the procedure is rerun whenever the load estimate is
refreshed.

Two pressures pull against each other. The load objective wants the freest possible assignment so
the maximum GPU load can be driven down. The network objective wants locality: GPUs inside a
node communicate over a faster interconnect than GPUs across nodes, so scattering one expert's
replicas over many nodes can increase slow cross-node traffic. A placement routine therefore has
to balance GPU-load flattening, node locality, and recomputation cost.

## Background

MoE models scale capacity by replacing a dense feed-forward block with many expert blocks and
a router that activates only a few experts per token. This keeps per-token compute roughly fixed
while growing parameter count, but it creates a systems problem: the learned router does not use
all experts uniformly on every workload. A few experts can become hot, other experts stay cold,
and the hot set can drift with traffic mix.

Expert parallelism makes that skew visible to hardware. If one GPU receives more routed tokens
than the others, the all-to-all, expert compute, and combine phase wait for that GPU. A useful
load-balance score is therefore `mean_load / max_load`, equal to 1 only when the loads are
perfectly equal and smaller when one GPU dominates. The same max-load logic applies at node
level when inter-node bandwidth is the scarcer resource.

Restricted routing is the standard communication control. Experts are divided into groups or
deployed on devices/nodes, and each token is allowed to contact only a limited number of those
groups/devices/nodes. This bounds communication fan-out and gives placement a natural grouping
structure: experts in the same routing group benefit from being co-located.

The placement subproblem contains a familiar scheduling primitive. Given weighted items and
`m` bins, minimizing the largest bin load is multiway number partitioning. The classical
Longest Processing Time rule sorts jobs by decreasing processing time and repeatedly assigns the
next job to the least-loaded machine; for ordinary identical-machine makespan it has the Graham
bound `C_LPT <= (4/3 - 1/(3m)) * OPT`. The serving interface adds an exact-cardinality
constraint: each bin must hold the same number of items. A capacity-aware greedy rule can keep
the same heaviest-first, least-loaded-with-room idea, but the classical LPT constant belongs to
ordinary LPT and should not be treated as a guarantee for every cardinality-constrained variant or
for a later non-greedy rank heuristic.

## Baselines

**Training-time auxiliary load balancing.** Early large MoE systems such as GShard use top-2
routing, expert capacity, local-group dispatch, and an auxiliary load-balancing loss to push expert
usage toward uniformity. This reduces collapse during training, but capacity overflow can drop or
skip tokens, and the mechanism shapes routing probabilities rather than solving a serving-time
placement problem for a fixed load vector.

**Switch-style top-1 routing with capacity.** Switch Transformer simplifies MoE routing to one
expert per token and uses a capacity factor plus a differentiable load-balancing loss
`alpha * N * sum_i f_i * P_i`, where `f_i` is the fraction of tokens dispatched to expert `i` and
`P_i` is the mean router probability mass for that expert. This makes sparse routing simpler and
stable at scale, but it is still a training/routing remedy; it does not decide which expert copies
live on which GPUs at serving time.

**Device-limited routing and device-level balance losses.** Device-limited and group-limited
routing bound the number of devices or nodes touched by one token. Additional device-level and
communication-balance losses encourage even received/sent traffic. These mechanisms reduce the
communication search space and explain why locality matters, but they cannot make a fixed
deployment exactly balanced on a drifting workload.

**Auxiliary-loss-free balancing.** Bias-based balancing adjusts a per-expert routing bias: experts
with heavy recent load have their bias decreased, and underloaded experts have it increased. The
bias affects top-K selection but not the gate value used to weight expert outputs. This avoids
auxiliary-loss interference with the modeling objective, but it still controls routing rather than the
physical placement of redundant expert copies.

**Greedy serving-time placement.** A direct inference-time primitive is to duplicate overloaded
experts within a fixed slot budget, compute each physical replica's effective load, sort replicas
from heaviest to lightest, and assign each to the currently least-loaded GPU that still has a free
slot. The same primitive can also pack expert groups to nodes. It is simple and usually balances
well, but the assignment is sequential: each choice depends on all previous bin loads, so a Python
loop over items and batches becomes expensive when placements are recomputed repeatedly.

## Evaluation settings

A placement routine is evaluated on generated or observed per-layer expert-load tensors. The
important deployment knobs are the number of logical experts, expert groups, nodes, GPUs, and
physical replica slots. Valid settings require divisibility conditions such as `E % num_groups == 0`,
`num_gpus % num_nodes == 0`, and `num_replicas % num_gpus == 0`.

Correctness checks are structural. `phy2log` must have shape `[num_layers, num_replicas]`;
`logcnt` must have shape `[num_layers, num_logical_experts]`; every logical expert must have at
least one replica; `logcnt.sum(-1)` must equal `num_replicas`; and each GPU must receive exactly
`num_replicas // num_gpus` physical slots.

Quality is measured by per-GPU balance, per-node balance, replica locality, and runtime. GPU and
node balance use `mean_load / max_load`. Locality counts how many nodes contain replicas of a
logical expert, often through a traffic-weighted `1 / nodes_per_expert` score. Runtime matters
because placement is recomputed from fresh load statistics, so a slow combinatorial loop can erase
the value of frequent rebalancing.

## Code framework

The harness provides the load tensor and expects three functions. The scaffold below is neutral:
the bodies are the open placement logic, while the signatures and output contract are fixed.

```python
from typing import Tuple
import torch


def balanced_packing(weight: torch.Tensor, num_packs: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack n weighted items into num_packs packs of equal size.

    Args:
        weight: [B, n], item weights for B independent packing problems.
        num_packs: number of packs.
    Returns:
        pack_index: [B, n], pack id for each original item.
        rank_in_pack: [B, n], slot index of each item within its pack.
    """
    B, n = weight.shape
    assert n % num_packs == 0
    # TODO: assign each item to one equal-size pack.
    pass


def replicate_experts(
    weight: torch.Tensor, num_phy: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand num_log logical experts into num_phy physical slots.

    Args:
        weight: [B, num_log], load per logical expert.
        num_phy: total physical slots after replication.
    Returns:
        phy2log: [B, num_phy], logical expert id for each physical slot.
        rank: [B, num_phy], copy index for each physical slot.
        logcnt: [B, num_log], number of copies per logical expert.
    """
    B, num_log = weight.shape
    # TODO: choose replica counts and record the physical slots.
    pass


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a legal expert placement from per-layer logical-expert loads.

    Args:
        weight: [L, E], token load per expert per layer.
        num_replicas: total physical slots, a multiple of num_gpus.
        num_groups: number of expert groups, dividing E.
        num_nodes: number of server nodes.
        num_gpus: total GPUs, a multiple of num_nodes.
    Returns:
        phy2log: [L, num_replicas], logical expert for each physical slot.
        log2phy: [L, E, max_rep], physical slots per logical expert (-1 = unused).
        logcnt: [L, E], replica count per logical expert.
    """
    L, E = weight.shape
    weight = weight.float().cpu()
    # TODO: compose replication and packing into a legal placement.
    pass
```
