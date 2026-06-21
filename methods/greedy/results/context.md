## Research question

A large Mixture-of-Experts (MoE) language model is served with **expert parallelism (EP)**:
every routed expert is a separate feed-forward network whose weights live on one GPU, and a
token's top-K routing decision sends its hidden state to the GPUs that own its chosen experts,
via an all-to-all *dispatch* before the expert FFNs and an all-to-all *combine* after. Because
the combine cannot start until every GPU has finished its experts' work, the per-layer latency
is set by the **most-loaded** GPU, not the average. Real serving traffic is skewed: a few "hot"
experts attract far more tokens than the rest, and which experts are hot drifts as the input
distribution changes.

The hardware is two-level. GPUs are grouped into **nodes**; the intra-node interconnect
(NVLink) is fast and plentiful, the inter-node fabric (InfiniBand) is slow and scarce. A token
whose experts span many nodes pays inter-node bandwidth proportional to the number of nodes it
touches. The placement question is posed against this setting: given an estimate of per-expert
token load and a fixed budget of physical expert slots (a multiple of the GPU count, with each
GPU hosting exactly the same number of slots), produce an expert-to-GPU placement plan, and do
it cheaply enough to re-run online as the observed loads change (on the order of every ten
minutes).

## Background

By this time, MoE is a dominant way to scale LLM capacity without scaling per-token compute:
each FFN block is replaced by `N_r` routed experts plus a small number `N_s` of always-on
shared experts, and a learned router selects, per token, the top `K_r` routed experts by a
token-to-expert affinity score. DeepSeekMoE (Dai et al. 2024) uses *fine-grained* experts —
many small experts, several activated per token — which sharpens specialization and increases
the number of distinct GPUs a token's experts can land on.

The facts that set up the placement problem:

- **Expert load is skewed and non-stationary.** Even a well-trained router does not send tokens
  uniformly: at serving time the live input distribution makes a handful of experts hot and
  leaves most cold, and the hot set shifts over time. This is the same skew that, during
  *training*, drives the **routing collapse** phenomenon (Shazeer et al. 2017), where a few
  experts dominate.

- **The bottleneck is the maximum, not the mean.** Synchronous all-to-all means the layer waits
  on the slowest GPU. Minimizing wasted compute is therefore a *makespan* problem: balance the
  peak load across the parallel units. The same logic applies one level up, at node
  granularity, because cross-node communication and node-level synchronization wait on the
  busiest node.

- **Inter-node traffic is the expensive resource, and routing respects group structure.**
  DeepSeek-V2 (DeepSeek-AI 2024a) introduced **device-limited routing**: each token's experts
  are restricted to at most `M` devices (chosen by the per-device top affinity sums), and
  empirically `M >= 3` matches unrestricted top-K quality while bounding the number of devices
  each token must reach. The node-limited variant applies the same idea at server granularity:
  each token reaches at most `M` nodes (`M = 4` in the serving stack here), with nodes picked by
  the sum of the highest `K_r/M` affinity scores of the experts on each node. So experts come in
  **groups**, and the routing assumes a token's experts cluster onto a few nodes.

- **Serving load is measured online.** Auxiliary-loss-free training balance (Wang et al. 2024)
  keeps *training* load balanced with a per-expert bias `b_i`: it is added to the routing scores
  only for the top-K selection (the gating value still uses the unbiased score), and after each
  step `b_i` is nudged down by `γ` if expert `i` was overloaded and up by `γ` if underloaded — a
  feedback controller that equalizes load without injecting the interfering gradients a balance
  loss would. At inference the load is whatever the live traffic produces; it is measured from
  online statistics, and it changes, so the physical layout is decided from observed loads
  periodically.

- **Classical scheduling theory.** Assigning weighted items to parallel units to minimize the
  maximum load is the identical-machines makespan problem `P||Cmax`: NP-hard in the strong sense
  (the two-machine case is exactly the PARTITION problem; the variable-machine case is strongly
  NP-hard by reduction from 3-PARTITION). Graham (1966, 1969) showed simple greedy rules come
  close for the unconstrained identical-machine version. **List scheduling** — take the items in
  any order, drop each onto the currently least-loaded machine — guarantees a makespan within a
  factor `2 − 1/m` of optimal on `m` machines. Sorting the items **largest-first** before the
  same greedy sweep (the **Longest Processing Time**, LPT, rule) tightens the guarantee to
  `4/3 − 1/(3m)`.

## Baselines

**GShard / Switch-style capacity + auxiliary loss (Lepikhin et al. 2020; Fedus et al. 2021).**
An early large-scale answer to expert imbalance. Each expert is given a fixed **capacity** (a
cap on tokens per batch); tokens beyond the cap are **dropped** (skipped via the residual), and
a differentiable **load-balance auxiliary loss** — proportional to the product of the fraction
of tokens routed to each expert and the mean router probability mass on it — is added to push
the router toward uniform usage. *Core math:* `L_aux = α · N · Σ_i f_i · P_i`, with `f_i` the
fraction of tokens dispatched to expert `i` and `P_i` the mean gate probability. Both levers act
at *training* time on the *router*.

**Auxiliary-loss-free training balance (Wang et al. 2024).** Replaces the aux-loss with the
per-expert bias feedback controller described above: add `b_i` to scores for top-K selection,
nudge `b_i` by `±γ` from observed overload/underload. It keeps the *training* distribution
balanced and leaves the model's gating values clean.

**Device-/node-limited routing.** Caps the number of devices/nodes a token's experts may span
(`M >= 3` is enough for the device-level version to track unrestricted top-K closely; the
node-level setting here uses `M = 4`). It is a routing constraint that bounds the communication
footprint per token and establishes the group structure.

**Greedy makespan scheduling — LPT / list scheduling (Graham 1966, 1969).** The classical
greedy assignment: (list scheduling) drop each item on the least-loaded machine, makespan
≤ `(2 − 1/m)·OPT`; (LPT) sort items descending first, makespan ≤ `(4/3 − 1/(3m))·OPT`. It
balances a fixed set of indivisible items, taken in some order onto the least-loaded machine.

## Evaluation settings

The yardsticks for a serving-time placement, all defined before any particular placement exists:

- **Deployment configurations** drawn from real MoE serving setups, each specified by the
  number of logical experts `E`, expert groups `G`, server nodes `N`, GPUs `D` (a multiple of
  `N`), and the physical-slot budget `R` (a multiple of `D`), e.g. a 256-expert / 8-group /
  8-node / 64-GPU layout with `R = 320` slots, a 128-expert / 4-node / 32-GPU layout, a
  160-expert variant, and a tight-budget 16-node stress layout with `groups_per_node = 2` and a
  long-tailed (Zipf) load. The per-expert token load that drives placement is itself an input,
  taken from online statistics; in evaluation it is synthesized from a skewed (Zipf) traffic
  model.
- **Constraints any valid plan must satisfy:** `E` divisible by `G`, `G` divisible by `N`, `D`
  divisible by `N`, `R` divisible by `D`; each GPU receives exactly `R/D` physical experts;
  every logical expert is represented at least once; the assigned slot counts sum to `R` per
  layer.
- **Metrics:** per-GPU balance `mean_gpu_load / max_gpu_load` (capped at 1.0 for a perfectly
  even peak); per-node balance `mean_node_load / max_node_load`; **locality**, a traffic-weighted
  score that for each (layer, expert) counts how many distinct nodes hold slots for that expert
  and credits `1 / nodes_per_expert` (so an expert whose slots all sit on one node scores 1.0, a
  uniformly scattered one scores `1/N`); and the algorithm's wall-clock **runtime**. The combined
  score weighs the four equally per configuration and takes the geometric mean across
  configurations.

## Code framework

The placement plan plugs into an MoE serving stack that already exists: a router that produces
per-token top-K affinities, an all-to-all dispatch/combine around the expert FFNs, and a layer
of online statistics that accumulates a per-expert token-load estimate. The algorithm that turns
that load estimate plus the cluster shape into a concrete expert-to-GPU assignment is what goes
in `place_experts`. The substrate is just tensor primitives (`torch`): summing/sorting/gathering
weights, and the three return maps the serving runtime consumes (which logical expert each
physical slot serves; the physical slots belonging to each logical expert; the slot count per
expert).

```python
from typing import Tuple
import torch


def place_experts(
    weight: torch.Tensor,       # [L, E]  estimated token load per expert, per layer
    num_slots: int,             # total physical expert slots (a multiple of num_gpus)
    num_groups: int,            # number of expert groups (divides E)
    num_nodes: int,             # number of server nodes
    num_gpus: int,              # total GPUs (a multiple of num_nodes)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Turn a per-expert load estimate and the cluster shape into a placement plan.

    Returns:
        phy2log:  [L, num_slots]         logical expert id served by each physical slot
        log2phy:  [L, E, max_slots]      physical slot ids per logical expert (-1 = unused)
        logcnt:   [L, E]                 number of physical slots assigned per logical expert

    Constraints the plan must satisfy: each GPU hosts exactly num_slots // num_gpus
    physical experts; every logical expert is represented at least once; logcnt sums to num_slots
    per layer.
    """
    L, E = weight.shape
    weight = weight.float().cpu()
    # TODO: fill the assignment policy that emits (phy2log, log2phy, logcnt).
    raise NotImplementedError
```

The serving runtime supplies `weight` (the online load estimate) and the cluster shape; the
body of `place_experts` is where the assignment logic will live, and the three returned maps are
exactly what the dispatch/combine machinery reads to know where each expert's tokens go.
