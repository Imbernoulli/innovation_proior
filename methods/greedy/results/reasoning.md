Let me start from what actually goes wrong when I serve this thing. I have an MoE model, and I'm running it with expert parallelism: every routed expert is a separate FFN sitting on some GPU, and when a token comes in, the router picks its top-K experts, the runtime does an all-to-all to ship that token's hidden state to whichever GPUs own those experts, the experts compute, and a second all-to-all brings the results back to combine. The thing that hurts is that this is synchronous. The combine can't begin until *every* GPU has finished its share of the expert work, so the per-layer latency is the time of the slowest GPU. Not the average GPU — the max. And the loads are nowhere near equal: the live traffic sends a few experts an enormous number of tokens and most experts almost nothing, and which experts are hot drifts around as the input distribution changes. So with one expert per GPU, the busy GPU is grinding through a mountain of tokens while the rest sit idle waiting for it, and I'm wasting most of the machine. The whole game is: lay out the experts across GPUs so that the *busiest* GPU is as light as possible.

So really I'm minimizing a maximum over parallel units given a pile of weighted things to distribute. That's a makespan problem — the identical-machines scheduling problem, assign jobs to machines to minimize the max machine load. And I know two things about that problem cold. One, it's NP-hard — the two-machine case is literally the PARTITION problem, split a multiset into two equal-sum halves, and the variable-machine case reduces from 3-PARTITION, so I'm not going to find the exact optimum in polynomial time, and anyway I have to re-run this online every few minutes as the loads drift, so I need something cheap, not something exhaustive. Two, greedy gets close in the unconstrained version. Take plain list scheduling: when the job that finishes last is placed, suppose it starts at time `t` and has size `p`. Because the algorithm put it on the least-loaded machine, all `m` machines already have load at least `t`, so the total work `W` is at least `m t + p`. Thus the final makespan `C = t + p <= W/m + (1 - 1/m)p`. Since any optimum is at least `W/m` and at least `p`, this gives `C <= (2 - 1/m) OPT`. Sorting biggest-first before the same greedy sweep — the longest-processing-time rule — should tighten that. Here's the argument I half-remember: now the job that finishes last is the smallest job in the prefix up to that placement, because the list is sorted descending. If that critical job were larger than `OPT/3`, then every job in the prefix would be larger than `OPT/3`, so an optimal schedule could put at most two of them on any one machine; in that at-most-two-jobs regime, the largest-first greedy pairing is already optimal on the prefix. So the only case where LPT can be worse than optimal has critical size `p <= OPT/3`, and plugging that into the same list-scheduling inequality gives `C <= OPT + (1 - 1/m)OPT/3 = (4/3 - 1/(3m)) OPT`.

I want to make sure I haven't fooled myself with that proof, because the `OPT/3` threshold is doing all the work and it's the kind of thing that's easy to state and wrong by a constant. Let me build a small instance where the bound should be tight and just compute the ratio. For `m = 2`, take jobs `[3, 3, 2, 2, 2]`; total is 12, so `OPT = 6` (one machine gets `3+3`, the other `2+2+2`). LPT sorts descending and drops onto the least-loaded machine: machine A gets the first 3, B gets the second 3, then A gets a 2 (A=5), B gets a 2 (B=5), then the last 2 goes onto A (A=7). Makespan 7, ratio `7/6 ≈ 1.1667`, and `4/3 - 1/(3·2) = 4/3 - 1/6 = 7/6` — exactly equal. For `m = 3`, jobs `[5,5,4,4,3,3,3]`, sum 27, `OPT = 9`: LPT lands at makespan 11, and `11/9 ≈ 1.2222` against `4/3 - 1/9 = 11/9`, again exactly the bound. So the proof isn't off by a constant; the worst case I can hand-construct sits right on `4/3 - 1/(3m)`, and the descending-sort really is what buys the improvement over the plain `2 - 1/m`. Good — that's the rule I'll carry forward, and the hardware-constrained version will keep the same ordering and least-loaded instinct while adding a capacity check.

Let me write that primitive down concretely, because there's a wrinkle from the hardware. I'm packing `n` items into `num_packs` packs, but every GPU has to host the *same number* of physical experts — that's a deployment constraint, the runtime allocates a fixed number of slots per GPU. So it's not free-form makespan; it's a balanced partition where every pack gets exactly `n / num_packs` items. Plain LPT would happily put six items on one machine and two on another if that balanced the *weight*; here I have to also balance the *count*. Fine — I keep the greedy rule but restrict the candidate machines to those that aren't already full. So: sort descending; for each item in that order, among the packs that still have room (count below `items_per_pack`), pick the one with the smallest current load; assign the item there, record its rank within the pack, bump that pack's load and count.

```python
def balanced_packing(weight: torch.Tensor, num_packs: int) -> Tuple[torch.Tensor, torch.Tensor]:
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(weight.size(-1), dtype=torch.int64, device=weight.device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    indices = weight.float().sort(-1, descending=True).indices.cpu()
    pack_index = torch.full_like(weight, fill_value=-1, dtype=torch.int64, device='cpu')
    rank_in_pack = torch.full_like(pack_index, fill_value=-1)
    for i in range(num_layers):
        pack_weights = [0] * num_packs
        pack_items = [0] * num_packs
        for group in indices[i]:
            pack = min((i for i in range(num_packs) if pack_items[i] < groups_per_pack),
                       key=pack_weights.__getitem__)
            assert pack_items[pack] < groups_per_pack
            pack_index[i, group] = pack
            rank_in_pack[i, group] = pack_items[pack]
            pack_weights[pack] += weight[i, group]
            pack_items[pack] += 1
    return pack_index, rank_in_pack
```

A couple of things I notice writing it. Each row `i` (a layer) packs independently, so there's an outer loop over the batch of layers. And there's a degenerate case: if `groups_per_pack == 1`, every pack gets exactly one item, so there's no balancing to do at all — item `i` just goes to pack `i` at rank 0, and I can return that directly without the inner loop. Worth special-casing because it'll come up. Also the inner loop is irreducibly sequential: each placement's "least loaded" decision depends on the running loads from all previous placements, so I can't vectorize the greedy choice. That's why I move the sorted indices and outputs to the CPU and let the Python loop carry the running `pack_weights` and `pack_items`. It'll be slow, but it mirrors the greedy decisions exactly and keeps the map construction simple.

Now I want to stress this primitive on the thing it's actually fighting — skew — and I have a worry. Greedy packing balances a fixed set of indivisible items. What happens when one item is, by itself, bigger than the fair share? Let me just construct that and measure. Say 16 items with total load 100 across 8 packs (so the fair share per pack is 12.5), but one item is hot at 20 and the other 15 split the remaining 80 evenly (≈5.33 each). I run the packer and sum the load that lands on each pack: the heaviest pack comes out at 25.33, and even the lightest possible outcome can't drop below 20, because the hot item is 20 and it has to land *somewhere* whole. The max pack load is floored at the single largest item, and that floor (20) sits above what balance demands (12.5). So no matter how I order or pack, the GPU holding that one expert is stuck at a fifth of the total while the average is an eighth. Greedy can't help here; *nothing* that just assigns whole experts can help. I've been thinking about this as "distribute the items," but the items themselves are the problem.

So the items have to become divisible. I can't literally split an expert's weights, but I *can* make copies of it. If the hot expert has `r` physical replicas and its requests are spread across those replicas, each copy carries about `w_i / r` of the load instead of the whole `w_i`. Let me check that this actually clears the floor on the instance that just defeated me. Take 8 logical experts, weights one-hot-at-20 and seven sharing 80 (≈11.43 each), and hand out replicas greedily, always to whichever expert currently has the largest per-replica load. With no extra replicas the max per-replica load is 20 — the floor I expected. Add a single replica and it goes to the hot expert, whose per-replica load drops from 20 to `20/2 = 10`; now the max per-replica load across all experts is `80/7 ≈ 11.43`, already *below* the fair share of 12.5. So one extra copy of the right expert is enough to break the floor on this instance, and more budget pushes it toward 10 (full duplication of everything). The effective item size is tunable: each extra copy lowers the floor imposed by that expert. This is the move classical scheduling never had — it assumed the jobs were fixed. Here the jobs are FFNs and I'm allowed to duplicate the heavy ones. So the algorithm needs a stage before packing: *replicate*.

But replication isn't free — I have a fixed budget. The runtime gives me `num_phy` physical slots total (a multiple of the GPU count, so every GPU gets the same number of slots), and `num_log` logical experts, so I have exactly `num_phy − num_log` extra replicas to hand out, and every logical expert needs at least one copy or its tokens have nowhere to go. The question is which experts get the extra copies. Picture handing out replicas one at a time. Right now expert `i` has some count `c_i` of replicas and load `w_i`, so each of its replicas carries `w_i / c_i`. If I give it one more, its per-replica load drops from `w_i / c_i` to `w_i / (c_i + 1)`. Which expert should get the next replica? My instinct says the one whose replicas are currently the most loaded — the largest `w_i / c_i` — because that is the per-replica load currently setting the ceiling. Let me make sure that instinct survives a second look rather than just sounding right. I'm trying to minimize the maximum over all replicas of the per-replica load. At any point that maximum is `max_i w_i / c_i`. If I add a replica to *any* expert that isn't currently at that argmax, the load on the argmax expert is untouched, so the maximum doesn't move at all — the replica was wasted on a non-bottleneck. The only move that can lower the current ceiling is to add a replica to one of the experts achieving it (and if several are tied, at least one of the tied ceilings comes down). So feeding the argmax of `w_i / c_i` is the only myopic step that attacks the current bottleneck. It's the discrete analogue of equalizing per-replica loads, and it's cheap. That matches the greedy I had in mind, and the small numeric check above already showed it does the right thing — the first replica went to the hot expert, exactly the argmax.

```python
def replicate_experts(weight: torch.Tensor, num_phy: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    device = weight.device
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    arangen = torch.arange(n, dtype=torch.int64, device=device)
    for i in range(num_log, num_phy):
        redundant_indices = (weight / logcnt).max(dim=-1).indices
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]
        logcnt[arangen, redundant_indices] += 1
    return phy2log, rank, logcnt
```

That loop is also sequential — `argmax` depends on the counts updated by all prior iterations — so again Python and CPU. Fine.

Now I have a second worry, and it's the one the pure-balance thinking completely misses. I said "put the replicas on different GPUs." But which GPUs? If I just throw all the replicas into one big global pack across every GPU in the cluster, greedy may make the GPU loads look attractive — and it can scatter expert `i`'s replicas across whatever GPUs happened to be least loaded, which means across *different nodes*. And that's a problem for a reason the balance number can't see. The GPUs aren't a flat pool; they're grouped into nodes, with fast NVLink inside a node and slow, scarce InfiniBand between nodes. The whole reason the router uses node-limited routing — each token restricted to at most a few nodes — is to keep the all-to-all mostly intra-node. If I scatter an expert's replicas across many nodes, then a token that wants that expert might have to cross to a far node to reach the particular replica it's assigned, and I've blown up exactly the inter-node traffic the routing was carefully designed to bound. So a placement that's strong on raw GPU balance can be terrible on communication, and the communication cost is invisible to the balance metric I've been optimizing. I need to balance the load *and* keep each expert's replicas on as few nodes as possible.

The experts already come in groups, and the routing already keeps a token's experts clustered onto a few nodes. So the locality I want is: keep a whole expert group, and all the replicas of the experts in it, on a single node. If group membership never crosses a node boundary, then an expert's replicas all live on its group's node, and an expert is never scattered across nodes. That reframes the whole thing as a nested, hierarchical packing instead of one flat global pack. First decide which groups go on which nodes; then, *within* each node, do the replication and the GPU packing. The replication and the GPU-level packing become intra-node operations, so by construction nothing leaves its node.

Three stages, then. Stage one: pack the groups onto the nodes. Each group has a total load — the sum of its experts' loads — so this is just my balanced-packing primitive again, items = groups, packs = nodes, `items_per_pack = num_groups / num_nodes` groups per node, balancing the per-node load. Stage two: within each node, replicate. Each node now owns `num_log / num_nodes` logical experts and gets `num_phy / num_nodes` physical slots, so I run `replicate_experts` per node on that node's slice. Stage three: within each node, pack the replicas onto that node's GPUs — balanced packing again, items = the node's physical replicas, packs = the node's GPUs, `items_per_pack = num_phy / num_gpus` (the per-GPU slot count). Three greedy packs, two of them with the same primitive, one replication in the middle, and locality should fall out for free because stages two and three never cross a node.

That "for free" is exactly the kind of thing I keep telling myself; I want to actually see it before I trust it. Let me build a small full instance and check that no expert ends up split across nodes. Take `E = 8` experts, `G = 4` groups, `N = 2` nodes, `D = 4` GPUs, budget `R = 12` slots, with two clearly hot experts. Running the three stages and reading out the final `phy2log` / `log2phy`, every node owns 6 slots, so a slot's node is `slot // 6`. Listing each expert's slots and the set of nodes they touch: every expert's replicas land on a *single* node — the two hot experts each get 3 replicas, and all 3 stay together on one node. No expert is split. So the hierarchy delivers the locality I was hoping for, and it's not an assumption — I can read it off the produced maps.

While I have a concrete instance running, let me also check the constraints the runtime demands, because it's cheap to verify and expensive to get wrong in production. On a `E=4, G=2, N=2, D=4, R=8` case with one very hot expert: `logcnt` sums to 8, which is exactly `num_replicas`, so the slot budget is spent precisely; every expert has `logcnt ≥ 1`, so nobody is left with no home; and `log2phy` is a correct inverse of `phy2log` — for every expert, the slots `log2phy` lists for it are exactly the slots whose `phy2log` points back at it, with `-1` in the leftover positions. So the three returned maps are mutually consistent, not just individually plausible.

Let me also sanity-check that the global, no-group case is just a degenerate instance of this rather than a separate algorithm. If I don't care about groups at all (say I'm decoding with a huge EP size where the locality story is different), I'd want to replicate and pack globally over all GPUs ignoring node structure. That's exactly this hierarchy with `num_groups = 1` and `num_nodes = 1`: one giant "node" holding everything, stage one is trivial (one group, one node), and stages two and three replicate and pack across the whole cluster. I can test this directly: take an instance where `num_groups % num_nodes != 0` so the entry point falls back to the global branch, and compare its output against an explicit `hierarchical(weight, R, 1, 1, num_gpus)` call — the `phy2log` and `logcnt` come out identical. So the hierarchical procedure with the group/node counts collapsed to one *is* the global policy. One algorithm, two regimes. For the deployment I care about here the node count divides the group count, so the genuine hierarchical branch runs.

Now I have to actually wire the three stages together, and this is where the bookkeeping gets fiddly, because each stage produces a permutation and I have to compose them to end up with the maps the serving runtime wants. Let me be precise about what I have to return. Three tensors: `phy2log`, of shape `[L, num_replicas]`, saying which logical expert each physical slot serves; `log2phy`, of shape `[L, E, max_replicas]`, the inverse, listing the physical slots for each logical expert with `-1` padding where an expert has fewer than the max number of replicas; and `logcnt`, `[L, E]`, the replica count per logical expert. Let me build them stage by stage and keep careful track of index spaces.

A tool I'll need repeatedly: given a permutation `perm` where `perm[k]` is "where element `k` went," I want its inverse, "what landed at position `k`." That's a scatter: `inv[perm[k]] = k`.

```python
def inv(perm):
    out = torch.empty_like(perm)
    out.scatter_(1, perm, torch.arange(perm.size(1), dtype=torch.int64,
                                       device=perm.device).expand(perm.shape))
    return out
```

Stage one. I compute the per-group load by reshaping the `[L, E]` weights into `[L, num_groups, group_size]` and summing the last axis: `tpg = weight.unflatten(-1, (num_groups, group_size)).sum(-1)`. Pack those groups onto nodes: `gpi, grk = balanced_packing(tpg, num_nodes)`, giving each group a node (`gpi`) and a rank within that node (`grk`). Now I want a *relabeling* of the logical experts so that experts on the same node sit in a contiguous block — that's what makes stages two and three able to slice "this node's experts" with a plain reshape. A group's position in the node-ordering is `gpi * groups_per_node + grk` (node index times groups-per-node, plus its rank within the node), and multiplying by `group_size` gives the starting expert offset of that group in the relabeled space; adding `arange(group_size)` spreads it across the group's experts:

```python
log2mlog = (((gpi * groups_per_node + grk) * group_size).unsqueeze(-1)
            + torch.arange(group_size, dtype=torch.int64, device=gpi.device)).flatten(-2)
mlog2log = inv(log2mlog)
```

So `log2mlog` sends each true logical expert to its slot in the node-major ordering, and `mlog2log` brings it back. The "mlog" space is "logical experts, reordered so each node's experts are contiguous."

Stage two. Gather the weights into node-major order and chop into per-node slabs: `tpm = weight.gather(-1, mlog2log).view(-1, experts_per_node)`, which is `[L * num_nodes, experts_per_node]` — each row is one node's experts in one layer. Replicate within each such row up to `replicas_per_node = num_phy / num_nodes` slots: `p2m, prk, mcnt = replicate_experts(tpm, replicas_per_node)`. Now `p2m` says, for each physical slot in a node, which node-major-logical expert it serves; `prk` is the replica rank; `mcnt` is the per-expert replica count, per node.

Stage three. Within each node, I want to pack its physical replicas onto its GPUs by the *per-replica* load. The per-replica load of a slot is its expert's total load divided by that expert's replica count, gathered to the physical-slot ordering: `tpp = (tpm / mcnt.float()).gather(-1, p2m)`. Pack onto the node's GPUs: `pi, ri = balanced_packing(tpp, gpus_per_node)`. A slot's final position within the node is `pi * phy_per_gpu + ri` (its GPU index times the per-GPU slot count, plus its rank on that GPU): `p2pp = pi * phy_per_gpu + ri`, and I invert it, `pp2p = inv(p2pp)`, to get "which slot landed at each final node-local position."

Now the composition, which is the part I have to get exactly right. I have, per node, a mapping from final node-local physical position back through the GPU packing to the slot, and from the slot to the node-major-logical expert. Compose them: `pp2m = p2m.gather(-1, pp2p)` — for each final position, follow it to its slot, then to the node-major-logical expert it serves. But that's still *node-local* logical ids, in the `experts_per_node` range; I have to lift them back to global. Reshape to `[L, num_nodes, replicas_per_node]` and add each node's base offset into the mlog space, which is `node_index * experts_per_node` — i.e. `arange(0, E, experts_per_node)` broadcast over the node axis — then flatten the node and slot axes back together:

```python
pp2m = (pp2m.view(L, num_nodes, -1)
        + torch.arange(0, E, experts_per_node, device=pp2m.device).view(1, -1, 1)).flatten(-2)
```

That gives, for every physical slot in the whole layer, the node-major-logical expert it serves. Map node-major-logical back to true logical with the stage-one inverse, and I have `phy2log`:

```python
pp2log = mlog2log.gather(-1, pp2m)
```

The replica rank of each final slot comes along the same path: `pprank = prk.gather(-1, pp2p).view(L, -1)`. And the per-logical-expert count: `mcnt` is per-node-major-logical, so gather it back through `log2mlog` to true-logical order: `logcnt = mcnt.view(L, -1).gather(-1, log2mlog)`.

Last, `log2phy`, the inverse map from logical expert to its physical slots. I allocate `[L, E, max_replicas]` filled with `-1`, where `max_replicas` is the largest replica count any expert ended up with, `logcnt.max()`. Then for each physical slot `s` (running `0..num_replicas - 1`), I know it serves logical expert `pp2log[s]` at replica rank `pprank[s]`, so it belongs at `log2phy[layer, pp2log[s], pprank[s]] = s`. Flatten the last two axes of `log2phy` and scatter: the flat target index is `pp2log * max_replicas + pprank`, and the values scattered there are `arange(num_replicas)`:

```python
mx = logcnt.max().item()
log2phy = torch.full((L, E, mx), -1, dtype=torch.int64)
log2phy.view(L, -1).scatter_(-1, pp2log * mx + pprank,
                             torch.arange(num_replicas, dtype=torch.int64,
                                          device=log2phy.device).expand(L, -1))
```

The `-1`s that survive are exactly the unused slots for experts with fewer than `max_replicas` copies — which is the padding contract the runtime expects. (And this is precisely the consistency I checked on the small instance above: the surviving slots per expert matched its `logcnt`, and each pointed back to that expert under `phy2log`.)

Let me put the whole thing together in the form I'd actually ship. I need the helper packer, the replica allocator, the hierarchical composition, and the public entry point that falls back to a single global group when the group/node divisibility needed for the hierarchy is absent. Each block ties back to the reasoning: stage one packs groups onto nodes for node balance and locality; stage two greedily replicates the hot experts within each node to shrink the un-balanceable items; stage three greedily packs the replicas onto each node's GPUs for per-GPU balance; the rest is index plumbing to assemble the maps.

```python
from typing import Tuple
import torch


def balanced_packing(weight: torch.Tensor, num_packs: int) -> Tuple[torch.Tensor, torch.Tensor]:
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(weight.size(-1), dtype=torch.int64, device=weight.device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    indices = weight.float().sort(-1, descending=True).indices.cpu()
    pack_index = torch.full_like(weight, fill_value=-1, dtype=torch.int64, device='cpu')
    rank_in_pack = torch.full_like(pack_index, fill_value=-1)
    for i in range(num_layers):
        pack_weights = [0] * num_packs
        pack_items = [0] * num_packs
        for group in indices[i]:
            pack = min((i for i in range(num_packs) if pack_items[i] < groups_per_pack),
                       key=pack_weights.__getitem__)
            assert pack_items[pack] < groups_per_pack
            pack_index[i, group] = pack
            rank_in_pack[i, group] = pack_items[pack]
            pack_weights[pack] += weight[i, group]
            pack_items[pack] += 1
    return pack_index, rank_in_pack


def replicate_experts(weight: torch.Tensor, num_phy: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    device = weight.device
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    arangen = torch.arange(n, dtype=torch.int64, device=device)
    for i in range(num_log, num_phy):
        redundant_indices = (weight / logcnt).max(dim=-1).indices
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

    # Stage 1: pack groups to nodes.
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(tokens_per_group, num_nodes)
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) * group_size).unsqueeze(-1) +
                torch.arange(group_size, dtype=torch.int64, device=group_pack_index.device)).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Stage 2: construct redundant experts within nodes.
    tokens_per_mlog = weight.gather(-1, mlog2log).view(-1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(tokens_per_mlog, num_physical_experts // num_nodes)

    # Stage 3: pack physical experts to GPUs.
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


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()
    if num_groups % num_nodes == 0:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus)
    maxlogcnt = logcnt.max().item()
    log2phy: torch.Tensor = torch.full((num_layers, num_logical_experts, maxlogcnt),
                                       -1, dtype=torch.int64, device=logcnt.device)
    log2phy.view(num_layers, -1).scatter_(
        -1, phy2log * maxlogcnt + phyrank,
        torch.arange(num_replicas, dtype=torch.int64, device=log2phy.device).expand(num_layers, -1))
    return phy2log, log2phy, logcnt

__all__ = ['rebalance_experts']
```

One last thing I want to see, since the whole motivation was the max-GPU bottleneck: does the replication budget actually move the per-GPU balance, or have I just built elaborate plumbing? Let me take a Zipf-skewed load over 16 experts on `G=4, N=2, D=4` and compute `mean_gpu_load / max_gpu_load` (1.0 is a perfectly even peak) for growing budgets. With `R = 16` — no replication room, one physical per logical — the balance is 0.69. Give it 4 extra replicas (`R = 20`) and it climbs to 0.74; 16 extra (`R = 32`) and it reaches 0.77. Monotone improvement, and exactly in the direction the bottleneck argument predicted: the extra slots go to the hot experts, their per-replica load falls, and the worst GPU lightens. So the replication stage isn't decorative — it's the part that actually buys headroom on the metric I care about, and the packing stages spend that headroom evenly.

Let me trace the causal chain once, end to end. The bottleneck in EP serving is the most-loaded GPU because the all-to-all is synchronous, and the loads are badly skewed, so I'm minimizing a maximum over parallel units — a makespan problem where the unconstrained greedy tools are cheap and well understood: arbitrary list scheduling has the `2 − 1/m` bound, and the largest-first LPT ordering tightens it to `4/3 − 1/(3m)`, a factor I checked sits exactly on my hand-built worst cases (`7/6` at `m=2`, `11/9` at `m=3`). The hardware forces equal item counts per unit, so I keep that largest-first least-loaded rule but constrain it to non-full packs rather than pretending the unconstrained theorem transfers unchanged. But greedy stalls when a single expert is much hotter than the fair share — I measured the max pack load floored at the hot item, above the fair share — so I make the item divisible by replicating it, spending a fixed budget of extra slots greedily on whichever expert currently has the largest per-replica load `w/c`, the only move that lowers the current ceiling, and the numeric check showed one well-placed replica already drops the worst per-replica load below the fair share. And packing the replicas globally can scatter an expert across nodes and detonate the inter-node traffic the node-limited routing was built to bound, so I nest the whole thing: pack groups onto nodes first, then replicate and pack onto GPUs *within* each node, which keeps every expert's replicas on one node — I read that locality straight off a worked instance, no expert split — and gives the global, group-agnostic policy as the same procedure with the group and node counts collapsed to one, which I confirmed produces identical maps. The rest is permutation bookkeeping — relabel logical experts into node-major order, replicate per node, pack per node, then compose the three permutations and scatter to assemble `phy2log`, `log2phy`, and `logcnt`, whose mutual consistency and budget/coverage constraints I verified on a small case. The greedy loops are sequential and so run on the CPU in plain Python: exact to this heuristic, simple, and slow, but faithful.
