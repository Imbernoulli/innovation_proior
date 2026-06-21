A Mixture-of-Experts model served online under expert parallelism keeps recomputing one thing: a placement plan that decides how many physical replicas each logical expert gets and which GPU each replica sits on. It has to, because the per-expert token load is wildly skewed — a Zipf-like tail means a few hot experts soak up most of the traffic — and worse, which experts are hot drifts as the workload changes. Since a layer cannot finish until its slowest GPU finishes, the most-loaded GPU sets the latency, and any imbalance is wasted silicon. So the rebalancer runs periodically off a moving average of recent per-expert token counts, splitting the hottest experts into replicas and spreading them so the per-GPU load flattens. The plan must do three things at once: balance load both across GPUs and across the server nodes that group them; keep each expert's replicas (and each routing group's experts) confined to a single node, because the intra-node interconnect (NVLink, $\sim160$ GB/s) is several times faster than the inter-node fabric (InfiniBand, $\sim50$ GB/s), and node-limited routing only pays off if placement honors that locality; and be cheap, because it runs over and over on the serving critical path.

The established hierarchical algorithm gets the first three right. Stage 1 sums each routing group's load and packs the groups onto nodes, balancing node load while keeping a group node-local. Stage 2 replicates hot experts within each node. Stage 3 packs the resulting physical replicas onto the GPUs inside each node. The trouble is purely speed: on a medium deployment a single rebalance takes on the order of $540$ ms, with the imbalance factor around $0.66$. That half-second is itself a latency tax. And the cost is not the arithmetic — the whole algorithm is shuffling small integer indices. It all goes into one routine, `balanced_packing`, written the textbook way: it takes $n$ weighted items, must split them into $P$ packs of exactly $n/P$ items each as evenly as possible, and does so with the LPT instinct — sort descending, then walk the items one at a time, and for each scan all $P$ packs to find the emptiest one that still has a free slot, and drop the item there. The control flow is an outer Python loop over the $L$ layers, an inner Python loop over the $n$ items, and a linear min-scan over $P$ packs inside that: $L\cdot n\cdot P$ interpreted iterations, each a `.item()` read off a tensor, on a machine that wants batched, data-parallel kernels. The flat (non-hierarchical) alternative — replicate globally and pack all replicas onto all GPUs in one shot — is no escape: it removes the group-to-node stage but scatters an expert's replicas across the whole cluster, giving up the very locality the node-limited routing was designed to buy.

The obstruction to simply vectorizing the loop is the whole problem, and it is worth being precise about it. The greedy choice for item $j$ is "which pack is emptiest right now," and "right now" means after items $1$ through $j-1$ have already been placed. Each decision depends on the running pack loads, which depend on every earlier decision — an inherently sequential prefix recurrence whose branch (the argmin) depends on accumulated state. There is no closed form for the current loads without simulating the whole prefix, which is exactly why greedy bin-packers do not parallelize.

The method I propose is Zigzag — a vectorized snake (boustrophedon) packing that replaces the greedy loop. The move is to stop reproducing greedy's step-by-step decisions and instead find an assignment of items to packs that is a fixed function of each item's sorted position alone, with no running state, so it can be computed for all items at once with index arithmetic. The crudest such rule is plain round-robin over the sorted items: rank $r$ to pack $r \bmod P$, so pack $p$ collects sorted-ranks $\{p, P+p, 2P+p, \dots\}$. It is purely positional, trivially vectorizable, and gives exactly $n/P$ items per pack for free. But it fails to balance, and the failure is informative. Group the sorted items into consecutive rounds of $P$. Round-robin hands every round to packs $0,1,\dots,P-1$ in the same order, so pack $0$ systematically gets the heaviest element of every round and pack $P-1$ the lightest of every round. The disparities do not cancel — they stack, by roughly the per-round spread times the number of rounds, and pack $0$ ends up badly overloaded.

The fix stares back from the failure mode: every round runs in the same direction, so reverse the direction each round. Run even rounds forward ($0,1,\dots,P-1$) and odd rounds backward ($P-1,P-2,\dots,1,0$). Now in round $0$, pack $0$ gets the heaviest item (rank $0$) and pack $P-1$ the lightest (rank $P-1$); but in round $1$, reversed, pack $0$ gets the lightest of that round (rank $2P-1$) and pack $P-1$ gets the heaviest (rank $P$). So pack $0$ pairs a heavy item with a light one while pack $P-1$ takes two adjacent middling ones, and the heavy-light pairing within each pack cancels the spread instead of accumulating it. This is the snake, and it is still purely positional. Writing $r=0,1,\dots,n-1$ for the sorted position, with round $\text{block}=\lfloor r/P\rfloor$ and offset $\text{off}=r \bmod P$,
$$\text{pack}(r) = \begin{cases}\text{off}, & \text{block even}\\ P-1-\text{off}, & \text{block odd}\end{cases}$$
i.e. `pack = where(block % 2 == 0, off, P - 1 - off)`. The within-pack rank — the item's slot index inside its pack — is just the round number $\text{block}$, because each pack receives exactly one item per round in round order, so the round number is how many items already went into that pack. Cardinality comes for free: with $n$ divisible by $P$ there are $n/P$ rounds, each dealing one item to each pack, so every pack ends with exactly $n/P$ items and ranks $0\dots n/P-1$ — the `counts[p] < items_per_pack` filter the greedy needed is satisfied by construction.

The honest question is whether this balances as well as greedy or just cheaply worse, and the answer is conditional. After sorting, the weights are monotonically non-increasing, so the first round is forced: greedy places ranks $0\dots P-1$ into empty packs, the same forward sweep as the snake. The snake then equals the greedy order exactly when each just-filled pack overtakes the next pack in the assumed alternating order — when, after pack $P-1$ receives rank $P$, pack $P-2$ is the emptiest, then after pack $P-2$ receives rank $P+1$, pack $P-3$ is the emptiest, and so on. A locally affine descending block has precisely this property, because over two consecutive rounds pack $p$ receives $w[p]+w[2P-1-p]$, and those pair-sums are constant across $p$, so the forward-plus-backward pair equalizes the pack sums and the canonical greedy tie-break can restart from pack $0$. A small trace makes it concrete: with $P=4$ and sorted weights $100,90,80,70,60,50,40,30,20,10,5,1$, the snake sends sorted-ranks $\{0,7,8\}$ to pack $0$, $\{1,6,9\}$ to pack $1$, $\{2,5,10\}$ to pack $2$, $\{3,4,11\}$ to pack $3$; after the first two rounds the pair-sums are all $130$, and tracing the capped greedy by hand gives the identical assignment with sums $150,140,135,131$. So under that alternating emptiest-bin condition the snake is LPT-greedy written in closed form. Where the condition only approximately holds — smooth long-tailed expert-load sequences, and especially Stage 3's per-replica loads after replication has shaved the peaks — it is the cheap positional approximation to the same behavior. It diverges from greedy only on arbitrary unsorted weights, or sorted weights with cliffs or strong curvature, because greedy reads the running loads and the snake does not; but I always sort first and the real traces are long-tailed rather than adversarial, so that is not the regime I serve.

The cost is now the win I was after. The greedy was $L\cdot n\cdot P$ interpreted iterations with per-element tensor reads; the snake is one batched `sort` over the $[B,n]$ tensor, a couple of `arange`/`where`/integer ops on length-$n$ vectors, and two `scatter_`s — a handful of kernels whose count is independent of $n$ ($n$ only sets the size of each kernel's contiguous work, which the hardware eats in parallel), with the per-layer dimension $B$ riding along as a batch. The one subtlety is the scatter direction: $\text{pack}(r)$ and $\text{rank}(r)$ are indexed by sorted position $r$, but the caller wants them indexed by original item id, and `sorted_idx[r]` is the original id of the item at sorted position $r$, so writing the per-sorted-position values into the slots named by `sorted_idx` — `pack_index.scatter_(-1, sorted_idx, pack_over_positions)` — yields `pack_index[original_id]` correctly. The index work stays on CPU (the reference casts `weight.float().cpu()` up front), because these are tiny integer permutations consumed by host-side dispatch; keeping them on CPU avoids launching a swarm of micro-kernels and stays deterministic.

Both `balanced_packing` calls — Stage 1 (groups to nodes) and Stage 3 (per-replica loads to GPUs) — become the snake, and doing both is what matters: Stage 3 packs more items than Stage 1, so snaking only Stage 1 would leave most of the Python tax in place. Stage 2, the replication, stays a sequential loop, and deliberately so. Its recurrence is genuinely sequential and has no positional trick — each new replica changes the per-replica load of the expert it joins ($\text{load}/2$, then $\text{load}/3$, …), so the argmax for the next replica depends on the last decision — but it iterates only over the redundant count $\text{num\_phy}-\text{num\_log}$, a few dozen to a couple hundred draws, not the whole tensor, and it runs after the two big packings are vectorized away. The rule itself is the right one: handing the extra replica to the expert with the largest per-replica load $\text{weight}/\text{logcnt}$ is the greedy that drives the peak per-replica load down, where dividing by the current replica count is what makes "overloaded" mean overloaded relative to its own copies. The index-composition bookkeeping that stitches the three stages together — folding Stage 1's (node, rank) into the node-major `log2mlog` permutation and its inverse, gathering loads per node for replication, building the GPU slot index `pack*phy_per_gpu + rank` and composing back to logical ids with the per-node expert offset, then scattering physical slots into the `log2phy` map — is left exactly as it already works, because that is index plumbing where an off-by-one silently mis-routes tokens. The only body I change is `balanced_packing`.

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
