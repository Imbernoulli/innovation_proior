Let me start from what actually hurts. I have an MoE model serving online, experts spread across GPUs under expert parallelism, and the per-expert token load is wildly uneven and keeps drifting — a few hot experts soak up most of the traffic, and which ones are hot changes with the workload. The layer can't finish until its slowest GPU finishes, so the most-loaded GPU sets the latency, and any imbalance is wasted silicon. So I periodically recompute a placement: how many replicas each logical expert gets, and which GPU each replica sits on, to flatten the per-GPU load. The hierarchical algorithm I already have does this well — it balances per-GPU and per-node load, and it keeps each expert's replicas on a single node so I don't pay the inter-node fabric, which on this hardware is several times slower than NVLink. The problem is purely that it's slow: on a medium deployment a single rebalance takes on the order of half a second, and this thing runs on the serving critical path, over and over, as the load shifts. Half a second of placement is itself a latency tax I can't afford. I want to keep the hierarchy, the replica-count rule, and the output maps, but replace the slow packing choice with something that preserves the same balance-and-locality intent without walking every item in Python.

So where does the time go? Not in the arithmetic. The whole algorithm is shuffling small integer indices around. The cost is in one routine, `balanced_packing`, and the way it's written. It takes n weighted items and has to split them into P packs of exactly n/P items each, as evenly as possible by weight — that's the group-to-node step (Stage 1) and the replica-to-GPU step (Stage 3). And it does it the textbook way: sort the items by weight descending, then walk them one at a time, and for each one scan all P packs to find the emptiest pack that still has a free slot, and drop the item there. This is the LPT instinct — largest items first, then the emptiest available bin — with an extra hardware slot cap. The classical LPT guarantee belongs to the unconstrained version, so I should not lean on that theorem as if it proved the capped routine; the thing I need from it is the operational rule that works well here. But look at the control flow: an outer Python loop over the L layers, an inner Python loop over the n items, and inside that a linear min-scan over the P packs. That's L·n·P iterations of interpreted Python, each one a `.item()` off a tensor and a min over a little list. Every quantity is tiny; the only thing that's expensive is that I'm doing it element-by-element in Python on a machine that wants batched, data-parallel tensor kernels. The GPU is sitting idle while Python crawls through indices.

The obvious instinct is "just vectorize the loop." But I can't, and it's worth being precise about *why*, because the obstruction is the whole problem. The greedy choice for item j is "which pack is emptiest right now" — and "right now" means after items 1 through j−1 have already been placed. The decision for each item depends on the running pack loads, which depend on every earlier decision. It's an inherently sequential recurrence: a prefix scan where each step's branch (the argmin) depends on the accumulated state. I can't turn "assign item j to the argmin of the current loads" into a single batched scatter, because there's no closed form for the current loads without simulating the whole prefix. This is the same reason you can't trivially parallelize a greedy bin-packer. Wall.

Let me back up and ask what I really need, not what the greedy does. The greedy is one *way* to get a balanced cardinality-constrained partition; it is not the only way, and I don't actually care about reproducing its decisions step for step — I care about getting packs whose sums are about as even as the greedy's, computed without a per-item running-load dependence. So the question becomes: is there an assignment of items to packs that I can write down as a *fixed function of each item's sorted position alone* — no running state — that still comes out balanced? If the pack an item goes to depends only on its rank in the sorted order and not on what happened to the other items, then I can compute it for all items at once with index arithmetic, and the sequential chain is broken.

Start with the crudest such fixed rule: plain round-robin over the sorted items. Item of rank 0 (heaviest) to pack 0, rank 1 to pack 1, …, rank P−1 to pack P−1, rank P back to pack 0, rank P+1 to pack 1, and so on — pack p gets sorted-ranks {p, P+p, 2P+p, …}. Purely positional, trivially vectorizable, and it gives exactly n/P items per pack for free since the ranks cycle. Does it balance? Let me think about the sums. Group the sorted items into consecutive rounds of P: round 0 is ranks 0..P−1, round 1 is ranks P..2P−1, etc. Round-robin hands round-0's items to packs 0,1,…,P−1 *in order*, so pack 0 gets the heaviest of that round and pack P−1 the lightest. Fine so far. But round 1 *also* runs in order 0,1,…,P−1, so pack 0 gets rank P (the heaviest of round 1) and pack P−1 gets rank 2P−1 (the lightest of round 1). And round 2 again. So pack 0 systematically collects the heaviest element of *every* round, and pack P−1 the lightest of every round. The disparities don't cancel — they *stack*. Pack 0 ends up overloaded, pack P−1 underloaded, by roughly the per-round spread times the number of rounds. That's worse than I can accept; round-robin's positional simplicity bought me nothing because it always pours the big end of each round into the same low-index pack. Wall again, but an *informative* one — the failure tells me exactly what to fix.

The fix stares back at me from the failure mode: the imbalance comes from every round running in the *same* direction, so the same pack always gets the heavy end. So alternate the direction. Run the even rounds forward — pack 0,1,…,P−1 — and the odd rounds *backward* — pack P−1,P−2,…,1,0. Now in round 0 pack 0 gets the heaviest item (rank 0) and pack P−1 the lightest (rank P−1); but in round 1, reversed, pack 0 gets the *lightest* of that round (rank 2P−1) and pack P−1 gets the *heaviest* of that round (rank P). So pack 0's two items are rank 0 and rank 2P−1 — a heavy one paired with a light one — while pack P−1's two items are rank P−1 and rank P, two middling adjacent ones. Round by round, each pack alternately gets the heavy end and the light end, so the heavy-light pairing within each pack cancels the spread instead of accumulating it. This is the snake: reverse direction each row. The assignment is still purely positional: pack of rank r depends only on r, through which round it's in and whether that round is even or odd. No running loads. So it vectorizes.

Let me write it as index math and convince myself it's exactly the right closed form. Walk the sorted items by their position r = 0,1,…,n−1. The round is `block = r // P` and the within-round offset is `off = r % P`. For an even block, the pack is just `off`; for an odd block, it's the mirror `P − 1 − off`. So

  pack(r) = off                  if block is even,
          = P − 1 − off          if block is odd,

i.e. `pack = where(block % 2 == 0, off, P − 1 − off)`. And the rank of the item *within* its pack — its slot index inside the pack — is just the round number, `block`, because pack p receives exactly one item per round, in round order, so the round number *is* how many items already went into that pack before it. Let me sanity-check the cardinality: with n divisible by P there are n/P rounds, each contributes exactly one item to each pack, so every pack ends with exactly n/P items and ranks 0..n/P−1. The capacity constraint that the greedy enforced with a `counts[p] < items_per_pack` filter is satisfied automatically here — the snake can never over-fill a pack, because the round structure deals one item to each pack per round. That's a small relief: I get the hard cardinality requirement for free instead of having to police it.

Now the real question — does this actually balance as well as the greedy, or did I just trade a good heuristic for a cheap-but-worse one? Let me reason about it on the kind of load sequence I actually feed it. After sorting, the weights are monotonically non-increasing, so the first round is forced: greedy places ranks 0..P−1 into empty packs, and with the usual lowest-index tie break that is the same forward sweep as the snake. After that I have to be more careful. Pack P−1 is the emptiest immediately after round 0, so greedy gives rank P to pack P−1, but the rest of the reversed round only follows if that new load has climbed high enough that pack P−2 is now the emptiest, then after pack P−2 receives rank P+1 it must climb high enough that pack P−3 is next, and so on. In other words, the snake is not a theorem for every sorted sequence; it is the closed form for the greedy order when each filled pack overtakes the next pack in the assumed alternating order. A locally affine descending block has exactly this property: over two consecutive blocks, pack p receives `w[p] + w[2P−1−p]`, and those sums are equal, so after the backward sweep the canonical greedy tie-break can start the next forward sweep from pack 0 again. Smooth sorted expert-load tails approximate the same condition, especially after the largest spikes have been split into replicas. So the precise claim is conditional: when the sorted sequence keeps the packs in that alternating emptiest-bin order, the snake is LPT written in closed form. When that condition only approximately holds, the snake is the fixed positional approximation to that greedy order.

I should check this in a small trace rather than just wave at "smoothness." Take P=4 and sorted weights 100,90,80,70,60,50,40,30,20,10,5,1. The snake sends sorted-ranks {0,7,8} to pack 0, {1,6,9} to pack 1, {2,5,10} to pack 2, {3,4,11} to pack 3 (rank 0 forward into pack 0; round 1 reversed so rank 4→pack 3, 5→pack 2, 6→pack 1, 7→pack 0; round 2 forward so 8→pack 0, 9→pack 1, 10→pack 2, 11→pack 3). After the first two rounds the pair sums are all 130, so the greedy tie-break can walk the next round in the same forward order; manually tracing the capped greedy gives the identical assignment and the same sums, 150, 140, 135, 131. This is the exact case I want the formula to capture: the sequential "emptiest bin" choices have collapsed to a deterministic alternating order.

Now I want to be honest about where it *isn't* the same, because a heuristic I don't understand the failure boundary of will bite me. Sorted order alone is not enough. If the weights have a cliff or strong curvature — a couple of huge items and then a long flat tail, or an exponential drop where the first item dwarfs the rest — greedy adapts by continuing to feed the genuinely emptiest pack, while the fixed pattern keeps following its alternating schedule. On arbitrary unsorted weights the two diverge even more, precisely because greedy looks at the running loads and the snake does not. But that is not the structure I am targeting. I sort the expert or replica loads first, the real traces are long-tailed rather than adversarially permuted, and Stage 3 is friendlier than the raw experts because the replication stage has already split hot experts into multiple copies and smoothed the per-replica loads. So the right way to see this is: the snake is a closed-form positional realization of LPT when the sorted sequence keeps the alternating emptiest-bin order, and a low-cost approximation to that order on the smoothed load sequences where I need speed.

There's one more thing the positional rule has to deliver besides the pack id: the within-pack rank, which downstream code uses to compute each item's flat slot index. The greedy returned `rank_in_pack` = how many items were already in the chosen pack. In the snake that's just the round number `block`, as I noted — and crucially it's still positional, so it scatters back just as cleanly. So `balanced_packing` returns, for every item, `pack_index` and `rank_in_pack`, both computed from the sorted position with arange/where, then scattered from sorted order back to the original item order with a single `scatter_`. Let me make sure I get that scatter direction right, because it's the kind of thing that silently corrupts the maps. I computed `pack(r)` and `rank(r)` indexed by *sorted position* r. But the caller wants them indexed by *original item id*. `sorted_idx[r]` is the original id of the item at sorted position r. So I have, in sorted order, a vector of pack assignments; I want to write the pack assignment for sorted-position r into output slot `sorted_idx[r]`. That's exactly `pack_index.scatter_(-1, sorted_idx, pack_assign_over_sorted_positions)` — scatter the per-sorted-position values into the locations named by `sorted_idx`. Same for the ranks. After that, `pack_index[original_id]` is the pack of that item. Good — one sort, a couple of arange/where ops, two scatters, no Python loop over items at all, and the per-layer dimension B just rides along as a batch dimension since the pattern is identical for every layer.

Let me make the cost concrete to confirm this is the kind of win I'm after. The greedy was L·n·P interpreted iterations with per-element tensor reads. The snake is: one batched `sort` over the [B, n] tensor, a couple of `arange`/`where`/integer ops on length-n vectors, and two `scatter_`s — a handful of kernels whose count is *independent of n* (n only sets the size of each kernel's contiguous work, which the hardware eats in parallel). That's the difference between Python crawling through items and tensor kernels doing the same index construction in bulk, while keeping the equal-cardinality guarantee by construction.

Now, the rest of the algorithm. There are two `balanced_packing` calls, Stages 1 and 3, and both become the snake — and I want to flag to myself that doing it in *both* stages is what matters. If I only snake-ified Stage 1 (groups to nodes) and left Stage 3 (replicas to GPUs) as the loop, I'd still be paying the Python tax on the larger of the two packings — Stage 3 packs num_replicas items, more than Stage 1's num_groups — so the bottleneck would barely move. The speedup only fully lands when *every* packing call is the vectorized snake. So: Stage 1, sum each routing group's load, snake-pack the G groups onto N nodes, which keeps each group's experts node-confined (locality) and balances node load. Stage 3, snake-pack the per-replica loads onto the gpus_per_node GPUs within each node, exactly n/P = phy_per_gpu replicas per GPU. Both the same routine.

Stage 2, the replication, is different, and I should think about whether it *also* needs vectorizing or whether I leave it as a loop. Its job: grow num_log experts to num_phy physical replicas so the maximum per-replica load is minimized, by repeatedly handing one more replica to whichever expert currently has the largest load-per-replica, weight/logcnt, then incrementing that expert's count. This is a different kind of recurrence than the packing one, and a genuinely sequential one: each new replica *changes* the per-replica load of the expert it's added to (load/2, then load/3, …), so the argmax for the next replica depends on the previous decision. There's no positional trick here — the whole point is that the decisions interact through the diminishing per-replica returns. But two things make this fine to leave as a loop. First, the number of iterations is only the *redundant* count, num_phy − num_log — the extra replicas, a few dozen to a couple hundred — not the whole tensor, so it's a short loop, and it runs after the two big packings are already vectorized away. Second, the rule itself is the right one and I don't want to perturb it: replicating the single most-overloaded expert each time is the greedy that drives the peak per-replica load down, and the per-replica-load metric (weight/logcnt) is what makes "overloaded" mean overloaded-relative-to-its-current-copies. So Stage 2 stays a sequential argmax loop, vectorized only across the batch (layers·nodes) dimension, not across the replica draws. The honest accounting: the packings were the L·n·P bottleneck, the replication is L·(num_redundant)·(cheap argmax) which is small, so killing the packing loops gets essentially all the speedup.

Then the bookkeeping that stitches the three stages together — and I want to keep it identical to what already works, because it's index plumbing where an off-by-one silently mis-routes tokens. Stage 1 produces, per item, a (node, rank-in-node) from the snake, which I fold into a permutation `log2mlog` taking a logical expert id to its position in a node-major "mlog" layout: `(node * groups_per_node + rank) * group_size + within_group_offset`. Its inverse `mlog2log` lets Stage 2 gather the node-major loads into a [L·N, experts_per_node] tensor and replicate within each node independently. Stage 3 computes per-replica loads `tpm/mcnt` gathered through the replication map, snake-packs them to GPUs, builds the slot index `pack*phy_per_gpu + rank`, inverts it, and composes the maps back out to logical ids, adding the per-node expert offset so node k's local ids map to the global range. Finally I scatter the physical slot indices into the `log2phy` map (logical expert → its physical replica slots, −1 where unused), sized by the max replica count. All of that composition is unchanged; the only thing I replaced is the body of `balanced_packing`.

One implementation detail to pin down: the packing does its index work on CPU. The sort, where, and scatter are tiny relative to the model, and the maps are integer permutations consumed by host-side dispatch logic; keeping them on CPU (the reference casts `weight.float().cpu()` up front) avoids launching a swarm of tiny GPU kernels for what is really index arithmetic, and keeps it deterministic. So `balanced_packing` returns its two index tensors on CPU. Not load-bearing for correctness, but it's the difference between "vectorized" and "vectorized *and* not death-by-kernel-launch."

So let me write the one slot I'm actually changing — `balanced_packing` — as the snake, and keep the rest of the three-stage harness exactly as it stands:

```python
from typing import Tuple
import torch


def balanced_packing(weight: torch.Tensor, num_packs: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # Partition each row's n items into num_packs packs of exactly n//num_packs items,
    # balancing pack sums — vectorized "snake" (boustrophedon) assignment, no per-item loop.
    B, n = weight.shape
    assert n % num_packs == 0

    if n // num_packs == 1:                          # one item per pack -> identity
        idx = torch.arange(n, dtype=torch.int64, device=weight.device).expand(B, -1)
        return idx, torch.zeros_like(idx)

    sorted_idx = weight.float().sort(-1, descending=True).indices   # heaviest first (LPT order)

    positions   = torch.arange(n, device=weight.device)
    block_id    = positions // num_packs             # which round of P this position is in
    pos_in_block = positions % num_packs             # offset within the round
    is_even     = block_id % 2 == 0
    # even rounds sweep packs 0..P-1; odd rounds sweep P-1..0  -> heavy/light pair per pack
    pack_assign = torch.where(is_even, pos_in_block, num_packs - 1 - pos_in_block)
    rank_assign = block_id                           # within-pack slot = round number

    # scatter the per-sorted-position assignments back to original item order
    pack_expanded = pack_assign.unsqueeze(0).expand(B, -1)
    rank_expanded = rank_assign.unsqueeze(0).expand(B, -1)
    pack_index   = torch.zeros(B, n, dtype=torch.int64, device=weight.device)
    rank_in_pack = torch.zeros(B, n, dtype=torch.int64, device=weight.device)
    pack_index.scatter_(-1, sorted_idx, pack_expanded)
    rank_in_pack.scatter_(-1, sorted_idx, rank_expanded)

    return pack_index.cpu(), rank_in_pack.cpu()


def replicate_experts(weight: torch.Tensor, num_phy: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Grow num_log experts to num_phy replicas; each extra replica goes to the expert with
    # the largest current per-replica load. Genuinely sequential in the replica draws (the
    # per-replica load changes as copies are added), but only num_phy-num_log iterations.
    B, num_log = weight.shape
    device = weight.device
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(B, 1)
    rank = torch.zeros(B, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(B, num_log, dtype=torch.int64, device=device)
    idx_b = torch.arange(B, dtype=torch.int64, device=device)
    for i in range(num_log, num_phy):
        eff = weight / logcnt.float()                # per-replica load
        top = eff.argmax(dim=-1)                     # most-overloaded expert
        phy2log[:, i] = top
        rank[:, i] = logcnt[idx_b, top]
        logcnt[idx_b, top] += 1
    return phy2log, rank, logcnt


def rebalance_experts(weight: torch.Tensor, num_replicas: int, num_groups: int,
                      num_nodes: int, num_gpus: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    L, E = weight.shape
    weight = weight.float().cpu()
    group_size = E // num_groups
    gpus_per_node = num_gpus // num_nodes
    phy_per_gpu = num_replicas // num_gpus
    groups_per_node = num_groups // num_nodes
    experts_per_node = E // num_nodes
    replicas_per_node = num_replicas // num_nodes

    def inv(perm):                                    # per-row inverse permutation
        out = torch.empty_like(perm)
        out.scatter_(1, perm, torch.arange(perm.size(1), dtype=torch.int64).expand(perm.shape))
        return out

    # Stage 1: snake-pack routing groups onto nodes (balances node load, keeps a group node-local)
    tpg = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    gpi, grk = balanced_packing(tpg, num_nodes)
    log2mlog = (((gpi * groups_per_node + grk) * group_size).unsqueeze(-1)
                + torch.arange(group_size)).flatten(-2)
    mlog2log = inv(log2mlog)

    # Stage 2: replicate hot experts within each node (sequential, short)
    tpm = weight.gather(-1, mlog2log).view(-1, experts_per_node)
    p2m, prk, mcnt = replicate_experts(tpm, replicas_per_node)

    # Stage 3: snake-pack the per-replica loads onto GPUs within each node
    tpp = (tpm / mcnt.float()).gather(-1, p2m)        # per-replica load, already smoothed
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

The causal chain: the rebalancer is correct and well-balanced but spends half a second per call because its bin-packing core is a Python loop over every item, each doing a running-load argmin — and that running-load dependence is exactly why the loop itself can't be vectorized. Round-robin over the sorted items breaks the dependence but stacks the heavy end of every round into the same pack and overloads it. Reversing the direction every round — the snake — pairs the heavy side of one round with the light side of the next, so the round spreads cancel when the sorted weights keep the alternating emptiest-bin order. Under that condition this is LPT-greedy without the loop; on the smoothed expert-load sequences it is the cheap positional approximation to the same behavior. Applying it to both packing stages, keeping the genuinely sequential but short replication loop, and leaving the index-composition bookkeeping untouched gives the final method.
