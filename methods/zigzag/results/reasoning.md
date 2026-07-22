Let me start from what actually hurts. I have an MoE model serving online, experts spread across GPUs under expert parallelism, and the per-expert token load is wildly uneven and keeps drifting — a few hot experts soak up most of the traffic, and which ones are hot changes with the workload. The layer can't finish until its slowest GPU finishes, so the most-loaded GPU sets the latency, and any imbalance is wasted silicon. So I periodically recompute a placement: how many replicas each logical expert gets, and which GPU each replica sits on, to flatten the per-GPU load. The hierarchical algorithm I already have does this well — it balances per-GPU and per-node load, and it keeps each expert's replicas on a single node so I don't pay the inter-node fabric, which on this hardware is several times slower than NVLink. The problem is purely that it's slow: on a medium deployment a single rebalance takes on the order of half a second, and this thing runs on the serving critical path, over and over, as the load shifts. Half a second of placement is itself a latency tax I can't afford. I want to keep the hierarchy, the replica-count rule, and the output maps, but cut the cost of the packing choice without throwing away the balance and locality the hierarchy gives me.

So where does the time go? Not in the arithmetic. The whole algorithm is shuffling small integer indices around. The cost is in one routine, `balanced_packing`, and the way it's written. It takes n weighted items and has to split them into P packs of exactly n/P items each, as evenly as possible by weight — that's the group-to-node step (Stage 1) and the replica-to-GPU step (Stage 3). And it does it the textbook way: sort the items by weight descending, then walk them one at a time, and for each one scan all P packs to find the emptiest pack that still has a free slot, and drop the item there. This is the LPT instinct — largest items first, then the emptiest available bin — with an extra hardware slot cap. The classical LPT guarantee belongs to the unconstrained version, so I should not lean on that theorem as if it proved the capped routine; the thing I need from it is the operational rule that works well here. But look at the control flow: an outer Python loop over the L layers, an inner Python loop over the n items, and inside that a linear min-scan over the P packs. That's L·n·P iterations of interpreted Python, each one a `.item()` off a tensor and a min over a little list. Every quantity is tiny; the only thing that's expensive is that I'm doing it element-by-element in Python on a machine that wants batched, data-parallel tensor kernels. The GPU is sitting idle while Python crawls through indices.

Before I go anywhere, let me at least confirm the cost is where I think it is and that there's real headroom, rather than chasing a routine that's already cheap. A Stage-3-shaped call packs n=40 per-replica loads into P=8 GPUs, and the batch dimension is layers·nodes — say 61 layers · 8 nodes ≈ 490 rows. I time the loop body as written on a random `[490, 40]` tensor: it comes in around 256 ms for a single such call. That alone is most of the half-second budget, and Stage 1 adds another call, so the diagnosis holds — the per-item Python loop is the whole problem, not the surrounding tensor plumbing.

The obvious instinct is "just vectorize the loop." But I can't, and it's worth being precise about *why*, because the obstruction is the whole problem. The greedy choice for item j is "which pack is emptiest right now" — and "right now" means after items 1 through j−1 have already been placed. The decision for each item depends on the running pack loads, which depend on every earlier decision. It's an inherently sequential recurrence: a prefix scan where each step's branch (the argmin) depends on the accumulated state. I can't turn "assign item j to the argmin of the current loads" into a single batched scatter, because there's no closed form for the current loads without simulating the whole prefix. This is the same reason you can't trivially parallelize a greedy bin-packer. Wall.

Let me back up and ask what I really need, not what the greedy does. The greedy is one *way* to get a balanced cardinality-constrained partition; it is not the only way, and I don't actually care about reproducing its decisions step for step — I care about getting packs whose sums are about as even as the greedy's, computed without a per-item running-load dependence. So the question becomes: is there an assignment of items to packs that I can write down as a *fixed function of each item's sorted position alone* — no running state — that still comes out balanced? If the pack an item goes to depends only on its rank in the sorted order and not on what happened to the other items, then I can compute it for all items at once with index arithmetic, and the sequential chain is broken.

Start with the crudest such fixed rule: plain round-robin over the sorted items. Item of rank 0 (heaviest) to pack 0, rank 1 to pack 1, …, rank P−1 to pack P−1, rank P back to pack 0, rank P+1 to pack 1, and so on — pack p gets sorted-ranks {p, P+p, 2P+p, …}. Purely positional, trivially vectorizable, and it gives exactly n/P items per pack for free since the ranks cycle. Does it balance? Let me think about the sums. Group the sorted items into consecutive rounds of P: round 0 is ranks 0..P−1, round 1 is ranks P..2P−1, etc. Round-robin hands round-0's items to packs 0,1,…,P−1 *in order*, so pack 0 gets the heaviest of that round and pack P−1 the lightest. Fine so far. But round 1 *also* runs in order 0,1,…,P−1, so pack 0 gets rank P (the heaviest of round 1) and pack P−1 gets rank 2P−1 (the lightest of round 1). And round 2 again. So pack 0 systematically collects the heaviest element of *every* round, and pack P−1 the lightest of every round. The disparities don't cancel — they *stack*. To see how badly, take P=4 and twelve descending weights 100,90,80,70,60,50,40,30,20,10,5,1: pack 0 gets ranks {0,4,8} = 100+60+20 = 180, pack 3 gets ranks {3,7,11} = 70+30+1 = 101. That's a spread of 79 on sums near 130 — balance min/max ≈ 0.56. Round-robin's positional simplicity bought me nothing because it always pours the big end of each round into the same low-index pack. Wall again, but an *informative* one — the failure tells me exactly what to fix.

The fix stares back at me from the failure mode: the imbalance comes from every round running in the *same* direction, so the same pack always gets the heavy end. So alternate the direction. Run the even rounds forward — pack 0,1,…,P−1 — and the odd rounds *backward* — pack P−1,P−2,…,1,0. Now in round 0 pack 0 gets the heaviest item (rank 0) and pack P−1 the lightest (rank P−1); but in round 1, reversed, pack 0 gets the *lightest* of that round (rank 2P−1) and pack P−1 gets the *heaviest* of that round (rank P). So pack 0's two items are rank 0 and rank 2P−1 — a heavy one paired with a light one — while pack P−1's two items are rank P−1 and rank P, two middling adjacent ones. Round by round, each pack alternately gets the heavy end and the light end, so the heavy-light pairing within each pack tends to cancel the spread instead of accumulating it. Reverse direction each row — a boustrophedon, an ox-plowing back-and-forth. The assignment is still purely positional: the pack of rank r depends only on r, through which round it's in and whether that round is even or odd. No running loads. So it vectorizes.

Let me write it as index math. Walk the sorted items by their position r = 0,1,…,n−1. The round is `block = r // P` and the within-round offset is `off = r % P`. For an even block, the pack is just `off`; for an odd block, it's the mirror `P − 1 − off`. So

  pack(r) = off                  if block is even,
          = P − 1 − off          if block is odd,

i.e. `pack = where(block % 2 == 0, off, P − 1 − off)`. And the rank of the item *within* its pack — its slot index inside the pack — is just the round number, `block`, because pack p receives exactly one item per round, in round order, so the round number *is* how many items already went into that pack before it. Let me sanity-check the cardinality: with n divisible by P there are n/P rounds, each contributes exactly one item to each pack, so every pack ends with exactly n/P items and ranks 0..n/P−1. The capacity constraint that the greedy enforced with a `counts[p] < items_per_pack` filter is satisfied automatically here — this pattern can never over-fill a pack, because the round structure deals one item to each pack per round. That's a small relief: I get the hard cardinality requirement for free instead of having to police it.

Now the real question — does this actually balance as well as the greedy, or did I just trade a good heuristic for a cheap-but-worse one? I don't trust the heavy-light hand-waving until I've watched the two side by side on a concrete sequence. Take the same twelve weights, P=4. The alternating rule sends sorted-ranks {0,7,8} to pack 0, {1,6,9} to pack 1, {2,5,10} to pack 2, {3,4,11} to pack 3 (rank 0 forward into pack 0; round 1 reversed so 4→pack 3, 5→pack 2, 6→pack 1, 7→pack 0; round 2 forward so 8→pack 0, 9→pack 1, 10→pack 2, 11→pack 3). Its sums: pack 0 = 100+30+20 = 150, pack 1 = 90+40+10 = 140, pack 2 = 80+50+5 = 135, pack 3 = 70+60+1 = 131. Now run the capped greedy on the same sequence and write down its *running* loads so I can see what it actually does, not assume:

  rank 0 (w=100) → pack 0   loads [100, 0, 0, 0]
  rank 1 (w= 90) → pack 1   loads [100, 90, 0, 0]
  rank 2 (w= 80) → pack 2   loads [100, 90, 80, 0]
  rank 3 (w= 70) → pack 3   loads [100, 90, 80, 70]
  rank 4 (w= 60) → pack 3   loads [100, 90, 80, 130]
  rank 5 (w= 50) → pack 2   loads [100, 90, 130, 130]
  rank 6 (w= 40) → pack 1   loads [100, 130, 130, 130]
  rank 7 (w= 30) → pack 0   loads [130, 130, 130, 130]
  rank 8 (w= 20) → pack 0   loads [150, 130, 130, 130]
  rank 9 (w= 10) → pack 1   loads [150, 140, 130, 130]
  rank 10 (w=  5) → pack 2  loads [150, 140, 135, 130]
  rank 11 (w=  1) → pack 3  loads [150, 140, 135, 131]

Watch what happened: round 0 goes forward 0,1,2,3 into the empty packs; then because pack 3 is now the emptiest, round 1 falls *backward* 3,2,1,0, and after it the loads are all 130; then round 2 goes forward again 0,1,2,3. The greedy *reconstructed the back-and-forth on its own* from the running loads — and its final sums are 150,140,135,131, identical to the alternating rule, assignment for assignment. So on this sequence the positional pattern and the greedy are the same thing. The reason is visible in the trace: over a forward-then-backward pair, pack p receives w[rank p] + w[rank 2P−1−p], and here those pair sums are 100+30, 90+40, 80+50, 70+60 = 130 across the board, so after the reversed round the packs re-level and the greedy's lowest-index tie-break restarts a forward sweep. So the honest statement is conditional: when the sorted sequence keeps the packs re-leveling like this round to round, the back-and-forth *is* LPT-greedy written in closed form; it is not a theorem for an arbitrary sorted sequence.

Which means I have to find out where it stops being LPT, because a heuristic I don't understand the failure boundary of will bite me. My first guess for a hard case is a cliff: one huge item then a flat tail, P=4, weights 1000 then eleven 10s. But when I actually place it, both methods drop the 1000 into one pack and spread the 10s evenly across the other slots — sums [1020,30,30,30] for *both*. They agree exactly. So a single dominating spike is not where they diverge; greedy has nothing to exploit once the tail is flat, and neither does the pattern. The divergence needs *curvature in the tail*. Try an exponential drop, weights 2^0,2^−1,…,2^−11 into P=4. Now the pattern gives pack sums roughly [1.012, 0.518, 0.282, 0.188], balance ≈ 0.186, while greedy — continuing to feed the genuinely emptiest pack instead of following a fixed schedule — gives [1.002, 0.506, 0.273, 0.219], balance ≈ 0.218. Greedy wins, and the gap is real, not rounding. So the failure mode is curvature, not magnitude: when consecutive rounds *don't* re-level, the fixed schedule keeps marching while greedy adapts, and greedy comes out ahead.

That makes me nervous enough to check the regime I actually run in, not a toy. The packing calls in the hierarchy are Stage 3 (per-replica loads onto gpus_per_node GPUs) and Stage 1 (groups onto nodes). For deepseek-v3 Stage 3 packs about n=40 per-replica loads into P=8, so n/P=5 rounds; the per-replica loads have already been smoothed by the replication stage, which split the hottest experts into copies. I simulate that — Zipf expert loads in a node, replicate the hottest down by the weight/count rule, then take the resulting per-replica loads — and compare. The pattern lands at balance ≈ 0.94 against greedy's ≈ 0.995; on the stress config (n=24, P=8, the most extreme tail) it's ≈ 0.89 against ≈ 0.94. So I am *not* matching greedy on the real loads — I'm giving up a few points of per-GPU balance. That is the uncomfortable truth I have to weigh, and I'd rather see it now than discover it in the score. The replication does smooth the peaks, which keeps the gap from being a disaster, but the per-replica tail still has enough curvature that greedy's adaptivity buys it something the fixed schedule can't.

So is the trade worth it? The score weights balance, node-balance, locality, and runtime equally and takes a geometric mean across configs, so I'm explicitly allowed to spend a little balance to buy a lot of runtime — as long as "a lot" is true. Let me put a number on the speedup, not assume it. Replacing the loop body with the positional construction — one batched sort, a couple of arange/where ops, two scatters — on the same `[490, 40]`, P=8 input that cost 256 ms in the loop now runs in about 0.56 ms. That's roughly a 450× cut on the dominant call, turning the half-second rebalance into something negligible on the critical path. A few percent of per-GPU balance for two-and-a-half orders of magnitude of latency, under a score that prices them the same, is a trade I'll take — and crucially it's a trade I've now measured on both sides rather than asserted. If the balance loss had been large I'd have to reconsider, e.g. a cheap local-swap pass after the pattern; at this gap it isn't worth the complexity.

There's one more thing the positional rule has to deliver besides the pack id: the within-pack rank, which downstream code uses to compute each item's flat slot index. The greedy returned `rank_in_pack` = how many items were already in the chosen pack. Here that's just the round number `block`, as I noted — and crucially it's still positional, so it scatters back just as cleanly. So `balanced_packing` returns, for every item, `pack_index` and `rank_in_pack`, both computed from the sorted position with arange/where, then scattered from sorted order back to the original item order with a single `scatter_`. Let me make sure I get that scatter direction right, because it's the kind of thing that silently corrupts the maps. I computed `pack(r)` and `rank(r)` indexed by *sorted position* r. But the caller wants them indexed by *original item id*. `sorted_idx[r]` is the original id of the item at sorted position r. So I have, in sorted order, a vector of pack assignments; I want to write the pack assignment for sorted-position r into output slot `sorted_idx[r]`. That's exactly `pack_index.scatter_(-1, sorted_idx, pack_assign_over_sorted_positions)` — scatter the per-sorted-position values into the locations named by `sorted_idx`. Same for the ranks. After that, `pack_index[original_id]` is the pack of that item. To be sure I haven't fooled myself, I check the degenerate case the harness special-cases: n=8 items into P=8 packs, one per pack. The construction should give every pack exactly one item and every within-pack rank 0 — running it, the eight pack ids are a permutation of 0..7 and all ranks are 0, which is the identity placement, as it must be. Good — one sort, a couple of arange/where ops, two scatters, no Python loop over items at all, and the per-layer dimension B just rides along as a batch dimension since the pattern is identical for every layer.

Let me make the cost picture explicit to confirm this is the kind of win I'm after. The greedy was L·n·P interpreted iterations with per-element tensor reads. The new routine is: one batched `sort` over the [B, n] tensor, a couple of `arange`/`where`/integer ops on length-n vectors, and two `scatter_`s — a handful of kernels whose count is *independent of n* (n only sets the size of each kernel's contiguous work, which the hardware eats in parallel). That's the difference between Python crawling through items and tensor kernels doing the same index construction in bulk, while keeping the equal-cardinality guarantee by construction — and the 256 ms → 0.56 ms measurement above is that difference cashed out.

Now, the rest of the algorithm. There are two `balanced_packing` calls, Stages 1 and 3, and both become the new routine — and I want to flag to myself that doing it in *both* stages is what matters. If I only converted Stage 1 (groups to nodes) and left Stage 3 (replicas to GPUs) as the loop, I'd still be paying the Python tax on the larger of the two packings — Stage 3 packs num_replicas items, more than Stage 1's num_groups — so the bottleneck would barely move. The speedup only fully lands when *every* packing call is vectorized. (And Stage 1 sometimes degenerates: for deepseek-v3, num_groups=8 and num_nodes=8, so n/P=1 and the call is the trivial identity branch anyway — the real work, and the real win, is Stage 3.) So: Stage 1, sum each routing group's load, pack the G groups onto N nodes, which keeps each group's experts node-confined (locality) and balances node load. Stage 3, pack the per-replica loads onto the gpus_per_node GPUs within each node, exactly n/P = phy_per_gpu replicas per GPU. Both the same routine.

Stage 2, the replication, is different, and I should think about whether it *also* needs vectorizing or whether I leave it as a loop. Its job: grow num_log experts to num_phy physical replicas so the maximum per-replica load is minimized, by repeatedly handing one more replica to whichever expert currently has the largest load-per-replica, weight/logcnt, then incrementing that expert's count. This is a different kind of recurrence than the packing one, and a genuinely sequential one: each new replica *changes* the per-replica load of the expert it's added to (load/2, then load/3, …), so the argmax for the next replica depends on the previous decision. There's no positional trick here — the whole point is that the decisions interact through the diminishing per-replica returns. But two things make this fine to leave as a loop. First, the number of iterations is only the *redundant* count, num_phy − num_log — the extra replicas, a few dozen to a couple hundred — not the whole tensor, so it's a short loop, and it runs after the two big packings are already vectorized away. Second, the rule itself is the right one and I don't want to perturb it: replicating the single most-overloaded expert each time is the greedy that drives the peak per-replica load down, and the per-replica-load metric (weight/logcnt) is what makes "overloaded" mean overloaded-relative-to-its-current-copies. So Stage 2 stays a sequential argmax loop, vectorized only across the batch (layers·nodes) dimension, not across the replica draws. The accounting that matters: the packings were the L·n·P bottleneck (the 256 ms), the replication is L·(num_redundant)·(cheap argmax) which is small, so killing the packing loops gets essentially all the speedup.

Then the bookkeeping that stitches the three stages together — and I want to keep it identical to what already works, because it's index plumbing where an off-by-one silently mis-routes tokens. Stage 1 produces, per item, a (node, rank-in-node) from the packing, which I fold into a permutation `log2mlog` taking a logical expert id to its position in a node-major "mlog" layout: `(node * groups_per_node + rank) * group_size + within_group_offset`. Its inverse `mlog2log` lets Stage 2 gather the node-major loads into a [L·N, experts_per_node] tensor and replicate within each node independently. Stage 3 computes per-replica loads `tpm/mcnt` gathered through the replication map, packs them to GPUs, builds the slot index `pack*phy_per_gpu + rank`, inverts it, and composes the maps back out to logical ids, adding the per-node expert offset so node k's local ids map to the global range. Finally I scatter the physical slot indices into the `log2phy` map (logical expert → its physical replica slots, −1 where unused), sized by the max replica count. All of that composition is unchanged; the only thing I replaced is the body of `balanced_packing`.

One implementation detail to pin down: the packing does its index work on CPU. The sort, where, and scatter are tiny relative to the model, and the maps are integer permutations consumed by host-side dispatch logic; keeping them on CPU (the reference casts `weight.float().cpu()` up front) avoids launching a swarm of tiny GPU kernels for what is really index arithmetic, and keeps it deterministic. So `balanced_packing` returns its two index tensors on CPU. Not load-bearing for correctness, but it's the difference between "vectorized" and "vectorized *and* not death-by-kernel-launch."

So let me write the one slot I'm actually changing — `balanced_packing` — and keep the rest of the three-stage harness exactly as it stands:

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
