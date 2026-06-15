The greedy run told me where the score went, and it went almost entirely into runtime. Locality is
pinned at exactly 1.000 on all four configs — the hierarchy did its job, no expert ever crosses a node
— and on the gentle real-model deployments the balance is solid: 0.940 / 0.947 on qwen3-moe, 0.925 /
0.931 on deepseek-v2. Those are not the problem. The problem is the wall-clock: 248 ms on deepseek-v3,
102 ms on qwen3-moe, 153 ms on deepseek-v2, 256 ms on stress-skew. Each config weights runtime equally
with the three quality metrics, and the runtime term is calibrated against the spread of the
baselines, so a quarter-second placement is scored as essentially worthless on that quarter of every
config — and the geometric mean across configs drags the whole task score down to 0.255, far below what
the perfect locality and the strong real-model balance would otherwise earn. The diagnosis is sharp and
it is not a balance problem: the algorithm is *correct*, it just spends hundreds of milliseconds doing
it, on the serving critical path, every time the load drifts. The hierarchy, the replica-count rule,
and the output maps are all worth keeping; what I have to kill is the time.

So where does the time go? Not in the arithmetic — the whole algorithm is shuffling small integer
indices. The cost is one routine, `balanced_packing`, and the way it is written. It takes `n` weighted
items and splits them into `P` packs of exactly `n/P` items each, as evenly as possible by weight, and
it does it the textbook way: sort descending, then walk the items one at a time, and for each one scan
all `P` packs to find the emptiest pack with a free slot and drop the item there. That is the
least-loaded-feasible greedy I leaned on. But look at the control flow: an outer Python loop over the
`L` layers, an inner Python loop over the `n` items, and inside that a linear min-scan over `P` packs —
`L·n·P` iterations of interpreted Python, each a `.item()` off a tensor and a `min` over a little list.
Every quantity is tiny; the only expensive thing is that I am doing it element-by-element in Python on a
machine that wants batched tensor kernels. And it is called twice — Stage 1 packs `num_groups` items
onto nodes, Stage 3 packs `num_replicas` per-replica loads onto GPUs. Stage 3 is the heavier of the two
because `num_replicas` is much larger than `num_groups`; on deepseek-v3 that is 320 replicas per layer
walked in Python across 61 layers. That is the 248 ms.

The obvious instinct is "just vectorize the loop," and it is worth being precise about why I cannot,
because the obstruction is the whole problem. The greedy choice for item `j` is "which pack is emptiest
right now" — and "right now" means after items `0..j−1` are already placed. The decision for each item
depends on the running pack loads, which depend on every earlier decision. It is an inherently
sequential recurrence: a prefix scan where each step's branch (the argmin over current loads) depends
on the accumulated state, with no closed form for the current loads without simulating the whole prefix.
This is the same reason you cannot trivially parallelize a greedy bin-packer. So I cannot batch away the
dependency; I have to *change the rule*.

Let me back up and ask what I actually need, not what the greedy does. The greedy is one *way* to get a
balanced, cardinality-constrained partition; it is not the only way, and I do not care about reproducing
its decisions step for step — I care about packs whose sums are about as even as the greedy's, computed
without a per-item running-load dependence. So the question is: is there an assignment of items to packs
I can write as a *fixed function of each item's sorted position alone*, no running state, that still
comes out balanced? If the pack an item goes to depends only on its rank in the sorted order, I can
compute it for all items at once with index arithmetic, and the sequential chain is broken.

Start with the crudest such fixed rule: plain round-robin over the sorted items. Heaviest (rank 0) to
pack 0, rank 1 to pack 1, …, rank `P−1` to pack `P−1`, rank `P` back to pack 0, and so on — pack `p`
gets sorted-ranks `{p, P+p, 2P+p, …}`. Purely positional, trivially vectorizable, and exactly `n/P`
items per pack for free since the ranks cycle. Does it balance? Group the sorted items into consecutive
rounds of `P`. Round-robin hands round 0's items to packs `0,1,…,P−1` in order, so pack 0 gets the
heaviest of that round and pack `P−1` the lightest. But round 1 *also* runs in order, so pack 0 gets
the heaviest of round 1, and round 2 again. Pack 0 systematically collects the heaviest element of
*every* round, and pack `P−1` the lightest of every round. The disparities do not cancel — they
*stack*. Pack 0 ends up overloaded, pack `P−1` underloaded, by roughly the per-round spread times the
number of rounds. That is worse than I can accept; the positional simplicity bought me nothing because
it always pours the big end of each round into the same low-index pack. But it is an *informative*
failure — it tells me exactly what to fix.

The fix stares back from the failure: the imbalance comes from every round running in the same
direction, so the same pack always gets the heavy end. So alternate the direction. Run even rounds
forward — packs `0,1,…,P−1` — and odd rounds *backward* — `P−1,P−2,…,1,0`. Now in round 0 pack 0 gets
the heaviest item (rank 0) and pack `P−1` the lightest (rank `P−1`); in round 1, reversed, pack 0 gets
the *lightest* of that round (rank `2P−1`) and pack `P−1` gets the *heaviest* (rank `P`). So pack 0's
two items are rank 0 and rank `2P−1` — a heavy one paired with a light one — while pack `P−1`'s two are
rank `P−1` and rank `P`, two middling adjacent ones. Round by round, each pack alternately gets the
heavy end and the light end, so the spread within each pack cancels instead of accumulating. This is the
snake — reverse direction each row. The assignment is still purely positional: the pack of rank `r`
depends only on `r`, through which round it is in and whether that round is even or odd. No running
loads. So it vectorizes: the round is `r // P`, the within-round offset is `r % P`, the pack is the
offset on even rounds and the mirror `P − 1 − offset` on odd rounds, and the within-pack rank is just
the round number, because each pack receives exactly one item per round in round order. The capacity
constraint the greedy enforced with a `counts[p] < items_per_pack` filter is now satisfied
automatically — the round structure deals one item to each pack per round, so with `n` divisible by `P`
every pack ends with exactly `n/P` items. I get the hard cardinality requirement for free.

Now the real question — does this balance as well as the greedy, or did I trade a good heuristic for a
cheap-but-worse one? After sorting, the weights are non-increasing, so the first round is forced:
greedy places ranks `0..P−1` into empty packs, the same forward sweep as the snake. After that I have
to be careful. Pack `P−1` is the emptiest after round 0, so greedy gives rank `P` to pack `P−1`; the
rest of the reversed round follows only if that new load has climbed enough that pack `P−2` is now
emptiest, then `P−3`, and so on. The snake is not a theorem for *every* sorted sequence; it is the
closed form for the greedy order *when each filled pack overtakes the next in the alternating order*. A
locally affine descending block has exactly this property: over two consecutive blocks pack `p`
receives `w[p] + w[2P−1−p]`, and those pair sums are equal across `p`, so after the backward sweep the
canonical tie-break starts the next forward sweep from pack 0 again. Smooth sorted load tails
approximate the same condition. So the precise claim is conditional: when the sorted sequence keeps the
packs in that alternating emptiest-bin order, the snake is LPT written in closed form; when it only
approximately holds, the snake is the fixed positional approximation to that greedy order.

And here is where the greedy feedback makes me cautious in exactly the right place. The condition is
friendliest where the load sequence is smooth — and Stage 3 *is* smoothed, because the replication
stage has already split the hot experts into multiple copies, so the per-replica loads I pack in Stage 3
have had their peaks shaved. The condition is *least* friendly where the sorted sequence has a cliff or
strong curvature — a couple of huge items and then a long flat tail. That is precisely the
`stress-skew` regime: a long-tail Zipf load at skew 0.95, and Stage 1 packing only `groups_per_node =
2` groups onto each of 16 nodes. Greedy already cratered there — balance 0.222, balance_node 0.336 —
because with two groups per node and extreme skew there is almost no freedom to even out the nodes, and
greedy made its locally-best choice and still landed badly. The snake does not read the running loads,
so on that cliffed Stage-1 sequence it cannot do *better* than greedy did, and it may do marginally
worse on the exact node assignment; what it will *not* do is fix the structural problem, because the
problem there is the hierarchy itself starving Stage 1 of freedom, not the packing rule. So I should
expect stress-skew balance to stay roughly where greedy left it — around 0.222 — and I should not
pretend the snake rescues it. The snake's job is runtime, not stress-config balance.

Two more decisions. First, both packing calls become the snake, and doing it in *both* is what matters.
If I only snake-ified Stage 1 and left Stage 3 as the loop, I would still pay the Python tax on the
*larger* packing — Stage 3 walks more items than Stage 1 — so the 248 ms would barely move. The
speedup only fully lands when every packing call is vectorized. Second, Stage 2, the replication, stays
a loop. Its recurrence is genuinely sequential in a way no positional trick fixes: each new replica
*changes* the per-replica load of the expert it joins (`w/2`, then `w/3`, …), so the argmax for the
next replica depends on the previous draw — the whole point is the diminishing per-replica returns. But
it iterates only over the *redundant* count, `num_phy − num_log` — a few dozen to a couple hundred — not
the whole tensor, and it runs after the two big packings are already vectorized away, so it is not the
bottleneck. And I do not want to perturb it: replicating the single most-overloaded expert each time is
the right greedy for driving the peak per-replica load down. So Stage 2 stays a sequential argmax loop,
batched across the layers·nodes dimension but not across the replica draws.

The index-composition bookkeeping that stitches the three stages — the node-major relabeling, the
per-node gathers, the slot-index construction and inverses, the final scatter into `log2phy` — I keep
*identical* to what already works, because it is plumbing where an off-by-one silently mis-routes
tokens. The only thing I replace is the body of `balanced_packing`. I also keep its index work on the
CPU: the sort, the `where`, and the two scatters are tiny relative to the model, the maps are integer
permutations consumed by host-side dispatch, and keeping them on CPU avoids launching a swarm of tiny
GPU kernels for what is really index arithmetic. The full scaffold module is in the answer; here the
move is just the snake in place of the loop, in both packing stages.

So the falsifiable expectations against the greedy numbers. Locality should stay exactly 1.000 on all
four configs — I have not touched the hierarchy, so no expert crosses a node. The runtime should
collapse by roughly two orders of magnitude: from 248 / 102 / 153 / 256 ms down to the low single
digits of milliseconds, because `L·n·P` interpreted iterations become one batched sort plus a handful
of `arange`/`where`/`scatter` kernels whose count is independent of `n`. The balance numbers should
land *at or just below* greedy's, not above: on the smooth real-model configs I expect to roughly hold
the line — qwen3-moe near 0.94, deepseek-v2 near 0.92, deepseek-v3's already-mediocre 0.68 staying near
there — because the snake matches LPT under the smoothness condition those configs roughly satisfy; and
on stress-skew I expect balance to stay pinned around 0.222, because the snake cannot fix what the
hierarchy's two-groups-per-node bottleneck imposed on greedy. If the runtime does not collapse, the
vectorization failed; if the balance falls *well* below greedy's on the real configs, the smoothness
assumption was wrong and the snake is a worse packer than I argued. The single number that should move,
and move enormously, is the task score: with locality held and balance roughly held but runtime cut
from a quarter-second to milliseconds, the score should jump from 0.255 toward the high-0.30s — the
runtime tax that crushed greedy simply removed.
