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

Before I do, I want to read the balance columns carefully enough to be sure I am not about to trade
runtime for a quality regression I could have seen coming. In every one of the four greedy rows
`balance` sits just below `balance_node` — 0.679 under 0.702 on deepseek-v3, 0.940 under 0.947 on
qwen3-moe, 0.925 under 0.931 on deepseek-v2, 0.222 under 0.336 on stress-skew — exactly the ordering the
partition geometry forces, since a GPU can never be lighter than an equal share of its node. So the
node-level number is the ceiling and the GPU-level number is tracking just under it; the small gap
between them is the only part Stage 3's packing quality actually controls, and it is a couple of points
at most. That tells me two things. The balance ceiling lives upstream, in whatever imbalance Stage 1
hands the nodes — a structural story I am deliberately *not* touching this rung. And the packing rule I
am about to swap is responsible only for that thin GPU-versus-node gap, so as long as a replacement holds
that gap it cannot move the balance numbers much either way. That is the licence to go after runtime
without fear: the balance is already pinned by structure, not by the loop I want to delete.

So where does the time go? Not in the arithmetic — the whole algorithm is shuffling small integer
indices. The cost is one routine, `balanced_packing`, and the way it is written. It takes `n` weighted
items and splits them into `P` packs of exactly `n/P` items each, as evenly as possible by weight, and
it does it the textbook way: sort descending, then walk the items one at a time, and for each one scan
all `P` packs to find the emptiest pack with a free slot and drop the item there. That is the
least-loaded-feasible greedy I leaned on. But look at the control flow: an outer Python loop over the
batch, an inner Python loop over the `n` items, and inside that a linear min-scan over `P` packs. It is
called twice, and I can count the interpreted iterations exactly. Stage 3 is the heavy call — it packs
each node's `replicas_per_node` per-replica loads onto that node's `gpus_per_node` GPUs, batched over
`L` layers times `num_nodes` nodes, so its inner-iteration count is `L · num_nodes · replicas_per_node ·
gpus_per_node`, and since `num_nodes · gpus_per_node = num_gpus` and `replicas_per_node = R / num_nodes`
that collapses to `L · R · D / num_nodes`. Plug the suite in: deepseek-v3 is `320·64/8 = 2560` per
layer, qwen3-moe `160·32/4 = 1280`, deepseek-v2 `192·32/4 = 1536`, stress-skew `384·128/16 = 3072`.
Stage 1 is negligible beside these — it packs only `num_groups` items, and on deepseek-v3 it hits the
one-item-per-pack degenerate branch and does no loop at all. So Stage 3 is the whole bill, and the model
predicts an ordering: qwen3 (1280) < deepseek-v2 (1536) < deepseek-v3 (2560) < stress-skew (3072). The
measured milliseconds are 102 < 153 < 248 < 256 — the *exact* same ordering, and roughly the same
spacing once I allow that the per-model layer counts differ (the two DeepSeek models run more layers
than qwen, which is why their milliseconds sit a touch above what the per-layer iteration count alone
would predict). That is a satisfying confirmation that the cost is the interpreted `L·R·D/N` sweep and
nothing else — no hidden allocation, no tensor op on the critical path — which means if I can compute
the same assignment without that per-item Python sweep, essentially all of the runtime falls away.

The obvious instinct is "just vectorize the loop," and it is worth being precise about why I cannot,
because the obstruction is the whole problem. The greedy choice for item `j` is "which pack is emptiest
right now" — and "right now" means after items `0..j−1` are already placed. The decision for each item
depends on the running pack loads, which depend on every earlier decision. It is an inherently
sequential recurrence: a prefix scan where each step's branch (the argmin over current loads) depends
on the accumulated state, with no closed form for the current loads without simulating the whole prefix.
This is the same reason you cannot trivially parallelize a greedy bin-packer. So I cannot batch away the
dependency; I have to *change the rule*.

Before I abandon the greedy's exact decisions, though, let me walk the cheaper escapes and make sure I
am not throwing away quality I could have kept. One option is to leave the algorithm alone and just make
the interpreter faster — push the loop into a compiled kernel, Cython or C++ or a jit. But the editable surface
*is* these three Python functions inside the frozen substrate, and even compiled the recurrence is still
`L·R·D/N` strictly-serial argmin steps; I would be paying to make a serial chain run its constant faster,
not to remove the chain, and the win tops out at a small factor while the code fights the harness. A
second option keeps the greedy decisions exactly but flips which axis I loop in Python: today the outer
loop is over the batch and the inner over items, so I pay `B` copies of the whole `n·P` sweep; instead I
could loop over items on the outside and vectorize the emptiest-pack min-scan *across* the batch, so
each of the `n·P` steps is one tensor op over all `B = L·num_nodes` rows at once. That genuinely cuts the
interpreted step count from `B·n·P` to `n·P` — on deepseek-v3 from `L·2560` down to `40·8 = 320` steps,
a real order-of-magnitude — but it does not get me where I need to be. Those 320 steps are still a serial
Python chain, each one a masked argmin plus a scatter that has to honour a *different* per-row feasible
set (each layer-node fills its packs at its own rate, so the `counts < items_per_pack` mask diverges
across the batch and I cannot share the branch), so the code is both slower than a handful of kernels and
much more delicate. It would land me in the tens of milliseconds, not the ones, and buy me a fragile
implementation for the trouble. And there is no parallel-scan trick to fall back on, because the greedy's
per-step operator is an argmin over running loads, which is not associative — there is no closed form for
the state after `j` placements without replaying them. So the honest conclusion is that keeping the
greedy's exact decisions caps me at a partial, awkward speedup; to reach a placement whose cost is
*independent of `n`* I have to give up reproducing its decisions and find a rule with no running-load
dependence at all.

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
*stack*. Let me watch it fail on a concrete instance: six items of weight 8, 7, 6, 5, 4, 3 into three
packs, so two rounds. Round-robin gives pack 0 = {8, 5} = 13, pack 1 = {7, 4} = 11, pack 2 = {6, 3} =
9; the max is 13 against a mean of 11, a balance of `11/13 = 0.846`. Pack 0 got the big end of both
rounds and pack 2 the small end of both, exactly the stacking I feared. The positional simplicity
bought me nothing because it always pours the big end of each round into the same low-index pack. But it
is an *informative* failure — it tells me exactly what to fix.

The fix stares back from the failure: the imbalance comes from every round running in the same
direction, so the same pack always gets the heavy end. So alternate the direction. Run even rounds
forward — packs `0,1,…,P−1` — and odd rounds *backward* — `P−1,P−2,…,1,0`. On the same six items:
round 0 forward puts 8, 7, 6 on packs 0, 1, 2; round 1 reversed puts 5, 4, 3 on packs 2, 1, 0. Now pack
0 holds {8, 3} = 11, pack 1 holds {7, 4} = 11, pack 2 holds {6, 5} = 11 — all three identical, a
balance of exactly 1.000 where round-robin managed 0.846. Round by round, each pack alternately gets the
heavy end and the light end, so the spread within each pack cancels instead of accumulating. This is the
snake — reverse direction each row. The assignment is still purely positional: the pack of rank `r`
depends only on `r`, through which round it is in and whether that round is even or odd. Concretely, for
rank `r` the round is `r // P`, the within-round offset is `r % P`, the pack is the offset on even
rounds and the mirror `P − 1 − offset` on odd rounds, and the within-pack rank is just the round number
— take `P = 8` and rank 10: round `10 // 8 = 1` (odd), offset `10 % 8 = 2`, so the pack is `8 − 1 − 2 =
5` at within-pack rank 1, all from arithmetic on `10`, no running loads anywhere. So it vectorizes into
one sort plus a handful of `arange`/`where`/`scatter` kernels whose count is independent of `n`. The
capacity constraint the greedy enforced with a `counts[p] < items_per_pack` filter is now satisfied
automatically — the round structure deals one item to each pack per round, so with `n` divisible by `P`
every pack ends with exactly `n/P` items, and the within-pack ranks it emits are `0, 1, …, n/P − 1`,
each used once per pack. On the six-item example pack 0 collected the round-0 item at within-pack rank 0
and the round-1 item at rank 1, pack 2 the same — every pack sees rank 0 then rank 1, so the
`(pack, rank)` pairs form a clean grid with no collision, which is exactly what the downstream scatter
into the slot map needs. I get the hard cardinality requirement for free, no filter needed. And I can
now count the whole cost: sort the batch once, build the positional `pack`/`rank` vectors with an
`arange`, a floor-divide, a modulo, a comparison and a `where`, then two `scatter`s back to item order
and a CPU hand-off — on the order of eight tensor ops per call, sixteen across the two packing stages,
and not one of them scales with `n`. Against the `L·R·D/N` interpreted iterations that produced the
quarter-second, that is the difference between thousands of serial Python steps and a fixed dozen-odd
kernel launches, which is why I expect the runtime to fall to the low single-digit milliseconds rather
than merely improve.

Now the real question — does this balance as well as the greedy, or did I trade a good heuristic for a
cheap-but-worse one? Run the greedy on those same six items and it produces {8,3}, {7,4}, {6,5} — the
*identical* partition the snake found, all packs at 11. That is not a coincidence. After sorting, the
weights are non-increasing, so the first round is forced: greedy places ranks `0..P−1` into empty packs,
the same forward sweep as the snake. After that, pack `P−1` is emptiest, so greedy gives rank `P` to
pack `P−1`; the rest of the reversed round follows exactly as long as each newly-loaded pack has climbed
just enough that the next pack in the reversed order is now the emptiest. A locally affine descending
block has precisely this property: over two consecutive rounds pack `p` receives `w[p] + w[2P−1−p]`, and
those pair-sums are constant across `p` (here `8+3 = 7+4 = 6+5 = 11`), so after the backward sweep the
canonical tie-break restarts the next forward sweep from pack 0 and the pattern holds. The snake is not
a theorem for *every* sorted sequence; it is the closed form for the greedy order *when each filled pack
overtakes the next in the alternating order*, which smooth sorted tails approximate. And I should be
honest that when they match, they match at whatever the greedy achieves, not at the optimum — on the
ramp `9,8,7,…,1` into three packs both snake and greedy land on `16 / 15 / 14`, a balance of `0.9375`,
even though the perfect `15 / 15 / 15` partition exists; the sorted-greedy family leaves that on the
table, and the snake inherits exactly that gap, no more and no less. So the precise claim is
conditional: where the sorted sequence keeps the packs in the alternating emptiest-bin order the snake
*is* LPT written in closed form; where it only approximately holds the snake is the fixed positional
approximation to it; nowhere does it beat LPT, and nowhere on smooth data does it fall meaningfully
behind.

And here is where the greedy feedback makes me cautious in exactly the right place. The condition is
friendliest where the load sequence is smooth — and Stage 3 *is* smoothed, because the replication
stage has already split the hot experts into multiple copies, so the per-replica loads I pack in Stage 3
have had their peaks shaved toward each other. The condition is *least* friendly where the sorted
sequence has a cliff or strong curvature — a couple of huge items and then a long flat tail. That is
precisely the `stress-skew` regime: a long-tail Zipf load at skew 0.95, and Stage 1 packing only
`groups_per_node = 2` groups onto each of 16 nodes. Greedy already cratered there — balance 0.222,
balance_node 0.336 — because with two groups per node and extreme skew there is almost no freedom to
even out the nodes, and greedy made its locally-best choice and still landed badly. The snake does not
read the running loads, so on that cliffed Stage-1 sequence it cannot do *better* than greedy did, and
it may do a hair worse on the exact node assignment; what it will *not* do is fix the structural
problem, because the problem there is the hierarchy itself starving Stage 1 of freedom, not the packing
rule. So I should expect stress-skew balance to stay essentially where greedy left it — around 0.222 —
and I should not pretend the snake rescues it. The snake's job is runtime, not stress-config balance.

Two more decisions. First, both packing calls become the snake, and doing it in *both* is what matters.
The cost model already told me Stage 3 is the whole `L·R·D/N` bill and Stage 1 is negligible, so if I
snake-ified only Stage 1 and left Stage 3 as the loop I would still pay the entire 248 ms — the speedup
only fully lands when the *heavy* packing is vectorized. Snaking both is free anyway since they call the
same primitive, and on deepseek-v3 Stage 1 does not even reach the snake body because its eight
groups-into-eight-nodes hits the one-item-per-pack shortcut, so there the change is purely a Stage-3
change. Second, Stage 2, the replication, stays a loop. Its recurrence is genuinely sequential in a way
no positional trick fixes: each new replica *changes* the per-replica load of the expert it joins
(`w/2`, then `w/3`, …), so the argmax for the next replica depends on the previous draw — the whole
point is the diminishing per-replica returns. But it iterates only over the *redundant* count, `num_phy
− num_log` — which is exactly eight per node on every config in the suite, a trivial number of draws —
and it runs after the two big packings are already vectorized away, so it is not the bottleneck. And I
do not want to perturb it: replicating the single most-overloaded expert each time is the right greedy
for driving the peak per-replica load down. So Stage 2 stays a sequential argmax loop, batched across
the layers·nodes dimension but not across the replica draws.

The index-composition bookkeeping that stitches the three stages — the node-major relabeling, the
per-node gathers, the slot-index construction and inverses, the final scatter into `log2phy` — I keep
*identical* to what already works, because it is plumbing where an off-by-one silently mis-routes
tokens, and none of it is where the time went. The only thing I replace is the body of
`balanced_packing`. I also keep its index work on the CPU: the sort, the `where`, and the two scatters
are tiny relative to the model, the maps are integer permutations consumed by host-side dispatch, and
keeping them on CPU avoids launching a swarm of tiny GPU kernels for what is really index arithmetic.
The move is just the snake in place of the loop, in both packing stages.

So the falsifiable expectations against the greedy numbers. Locality should stay exactly 1.000 on all
four configs — I have not touched the hierarchy, so no expert crosses a node, and if it moves off 1.000
at all I have broken the bookkeeping. The runtime should collapse by roughly two orders of magnitude:
from 248 / 102 / 153 / 256 ms down to the low single digits of milliseconds, because the `L·R·D/N`
interpreted iterations become one batched sort plus a fixed handful of index kernels whose count no
longer depends on `n`; if the runtime does *not* collapse, the vectorization failed and the sequential
sweep is still hiding somewhere. The balance numbers should land *at or just below* greedy's, not above:
on the smooth real-model configs I expect to roughly hold the line — qwen3-moe near 0.94, deepseek-v2
near 0.92, deepseek-v3's already-mediocre 0.68 staying near there — because the snake matches LPT under
the smoothness those configs roughly satisfy; and on stress-skew I expect balance to stay pinned around
0.222, because the snake cannot fix what the hierarchy's two-groups-per-node bottleneck imposed. And
balance_node in particular should barely move off greedy's 0.947 / 0.931 / 0.702 / 0.336: on deepseek-v3
Stage 1 is the identity for both rules so the node assignment is bit-for-bit the same, and on the others
the snake only replaces a Stage-1 pack of smooth-ish group sums where it tracks greedy closely — so if
balance_node shifts by more than a whisker I have learned something surprising about how different the
snake's node partition really is. If the balance falls *well* below greedy's on the real configs, the
smoothness assumption was wrong and the snake is a worse packer than I argued. The single number that should move, and move enormously, is the
task score: with locality held and balance roughly held but runtime cut from a quarter-second to
milliseconds, the score should jump from 0.255 toward the high-0.30s — the runtime tax that crushed
greedy simply removed. What it will *not* do is touch the balance ceiling on the two hard configs, and
that is the wound the next rung will have to read off the numbers and open.
