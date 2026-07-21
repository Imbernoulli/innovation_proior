The greedy run told me where the score went, and it went almost entirely into runtime. Locality is
pinned at exactly 1.000 on all four configs — the hierarchy did its job, no expert ever crosses a node
— and on the gentle real-model deployments the balance is solid: 0.940 / 0.947 on qwen3-moe, 0.925 /
0.931 on deepseek-v2. Those are not the problem. The problem is the wall-clock: 248 ms on deepseek-v3,
102 on qwen3-moe, 153 on deepseek-v2, 256 on stress-skew. Each config weights runtime equally with the
three quality metrics, calibrated against the spread of the baselines, so a quarter-second placement is
scored as essentially worthless on that quarter of every config — and the geometric mean drags the task
score to 0.255, far below what the perfect locality and strong balance would otherwise earn. The
diagnosis is sharp and it is not a balance problem: the algorithm is *correct*, it just spends hundreds
of milliseconds doing it, on the serving critical path, every time the load drifts. The hierarchy, the
replica-count rule, and the output maps are all worth keeping; what I have to kill is the time.

The balance columns confirm I can go after runtime without trading away quality. In every greedy row
`balance` sits just below `balance_node` — 0.679 under 0.702, 0.940 under 0.947, 0.925 under 0.931,
0.222 under 0.336 — exactly the ordering the partition geometry forces. So the node-level number is the
ceiling and the GPU-level number tracks just under it; the small gap between them is the only part
Stage 3's packing quality controls, a couple of points at most. The balance ceiling lives upstream, in
whatever imbalance Stage 1 hands the nodes — a structural story I am deliberately *not* touching now —
and the packing rule I am about to swap is responsible only for that thin GPU-versus-node gap, so a
replacement that holds the gap cannot move the balance much either way.

So where does the time go? Not in the arithmetic — the whole algorithm shuffles small integer indices.
It is one routine, `balanced_packing`: an outer Python loop over the batch, an inner Python loop over
the `n` items, and inside that a linear min-scan over `P` packs. It is called twice, and I can count
the interpreted iterations. Stage 3 is the heavy call — it packs each node's `replicas_per_node`
per-replica loads onto that node's `gpus_per_node` GPUs, batched over `L` layers times `num_nodes`, so
its inner-iteration count is `L · num_nodes · replicas_per_node · gpus_per_node`, which collapses to
`L · R · D / num_nodes`. Plug the suite in: deepseek-v3 is `320·64/8 = 2560` per layer, qwen3-moe
`160·32/4 = 1280`, deepseek-v2 `192·32/4 = 1536`, stress-skew `384·128/16 = 3072`. Stage 1 is
negligible beside these (it packs only `num_groups` items, and on deepseek-v3 hits the
one-item-per-pack branch and loops not at all). The model predicts the ordering qwen (1280) <
deepseek-v2 (1536) < deepseek-v3 (2560) < stress-skew (3072), and the measured milliseconds are 102 <
153 < 248 < 256 — the same ordering, the two DeepSeek models sitting a touch high because they run more
layers than qwen. So the cost is the interpreted `L·R·D/N` sweep and nothing else: no hidden
allocation, no tensor op on the critical path. Compute the same assignment without that per-item Python
sweep and essentially all of the runtime falls away.

The obvious instinct is "just vectorize the loop," and I cannot, because the obstruction is the whole
problem. The greedy choice for item `j` is "which pack is emptiest right now," and "right now" means
after items `0..j−1` are placed. Each decision depends on the running pack loads, which depend on every
earlier decision — an inherently sequential recurrence, a prefix scan whose per-step operator is an
argmin over accumulated state, with no closed form for the current loads without simulating the whole
prefix. This is the same reason you cannot trivially parallelize a greedy bin-packer. I cannot batch
away the dependency; I have to *change the rule*.

The cheaper escapes do not reach where I need to be. Pushing the loop into a compiled kernel leaves it
`L·R·D/N` strictly-serial argmin steps — a constant-factor win, not a removal. Flipping the loop axis —
loop over items outside, vectorize the emptiest-pack min-scan across the batch — cuts the interpreted
step count from `B·n·P` to `n·P` (on deepseek-v3, `40·8 = 320` steps), but those 320 steps are still a
serial Python chain, each a masked argmin plus a scatter honouring a *different* per-row feasible set
(each layer-node fills at its own rate, so the capacity mask diverges and I cannot share the branch).
That lands me in tens of milliseconds, not ones. To reach a cost *independent of `n`* I have to give up
reproducing greedy's decisions and find a rule with no running-load dependence at all.

So back up: what do I actually need? Not greedy's decisions step for step — I need packs whose sums are
about as even as greedy's, computed without a per-item running-load dependence. Is there an assignment
I can write as a *fixed function of each item's sorted position alone*? If the pack an item goes to
depends only on its rank in the sorted order, I compute it for all items at once with index arithmetic
and the sequential chain is broken.

Start with the crudest such rule: plain round-robin over the sorted items, pack `p` getting sorted-ranks
`{p, P+p, 2P+p, …}`. Purely positional, trivially vectorizable, exactly `n/P` items per pack for free.
Does it balance? Group the sorted items into consecutive rounds of `P`. Round 0 hands its heaviest to
pack 0 and lightest to pack `P−1` — but round 1 runs in the same direction, so pack 0 again gets the
heaviest of round 1, and round 2 again. Pack 0 systematically collects the heaviest of *every* round;
the disparities stack instead of cancelling. On six items 8, 7, 6, 5, 4, 3 into three packs: pack 0 =
{8,5} = 13, pack 1 = {7,4} = 11, pack 2 = {6,3} = 9, max 13 against mean 11, balance `0.846`. An
informative failure — it tells me exactly what to fix.

The fix stares back: the imbalance comes from every round running the same direction. So alternate.
Even rounds forward — packs `0…P−1` — odd rounds backward — `P−1…0`. On the same six items, round 0
forward puts 8, 7, 6 on packs 0, 1, 2 and round 1 reversed puts 5, 4, 3 on packs 2, 1, 0: pack 0 =
{8,3} = 11, pack 1 = {7,4} = 11, pack 2 = {6,5} = 11 — all identical, balance exactly 1.000 where
round-robin managed 0.846. Each pack alternately gets the heavy end and the light end, so the per-round
spreads cancel within it. This is the snake — reverse direction each row. The assignment is still
purely positional: for rank `r` the round is `r // P`, the offset `r % P`, the pack is the offset on
even rounds and `P − 1 − offset` on odd rounds, and the within-pack rank is just the round number. So
it vectorizes into one sort plus a handful of `arange`/`where`/`scatter` kernels whose count is
independent of `n`. The cardinality constraint the greedy enforced with a `counts < items_per_pack`
filter is now automatic — the round structure deals one item to each pack per round, so every pack ends
with exactly `n/P` items and emits within-pack ranks `0, 1, …, n/P − 1` once each, a clean
`(pack, rank)` grid with no collision for the downstream scatter. Sort the batch once, build the
positional vectors, two scatters back to item order and a CPU hand-off: on the order of eight tensor
ops per call, sixteen across the two stages, none scaling with `n`. Against the `L·R·D/N` interpreted
iterations that produced the quarter-second, that is the difference between thousands of serial Python
steps and a fixed dozen-odd kernel launches — the runtime should fall to the low single-digit
milliseconds, not merely improve.

Does this balance as well as greedy, or did I trade a good heuristic for a cheap worse one? Run greedy
on those six items and it produces {8,3}, {7,4}, {6,5} — the *identical* partition. Not a coincidence:
after sorting, the first round is forced (greedy fills empty packs `0..P−1`, the snake's forward
sweep), and thereafter pack `P−1` is emptiest so greedy gives rank `P` to it, the reversed round
following exactly as long as each newly-loaded pack climbs just enough that the next in reversed order
becomes emptiest. A locally affine descending block has precisely this property: over two rounds pack
`p` receives `w[p] + w[2P−1−p]`, constant across `p` (`8+3 = 7+4 = 6+5 = 11`). So the snake is the
closed form for the greedy order *when each filled pack overtakes the next in the alternating order*,
which smooth sorted tails approximate. And when they match they match at whatever greedy achieves, not
the optimum — on the ramp `9,8,…,1` into three packs both land on `16/15/14`, balance `0.9375`, leaving
the perfect `15/15/15` on the table. The snake inherits exactly that gap, no more, no less: nowhere
does it beat LPT, and nowhere on smooth data does it fall meaningfully behind.

And here the greedy feedback makes me cautious in the right place. The condition is friendliest where
the load sequence is smooth — and Stage 3 *is* smoothed, because replication has already split the hot
experts into copies, shaving the per-replica peaks toward each other. It is *least* friendly where the
sorted sequence has a cliff: a couple of huge items and a long flat tail. That is precisely
`stress-skew` — long-tail Zipf at skew 0.95, Stage 1 packing only `groups_per_node = 2` groups onto
each of 16 nodes. Greedy already cratered there, 0.222 / 0.336, because two groups per node under
extreme skew leaves almost no freedom. The snake reads no running loads, so on that cliffed Stage-1
sequence it cannot do *better* than greedy and may do a hair worse on the exact node assignment; what
it will not do is fix the structural problem, which is the hierarchy starving Stage 1, not the packing
rule. So I expect stress-skew balance to stay essentially where greedy left it, around 0.222. The
snake's job is runtime, not stress-config balance.

Two more decisions. Both packing calls become the snake, and doing *both* is what matters: the cost
model says Stage 3 is the whole `L·R·D/N` bill, so snaking only Stage 1 would still pay the 248 ms.
Snaking both is free anyway since they share the primitive, and on deepseek-v3 Stage 1 hits the
one-item-per-pack shortcut so there the change is purely Stage 3. Second, Stage 2 replication stays a
loop: its recurrence is genuinely sequential — each new replica *changes* the per-replica load of the
expert it joins (`w/2`, then `w/3`, …), so the argmax for the next depends on the previous draw, which
is the whole diminishing-returns point — but it iterates only over the redundant count, eight per node
on every config, and runs after the two big packings are vectorized away, so it is not the bottleneck.
The index-composition bookkeeping — node-major relabeling, per-node gathers, slot-index construction
and inverses, the final scatter into `log2phy` — I keep *identical*, because it is plumbing where an
off-by-one silently mis-routes tokens and none of it is where the time went. I also keep the maps on
the CPU: the sort, `where`, and scatters are tiny integer work, and keeping them host-side avoids
launching a swarm of tiny GPU kernels. The move is just the snake in place of the loop, in both stages.

So the falsifiable expectations. Locality should stay exactly 1.000 on all four — the hierarchy is
untouched, so any drift off 1.000 means broken bookkeeping. Runtime should collapse by roughly two
orders of magnitude, from 248/102/153/256 ms into the low single digits, as the `L·R·D/N` iterations
become one sort plus a fixed handful of kernels; if it does not, the sequential sweep is still hiding
somewhere. Balance should land *at or just below* greedy's, never above — near 0.94 on qwen, 0.92 on
deepseek-v2, deepseek-v3 near its mediocre 0.68, stress-skew pinned around 0.222 — because the snake
matches LPT under the smoothness the real configs roughly satisfy and cannot fix stress-skew's Stage-1
bottleneck. `balance_node` in particular should barely move off greedy's numbers, since on deepseek-v3
Stage 1 is the identity for both rules and elsewhere the snake replaces a Stage-1 pack of smooth group
sums it tracks closely. The number that should move enormously is the task score, from 0.255 toward the
high-0.30s. What none of this touches is the balance ceiling on the two hard configs — the wound to
open next.
