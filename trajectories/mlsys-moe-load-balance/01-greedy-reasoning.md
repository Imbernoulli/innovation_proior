The layer waits for the busiest GPU. Under expert parallelism the combine all-to-all cannot begin
until every GPU has finished its share of the expert work, so the per-layer latency is the load of the
single most-loaded GPU, not the average — and the live token load is wildly skewed, a handful of hot
experts soaking up most of the traffic while the rest sit nearly idle, with the hot set drifting as the
input distribution changes. Put one expert on each of the sixty-four GPUs of the largest deployment
here and let a single card soak up five percent of the batch while the fair share is one sixty-fourth —
a hair over one and a half percent — and that layer runs at better than three times its ideal latency,
every layer, for as long as the skew holds. The whole game is to lay out experts across GPUs so the
*busiest* GPU is as light as possible, and to recompute that layout cheaply, online, every time the
observed loads shift.

That is minimizing a maximum over parallel units given a pile of weighted things to distribute — the
identical-machines scheduling problem `P||Cmax`. It is NP-hard — the two-machine case is exactly
PARTITION, the variable-machine case reduces from 3-PARTITION — so I will not find the exact optimum in
polynomial time, and since I re-run this every few minutes as loads drift I need something cheap, not
exhaustive. Greedy gets close. Plain list scheduling — drop each job on the least-loaded machine — is
within `(2 − 1/m)·OPT`: when the last-finishing job of size `p` is placed it starts at some time `t`,
and because it went on the least-loaded machine all `m` machines already carry at least `t`, so
`W ≥ m·t + p` and `C = t + p ≤ W/m + (1 − 1/m)·p`, which against `OPT ≥ W/m` and `OPT ≥ p` gives the
factor. Sorting biggest-first before the same sweep — the Longest Processing Time rule — sharpens it to
`(4/3 − 1/(3m))·OPT`, since the critical last job is now the smallest in its prefix. That `4/3` is
tight, so even where greedy is at its best I should expect a few points of residual imbalance baked in
before any hardware constraint makes it worse. And the bound belongs to the *unconstrained* rule; my
hardware adds a constraint, so I keep the ordering and the least-loaded instinct but must not pretend
the bound transfers unchanged.

What would beat LPT's constant? Exact optimization by DP or an integer program is off the table —
NP-hard, and with float loads over a wide range a subset-sum DP is not even cleanly pseudo-polynomial,
run on the serving critical path over hundreds of experts per layer. Karmarkar–Karp largest-differencing
has a far better constant on random instances, but it produces a partition with *no cardinality control*
— differencing hands one machine many items and another few, exactly what my hardware forbids — so its
win is unreachable here. It is constrained greedy LPT.

The constraint is concrete. Every GPU hosts the *same* fixed number of physical experts, so this is not
free-form makespan but a *balanced* partition where every pack gets exactly `n / num_packs` items.
Plain LPT would put six items on one machine and two on another if that balanced the weight; here I
must balance the count too. The fix is small: keep the greedy rule but at each step restrict the
candidate packs to those not already full. Sort descending; for each item pick the least-loaded pack
with a free slot, record its rank within the pack, bump that pack's load and count. That is
`balanced_packing`, the primitive I lean on twice. There is a degenerate case worth special-casing:
when there is exactly one item per pack there is nothing to balance, so item `i` goes to pack `i` at
rank 0 with no inner loop — and that branch is not academic, it fires on a real config, because
`deepseek-v3` has eight groups and eight nodes, so Stage 1 packs eight groups into eight packs of one,
the identity, no balancing done. Worth remembering when I read the numbers back. The inner loop is
irreducibly sequential — each placement's least-loaded decision reads the loads left by all earlier
placements — so I move the sorted indices to the CPU and let a Python loop carry the running loads.
Slow, but it mirrors the greedy decisions exactly.

Now the wall the pure packing thinking misses. Greedy balances a *fixed* set of indivisible items. But
what if one item, by itself, is bigger than the fair share? Suppose one expert carries a fifth of all
tokens and I have eight GPUs. Perfect balance would put one-eighth on each, but this expert alone is a
fifth, more than an eighth, and it cannot be split — it is a single set of weights on a single GPU. So
no matter how I pack, the GPU holding it is at least a fifth and my max is floored there, a balance
ceiling of `(1/8)/(1/5) = 0.625` no packer can breach. *Nothing* that assigns whole experts can help;
the items themselves are the problem.

So the items have to become divisible. I cannot split an expert's weights, but I *can* make copies of
it: if a hot expert has `r` replicas and its requests spread across them, each copy carries about
`w_i / r`. The effective item size becomes tunable — each extra copy lowers the floor that expert
imposes. But replication is not free — the runtime gives me exactly `num_phy − num_log` extra copies,
and every logical expert needs at least one. Which experts get the extras? Expert `i` with count `c_i`
and load `w_i` has per-replica load `w_i / c_i`; one more copy drops it to `w_i / (c_i + 1)`. That
per-replica load of the current worst expert is the ceiling, and adding a replica anywhere except a
current argmax leaves the ceiling untouched — so I repeatedly feed the argmax of `w_i / c_i`, doing it
`num_phy − num_log` times. That is the discrete water-fill: three experts of load 10, 1, 1 with two
spare copies send both to the 10 (dropping its ceiling 10 → 5 → 3.33) and leave the cold experts at one
copy each, because feeding them would not have touched the governing ceiling. The loop is sequential —
the next argmax depends on the counts updated by prior draws — so again Python and CPU, but it iterates
only over the extras.

Now the second wall, which pure-balance thinking misses entirely. I said put the replicas on different
GPUs — which GPUs? The GPUs are not a flat pool; they are grouped into nodes, fast NVLink inside and
slow, scarce InfiniBand between, and the router uses node-limited routing precisely to keep the
all-to-all mostly intra-node. If a global packer scatters a hot expert's five replicas onto GPUs that
happen to live on five different nodes, a token wanting that expert may have to cross to a far node to
reach its assigned replica — blowing up exactly the inter-node traffic the routing was designed to
bound. That expert scores `1/5` on the traffic-weighted node-locality metric while carrying heavy
traffic, dragging the weighted average down hard. A placement strong on raw GPU balance can be terrible
on communication, and that cost is invisible to the balance number. The `locality` metric is there to
make this second objective count, so I cannot trade it away.

The experts already come in groups, and the routing already clusters a token's experts onto a few
nodes. So the locality I want is: keep a whole group, and all replicas of its experts, on one node. If
group membership never crosses a node boundary, an expert's replicas all live on its group's node and
nothing is ever scattered — locality falls out at a guaranteed 1.0, bought structurally rather than
optimized for. That reframes the whole thing as nested packing: first decide which groups go on which
nodes; then, *within* each node, replicate and pack onto GPUs. Three stages. Stage 1 packs the groups
onto the nodes (items = groups by total load, packs = nodes, `groups_per_node` each). Stage 2
replicates within each node (each node owns `num_log / num_nodes` logical experts and gets
`num_phy / num_nodes` slots). Stage 3 packs each node's replicas onto its GPUs (items weighted by
per-replica load, `num_phy / num_gpus` slots per GPU). Two packs of the same primitive, one
replication between, locality for free.

Reading the four configs through this decomposition tells me where the pressure falls. Every config
lands on the *same* replication headroom per node — eight spare copies everywhere: `deepseek-v3` 40
slots for 32 experts, `qwen3-moe` 40 for 32, `deepseek-v2` 48 for 40, `stress-skew` 24 for 16. So the
divisibility I can manufacture is fixed at eight per node, but the tail it must tame is not: under
Zipf-0.5 the eight copies comfortably cover the warm experts, while under `stress-skew`'s
Zipf-1.0/skew-0.95 the single hottest expert alone might want a dozen copies and can never get them,
its node having only eight to share across sixteen experts. (The global group-agnostic policy is just
this hierarchy with `num_groups = num_nodes = 1`, one giant node; here `num_nodes` divides `num_groups`
so the genuine hierarchical branch runs.)

The rest is permutation bookkeeping, fiddly because each stage produces a permutation and I compose
them to emit the three maps: `phy2log [L, num_replicas]`, `log2phy [L, E, max_rep]` (the inverse, `−1`
padded), and `logcnt [L, E]`. Stage 1 relabels logical experts into node-major order — a group's
position is `node · groups_per_node + rank`, times `group_size`, plus the within-group offset — so
Stage 2 can slice a node's experts with a reshape; Stage 2 replicates per node; Stage 3 builds each
slot's final position as `gpu · phy_per_gpu + rank`, inverts it, and composes back through the
node-major inverse, lifting node-local ids to global by adding each node's base offset. Finally
`log2phy` is assembled by scattering each physical slot into `[E, max_rep]` at `(its logical expert,
its replica rank)`, `−1` elsewhere. It is all index plumbing where an off-by-one silently mis-routes
tokens, so I keep it exact.

One relationship between the two balance numbers constrains what any placement can achieve. `balance` is
`mean_gpu / max_gpu`, `balance_node` the same at node granularity. The busiest node carries load `M`;
its `gpus_per_node` GPUs average `M / gpus_per_node`, so at least one is at least that heavy:
`max_gpu ≥ max_node / gpus_per_node`. Dividing the per-GPU mean (which is the per-node mean over
`gpus_per_node`) by that, the `gpus_per_node` cancels and `balance ≤ balance_node`. The finer partition
is always at least as imbalanced as the coarser one, so `balance_node` is the governing ceiling:
whatever imbalance Stage 1 leaves between nodes, Stage 3 can only match, never repair, because a GPU
cannot be lighter than an equal share of its node. Every row I get back must show `balance ≤
balance_node`, or my model of the metric is wrong.

This default fill answers the placement problem completely: it balances per-GPU and per-node load and
keeps every expert's replicas on one node, so `locality` should be a perfect 1.0 on every config, and
the greedy packing is near-optimal for the makespan core. But the config reading says balance will not
be uniform. On the two mildest deployments — `qwen3-moe` at Zipf 0.5 and `deepseek-v2` at Zipf 0.6, both
with two groups per node giving Stage 1 room and eight copies easily covering their tails — I expect
balance in the low-to-mid 0.9s. `deepseek-v3` I am least sure of: its Stage 1 is the degenerate
identity, so `balance_node` is nothing but the raw spread of the eight group sums, which Stages 2–3
cannot touch — softer than the other reals, but I will let the columns say the number. `stress-skew` is
where I expect balance to crater: two groups per node gives Stage 1 almost no freedom, the 1.5× budget
leaves only eight spare copies, and the extreme tail keeps the hottest expert's per-replica load several
times the fair share — a floor no downstream packing can lift, so balance well under a half. And the one
thing this method is *not* is fast: `balanced_packing` is a Python loop over every item with an `O(P)`
min-scan inside, run across `L` layers and, in Stage 3, across `num_replicas` items per layer-node, so
the wall time will be hundreds of milliseconds on the larger configs, paid on the serving critical path.
So a split — `locality` pinned at 1.0, balance solid on the gentle configs and cratered on `stress-skew`,
`runtime_ms` an open wound across the board — and whichever of the stress-config balance or the
universal slowness dominates the score is what to attack first.
