The layer waits for the busiest GPU. Under expert parallelism the combine all-to-all cannot begin
until every GPU has finished its share of the expert work, so the per-layer latency is the load of the
single most-loaded GPU, not the average — and the live token load is wildly skewed, a handful of hot
experts soaking up most of the traffic while the rest sit nearly idle, with the hot set drifting as the
input distribution changes. So with one expert per GPU the busy GPU grinds through a mountain of tokens
while the others wait, and most of the machine is wasted. The whole game is to lay out experts across
GPUs so the *busiest* GPU is as light as possible, and to recompute that layout cheaply, online, every
time the observed loads shift.

That is minimizing a maximum over parallel units given a pile of weighted things to distribute — a
makespan problem, the identical-machines scheduling problem `P||Cmax`: assign jobs to machines to
minimize the max machine load. Two facts about it I know cold. One, it is NP-hard — the two-machine
case is exactly PARTITION, split a multiset into two equal-sum halves, and the variable-machine case
reduces from 3-PARTITION — so I will not find the exact optimum in polynomial time, and since I have to
re-run this every few minutes as loads drift, I need something cheap, not exhaustive. Two, greedy gets
close. Plain list scheduling — drop each job on the currently least-loaded machine — has a clean bound:
when the job that finishes last is placed, suppose it starts at time `t` with size `p`; because it went
on the least-loaded machine, all `m` machines already carry at least `t`, so total work `W ≥ m·t + p`,
hence makespan `C = t + p ≤ W/m + (1 − 1/m)·p`. Since any optimum is at least `W/m` and at least `p`,
this gives `C ≤ (2 − 1/m)·OPT`. Sorting biggest-first before the same sweep — the Longest Processing
Time rule — sharpens it: the critical last job is now the smallest in its prefix, and either it is
larger than `OPT/3`, in which case every prefix job exceeds `OPT/3` so an optimum can hold at most two
per machine and the largest-first pairing is already optimal there, or it is at most `OPT/3`, and
plugging that into the same inequality yields `C ≤ (4/3 − 1/(3m))·OPT`. That theorem belongs to the
*unconstrained* rule; my hardware adds a constraint, so I keep the ordering and the least-loaded
instinct but I must not pretend the bound transfers unchanged.

The constraint is concrete. Every GPU hosts the *same number* of physical experts — the runtime
allocates a fixed slot count per GPU. So it is not free-form makespan; it is a *balanced* partition
where every pack gets exactly `n / num_packs` items. Plain LPT would happily put six items on one
machine and two on another if that balanced the *weight*; here I have to balance the *count* too. The
fix is small: keep the greedy rule but restrict the candidate packs at each step to those not already
full. Sort descending; for each item in that order, among the packs with a free slot pick the one with
the smallest current load; assign it there, record its rank within the pack, bump that pack's load and
count. That gives me the primitive I will lean on twice — `balanced_packing`, items into equal-count
packs by least-loaded-feasible greedy. There is a degenerate case worth special-casing: when there is
exactly one item per pack, there is nothing to balance, so item `i` simply goes to pack `i` at rank 0
and I return that without the inner loop. And the inner loop is irreducibly sequential — each
placement's "least loaded" decision reads the running loads from all earlier placements — so I cannot
vectorize the greedy choice; I move the sorted indices to the CPU and let a Python loop carry the
running loads. Slow, but it mirrors the greedy decisions exactly.

Now the wall the pure packing thinking misses. Greedy balances a *fixed* set of indivisible items. But
what if one item, by itself, is bigger than the fair share? Suppose one expert is so hot it carries a
fifth of all the tokens and I have eight GPUs. Perfect balance would put one-eighth on each GPU — but
this one expert alone is a fifth, more than an eighth, and it cannot be split, because it is a single
set of weights on a single GPU. So no matter how I pack, the GPU holding it is at least a fifth, and my
max is floored at a fifth even though the average is an eighth. Greedy cannot help; *nothing* that
assigns whole experts can. The max GPU load is floored by the single largest item, and that floor sits
above what balance demands. The items themselves are the problem.

So the items have to become divisible. I cannot split an expert's weights, but I *can* make copies of
it. If the hot expert has `r` physical replicas and its requests spread across them, each copy carries
about `w_i / r` instead of the whole `w_i`. Now the effective item size is tunable: each extra copy
lowers the floor that expert imposes, and with enough budget it pulls the worst items back toward the
fair share before packing. This is the move classical scheduling never had — it assumed fixed jobs.
Here the jobs are FFNs and I am allowed to duplicate the heavy ones, so the algorithm grows a second
stage before packing: *replicate*. But replication is not free — the runtime gives me `num_phy`
physical slots total and `num_log` logical experts, so I have exactly `num_phy − num_log` extra copies
to hand out, and every logical expert needs at least one or its tokens have nowhere to go. Which
experts get the extras? Picture handing out replicas one at a time. Expert `i` currently has count
`c_i` and load `w_i`, so each replica carries `w_i / c_i`; one more copy drops that to `w_i / (c_i +
1)`. The expert I should feed is the one whose replicas are *currently the most loaded* — the largest
`w_i / c_i` — because that per-replica load is the ceiling. Adding a replica anywhere except a current
argmax leaves the ceiling untouched; adding it to an argmax is the only move that can lower it, and on a
tie it attacks one of the tied ceilings. So repeatedly feed the argmax of `w_i / c_i`, do it `num_phy −
num_log` times, and I have the discrete water-filling that equalizes per-replica load. That loop is also
sequential — the next argmax depends on the counts updated by all prior draws — so again Python and CPU.

Now the second wall, the one pure-balance thinking completely misses. I said "put the replicas on
different GPUs." Which GPUs? If I throw all replicas into one global pack across every GPU in the
cluster, greedy may make the GPU loads look attractive while scattering expert `i`'s replicas across
whatever GPUs happened to be least loaded — across *different nodes*. That is a disaster for a different
reason. The GPUs are not a flat pool; they are grouped into nodes, with fast NVLink inside a node and
slow, scarce InfiniBand between nodes. The whole reason the router uses node-limited routing — each
token restricted to a few nodes — is to keep the all-to-all mostly intra-node. If I scatter an expert's
replicas across many nodes, a token wanting that expert may have to cross to a far node to reach its
assigned replica, and I have blown up exactly the inter-node traffic the routing was designed to bound.
A placement strong on raw GPU balance can be terrible on communication, and that cost is invisible to
the balance number. I need to balance the load *and* keep each expert's replicas on as few nodes as
possible — and the task's `locality` metric is precisely there to make that second objective count, so I
cannot trade it away the way a balance-only objective would let me.

The experts already come in groups, and the routing already clusters a token's experts onto a few
nodes. So the locality I want is: keep a whole expert group, and all replicas of its experts, on a
single node. If group membership never crosses a node boundary, an expert's replicas all live on its
group's node and an expert is never scattered. That reframes the whole thing as a nested, hierarchical
packing rather than one flat global pack. First decide which groups go on which nodes; then, *within*
each node, do the replication and the GPU packing. The replication and GPU-level packing become
intra-node operations, so by construction nothing leaves its node and locality falls out for free.

Three stages, then. Stage 1: pack the groups onto the nodes — each group has a total load (the sum of
its experts' loads), so this is the balanced-packing primitive with items = groups, packs = nodes,
`groups_per_node = num_groups / num_nodes` per node, balancing per-node load. Stage 2: within each node,
replicate — each node owns `num_log / num_nodes` logical experts and gets `num_phy / num_nodes` slots,
so I run the replication rule per node on its slice. Stage 3: within each node, pack its replicas onto
its GPUs — balanced packing again, items = the node's physical replicas weighted by per-replica load,
packs = the node's GPUs, exactly `num_phy / num_gpus` slots per GPU. Two packs with the same primitive,
one replication between them, locality for free. As a sanity check, the global group-agnostic policy is
just this hierarchy with `num_groups = num_nodes = 1`: one giant "node" holding everything, Stage 1
trivial, Stages 2–3 replicate and pack across the whole cluster — one algorithm, two regimes. For the
deployments here `num_nodes` divides `num_groups`, so the genuine hierarchical branch runs.

The rest is permutation bookkeeping, and it is fiddly because each stage produces a permutation and I
have to compose them to emit the three maps the runtime wants: `phy2log [L, num_replicas]` (logical
expert per physical slot), `log2phy [L, E, max_rep]` (the inverse, with `−1` padding), and `logcnt [L,
E]` (replica count per expert). Stage 1 relabels logical experts into a node-major order so each node's
experts are a contiguous block — a group's position is `node · groups_per_node + rank`, times the group
size, plus the within-group offset — which lets Stage 2 slice "this node's experts" with a plain
reshape. Stage 2 replicates per node. Stage 3 packs per node, builds each slot's final position as
`gpu · phy_per_gpu + rank`, inverts it, then composes back: follow each final position to its slot, to
its node-major-logical expert, lift node-local ids to global by adding each node's base offset, and map
node-major back to true logical with the Stage-1 inverse. Finally I assemble `log2phy` by scattering
each physical slot index into `[E, max_rep]` at `(its logical expert, its replica rank)`, leaving `−1`
where an expert has fewer than the max copies — exactly the padding contract. The whole composition is
index plumbing; an off-by-one silently mis-routes tokens, so I keep it exact.

Why start the ladder here, with this default fill of the scaffold. It is the correct, complete answer
to the *placement* problem: it balances per-GPU load, it balances per-node load through Stage 1, and it
keeps every expert's replicas on one node, so on the `locality` metric it should score a perfect 1.0 on
every config — the hierarchy guarantees no expert ever crosses a node. The greedy packing is the
near-optimal heuristic for the NP-hard makespan core, so the balance numbers should be respectable where
the configuration is forgiving. The one thing it is not is fast: `balanced_packing` is a Python loop
over every item with an O(P) min-scan inside, run across `L` layers, and Stage 3 walks `num_replicas`
items per layer-node — so the wall time will be on the order of hundreds of milliseconds on the larger
configs, paid on the serving critical path. So what to watch is a split: locality pinned at 1.0
everywhere; balance and balance_node solid on the gentle real-model configs but exposed wherever the
hierarchy is forced into a corner — the `stress-skew` config, with only two groups per node, a 1.5×
replication budget, and a long-tail Zipf load, is where I expect the balance to crater, because Stage 1
has almost no freedom to even out the nodes and the tight budget leaves the hottest experts
under-replicated; and the runtime is the open wound across the board. Whichever of those two — the
stress-config balance or the universal slowness — turns out to dominate the score is what the next rung
has to attack first.
