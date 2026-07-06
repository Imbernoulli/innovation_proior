The zigzag run did exactly what I asked of it and exposed exactly what it could not do. Runtime
collapsed: 248 / 102 / 153 / 256 ms became 1.6 / 0.85 / 1.2 / 1.6 ms — two orders of magnitude, just as
the vectorization argument predicted, the `L·R·D/N` interpreted loop replaced by a sort and a handful of
index kernels. Locality held at exactly 1.000 on all four configs — the hierarchy was untouched, so no
expert crosses a node. And the balance came in right where I said it would: at or just below greedy's on
the smooth real-model configs, and pinned on stress-skew. It is worth reading the exact gap, because it
confirms the mechanism. On deepseek-v3 balance slipped from greedy's 0.679 to 0.659, on qwen3-moe from
0.940 to 0.919, on deepseek-v2 from 0.925 to 0.906 — about two points each, the snake paying a small
premium over the adaptive greedy on the thin GPU-versus-node packing gap, exactly the "at or just below,
never above" I argued. On stress-skew it did not move at all, 0.222 to 0.222. The task score jumped from
0.255 to 0.375, almost all of it the removed runtime tax. So the runtime wound is closed. But the score
is now bottlenecked by *balance*, and the numbers tell me precisely where: balance is mediocre on
deepseek-v3 (0.659) and catastrophic on stress-skew (0.222), and on those two configs it is dragging the
geometric mean down hard while qwen3-moe and deepseek-v2 sit comfortably near 0.91. The question for this
rung is no longer speed. It is: why is the balance capped, and what is actually capping it.

Look at what greedy and zigzag share, because it is a genuinely diagnostic coincidence. They score
*identical* balance_node to six digits: 0.702209 on deepseek-v3 for both, 0.335967 on stress-skew for
both, 0.946751 on qwen3-moe for both, and on deepseek-v2 they differ only in the sixth decimal
(0.930967 versus 0.930966), a rounding artifact. That is the tell, and it is a strong one. Greedy and
snake are *different packers* — one an adaptive least-loaded sweep that reads running loads, the other a
fixed positional deal that reads nothing — and yet they place the load across nodes identically. If the
node-level balance were set by the packing rule, two rules this different would not agree to six digits.
So the node balance is not being decided by the packing at all. It is being decided *upstream* of the
packing, by the one thing greedy and snake share unchanged: Stage 1, the group-to-node assignment, and
the hierarchy that forces its shape. The GPU-level balance dropped a couple of points when I swapped
packers, because the packer does control the thin gap between GPU and node balance; the node-level
balance did not budge, because the packer does not control it. The cap is Stage 1.

Let me make that concrete config by config, because the mechanism differs in degree. On deepseek-v3
there are eight groups and eight nodes, so `groups_per_node = 1` — Stage 1 has *no freedom whatsoever*:
one group per node, the assignment is the one-item-per-pack identity, and whatever node-level imbalance
the eight group loads happen to carry is locked in before replication and GPU packing even begin.
balance_node 0.702 is not a packing failure; it is the raw spread of the eight group sums, which Stage 1
literally cannot touch because there is exactly one group per node and no choice to make. Read the number
back into physical terms: balance_node `= mean_node / max_node = 0.702` means the busiest node runs about
`1/0.702 ≈ 1.42×` the average node — a group forty-two percent above the mean, sitting on its own node,
with the other seven waiting on it every layer. That is entirely a property of where the hot experts
landed among the eight fixed groups, and nothing in Stages 2 or 3 can move a single token off that node. On stress-skew
it is worse: `groups_per_node = 2`, a long-tail Zipf at skew 0.95, so Stage 1 packs 32 wildly-skewed
group loads two to a node across 16 nodes, and with only two slots per node there is almost no room to
even them out — hence balance_node 0.336, and a balance of 0.222 that no downstream packing can rescue,
because the node a replica lands on was decided by a Stage 1 that had its hands tied. And recall the
partition geometry from the start of the ladder: balance can never exceed balance_node, because a GPU
cannot be lighter than an equal share of its node. So a frozen balance_node of 0.702 caps deepseek-v3's
balance at 0.702 no matter how perfectly Stage 3 packs, and a frozen 0.336 caps stress-skew's balance at
0.336 — and indeed the measured balances, 0.659 and 0.222, sit under those ceilings exactly as the
inequality demands. The hierarchy is *buying* the perfect locality by *spending* balance: by insisting
every group, and therefore every expert's replicas, stays whole on one node, it hands Stage 1 a rigid,
low-freedom partition and then asks Stages 2 and 3 to balance within a node structure that is already
lopsided.

So what are my moves, given that the cap is Stage 1 and not the packer? I can see four, and three of
them die on inspection. I could put a *better* Stage-1 packer in — but the whole diagnosis is that two
very different packers produced identical balance_node, so a third packer, however clever, packs the same
32 skewed group loads into the same two-per-node boxes and lands in the same place; the freedom simply is
not there to be exploited. I could *re-group* the experts so the group loads come out even before
packing — but the groups are fixed by the router's node-limited routing; they are a property of the
serving stack I do not own, not a knob on the placement. I could relax the confinement *partially*,
letting a few of the hottest experts spill their replicas across a node boundary while keeping everyone
else local — but that is a delicate, tunable middle path, and before I invest in it I want to know
whether removing the confinement helps *at all*. Which is the fourth move, and the cheapest test of the
hypothesis: drop the node hierarchy entirely. Instead of "pack groups onto nodes, then replicate and
pack within each node," do a single global pass — replicate over *all* logical experts at once, then
pack *all* the physical replicas directly across *all* the GPUs in the cluster, ignoring group and node
structure. This is not a new algorithm; it is the established global policy, the same procedure with the
group and node counts collapsed to one — precisely the `num_groups = num_nodes = 1` regime I noted at
the very start as living inside the same code path, one giant pool. The hierarchy's official fallback
when group-to-node divisibility is absent *is* this flat pack. Here I am choosing it deliberately, as the
explicit global branch, even though the hierarchy is available, because I want a clean read on whether
the hierarchy is what is capping balance. And I can check the two are truly the same procedure by
collapsing the hierarchy on paper: set `num_nodes = 1` and Stage 1 packs every group onto a single node,
the trivial identity; `experts_per_node` becomes all `E` experts and `replicas_per_node` becomes all
`num_replicas`, so Stage 2 replicates globally; `gpus_per_node` becomes all `num_gpus`, so Stage 3 packs
every replica across the whole cluster. That is line-for-line the flat construction, node-major
relabeling and its inverse stripped out as no-ops. So the flat path is not new, untested code — it is the
exercised global branch with its trivial wrappers removed, which is reassuring for a change whose whole
risk is silent mis-routing.

Why would a flat global pack balance better? Because the packer now sees the whole problem at once. In
the hierarchy, Stage 3 packs only one node's `replicas_per_node` items onto that node's `gpus_per_node`
GPUs — a small, isolated subproblem stuck with whatever lopsided load that node inherited from Stage 1;
on stress-skew that is 24 replicas onto 8 GPUs inside a node whose two groups may both be hot. Flat, the
packer sees all `num_replicas` items and all `num_gpus` packs together, so a heavy replica on what would
have been an overloaded node can be balanced against a light slot *anywhere* in the cluster, not just
within its node. The single largest constraint on the achievable max-GPU-load — that each GPU's load is
bounded below by the average of its *node's* inherited load, the very floor the balance-≤-balance_node
inequality encodes — disappears, because there is no node structure to inherit from. So balance_node
should climb off the frozen 0.702 and 0.336, and balance should climb with it, because the node-level
imbalance was a Stage 1 artifact and there is no Stage 1 now. The zigzag packing rule itself I keep,
because the runtime lesson from the last rung stands: it is the vectorized snake, one sort and index
arithmetic, no per-item loop, and run globally over all replicas and all GPUs it is still cheap.

And there is a second freedom the flat pass unlocks that the packing story alone misses, one that hits
exactly the config the hierarchy strangled worst. In the hierarchy, replication is *also* confined:
Stage 2 runs per node, so the hottest expert on a node can only draw from that node's spare slots — and
every node has exactly eight spare copies to share across all its experts, so however hot one expert is,
it caps at about nine replicas. Recall the balance ceiling that motivated replication in the first place:
a max-GPU load floored by the hottest per-replica load `w / r`. Under the hierarchy that `r` is bounded
by the node budget. Go flat and the hottest expert competes for the *whole* extra budget — 64 copies on
deepseek-v3, 128 on stress-skew — so water-filling can pour as many replicas into it as it takes to bring
its per-replica load down to the global equilibrium, unbounded by any node's local eight. Do the
stress-skew arithmetic: if the single hottest expert carries something like a sixth of the traffic, the
hierarchy holds it at nine replicas for a per-replica load around `0.16/9 ≈ 1.8%` against a fair share of
`1/128 ≈ 0.78%`, a floor that alone caps balance near 0.43 and in practice far lower; flat can hand that
same expert twenty-odd replicas, dropping its per-replica load toward `0.16/20 ≈ 0.8%`, right at the fair
share. So the flat pass attacks the balance ceiling from both sides at once — it frees the replication to
shave the hottest per-replica load *and* frees the packing to place those shavings anywhere — which is
why I expect the largest swing precisely on stress-skew, where both confinements bit hardest.

It is worth asking where the flat balance actually tops out, so I do not expect a perfect 1.0 and read a
shortfall as failure. Global water-filling equalizes per-replica loads across the experts it chooses to
replicate: it keeps feeding the current argmax until the extras run out, so at the end the heaviest
per-replica load is about the average per-replica load over the *replicated* set, which for a modest
budget sits a little above the global fair share `total / num_replicas`. Two residuals keep balance shy
of 1.0. First, the cold tail is still indivisible — an expert carrying less than a fair share gets one
copy and one GPU, and if it is lighter than the pack average it drags the *mean* down relative to the
max, a small permanent gap. Second, the snake is still a heuristic, and I already measured it conceding
about two points against adaptive greedy on the GPU-versus-node packing gap; that concession rides along
here too. So the honest ceiling on the real configs is high but not perfect — I expect balance up in the
high 0.97s on deepseek-v3, where 64 extras over 256 experts is enough headroom to flatten the moderate
Zipf-0.7 tail, and a shade higher on the gentler qwen and v2 loads — while stress-skew, with the fiercest
skew even after the replication is freed, climbs a lot but should top out lower, in the low 0.9s rather
than the high 0.97s, because 128 extras still cannot fully divide a Zipf-1.0/skew-0.95 tail down to a
128-way fair share.

Let me be concrete about the construction, because the global path is simpler than the hierarchy and I
want to be sure the maps still come out right. Replicate first: start every logical expert with one copy
and hand each of the `num_replicas − E` extra slots to the expert with the largest current per-replica
load `weight / logcnt` — the same water-filling rule, now over all `E` experts at once rather than per
node. Then the load of each physical slot is `weight / logcnt` gathered through `phy2log` (which says
which logical expert each slot represents), and I snake-pack those `num_replicas` per-replica loads
across `num_gpus` packs in one shot. Trace the shapes on deepseek-v3 to be sure they close: `phy2log` is
`[L, 320]`, the per-slot loads `[L, 320]`, the pack sends them into 64 GPUs, and `phy_per_gpu = 320/64 =
5`, so the GPU-ordered slot id is `pack_index · 5 + rank_in_pack`, a value in `0..319` — a genuine
permutation of the 320 slots, which I invert to get "which original slot landed at each packed
position," gather `phy2log` and the replica ranks through the inverse, and scatter the packed slot ids
into `log2phy[e, r]` sized by the max replica count. Equal cardinality is automatic — the snake puts
exactly one item per pack per round, so each GPU gets exactly 5 slots — and every expert keeps ≥ 1 copy
because replication only adds. The `num_groups` and `num_nodes` arguments are still accepted by the entry
point but the flat path simply does not consult them; the node-major relabeling that Stage 1 used, and
its inverse, are gone, which also means the composition is shorter and there is less index plumbing to
get wrong.

Now the cost, the part the metric will make me pay, and I have to be honest that it is not free. The
locality metric counts, for each expert, how many distinct nodes hold its replicas and credits `1 /
nodes_per_expert`, traffic-weighted. The hierarchy scored a perfect 1.000 everywhere because every
expert's replicas were confined to one node by construction. The flat pack has no such confinement: it
scatters an expert's replicas across whatever GPUs the snake assigns, and those GPUs can sit on different
nodes. So I am about to *lose* the one metric the hierarchy nailed, and the question is how much. The
replica-count distribution is what saves it from collapse. On deepseek-v3 there are 64 extra copies over
256 experts, so at most 64 experts are replicated and at least 192 keep exactly one copy — and a
one-replica expert lives on exactly one GPU, hence one node, hence locality 1.0, untouchable. Only the
replicated hot experts can scatter, and even a two-replica expert touches at most two of the eight nodes,
scoring at worst 0.5. The traffic weighting cuts both ways: the hot experts that scatter widest are
exactly the ones carrying the most traffic, so they pull the weighted average down more than their count
suggests — but the long tail of single-replica experts, perfectly local, still holds a large share of the
weighted mass. So I expect locality to stay in the low-to-mid 0.90s on the real configs, a few points
lost, not collapsed. On stress-skew the trade is steeper: 16 nodes means the `1 / num_nodes` floor is
`0.0625`, the replication budget spreads 128 extra copies so the hot experts get many replicas and
scatter the widest, and the extreme skew concentrates traffic on exactly those scattered experts — so I
expect locality there to fall the hardest, into the low 0.7s.

Whether this is a net win is arithmetic on four equally-weighted terms under a geometric mean, and the
shape of that arithmetic is what makes me expect a win rather than merely hope for one. The hierarchy's
balance deficits are *large and concentrated* — 0.659 and 0.222 on two configs — while the locality I am
giving up is *bounded* by the replica structure, a handful of points on the real configs and into the
0.7s on the stress config. Trading a 0.222 up toward the low 0.9s on stress-skew, against a locality drop
from 1.0 to the low 0.7s on that same config, is trading a factor-of-four gain for a quarter loss on
terms that count equally — the balance swing dwarfs the locality swing. On deepseek-v3, balance from
0.659 toward the high 0.97s against locality from 1.0 to the low-to-mid 0.90s is again a large gain for a
small loss. On qwen3-moe and deepseek-v2, already near 0.91 on balance, the gain is smaller but still
positive and the locality loss smaller still. So I expect the task score to climb past zigzag's 0.375,
into the high 0.30s. As for runtime, it should stay cheap — the packing is still the vectorized snake,
now over the global pool, with the Stage 1 group-to-node work removed. The one thing that grows is the
replication loop: flat runs `num_replicas − E` sequential argmax draws globally — 64 on deepseek-v3, 128
on stress-skew — where the hierarchy ran only its eight-per-node draws, so on the smaller configs the one
big pool can cost a touch more than several tiny per-node packs. But it is a low-hundreds of sequential
draws at most, batched over layers, so it must stay in the low single digits of milliseconds, nowhere
near greedy's hundreds.

There is one thing I want to be honest with myself about, because it is the seam where this rung could be
beaten. By dropping the hierarchy I have *decoupled* the two objectives the hierarchy had fused. The
hierarchy got locality for free and paid in balance; the flat pack gets balance and pays in locality.
Neither is doing the thing the task actually rewards, which is *both at once*. A method that kept the
node structure — so locality stayed near 1.0 — while somehow finding more node-level balancing freedom
than the rigid one-group-per-node Stage 1 allows would dominate both of these rungs. I do not have that
method in hand, and the partial-confinement idea I set aside earlier is where I would go looking for it;
the flat pack is the move the zigzag feedback points to *now*, because the feedback said the cap was the
hierarchy and the cheapest way to test that is to remove it. But I am noting the seam, because a win here
that trades one perfect metric for another is a win with a soft underbelly.

So the falsifiable expectations against the zigzag numbers. Runtime should stay in the low single-digit
milliseconds — I would not be surprised if it is marginally slower than the hierarchical snake on the
smaller configs because of the longer global replication loop, but it must not creep back toward greedy's
hundreds. Balance should rise substantially on the two configs the hierarchy strangled: deepseek-v3 from
0.659 toward the high 0.97s, stress-skew from 0.222 up into the low 0.9s — and balance_node should rise
in lockstep off the 0.702 and 0.336 the hierarchy locked in, because there is no Stage 1 to freeze it,
and it must stay at or above balance the whole way, or the partition geometry I trust is wrong. Locality
should *drop* off the perfect 1.000 — into the low-to-mid 0.90s on the real configs and into the low 0.7s
on stress-skew — and that drop is the price. The net: the balance gains are large and land on exactly the
configs that were dragging the geometric mean, the locality losses are bounded, so the task score should
climb past 0.375 into the high 0.30s. If locality collapses further than I expect on the real configs, or
if balance does *not* climb because something other than the hierarchy was capping it, then the trade
fails and the hierarchy should be kept; that is exactly what the next comparison will tell me.
