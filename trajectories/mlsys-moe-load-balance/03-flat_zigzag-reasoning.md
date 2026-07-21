The zigzag run did exactly what I asked and exposed exactly what it could not do. Runtime collapsed:
248 / 102 / 153 / 256 ms became 1.6 / 0.85 / 1.2 / 1.6 — two orders of magnitude, the `L·R·D/N`
interpreted loop replaced by a sort and index kernels. Locality held at exactly 1.000 on all four. And
the balance came in right where I said: at or just below greedy's on the smooth real-model configs —
0.679 → 0.659 on deepseek-v3, 0.940 → 0.919 on qwen3-moe, 0.925 → 0.906 on deepseek-v2, about two
points each, the snake's small premium over adaptive greedy on the thin GPU-versus-node gap — and
unmoved on stress-skew, 0.222 → 0.222. The task score jumped 0.255 → 0.375, almost all of it the
removed runtime tax. The runtime wound is closed. But the score is now bottlenecked by *balance*, and
the numbers say precisely where: mediocre on deepseek-v3 (0.659), catastrophic on stress-skew (0.222),
those two dragging the geometric mean down while qwen and v2 sit near 0.91. The question is no longer
speed. It is why the balance is capped, and what is capping it.

Look at what greedy and zigzag share, because it is a diagnostic coincidence. They score *identical*
balance_node to six digits — 0.702209 on deepseek-v3 for both, 0.335967 on stress-skew for both,
0.946751 on qwen3-moe for both, and on deepseek-v2 differing only in the sixth decimal. That is a
strong tell. Greedy and snake are *different packers* — one an adaptive least-loaded sweep, the other a
fixed positional deal — and yet they place the load across nodes identically. If the node-level balance
were set by the packing rule, two rules this different would not agree to six digits. So the node
balance is decided *upstream* of the packing, by the one thing they share unchanged: Stage 1, the
group-to-node assignment, and the hierarchy that forces its shape. GPU-level balance dropped a couple
of points when I swapped packers because the packer controls the thin GPU-versus-node gap; node-level
balance did not budge because the packer does not control it. The cap is Stage 1.

Config by config the mechanism differs in degree. On deepseek-v3, eight groups and eight nodes means
`groups_per_node = 1` — Stage 1 has *no freedom*: one group per node, the identity, whatever imbalance
the eight group loads carry locked in before replication and packing begin. balance_node 0.702 means
the busiest node runs `1/0.702 ≈ 1.42×` the average, a group forty-two percent over the mean sitting on
its own node with the other seven waiting on it every layer — entirely a property of where the hot
experts landed among the fixed groups, untouchable downstream. On stress-skew it is worse:
`groups_per_node = 2`, long-tail Zipf at skew 0.95, so Stage 1 packs 32 wildly-skewed group loads two
to a node across 16 nodes with almost no room to even them — hence balance_node 0.336 and a balance of
0.222 no downstream packing can rescue. And with `balance ≤ balance_node` from the partition geometry, a
frozen balance_node of 0.702 caps deepseek-v3's balance there and a frozen 0.336 caps stress-skew's, and
the measured 0.659 and 0.222 sit under those ceilings exactly as the inequality demands. The hierarchy
is *buying* perfect locality by *spending* balance: insisting every group stays whole on one node hands
Stage 1 a rigid, low-freedom partition, then asks Stages 2–3 to balance within a node structure that is
already lopsided.

So what are my moves, given the cap is Stage 1 and not the packer? Three die on inspection. A *better*
Stage-1 packer is pointless — two very different packers already produced identical balance_node, so a
third packs the same skewed group loads into the same two-per-node boxes; the freedom is not there.
*Re-grouping* the experts to even the group loads is not mine to do — the groups are fixed by the
router's node-limited routing, a property of the serving stack, not a knob on placement. *Partial*
relaxation — letting a few hottest experts spill replicas across a node boundary while everyone else
stays local — is a delicate tunable middle path, and before I invest in it I want to know whether
removing the confinement helps *at all*. That is the fourth move and the cheapest test of the
hypothesis: drop the node hierarchy entirely. Instead of "pack groups onto nodes, then replicate and
pack within each node," do a single global pass — replicate over *all* logical experts at once, then
pack *all* physical replicas directly across *all* GPUs, ignoring group and node structure. This is the
`num_groups = num_nodes = 1` regime I noted at the start as living in the same code path, one giant
pool — the hierarchy's own fallback when group-to-node divisibility is absent, with the node-major
relabeling and its inverse collapsing to no-ops. So the flat path is not new, untested code but the
exercised global branch with its trivial wrappers removed, reassuring for a change whose whole risk is
silent mis-routing.

Why would a flat pack balance better? Because the packer now sees the whole problem. In the hierarchy,
Stage 3 packs only one node's `replicas_per_node` items onto that node's `gpus_per_node` GPUs — a small
subproblem stuck with whatever lopsided load that node inherited from Stage 1 (on stress-skew, 24
replicas onto 8 GPUs inside a node whose two groups may both be hot). Flat, the packer sees all
`num_replicas` items and all `num_gpus` packs together, so a heavy replica can be balanced against a
light slot *anywhere*. The single largest constraint on the achievable max-GPU load — that each GPU is
bounded below by the average of its node's inherited load, the floor `balance ≤ balance_node` encodes —
disappears with the node structure, so balance_node should climb off the frozen 0.702 and 0.336 and
balance with it. The zigzag packing rule itself I keep — the runtime lesson stands — and run globally
over all replicas and all GPUs it is still cheap.

There is a second freedom the flat pass unlocks, and it hits exactly the config the hierarchy strangled
worst. In the hierarchy, replication is *also* confined: Stage 2 runs per node, drawing only from that
node's eight spare copies, so however hot one expert is it caps at about nine replicas. Recall the
balance ceiling that motivated replication — a max-GPU load floored by the hottest per-replica load
`w/r`. Under the hierarchy that `r` is bounded by the node budget; go flat and the hottest expert
competes for the *whole* extra budget — 64 copies on deepseek-v3, 128 on stress-skew — so water-filling
can pour as many replicas into it as needed. The stress-skew arithmetic: if the single hottest expert
carries something like a sixth of the traffic, the hierarchy holds it at nine replicas for a per-replica
load around `0.16/9 ≈ 1.8%` against a fair share of `1/128 ≈ 0.78%`, a floor that alone caps balance
below a half; flat can hand it twenty-odd replicas, dropping its per-replica load toward `0.16/20 ≈
0.8%`, right at the fair share. So the flat pass attacks the balance ceiling from both sides — it frees
replication to shave the hottest per-replica load *and* frees packing to place the shavings anywhere —
which is why I expect the largest swing on stress-skew, where both confinements bit hardest.

Where does the flat balance top out, so I do not read a shortfall as failure? Global water-filling
equalizes per-replica loads across the experts it chooses to replicate, so the heaviest per-replica
load ends near the average over the *replicated* set, a little above the global fair share
`total / num_replicas`. Two residuals keep balance shy of 1.0: the cold tail is still indivisible — an
expert under a fair share gets one copy on one GPU and drags the mean down against the max — and the
snake is still a heuristic conceding its couple of points on the GPU-versus-node gap. So the honest
ceiling is high but not perfect: on the real configs, up into the high 0.9s where the extra budget is
enough to flatten a moderate Zipf tail; stress-skew climbs a lot but should top out clearly lower,
because 128 extras still cannot fully divide a Zipf-1.0/skew-0.95 tail down to a 128-way fair share.

The construction is simpler than the hierarchy. Replicate first — every logical expert starts with one
copy, each of the `num_replicas − E` extras goes to the largest current `weight / logcnt`, now over all
`E` experts at once. Then each physical slot's load is `weight / logcnt` gathered through `phy2log`, and
I snake-pack those `num_replicas` loads across `num_gpus` packs in one shot. On deepseek-v3: `phy2log`
is `[L, 320]`, per-slot loads `[L, 320]`, `phy_per_gpu = 320/64 = 5`, so the GPU-ordered slot id is
`pack_index · 5 + rank_in_pack` in `0..319`, a genuine permutation of the 320 slots, which I invert,
gather `phy2log` and ranks through, and scatter into `log2phy[e, r]` sized by the max replica count.
Equal cardinality is automatic (the snake puts one item per pack per round, so each GPU gets 5 slots),
every expert keeps ≥ 1 copy since replication only adds, and `num_groups`/`num_nodes` are still accepted
but not consulted — the node-major relabeling and its inverse are gone, so there is less plumbing to get
wrong.

Now the cost the metric makes me pay, and it is not free. `locality` credits each expert `1 /
nodes_per_expert`, traffic-weighted. The hierarchy scored a perfect 1.000 because every expert was
confined to one node; the flat pack scatters replicas across whatever GPUs the snake assigns, which can
sit on different nodes. So I am about to *lose* the metric the hierarchy nailed — how much? The
replica-count distribution saves it from collapse. On deepseek-v3, 64 extra copies over 256 experts
means at most 64 experts are replicated and at least 192 keep exactly one copy — one GPU, one node,
locality 1.0, untouchable. Only replicated hot experts scatter, and a two-replica expert touches at
most two of eight nodes, scoring at worst 0.5. Traffic weighting cuts both ways: the widest-scattering
experts carry the most traffic and pull the average down more than their count, but the long tail of
single-replica experts still holds a large share of the weighted mass. So on the real configs I expect
locality to hold in the low-to-mid 0.90s, a few points lost, not collapsed. On stress-skew the trade is
steeper — 16 nodes so the `1/num_nodes` floor is `0.0625`, 128 extras spread the hot experts widest,
and extreme skew concentrates traffic on exactly those — so locality there should fall the hardest,
well below the reals.

Whether this is a net win is arithmetic on four equally-weighted terms under a geometric mean, and its
shape makes me expect a win rather than hope for one. The hierarchy's balance deficits are large and
concentrated — 0.659 and 0.222 — while the locality I give up is bounded by the replica structure, a
handful of points on the reals and steeper on stress-skew. Trading stress-skew's 0.222 up toward the
low 0.9s against a locality drop from 1.0 into the 0.7s is trading a factor-of-four gain for a quarter
loss on terms that count equally; on deepseek-v3, balance 0.659 up into the high 0.9s against locality
1.0 → low-to-mid 0.90s is again a large gain for a small loss; on qwen and v2, already near 0.91, the
gain is smaller but positive and the loss smaller still. So I expect the task score past zigzag's 0.375,
into the high 0.30s. Runtime should stay cheap — still the vectorized snake, now global, with the
Stage-1 work removed — but the one thing that grows is the replication loop: flat runs `num_replicas −
E` sequential argmax draws (64 on deepseek-v3, 128 on stress-skew) where the hierarchy ran only eight
per node, so on the smaller configs the one big pool can cost a touch more than several tiny per-node
packs. Low-hundreds of draws at most, batched over layers, so it stays in the low single-digit
milliseconds, nowhere near greedy's hundreds.

One thing to be honest about, because it is the seam where this approach could be beaten. Dropping the
hierarchy *decouples* the two objectives it had fused. The hierarchy got locality free and paid in
balance; the flat pack gets balance and pays in locality. Neither does the thing the task actually
rewards, which is *both at once*. A method that kept the node structure — locality near 1.0 — while
finding more node-level balancing freedom than the rigid one-group-per-node Stage 1 allows would
dominate both. I do not have it in hand, and the partial-confinement idea I set aside is where I would
look; the flat pack is the move the feedback points to *now*, because it said the cap was the hierarchy
and the cheapest way to test that is to remove it. But I note the seam: a win that trades one perfect
metric for another has a soft underbelly.

So the sharp test. Balance should rise substantially on the two strangled configs — deepseek-v3's 0.659
and stress-skew's 0.222 both well up — with balance_node rising in lockstep off the frozen 0.702 and
0.336 and staying at or above balance throughout, or the partition geometry I trust is wrong. Locality
is the price, dropping off 1.000 into the low-to-mid 0.90s on the reals and hardest on stress-skew;
runtime stays in the low single digits. If instead locality collapses further than that, or balance does
*not* climb because something other than the hierarchy was capping it, then the trade fails and the
hierarchy should be kept.
