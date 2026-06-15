The zigzag run did exactly what I asked of it and exposed exactly what it could not do. Runtime
collapsed: 248 / 102 / 153 / 256 ms became 1.6 / 0.85 / 1.2 / 1.6 ms — two orders of magnitude, just as
the vectorization argument predicted, the `L·n·P` interpreted loop replaced by a sort and a handful of
index kernels. Locality held at exactly 1.000 on all four configs — the hierarchy was untouched, so no
expert crosses a node. And the balance came in right where I said it would: roughly at or just below
greedy's on the smooth real-model configs (qwen3-moe 0.919, deepseek-v2 0.906, deepseek-v3 0.659), and
pinned at 0.222 on stress-skew, because the snake cannot fix a problem the hierarchy itself imposes. The
task score jumped from 0.255 to 0.375, almost all of it the removed runtime tax. So the runtime wound
is closed. But the score is now bottlenecked by *balance*, and the numbers tell me precisely where: the
balance term is mediocre on deepseek-v3 (0.659) and catastrophic on stress-skew (0.222), and on those
two configs it is dragging the geometric mean down hard while qwen3-moe and deepseek-v2 sit comfortably
near 0.92. The question for this rung is no longer speed. It is: why is the balance capped, and what is
actually capping it.

Look at what greedy and zigzag share. They both score *identical* balance_node — 0.702 on deepseek-v3,
0.336 on stress-skew — and nearly identical balance. That is the tell. The bottleneck is not the
packing rule at all; greedy and snake are different packers and they land in the same place. The
bottleneck is upstream of the packing: it is **Stage 1**, the group-to-node assignment, and the
hierarchy that forces it. On deepseek-v3 there are 8 groups and 8 nodes, so `groups_per_node = 1` —
Stage 1 has *no freedom whatsoever*: one group per node, the assignment is forced, and whatever
node-level imbalance the group loads happen to have is locked in before replication and GPU packing even
begin. balance_node 0.702 is not a packing failure; it is the raw spread of the eight group loads,
which Stage 1 cannot touch because there is exactly one group per node. On stress-skew it is worse:
`groups_per_node = 2`, a long-tail Zipf at skew 0.95, so Stage 1 packs 32 wildly-skewed group loads two
to a node across 16 nodes, and with two slots per node there is almost no room to even them out — hence
balance_node 0.336 and a balance of 0.222 that no downstream packing can rescue, because the node a
replica lands on was decided by a Stage 1 that had its hands tied. The hierarchy is *buying* the perfect
locality by *spending* balance: by insisting every group — and therefore every expert's replicas —
stays whole on one node, it hands Stage 1 a rigid, low-freedom partition and then asks Stages 2 and 3 to
balance within a node structure that is already lopsided.

So the move this rung has to consider is the one the hierarchy forbids: give the placement back its
freedom by *dropping the node hierarchy entirely*. Instead of "pack groups onto nodes, then replicate
and pack within each node," do a single global pass — replicate over *all* logical experts at once, then
pack *all* the physical replicas directly across *all* the GPUs in the cluster, ignoring group and node
structure. This is not a new algorithm; it is the established global policy, the official fallback the
hierarchical entry point uses when the group-to-node divisibility needed for the hierarchy is absent. It
is the same procedure with the group and node counts collapsed to one — one giant pool. Here I am
choosing it deliberately, as the explicit global branch, even though the hierarchy *is* available,
because I want to test the hypothesis that the hierarchy is what is capping balance.

Why would a flat global pack balance better? Because the packer now sees the whole problem at once. In
the hierarchy, Stage 3 packs only one node's `replicas_per_node` items onto that node's `gpus_per_node`
GPUs — a small, isolated subproblem with whatever lopsided load that node inherited from Stage 1. Flat,
the packer sees all `num_replicas` items and all `num_gpus` packs together, so a heavy replica on what
would have been an overloaded node can be balanced against a light slot anywhere in the cluster, not
just within its node. The single largest constraint on the achievable max-GPU-load — that each GPU's
load is bounded below by the average of its *node's* inherited load — disappears. The zigzag packing
rule itself I keep, because the runtime lesson from the last rung stands: it is the vectorized snake,
one sort and index arithmetic, no per-item loop. Run globally over all replicas and all GPUs, it is
still cheap. So the flat method is: replicate globally, then snake-pack the global per-replica loads
across all GPUs.

Let me be concrete about the construction, because the global path is simpler than the hierarchy and I
want to be sure the maps still come out right. Replicate first: start every logical expert with one
copy and hand each of the `num_replicas − E` extra slots to the expert with the largest current
per-replica load `weight / logcnt` — the same water-filling rule, now over all `E` experts at once
rather than per node. This is unchanged from the hierarchical Stage 2 except that it runs once over the
whole model; it is the genuinely-sequential argmax loop, but only over the redundant count, so it is not
the cost center. Then the load of each physical slot is `weight / logcnt` gathered through `phy2log`
(which says which logical expert each slot represents), and I snake-pack those `num_replicas` per-replica
loads across `num_gpus` packs in one shot. The packing returns, per slot, its GPU and its rank within
that GPU; the GPU-ordered linear slot id is `pack_index · phy_per_gpu + rank_in_pack`, I invert that
permutation to get "which original slot landed at each packed position," gather `phy2log` and the
replica ranks through the inverse, and scatter the packed slot ids into `log2phy[e, r]` sized by the max
replica count. Equal cardinality is automatic — the snake puts exactly one item per pack per round, so
each GPU gets exactly `phy_per_gpu` slots — and every expert keeps ≥ 1 copy because replication only
adds. The `num_groups` and `num_nodes` arguments are still accepted by the entry point but the flat path
simply does not consult them; the group-to-node relabeling that Stage 1 used is gone.

Now the cost, the part the metric will make me pay. The locality metric counts, for each expert, how
many distinct nodes hold its replicas and credits `1 / nodes_per_expert`, traffic-weighted. The
hierarchy scored a perfect 1.000 everywhere because every expert's replicas were confined to one node by
construction. The flat pack has no such confinement: it scatters an expert's replicas across whatever
GPUs the snake assigns, and those GPUs can sit on different nodes. A hot expert with many replicas,
spread by the packer across the cluster for balance, will touch several nodes, so its locality drops
toward `1 / num_nodes`. So I am about to *lose* the one metric the hierarchy nailed, in exchange for the
balance it was strangling. This is a genuine trade, and whether it is a net win depends on the
arithmetic of the four equally-weighted terms. The reason I expect it to win: the hierarchy's balance
deficits are *large and concentrated* — 0.659 and 0.222 on two configs — while the locality I am giving
up is bounded. On the real-model configs locality will not fall to `1/N`; replication factors are modest
(the budget is roughly 1.25–2×), so most experts have one or two replicas and a one- or two-replica
expert can only touch one or two nodes, keeping its locality high; only the heavily-replicated hot
experts scatter, and they are a minority of the traffic-weighted mass. So I expect locality to stay in
the low-to-mid 0.90s on the real configs — a few points lost, not collapsed — while balance climbs from
the 0.66–0.92 range up toward the high 0.97s, because the packer finally has the whole GPU pool. On
stress-skew the trade is steeper: 16 nodes means the `1/N` floor is `0.0625`, the replication budget is
tightest (1.5×), and the skew is extreme, so the hot experts get the most replicas and scatter the
widest — I expect locality there to fall the hardest, into the low 0.7s, but balance to climb from the
floored 0.222 up toward the low 0.9s, which on a config where balance was the catastrophe is a far
larger swing than the locality loss.

There is one thing I want to be honest with myself about, because it is the seam where this rung could
be beaten later. By dropping the hierarchy I have *decoupled* the two objectives the hierarchy had
fused. The hierarchy got locality for free and paid in balance; the flat pack gets balance and pays in
locality. Neither is doing the thing the task actually rewards, which is *both at once*. A method that
kept the node structure — so locality stayed near 1.0 — while finding more node-level balancing freedom
than the rigid one-group-per-node Stage 1 allows would dominate both of these rungs. I do not have that
method in hand; the flat pack is the move that the zigzag feedback points to, because the feedback said
the cap was the hierarchy and the cheapest way to test that is to remove it. But I am noting the seam.

So the falsifiable expectations against the zigzag numbers. Runtime should stay cheap — single-digit
milliseconds — because the packing is still the vectorized snake, now over the global pool, with the
Stage 1 group-to-node work removed; I would not be surprised if it is marginally slower than the
hierarchical snake on the smaller configs (replicating and packing one large pool can cost a touch more
than several tiny per-node packs) but it must stay in the low single digits, nowhere near greedy's
hundreds. Balance should rise substantially on the two configs the hierarchy strangled: deepseek-v3
from 0.659 toward the high 0.97s, and stress-skew from 0.222 up into the low 0.9s — and balance_node
should rise in lockstep, off the 0.702 and 0.336 the hierarchy locked in, because the node-level
imbalance was a Stage 1 artifact and there is no Stage 1 now. Locality should *drop* off the perfect
1.000 — into the low-to-mid 0.90s on the real configs and into the low 0.7s on stress-skew — and that
drop is the price. The net: the balance gains are large and on the configs that were dragging the
geometric mean, the locality losses are bounded, so the task score should climb past zigzag's 0.375 —
into the high 0.30s. If locality collapses further than I expect on the real configs, or if balance
does not climb because something other than the hierarchy was capping it, the trade fails and the
hierarchy should be kept; that is exactly what the next comparison will tell me.
