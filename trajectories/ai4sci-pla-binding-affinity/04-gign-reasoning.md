EHIGN landed exactly where I said it would, and reading its row carefully tells me the heavy machinery
is buying its gain in a narrow place — which is the opening for a leaner method. The headline came in
as predicted: on 2016 EHIGN jumped to RMSE 1.2426 / Rp 0.8218, by far the best 2016 on the ladder,
recovering the distance resolution EGNN lost (1.4213 / 0.7646) and clearing past SchNet's 0.7767. The
full 17/11-dim edge geometry in the messages plus the sum-of-contacts readout did the job on the
near-training core set. But the open question I flagged for 2019 resolved against the heavy machinery:
2019 came in at RMSE 1.4604 / Rp 0.6213 — the Rp barely moved past EGNN's 0.6175, and the RMSE actually
got *worse* than EGNN's 1.4422. And 2013 was essentially flat versus EGNN: 1.4117 / 0.8066 against
EGNN's 1.4114 / 0.8093 — EHIGN even lost a hair of 2013 Rp. So the picture is sharp: EHIGN's dual-head,
attention-bias-corrected, consistency-trained apparatus delivers a big win on 2016, a wash on 2013, and
a slight *regression* on the temporally distant 2019. That is the signature of capacity spent on the
training-era chemistry — the bias-correction heads and the rich covalent-edge features (bond type,
conjugation, ring membership) sharpen the familiar core sets but do not transfer to the held-out
temporal split, where the RMSE drifted up. The diagnosis is the inverse of the usual "add more": the
question is now whether all that machinery is *necessary*, or whether a leaner geometric core can hold
the wins where they matter while not paying the 2019 cost.

Let me re-derive from the physics with that question in front of me, because the lean answer is not "do
less of EHIGN" — it is a differently-shaped model. Binding affinity is set by interactions of two
distinct kinds — covalent bonds inside each molecule, non-covalent contacts across the interface — and
the whole point of a heterogeneous layer is to let a single atom, in a single step, fuse both its
covalent and its non-covalent neighbourhood into one updated representation, rather than process them in
separate stages. EHIGN does fuse them per step, but it does so with two *different* convolution
mechanisms (additive-edge sum for covalent, gated-mean for non-covalent) and then reads out only through
the interface. I want to ask whether a *single, uniform* message-passing primitive — applied to all
three edge sets with the same form — combined with a *two-channel readout* that uses both the interface
and the whole-complex pooled representation, generalizes better. The intuition is that a uniform
primitive has fewer ways to overfit a particular interaction regime, and a readout that does not bet
everything on the interface sum has a fallback when the interface scoring transfers poorly.

The primitive I reach for is the GIN-style update, because it is the most expressive simple
message-passing form and it injects the edge cleanly. For an intra-molecular edge I project the edge
feature to the node width and add it to the source, `msg = x_src + edge_proj(edge_attr)`, sum over
neighbours, and update as `mlp((1 + ε) · x + agg)` with a learnable scalar `ε` — the GIN form, where
`(1+ε)·x` keeps the centre node distinguishable from its aggregated neighbourhood (the thing that makes
GIN as discriminative as the Weisfeiler-Lehman test) and the MLP is `Linear → BatchNorm → ReLU →
Linear`. The edge projection is what carries the geometry: the same 17-dim covalent features and 11-dim
contact features EHIGN used, but here added into the message through a single learned projection rather
than gating it. I run this same GIN layer on the ligand's covalent graph and the pocket's covalent graph
with separate weights, and I run a sibling inter-molecular GIN layer on the ligand→pocket contacts. The
inter layer differs only in that source and destination are different node types, so it projects the
contact edge, adds it to the source, mean-aggregates over the variable contact degree (mean, not sum,
for the same degree-variance reason as before), and updates the destination by `mlp([x_dst, agg])` on
the concatenation. Three layers, hidden width 256, each with a residual `x ← layer(x) + x`. Crucially I
apply the intra layers to *both* molecules and the inter layer *into the pocket* every step, so a pocket
atom's representation accumulates both its own covalent context and the ligand contacts pressing on it,
fused in one step — the heterogeneous fusion, but built from one uniform primitive.

Now the readout, which is where this rung most deliberately departs from EHIGN. EHIGN bet the whole
prediction on a sum of per-contact scores minus an attention bias, in two directions, with a consistency
loss tying them. That is a strong, interpretable bet, and 2019 suggests it is also a *brittle* one — the
interface sum is exactly the quantity most sensitive to which contacts happen to fall inside the 5 Å
cutoff, and on held-out chemistry that cutoff population shifts. So I hedge with two readout channels and
average them. The first is the interface channel, kept in spirit but stripped of the bias correction and
the dual-head consistency machinery: for each ligand→pocket contact, score it from the concatenation of
the final ligand-atom feature, the final pocket-atom feature, and the *raw* contact edge feature, through
`Linear(2H + inter_edge_dim, H) → ReLU → Linear(H, 1)`, and sum the scores over a complex's contacts via
`inter_batch`. This is still "affinity as a sum over interface contacts," but a single direction and no
learned offset to subtract — leaner, with fewer parameters to overfit the training-era contact
population. The second is a graph channel that EHIGN does not have at all: mean-pool the final ligand
features and the final pocket features over their atoms, concatenate, and regress through `Linear(2H, H)
→ ReLU → Dropout(0.1) → Linear(H, 1)`. The graph channel is the safety net — even if the interface sum
transfers poorly to a held-out complex, a pooled whole-complex representation still carries
size-and-composition signal that correlates with affinity. The prediction is the *average* of the two
channels, `(inter_pred + graph_pred) / 2`. Averaging a sharp-but-brittle interface estimate with a
smooth-but-coarse graph estimate is exactly the variance-reduction move I want on the held-out split:
when the interface channel is reliable the average tracks it, and when the interface channel drifts the
graph channel anchors it.

Two design choices I want to be explicit about because they are the lean-vs-heavy decisions. First, no
bias correction. EHIGN subtracted an attention-normalized offset to kill the size-dependence of the raw
contact sum; I instead let the graph channel — which is itself a pooled, size-aware quantity — and the
interface channel's own `Linear` learn to absorb that offset, and I keep the readout to two `Linear`
layers per channel. The bet is that the attention-bias apparatus was capacity that helped 2016 (familiar
contacts) but did not transfer to 2019, so dropping it should cost little on the core sets and stop
paying the held-out penalty. Second, no `compute_loss` hook and no consistency term. This fill produces
a single `forward` output — the average of the two channels — so the harness's default plain
`F.mse_loss(pred, labels)` is exactly right, and there is no second head to make consistent. The whole
two-head, three-term EHIGN objective collapses to one prediction and one MSE. Leaner training, and I am
betting the consistency regularizer was buying agreement between two views that I am simply not building.

Let me check invariance survives, because it must. Every input is frame-independent: the 35-dim atom
one-hots, the 17-dim covalent edge features, the 11-dim contact features (all built from angles,
triangle areas, and distances, which are rigid-motion invariants). The GIN message adds a projection of
those invariant edge features to invariant node features; the aggregations are sums and means;
the interface readout concatenates invariant atom features with the invariant raw edge feature; the
graph readout pools invariant features. Coordinates never appear. So the prediction is invariant by
construction, no augmentation, exactly as required.

Now the falsifiable expectations against EHIGN's numbers, which is the real test of "leaner generalizes
better." The bet is that dropping the bias-correction and consistency machinery and adding a graph
readout channel will *hold* the core-set quality while *helping* the splits where EHIGN's machinery did
not transfer. Concretely: I expect 2013 to *improve* over EHIGN — the graph channel's whole-complex
signal should sharpen the small CASF-2013 set, so I would expect 2013 RMSE to drop below EHIGN's 1.4117
(toward ~1.32) and Rp to rise above its 0.8066 (toward ~0.83), recovering the 2013 sharpness EHIGN was
flat on. On 2016 the honest expectation is that I give a little back: EHIGN's heavy interface apparatus
genuinely won the familiar core set (1.2426 / 0.8218), and a leaner single-direction interface sum will
likely land slightly worse — I expect 2016 RMSE a touch above EHIGN's (around 1.30) and Rp a little
below 0.8218 — but still well ahead of EGNN and SchNet. The crux is 2019: if the lean hypothesis is
right, the average-of-two-channels readout should be *no worse* than EHIGN there and possibly the place
the graph channel earns its keep; but it is entirely possible that the GIN primitive's coarser edge
handling and the absence of the bias correction leave 2019 essentially unchanged or even slightly worse
in RMSE while Rp holds. The decisive claim is the *overall* one: averaged across the three benchmarks
with both metrics, the lean two-channel GIN model should edge ahead of EHIGN — winning 2013 by enough to
more than offset a small 2016 give-back, with 2019 roughly a wash. If instead 2013 fails to improve and
2016 drops sharply, the lean hypothesis is wrong and EHIGN's machinery was load-bearing after all. I
expect the former: this is the rung where doing *less*, but with a second readout channel as a hedge,
generalizes better than doing more. This is the strongest baseline on the ladder, and the bar it has to
clear is EHIGN's row, benchmark by benchmark. (The full scaffold module is in the answer.)
