EHIGN landed where I said it would on the headline, and reading its row carefully shows the heavy
machinery is buying its gain in a narrow place — which is the opening for a leaner method. The 2016
prediction came in strong: RMSE 1.4213 → 1.2426 (down 0.1787), Rp 0.7646 → 0.8218, by far the best 2016
on the ladder — the full 17/11-dim geometry plus the multiplicative sum-of-contacts readout did exactly
the job on the near-training core set. But the other two benchmarks tell the cautionary half. 2013 was
essentially flat versus EGNN (RMSE 1.4114 → 1.4117, Rp 0.8093 → 0.8066), the calibration caution coming
true: 2013's error was dynamic-range compression, and richer messages did not sharpen the range. And the
open 2019 question resolved against the heavy machinery: RMSE 1.4422 → 1.4604 (worse) with Rp barely
moving, 0.6175 → 0.6213.

Put those rows through the one lens that has been most diagnostic here — the gap between the near-training
2016 RMSE and the held-out 2019 RMSE, which measures how much core-set skill survives distribution shift.
SchNet's gap was 0.192, a wide chasm; EGNN closed it almost entirely to 0.021 by trading a little 2016
for a lot of 2019; EHIGN blew it back open to 1.4604 − 1.2426 = 0.218, *wider than SchNet's* and an order
of magnitude wider than EGNN's. That single comparison is the whole story of EHIGN's row: its dual-head,
attention-bias-corrected, consistency-trained apparatus and its rich covalent-edge features sharpened the
familiar core set dramatically and carried none of that sharpening to the held-out split. That is the
signature of capacity spent on *training-era chemistry*. So the question inverts the usual "add more": is
all that machinery *necessary*, or can a leaner core hold the win where it matters while not paying —
ideally reversing — the 2019 penalty?

"Leaner" cannot mean "do less of EHIGN": trimming heads or dropping the consistency term leaves its shape
intact — two bespoke convolutions and a readout that bets the whole prediction on a sum of per-contact
scores. If the interface sum is itself the brittle quantity, shrinking EHIGN around it does not help. So
the lean answer is a differently-shaped model, derived from the two soft spots the gap-analysis exposes:
the *convolution* may overfit by having two specialized mechanisms with many ways to memorize a regime,
and the *readout* may be brittle by resting everything on the interface contact sum. I address the first
with a single uniform message-passing primitive applied to all three edge sets, and the second with a
two-channel readout that does not bet everything on the interface.

On the convolution the choice is between EHIGN's two bespoke forms plus harder regularization, or one
uniform primitive applied identically to covalent and non-covalent edges. A single mechanism has strictly
fewer parameters and fewer ways to carve a regime-specific memorization, and the heterogeneous *layer* —
letting one atom fuse its covalent and non-covalent neighbourhoods into one update per step — does not
actually *require* two convolution forms, only that both edge sets feed the same atom's update. The
primitive I reach for is the GIN-style update, the most discriminative of the simple forms, with a clean
edge injection. For an intra-molecular edge, `msg = x_src + edge_proj(edge_attr)`, sum over neighbours,
update `mlp((1 + ε) · x + agg)` with a learnable scalar `ε`. The `(1 + ε) · x` term is load-bearing: a
plain `x + Σ_neighbours` cannot always tell a node from its own neighbourhood — if center and aggregate
coincide the update collapses the distinction — whereas weighting the center by `(1 + ε)` and each
neighbour by 1 makes the combined multiset map injective, the property that gives GIN discriminative power
matching the Weisfeiler-Lehman test. The edge projection carries the geometry — the same 17-dim covalent
and 11-dim contact features EHIGN used, added into the message rather than gating it — and the `mlp` is
`Linear → BatchNorm → ReLU → Linear`. Separate weights on the ligand's and pocket's covalent graphs, and
a sibling inter-molecular layer on the ligand→pocket contacts.

The inter layer differs only where it must. Source and destination are different node types, so it
projects the contact edge, adds it to the source, and — the one distinction I keep from EHIGN's
uniformity — *mean*-aggregates over the variable contact degree rather than summing, for the reason
established before: a pocket atom's contact count is a crowd-size nuisance, and the mean asks typical
strength rather than letting twenty grazing contacts outvote three real ones. It then updates the
destination by `mlp([x_dst, agg])`. Three layers, width 256, each wrapped in a residual `x ← layer(x) +
x`, applying the two intra layers to both molecules and the inter layer into the pocket every step, so a
pocket atom accumulates both its covalent context and the ligand contacts pressing on it, fused in one
step. The heterogeneous fusion EHIGN prized, rebuilt from one uniform primitive with a single justified
exception on the contact aggregation.

Place this message on the expressiveness axis the last three models have walked, because the placement is
a bet. EGNN's message was additive-separable in the endpoints; EHIGN's non-covalent message was
multiplicatively *gated*; this GIN message is additive again (`x_src + edge_proj(e)`), deliberately
giving up EHIGN's message-level multiplicativity. That is the hypothesis that the gated, regime-specific
message was part of what overfit 2016, and that a plain additive injection plus depth plus the `(1 + ε)`
center-weighting recovers enough discriminative power while carving fewer grooves to memorize into. The
sharp *pairwise* interaction reappears where it is cheapest and least prone to overfit: the interface
readout's concatenation of the two atom features with the raw edge, scored by an MLP. And the budget moves
the same way — three GIN layers and two small readout MLPs, no bias-correction heads, no dual scoring, no
second directional pass — materially smaller than EHIGN, and against ~10⁴ complexes fewer parameters is
itself a generalization lever. Lean in two senses at once, fewer mechanisms and fewer weights, both
cutting the way the held-out split rewards.

Now the readout, where this model most deliberately departs from EHIGN and where I earn the generalization
back. EHIGN bet the whole prediction on a sum of per-contact scores minus an attention bias, in two
directions with a consistency loss — strong and interpretable, but the gap-analysis says brittle, and here
is the mechanism. The interface sum is exactly the quantity most sensitive to *which contacts fall inside
the 5 Å cutoff*, and on held-out chemistry the population in that shell shifts — different residue types
line the pocket, different rotamers place atoms just inside or outside 5 Å — so a raw sum over that
shifting population inherits the shift. Concretely: a held-out pocket lined with flexible lysine side
chains whose terminal amines sit right around 5 Å from the ligand; a small conformational difference
between two near-identical structures flips several contacts in or out of the shell, so the sum over `k`
contacts swings by several per-contact scores between two poses that are physically almost the same
complex. Now ask what a whole-complex graph channel does on those same two structures: it mean-pools over
hundreds of pocket atoms, and flipping three or four boundary contacts changes which atoms *contact* the
ligand but not the pooled composition of the pocket, so its prediction barely moves. On precisely the
complexes where the interface channel is unstable, the graph channel is stable — the two fail on different
structures, the imperfect-error-correlation precondition the variance-reduction argument needs, made
concrete rather than assumed.

So I hedge with two readout channels and average them. The first is the interface channel, kept in spirit
but stripped of the bias correction and dual-head consistency: for each ligand→pocket contact, score it
from the concatenation of the final ligand-atom feature, the final pocket-atom feature, and the *raw*
contact edge feature through `Linear(2H + inter_edge_dim, H) → ReLU → Linear(H, 1)`, summed over a
complex's contacts. Still "affinity as a sum over interface contacts," but a single direction and no
learned offset — leaner, fewer parameters to overfit the training-era contact population. The second is a
channel EHIGN does not have: a graph channel that mean-pools the final ligand and pocket features,
concatenates, and regresses through `Linear(2H, H) → ReLU → Dropout → Linear(H, 1)`. The prediction is
the *average* of the two, `(inter_pred + graph_pred) / 2`.

And the averaging earns a number. Treat the two channels as estimators `A`, `B` of
the same target with error variances `σ_A²`, `σ_B²` and correlation `ρ`; the average has variance
`(σ_A² + σ_B² + 2ρ σ_A σ_B)/4`. In the symmetric case `σ_A = σ_B = σ` that is `σ²(1 + ρ)/2`, below `σ²`
for any `ρ < 1`: independent errors (`ρ = 0`) halve the variance, cutting error std by ~29%, and even
`ρ = 0.5` gives `0.75σ²`, ~13% off. Because the two channels fail on *different* complexes — I just
argued the graph channel is stable exactly where the interface sum is not — this is genuine variance
reduction, paying most on the held-out split where the interface channel's variance is highest.

The cleaner-looking route to that reduction — a *true* ensemble of two separately-trained models — I
reject on cost and honesty. Cost, because two models double the frozen harness's training budget.
Honesty, because I do not need two independent networks, only two estimators with imperfectly-correlated
errors, and a two-channel readout on a *shared* GIN encoder gives that for one forward pass, branching
only at the head. The caveat I own: because the channels share the encoder their errors are *more*
correlated than two independent models' would be, so `ρ` is larger and the hedge buys less than the
`ρ = 0` bound promises — a cheap internal ensemble, not a free lunch. I take it because the point is not
maximal variance reduction but stopping the interface sum from being a single point of failure on the
held-out split, which even a partially-correlated second channel does.

The graph channel is uncomfortably close to something I called a
shortcut at the floor: reading affinity off the *marginal* shapes of the two molecules, pooled statistics
that correlate with affinity for reasons that are not binding. Am I smuggling the shortcut back in? The
distinction is that at the floor the pooled marginals were the *entire* predictor with no interface term,
whereas here the graph channel is a *hedge* averaged against a genuine interface channel that reads the
contacts — its job is variance reduction when the interface channel is unstable, not to be the primary
predictor. But the honesty cuts both ways: if training leans on the graph channel because it is easier to
fit, the average drifts toward the shortcut and held-out behaviour regresses toward marginal-shape
guessing. I cannot rule that out from the design, and the way I will know is the 2019 number: if the graph
channel is anchoring, 2019 holds or improves; if it is shortcutting, 2019 Rp sags even as the core sets
look fine. A sharper falsifier than "overall it edges ahead."

Two lean-versus-heavy decisions. No bias correction: EHIGN subtracted an
attention-normalized offset to counter the size dependence of the raw contact sum; I let the graph channel
— itself a pooled, size-aware quantity — and the interface channel's own `Linear` absorb that offset. The
bet is that the attention-bias apparatus was capacity that helped 2016's familiar contacts but did not
transfer, so dropping it costs little on the core sets and stops paying the held-out penalty. And no
`compute_loss` hook: this fill produces a single `forward` output, the average of two channels, so the
harness's default plain `F.mse_loss` is exactly right, with no second head to make consistent. I am not
discarding EHIGN's consistency regularization so much as *replacing* it — the variance reduction moves
from the loss into the readout. Invariance survives: every input is frame-independent, the GIN message
adds a projection of invariant edge features to invariant node features, aggregations are sums and means,
both readouts operate on invariant features, and no coordinate ever appears.

Now the falsifiable expectations against EHIGN's row. The bet is that dropping the bias-correction and
consistency machinery and adding a graph channel *holds* the core-set quality while *helping* the splits
EHIGN's machinery did not transfer to. I expect 2013 to *improve* over EHIGN — the graph channel's
whole-complex signal sharpening the small CASF-2013 set EHIGN was flat on, so RMSE below 1.4117 and Rp
above 0.8066, finally recovering the 2013 range. On 2016 the honest expectation is I give a little back:
EHIGN's heavy interface apparatus genuinely won the familiar core set (1.2426 / 0.8218), and a leaner
single-direction sum lands slightly worse — RMSE a touch above EHIGN's, Rp a little below 0.8218 — but
still well ahead of EGNN and SchNet. The crux is 2019: if the lean hypothesis is right the averaged
readout should be *no worse* than EHIGN there and possibly the place the graph channel earns its keep,
narrowing the gap EHIGN blew open; but the coarser uniform primitive and the absent bias correction could
leave 2019 essentially unchanged or a hair worse in RMSE while Rp holds. The decisive claim is the
*overall* one: averaged across three benchmarks and both metrics, the lean two-channel GIN model should
edge ahead of EHIGN — winning 2013 by enough to more than offset a small 2016 give-back, with 2019 roughly
a wash. If instead 2013 fails to improve and 2016 drops sharply, the lean hypothesis is wrong and EHIGN's
machinery was load-bearing after all. I expect the former: doing *less* on the convolution and the loss,
but hedging the readout with a whole-complex channel, generalizes better than doing more. (The full
scaffold module is in the answer.)
