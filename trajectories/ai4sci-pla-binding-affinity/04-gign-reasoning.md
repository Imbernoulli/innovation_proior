EHIGN landed where I said it would on the headline, and reading its row carefully tells me the heavy
machinery is buying its gain in a narrow place — which is the opening for a leaner method. Let me
difference the rows before I theorize. The 2016 prediction came in strong: RMSE 1.4213 → 1.2426, a drop
of 0.1787 (about 12.6%), and Rp 0.7646 → 0.8218, by far the best 2016 on the ladder, recovering the
distance resolution EGNN lost and clearing past SchNet's 0.7767. The full 17/11-dim edge geometry in the
messages plus the multiplicative sum-of-contacts readout did exactly the job on the near-training core
set. But the two other benchmarks tell the cautionary half. 2013 was essentially flat versus EGNN: RMSE
1.4114 → 1.4117 (worse by 0.0003) and Rp 0.8093 → 0.8066 (worse by 0.0027) — EHIGN even shed a hair of
2013 quality, which is the calibration caution from two rungs back coming true: 2013's error was
dynamic-range compression, not comprehension, and pouring in richer messages did not sharpen the range.
And the open question I flagged for 2019 resolved against the heavy machinery: 2019 came in at RMSE
1.4604 (worse than EGNN's 1.4422 by 0.0182) with Rp 0.6213 (up a negligible 0.0038 from 0.6175). So the
Rp barely moved and the RMSE actually *regressed* on the temporally distant set.

Now put those three rows through the one lens that has been most diagnostic on this ladder — the gap
between the near-training 2016 RMSE and the held-out 2019 RMSE, which measures how much of a model's
core-set skill survives distribution shift. SchNet's gap was 1.5624 − 1.3702 = 0.192, a wide chasm.
EGNN closed it almost entirely to 1.4422 − 1.4213 = 0.021 by trading a little 2016 for a lot of 2019.
EHIGN blew it back open to 1.4604 − 1.2426 = 0.218 — *wider than SchNet's*, and an order of magnitude
wider than EGNN's. That single comparison is the whole story of EHIGN's row: its dual-head,
attention-bias-corrected, consistency-trained apparatus and its rich covalent-edge features (bond type,
conjugation, ring membership) sharpened the familiar core set dramatically and then failed to carry any
of that sharpening to the held-out split, so the generalization gap did not just persist, it grew. That
is the signature of capacity spent on *training-era chemistry*: the covalent-motif features and the
heavy interface head learn the 2016-era interactions in fine detail and do not transfer to 2019's later
chemistry. So the diagnosis inverts the usual "add more." The question is now whether all that machinery
is *necessary*, or whether a leaner core can hold the win where it matters while not paying — and
ideally reversing — the 2019 penalty.

Let me be careful about what "leaner" means, because the wrong reading is "do less of EHIGN," and that
is not the fix. Trimming heads or dropping the consistency term from EHIGN leaves its fundamental shape
intact: two bespoke convolutions and a readout that bets the entire prediction on a sum of per-contact
scores. If the interface sum is itself the brittle quantity, shrinking EHIGN around it does not help.
So the lean answer is a differently-shaped model, and I derive its shape from the two soft spots the
gap-analysis exposed: the *convolution* may be overfitting by having two specialized mechanisms with
many ways to memorize a particular interaction regime, and the *readout* may be brittle by resting
everything on the interface contact sum. I will address the first with a single uniform message-passing
primitive applied to all three edge sets, and the second with a two-channel readout that does not bet
everything on the interface. Let me take them in turn, because each is a real design decision with a
cheaper alternative I have to reject.

On the convolution, the alternatives are: keep EHIGN's two bespoke convolutions (covalent additive-sum,
non-covalent gated-mean) and just regularize harder; or replace both with one uniform primitive applied
identically to covalent and non-covalent edges. The intuition for the uniform primitive is capacity
control: a single mechanism has strictly fewer parameters and fewer ways to carve out a
regime-specific memorization than two specialized ones, and the whole point of a heterogeneous *layer*
— letting one atom fuse its covalent and its non-covalent neighbourhood into one updated representation
in a single step — does not actually *require* two different convolution forms, only that both edge
sets feed the same atom's update. EHIGN used two forms and read out only through the interface; I want
to test whether one form, fusing both neighbourhoods per step and read out through *both* the interface
and the whole complex, generalizes better. The primitive I reach for is the GIN-style update, because
it is the most discriminative of the simple message-passing forms and it injects the edge cleanly. For
an intra-molecular edge I project the edge feature to the node width and add it to the source, `msg =
x_src + edge_proj(edge_attr)`, sum over neighbours, and update as `mlp((1 + ε) · x + agg)` with a
learnable scalar `ε`. The `(1 + ε) · x` term is the load-bearing piece and worth spelling out: a plain
aggregator `x + Σ_neighbours` cannot always tell a node apart from its own neighbourhood — if the
center and the aggregated neighbours coincide, the update collapses the distinction — whereas weighting
the center by `(1 + ε)` and each neighbour by 1 makes the combined multiset map injectively, which is
what gives GIN discriminative power matching the Weisfeiler-Lehman graph-isomorphism test. The edge
projection is what carries the geometry: the same 17-dim covalent features and 11-dim contact features
EHIGN used, but added into the message through a single learned projection rather than gating it, and
the `mlp` is `Linear → BatchNorm → ReLU → Linear`. I run this same GIN layer on the ligand's covalent
graph and the pocket's covalent graph with separate weights, and a sibling inter-molecular GIN layer on
the ligand→pocket contacts.

The inter layer differs from the intra one only where it must. Source and destination are different
node types, so it projects the contact edge, adds it to the source, and — this is the one place I keep
a distinction from EHIGN's uniformity — *mean*-aggregates over the variable contact degree rather than
summing, for the same reason as before: a pocket atom's contact count is a crowd-size nuisance, and the
mean asks the typical contact strength rather than letting twenty grazing contacts outvote three real
ones by count. It then updates the destination by `mlp([x_dst, agg])` on the concatenation. Three
layers, hidden width 256, each wrapped in a residual `x ← layer(x) + x`, and crucially I apply the two
intra layers to both molecules *and* the inter layer into the pocket every step, so a pocket atom's
representation accumulates both its own covalent context and the ligand contacts pressing on it, fused
in one step. That is the heterogeneous fusion EHIGN prized, rebuilt from one uniform primitive with the
single justified exception of the aggregation on the contact edges.

It is worth placing this message on the expressiveness axis the last three rungs have been walking,
because the placement is itself a bet. EGNN's message was additive-separable in the endpoints
(`mlp_u(src) + mlp_v(dst)`); EHIGN's non-covalent message was multiplicatively *gated* (`h_src ⊙ e`);
this GIN message is additive again (`x_src + edge_proj(e)`), deliberately giving up EHIGN's
message-level multiplicativity. That is not an oversight — it is the hypothesis that the gated,
regime-specific message was part of what overfit 2016, and that a plain additive injection plus depth
(three layers) plus the `(1 + ε)` center-weighting recovers enough discriminative power while carving
fewer regime-specific grooves to memorize into. The sharp *pairwise* interaction I am not building into
the message reappears where it is cheapest and least prone to overfit: in the interface readout's
concatenation of the two atom features with the raw edge, scored by an MLP. And the budget moves the
same direction as the shape. EHIGN carried, per layer, two bespoke convolutions across four relations,
and at readout two full scoring heads and two attention-bias heads each ending in an `FC(H, 200)` — a
large parameter count concentrated in machinery that the gap-analysis says overfit. This model carries
three GIN layers (two intra, one inter) and two small readout MLPs, with no bias-correction heads, no
dual scoring, and no second directional pass. It is materially smaller, and against a training set on
the order of 10⁴ complexes, fewer parameters is itself a generalization lever — less to overfit —
independent of the architectural reshaping. The lean move is thus lean in two senses at once, fewer
mechanisms and fewer weights, and both cut in the direction the held-out split rewards.

Now the readout, which is where this rung most deliberately departs from EHIGN, and where I earn the
generalization back. EHIGN bet the whole prediction on a sum of per-contact scores minus an attention
bias, in two directions, with a consistency loss tying them — a strong, interpretable bet, and the
gap-analysis says it is also a *brittle* one. Here is the mechanism of the brittleness: the interface
sum is exactly the quantity most sensitive to *which contacts fall inside the 5 Å cutoff*, and on
held-out chemistry the population of contacts in that shell shifts — different residue types line the
pocket, different rotamers place different atoms just inside or just outside 5 Å — so a readout that is
a raw sum over that shifting population inherits the shift. Make it concrete: imagine a held-out pocket
lined with flexible lysine side chains whose terminal amines sit right around 5 Å from the ligand. A
small conformational difference between two near-identical structures flips several of those contacts
in or out of the shell, so the interface sum over `k` contacts swings by several per-contact scores
between two poses that are, physically, almost the same complex — a high-variance readout on exactly
the structures where the cutoff is crowded. Now ask what the graph channel does on those same two
structures: it mean-pools over hundreds of pocket atoms, and flipping three or four boundary contacts
in or out changes which atoms *contact* the ligand but not the pooled composition of the pocket, so the
graph prediction barely moves. So on precisely the complexes where the interface channel is unstable,
the graph channel is stable — the two channels fail on different structures, which is the
imperfect-error-correlation precondition the variance-reduction argument needs, made concrete rather
than assumed. I hedge with two readout channels and average them. The first is the interface channel, kept in spirit but stripped of the bias correction
and the dual-head consistency machinery: for each ligand→pocket contact, score it from the
concatenation of the final ligand-atom feature, the final pocket-atom feature, and the *raw* contact
edge feature, through `Linear(2H + inter_edge_dim, H) → ReLU → Linear(H, 1)`, and sum the scores over a
complex's contacts via `inter_batch`. This is still "affinity as a sum over interface contacts," but a
single direction and no learned offset to subtract — leaner, with fewer parameters to overfit the
training-era contact population. The second is a channel EHIGN does not have at all: a graph channel
that mean-pools the final ligand features and the final pocket features over their atoms, concatenates,
and regresses through `Linear(2H, H) → ReLU → Dropout(0.1) → Linear(H, 1)`. The graph channel is the
safety net — even if the interface sum transfers poorly to a held-out complex, a pooled whole-complex
representation still carries size-and-composition signal that correlates with affinity. The prediction
is the *average* of the two channels, `(inter_pred + graph_pred) / 2`.

Let me make the variance-reduction argument quantitative, because "averaging helps" is the kind of
claim that deserves a number. Treat the two channels as two estimators `A` (interface) and `B` (graph)
of the same target with per-complex error variances `σ_A²`, `σ_B²` and error correlation `ρ`. The
average has variance `Var((A + B)/2) = (σ_A² + σ_B² + 2ρ σ_A σ_B)/4`. Take the symmetric case `σ_A =
σ_B = σ` for intuition: the average's variance is `σ²(1 + ρ)/2`, which is below `σ²` for any `ρ < 1`.
If the two channels made *independent* errors (`ρ = 0`) the variance would halve, cutting the error
standard deviation by about 29%; even at `ρ = 0.5` the variance falls to `0.75σ²`, a ~13% reduction in
error standard deviation. The interface channel (sharp but brittle, sensitive to the cutoff population)
and the graph channel (smooth but coarse, driven by pooled composition) fail on *different* complexes —
their errors are far from perfectly correlated — so the average is a genuine variance-reduction move,
and it is a move that pays exactly on the held-out split, where the interface channel's variance is
highest. When the interface channel is reliable the average tracks it; when it drifts, the graph channel
anchors it. That is the whole bet: two imperfectly-correlated estimators beat either alone precisely
where one of them is unstable.

The cleaner-looking way to get that variance reduction would be a *true* ensemble: train an
interface-heavy model and a pooled model separately and average their predictions. I reject it on cost
and honesty. Cost, because two independent models double the training budget the frozen harness would
have to run. Honesty, because I do not actually need two independent networks — I need two estimators
whose errors are imperfectly correlated, and a two-channel readout on a *shared* GIN encoder gives me
that at the price of one forward pass, branching only at the head. There is a real caveat I should own,
though: because the two channels share the encoder, their errors will be *more* correlated than two
independently-trained models' would be — the `ρ` in the variance formula is larger than it would be for
a true ensemble — so the hedge buys less than the `ρ = 0` bound promises. It is a cheap internal
ensemble, not a free lunch; the shared backbone caps the upside. I take that trade because the point is
not to squeeze out the maximal variance reduction but to stop the interface sum from being a single
point of failure on the held-out split, and even a partially-correlated second channel does that.

I owe one honest reckoning with my own earlier reasoning, because the graph channel is uncomfortably
close to something I called a shortcut at the very start. At the floor I distrusted a model that reads
affinity off the *marginal* shapes of the two molecules — how big the ligand, how many rings — because
those pooled statistics correlate with affinity on the training distribution for reasons that are not
binding and will not survive a held-out split. The graph channel is exactly a pooled, whole-complex,
marginal-shape signal. So am I smuggling the shortcut back in? The distinction I hold onto is that at
the floor the pooled marginals were the *entire* predictor, with no interface term at all, whereas here
the graph channel is a *hedge* averaged against a genuine interface channel that does read the contacts.
Its job is variance reduction when the interface channel is unstable, not to be the primary predictor.
But the honesty cuts the other way too: if training leans on the graph channel because it is the
easier-to-fit signal, the average drifts toward the shortcut and the model's held-out behaviour would
regress toward marginal-shape guessing. I cannot rule that out from the design alone — it is a real
risk the averaged readout carries — and the way I will know is the 2019 number: if the graph channel is
anchoring rather than shortcutting, 2019 should hold or improve; if it is shortcutting, 2019 Rp should
sag even as the core sets look fine. That is a sharper falsifier than "overall it edges ahead," and I
will read the held-out split with it in mind.

Two design choices deserve to be stated explicitly because they are the lean-versus-heavy decisions.
First, no bias correction. EHIGN subtracted an attention-normalized offset to counter the size
dependence of the raw contact sum; I instead let the graph channel — which is itself a pooled,
size-aware quantity — and the interface channel's own `Linear` learn to absorb that offset, keeping the
readout to two `Linear` layers per channel. The bet is that the attention-bias apparatus was capacity
that helped 2016 (familiar contacts) but did not transfer to 2019, so dropping it should cost little on
the core sets and stop paying the held-out penalty. Second, no `compute_loss` hook and no consistency
term. This fill produces a single `forward` output — the average of the two channels — so the harness's
default plain `F.mse_loss(pred, labels)` is exactly right, and there is no second head to make
consistent. The whole two-head, three-term EHIGN objective collapses to one prediction and one MSE. It
is worth noting what I am giving up: EHIGN's consistency term was a free variance-reducing regularizer
tying two views of the interface. I am *replacing* that variance reduction with a different one — the
average of an interface and a graph channel — rather than simply discarding it, so the leaner training is
not naked; the hedge lives in the readout instead of the loss.

Invariance has to survive, and it does. Every input is frame-independent: the 35-dim atom one-hots, the
17-dim covalent edge features, the 11-dim contact features, all built from angles, triangle areas, and
distances, which are rigid-motion invariants. The GIN message adds a projection of those invariant edge
features to invariant node features; the aggregations are sums and means; the interface readout
concatenates invariant atom features with the invariant raw edge feature; the graph readout pools
invariant features. Coordinates never appear anywhere, so the prediction is invariant by construction,
no augmentation, exactly as required.

Now the falsifiable expectations against EHIGN's row, which is the real test of "leaner generalizes
better." The bet is that dropping the bias-correction and consistency machinery and adding a graph
channel will *hold* the core-set quality while *helping* the splits where EHIGN's machinery did not
transfer. Concretely: I expect 2013 to *improve* over EHIGN — the graph channel's whole-complex signal
should sharpen the small CASF-2013 set that EHIGN was flat on, so I would expect 2013 RMSE to drop below
EHIGN's 1.4117 and Rp to rise above 0.8066, finally recovering the 2013
range that pouring in richer messages failed to fix. On 2016 the honest expectation is that I give a
little back: EHIGN's heavy interface apparatus genuinely won the familiar core set (1.2426 / 0.8218),
and a leaner single-direction interface sum will likely land slightly worse — I expect 2016 RMSE a touch
above EHIGN's and Rp a little below 0.8218 — but still well ahead of EGNN and SchNet. The
crux is 2019: if the lean hypothesis is right, the average-of-two-channels readout should be *no worse*
than EHIGN there and possibly the place the graph channel earns its keep, narrowing the generalization
gap EHIGN blew open; but it is entirely possible that the uniform GIN primitive's coarser edge handling
and the absence of the bias correction leave 2019 essentially unchanged or even a hair worse in RMSE
while Rp holds. The decisive claim is the *overall* one: averaged across the three benchmarks with both
metrics, the lean two-channel GIN model should edge ahead of EHIGN — winning 2013 by enough to more than
offset a small 2016 give-back, with 2019 roughly a wash. If instead 2013 fails to improve and 2016 drops
sharply, the lean hypothesis is wrong and EHIGN's machinery was load-bearing after all. I expect the
former: this is the rung where doing *less* on the convolution and the loss, but hedging the readout with
a second whole-complex channel, generalizes better than doing more. This has to be the strongest baseline
on the ladder, and the bar it must clear is EHIGN's row, benchmark by benchmark. (The full scaffold
module is in the answer.)
