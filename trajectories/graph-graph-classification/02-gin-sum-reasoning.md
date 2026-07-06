The stripped DiffPool came back and its numbers say exactly what I suspected when I built it: the
learned soft pool, with no link or entropy auxiliary to lean on, behaves like a noisy mean and pools
*less* informatively than it should. Before I decide what to do next I want to read the three tables
arithmetically rather than by eye, because the seed spreads and the gap between the two metrics are where
the mechanism actually shows itself, and I would rather derive the next move from the numbers than from a
hunch. Start with PROTEINS, the three seeds {70.98, 71.44, 70.52}: the mean is 70.98, the full range is
71.44 − 70.52 = 0.92 points, and the population standard deviation works out to about 0.38 (the deviations
are 0, +0.46, −0.46, so the variance is (0 + 0.21 + 0.21)/3 ≈ 0.14 and its root ≈ 0.38). That band is
*tiny*. It is the fingerprint of a readout that is not learning a seed-distinguishing partition but
converging to the same diffuse, mean-like answer every single run — which is precisely the diffuse-softmax
degeneration I proved the stripped pool collapses to when nothing holds the assignment crisp. And there is
a second thing about PROTEINS that sharpens the reading: it is the *lowest* of the three datasets for this
readout. PROTEINS has the larger graphs, the ones where a genuine functional-group hierarchy would help
most, and the stripped pool extracts the *least* from exactly the graphs that should reward hierarchy the
most. That is the diagnosis confirming itself: a pool reaching for structure that gets nothing out of the
most-structured set is a pool that never found any structure.

Now MUTAG, the opposite fingerprint: {85.64, 78.71, 79.77}, mean 81.37, range 85.64 − 78.71 = 6.93 points,
population standard deviation about 3.05. That is nearly an *eight-fold* wider spread than PROTEINS (3.05
against 0.38, a ratio of 8.0), and it is not a method with a stable answer — it is the coin-flip I
predicted on a 188-graph set where K = 25 clusters exceed the per-graph node count and the assignment has
almost no data to fit. The column softmax over ~18 real nodes has to fill 25 cluster slots, several of
them near-redundant averages of the same handful of atoms; the assignment is forced diffuse by sheer
arithmetic before training ever gets a vote, and on so few graphs the little signal that remains is at the
mercy of the fold split. NCI1 sits in between at 78.25 with a range of 79.25 − 77.52 = 1.73 — respectable,
unremarkable, and again fairly tight, which fits a large balanced set where the mean-like pool at least
converges to the same place each run.

There is a second, independent reading hiding in the gap between test_acc and macro_f1, and it agrees with
the seed reading, which is the kind of corroboration I trust. On NCI1 the accuracy is 78.25 and the
macro_f1 is 78.21 — a gap of 0.04, essentially zero, telling me NCI1's two classes are balanced enough
that accuracy and macro-F1 measure the same thing. On PROTEINS the gap is 70.98 − 68.87 = 2.11, moderate.
On MUTAG it is 81.37 − 76.83 = 4.54, the largest of the three. So the acc-minus-F1 gap orders the datasets
MUTAG (4.54) > PROTEINS (2.11) > NCI1 (0.04), and that order is exactly the order of how count-blind a
majority-leaning mean would look on each: a mean that discards multiplicities buys accuracy on the frequent
class while the minority class's F1 sags, and the effect is loudest on the most imbalanced, smallest set.
The macro_f1 seed spread on MUTAG is itself telling — {83.25, 72.97, 74.26}, a range of 10.28, wider even
than the accuracy spread — which says the minority-class performance is what is swinging most from seed to
seed. So the two-metric picture and the seed picture point the same way: on the balanced, larger set the
pool is stable but merely mean-like; on the small imbalanced set it is a high-variance majority-leaner.

The diagnosis is clean and it is not a tuning problem. The one idea I kept from DiffPool — a learned soft
clustering pooled as Sᵀ X — needed the auxiliaries the harness would not let me wire in, and without them
it discards counts (it is a convex-combination pool, a weighted mean, as I proved) and, worse, it makes no
use whatsoever of the per-layer node embeddings the scaffold hands me in `layer_outputs`. I threw away two
distinct things at once: the *injective* reduction that keeps multiplicities, and the *multi-scale* signal
sitting unused in the layer stack. Any correction that recovers only one of the two leaves points on the
table.

So let me lay out what is actually on the menu now, because I want to choose the next readout by
elimination rather than by reflex. I have four candidates that all fit the post-backbone slot. The first
is a plain global *sum* over the last layer only — the cheapest possible fix, it recovers the counts a
mean discards but reads a single depth, so it addresses injectivity and does nothing about scale. The
second is to *sum every layer and concatenate* — recovers counts and reads every depth, addressing both
failures at once. The third is a *learned attention-weighted sum* over nodes, a gated soft attention that
re-weights each node before summing — it keeps every node but re-introduces a learned softmax over nodes,
which is exactly the kind of fragile learned pooling I just watched go diffuse in DiffPool; with no
structural prior the attention weights have the same freedom to flatten, and it still reads a single
depth, so it re-litigates the risk I just lost to without addressing the multi-scale miss. The fourth is a
set-aggregation network in the Set2Set family — non-destructive and expressive, but it is a single learned
recurrent aggregation over one layer, sequential and parameter-heavy (it is the very readout the parameter
budget is sized against), and again single-scale. Three of the four leave the multi-scale signal on the
floor, and two of the four spend fresh learned capacity on precisely the kind of soft pooling that just
failed. The one candidate that repairs *both* diagnosed failures while spending *no* fragile learned
capacity on the pooling itself is the second: sum every layer, concatenate. I will still have to earn that
choice on the aggregator side, but the elimination already points there.

Go back to first principles on the *flat* reduction, because that is what I actually have a clean theory
for. The question is narrow: among permutation-invariant reductions over the multiset of node embeddings,
which one keeps the most information? DiffPool's failure mode was that it became a mean, and a mean has a
precise defect I can exhibit on a two-element example. Take the multiset {a, b} and its inflated copy
{a, a, b, b}: the mean of the first is (a+b)/2 and the mean of the second is (2a+2b)/4 = (a+b)/2 —
*identical*. The sum, by contrast, is a+b versus 2a+2b, differing by exactly the factor 2 the inflation
introduced. So a mean captures only the *distribution* — the proportions of neighbor features — and is
blind to proportional inflation of counts; a sum sees the counts. Max is worse still: max over {a, b} and
max over {a, a, b, b} are the same element, and in fact max over any duplication of a set is unchanged, so
max depends only on which *distinct* elements are present — it sees the *support* and loses both counts and
proportions. The sum, meanwhile, is *injective* on bounded multisets, and I can watch why with a concrete
positional trick: map each integer-valued node feature x to 10^{−x} and sum. The multiset {1, 1, 2} gives
10⁻¹ + 10⁻¹ + 10⁻² = 0.21, whose digits read off "two ones, one two"; the multiset {1, 2, 2} gives
10⁻¹ + 10⁻² + 10⁻² = 0.12, "one one, two twos." Distinct multisets, distinct sums — the sum has
positionally encoded the exact multiplicity profile in base ten. And to see that it distinguishes what a
max cannot even when the *support* is identical: {1, 2} sums to 10⁻¹ + 10⁻² = 0.11 while {1, 1, 2} sums to
0.21 — same support, different counts, different sums, whereas max gives 2 for both. So among {sum, mean,
max} there is a strict pecking order in discriminative power — sum ⊐ mean ⊐ max — and DiffPool, by drifting
to a mean, sat exactly one rung below the top.

This is not an abstract preference here; it is the whole reason the backbone in front of me exists. The
GIN backbone was built precisely so that an injective sum readout makes the entire network as
discriminating as the Weisfeiler–Lehman test — the theoretical ceiling for any message-passing GNN. Each
GIN layer applies an MLP to (1+ε)·h_v + Σ_{u∈N(v)} h_u, an injective update on the multiset of a node's
neighbor features, so message passing preserves the WL distinguishing power layer by layer; but that
preservation is only cashed out at the readout if the readout is *also* injective on the multiset of node
features. Use a mean or a max at the readout and you throw away, in the last step, the very
distinguishing power the backbone spent five injective layers building. So using anything weaker than a sum
at the readout *wastes the backbone's expressivity* — a mean readout would hand a WL-capable backbone a
sub-WL answer. That settles the aggregator: sum, not mean, not max. It also disqualifies, cleanly, the
mean-side of what DiffPool became.

But a single sum over the final layer is still flat in the *other* sense DiffPool was reaching for, and
here I can finally do something about it without needing a clustering at all. The scaffold hands me
`layer_outputs`, a list of the per-layer node embeddings — one `[N_total, hidden_dim]` tensor for each of
the five GIN layers — and the DiffPool rung used *none* of it. Why should that list matter? Because a
node's embedding after k rounds of message passing is a learned summary of its rooted subtree of height k:
the height-1 embedding sees immediate neighbors, the height-5 embedding sees five hops out. As depth grows,
node representations get more global and more discriminative — so I want enough depth for power. But the
deepest features are also the most *specialized*, and worse, GINs over-smooth: after several rounds every
node's embedding drifts toward a common value as neighborhoods overlap, and on small datasets like MUTAG
the shallower, more local features often generalize better than a washed-out last layer. Reading only the
last layer — the `x` argument — is a *gamble* on one depth being the right one for all three datasets at
once, and there is no reason a single depth is right for tiny MUTAG molecules and hundred-node NCI1
compounds simultaneously. Rather than gamble, use *all* of them: let the final graph representation reach
back to every depth. The multi-scale signal DiffPool wanted from a learned hierarchy is already present,
for free, in the layer stack that the backbone hands me — I do not have to *learn* a hierarchy, I only
have to *read* the one that message passing already computed.

So the readout writes itself: sum-pool each layer's node embeddings *independently* into a per-layer graph
vector, then *concatenate* across layers, h_G = CONCAT( Σ_{v∈G} h_v^{(k)} : k = 1..K ). Each per-layer sum
is injective on its own multiset of node features; concatenation keeps all K of them side by side rather
than mixing them, so the downstream classifier — not a fragile assignment softmax — can weight the depths
however it likes. And there is a clean reading of *what* this computes: a node's height-k embedding is a
learned encoding of a height-k rooted subtree, so summing them is the learnable analogue of *counting
subtrees* — exactly what the WL subtree kernel does by hand, except the subtrees are embedded in a
continuous space, so *similar* subtrees land near each other, something one-hot WL labels can never
express. The readout therefore generalizes both WL and the WL subtree kernel rather than merely matching
them. The output width is hidden_dim × num_layers = 5 × 64 = 320 here, with no projection bottleneck, so
every depth's full signal reaches the classifier undiluted.

Before I commit to that 320 width I should check the budget, since a wide concat is where this kind of
readout usually pays a tax and I would rather find out now than have the harness reject the edit. The
fixed classifier's first layer is `Linear(output_dim, 64)`. At DiffPool's output_dim = 64 that layer had
64·64 + 64 = 4,160 parameters; at 320 it becomes 320·64 + 64 = 20,544, an increase of exactly 16,384. The
per-layer batch-norms I am about to add are 5 × BatchNorm1d(64) = 5 × (2·64) = 640. So the whole readout
costs roughly 16,384 + 640 ≈ 17,000 extra parameters over the plain-sum baseline, comfortably inside the
10·H² + 9·H = 41,536 headroom the budget allows — I am spending about forty percent of the allowance. The
width is free; I do not need a projection bottleneck, and adding one would only throw away the very
side-by-side separation of depths I am building the concat to preserve. Good — no compromise forced there,
and unlike the previous rung I am now actually *using* the budget rather than under-spending it.

There is one wrinkle I do have to handle, and it is the single place this task's readout departs from the
bare textbook JK-sum, so let me reason it out rather than copy it. Concatenating per-layer *sum* pools
means concatenating vectors at very *different scales*. The dominant source of scale is graph size: a sum
over a graph's nodes grows linearly with the node count, so a sum over a ~100-node NCI1 molecule is on the
order of six times the magnitude of a sum over a ~17-node MUTAG molecule, purely from N, before any
content enters — and PROTEINS, with graphs ranging into the hundreds of nodes, spreads that size factor
wider still *within a single batch*. On top of the size effect, the deeper GIN layers, after several rounds
of neighbor summation plus the (1+ε) self-weighting inside `GINConv` and the per-layer BatchNorm→ReLU, can
carry systematically different magnitudes than the shallow ones. So the concatenated vector mixes wildly
different magnitudes across both layers and graph sizes at once. Feed that straight into the classifier and
the large-magnitude coordinates dominate the first linear layer's gradient, the small-magnitude coordinates
are effectively ignored, and some folds simply fail to converge — the optimization stalls on the scale
mismatch, not on anything about the graphs themselves. I want to be careful here that the fix does not
undo the injectivity I just fought for. The fix that keeps injectivity intact is to
**batch-normalize each layer's graph-level embedding before concatenating**: a `BatchNorm1d(hidden_dim)`
per layer, applied to the `[B, hidden_dim]` pooled vectors. BatchNorm subtracts a per-channel mean and
divides by a per-channel standard deviation computed across the batch of graphs, which is exactly what
removes the O(N) size scaling and the per-depth magnitude drift, putting every layer's graph embedding on a
common, well-conditioned scale. Crucially it is an affine, invertible (at fixed statistics) rescaling of
each channel — a strictly monotone affine map — so it does *not* collapse the multiset distinctions the sum
encoded: a monotone affine map of an injective code is still injective. It only conditions the input so the
classifier sees all five depths on equal footing and every fold trains. This per-layer graph-BN is the one
piece I add beyond "sum each layer and concatenate," and it is there for optimization stability, not for
expressivity — I should not credit it with any of the accuracy this readout earns.

Let me quickly check the shapes end to end so I know the module is well-typed before I trust it. Each
`layer_outputs[k]` is `[N_total, 64]`; `global_add_pool(·, batch)` reduces it to `[B, 64]`; the k-th
`BatchNorm1d(64)` maps `[B, 64] → [B, 64]`; stacking the five and concatenating on the last axis gives
`[B, 320]`; and setting `self.output_dim = hidden_dim * num_layers = 320` is exactly what tells the fixed
classifier head to expect a 320-wide input. The permutation invariance I need is inherited for free: each
per-layer sum is invariant to node relabeling, BatchNorm acts per graph-level channel and cannot see node
order, and concatenation of invariants is invariant. Note I read from `layer_outputs` and ignore `x`
entirely — `x` is just `layer_outputs[-1]`, already included — and I touch neither `edge_index` nor any
dense batching nor any clusters. The whole readout is a per-layer sum, a per-layer BN, and a concat.

So the delta from the DiffPool rung is precise and points in the opposite direction from what I tried
first. DiffPool spent its capacity trying to *learn* structure — a soft clustering — with none of the
support that makes that pay, and ended up mean-like, discarding counts and ignoring depth. This rung spends
*no* learned capacity on the pooling itself: it takes the provably-injective sum, applies it at every
depth, and lets the downstream classifier decide how to weigh the scales. The only learned parameters in
the readout are the per-layer BNs, and they exist for conditioning, not clustering.

Reading DiffPool's shape, here is what I expect and where it is falsifiable against its numbers, using only
metrics the harness reports. PROTEINS is the cleanest test: DiffPool's diffuse mean-like pool flattened to
70.98 with that near-zero 0.92-point seed spread, so if the diagnosis is right — that a mean discards
counts and a single depth misses scale — then summing over every layer should *lift* PROTEINS clearly above
71, and because the readout is now deterministic rather than a learned-and-stuck softmax, I expect it to
stay tight but at a higher level. NCI1, the largest set where the injective-sum-feeds-WL-power argument
bites hardest, should also rise above DiffPool's 78.25 — this is the dataset where keeping counts and
reading every depth should matter most, and where the near-zero acc-minus-F1 gap means any gain shows up
cleanly on both metrics at once, with no majority-leaning wedge between them. MUTAG is the one I am least
sure about: the deterministic readout removes the assignment-softmax coin-flip, so I expect its seed
variance to *shrink* markedly from DiffPool's 6.93-point range, but on 188 graphs noise will remain and the
mean could land near or modestly above DiffPool's 81.37 without the gap being decisive; because MUTAG is the
imbalanced set, I would also watch whether the injective sum *narrows* DiffPool's 4.54-point acc-minus-F1
gap by serving the minority class better than a majority-leaning mean did. The falsifiable claim is the
ranking: if reading every layer with an injective sum is the right correction, this readout should beat
DiffPool on the aggregate and most clearly on PROTEINS and NCI1 — the datasets where counts and depth, not
luck, decide. If PROTEINS does *not* move off 70.98, my whole "DiffPool became a mean" story is wrong and I
would have to look elsewhere for the failure.

The causal chain in one breath: DiffPool's stripped soft pool became a noisy *mean* — flat at 70.98 on
PROTEINS with a 0.92-point seed spread, a 6.93-point coin-flip on MUTAG, an acc-minus-F1 gap ordered
4.54 > 2.11 > 0.04 across the three sets — discarding counts and ignoring the per-layer embeddings entirely
→ so go back to the flat reduction with a clean theory: among {sum, mean, max} only the *sum* is injective
on the multiset of node features ({a,b} and {a,a,b,b} share a mean but not a sum; max sees only the
support), and an injective sum makes the GIN backbone as powerful as WL, so anything weaker wastes the
backbone → recover the missing scale not by learning a hierarchy but by reading the one the backbone
already built: sum-pool *each layer* (`layer_outputs`) and concatenate (jumping knowledge, the learnable
WL-subtree-kernel), 320 wide, ~17k extra params inside the 41.5k budget, chosen over an attention-weighted
sum or a Set2Set that would re-spend learned capacity on the soft pooling that just failed → batch-normalize
each layer's graph embedding first so the O(N) size scaling and per-depth magnitude drift don't stall
optimization, an affine invertible map that preserves the sum's injectivity → drop it into `GraphReadout`
with `output_dim = hidden_dim × num_layers`, expecting PROTEINS and NCI1 to clear DiffPool's 70.98 and
78.25 and MUTAG's seed variance to tighten off its 6.93-point range.
