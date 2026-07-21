The stripped DiffPool came back and its numbers say exactly what I suspected when I built it: the
learned soft pool, with no link or entropy auxiliary to lean on, behaves like a noisy mean. The seed
spreads and the gap between the two metrics are where the mechanism shows itself, so let me read them.
PROTEINS's three seeds {70.98, 71.44, 70.52} span just 0.92 points (population σ ≈ 0.38) — a *tiny* band,
the fingerprint of a readout converging to the same diffuse, mean-like answer every run rather than
learning a seed-distinguishing partition, precisely the diffuse-softmax degeneration I proved the
stripped pool collapses to. And PROTEINS is the *lowest* of the three: it has the larger graphs where a
genuine functional-group hierarchy would help most, and the pool extracts the *least* from exactly the
graphs that should reward hierarchy. A pool reaching for structure that gets nothing out of the
most-structured set never found any structure.

MUTAG is the opposite fingerprint: {85.64, 78.71, 79.77}, mean 81.37, range 6.93 (σ ≈ 3.05) — an
eight-fold wider spread than PROTEINS, the coin-flip I predicted on a 188-graph set where K = 25 clusters
exceed the per-graph node count. The column softmax over ~18 real nodes fills 25 slots with near-redundant
averages, forced diffuse by arithmetic before training votes, and on so few graphs the residue is at the
mercy of the fold split. NCI1 sits between at 78.25, range 1.73 — unremarkable and fairly tight, fitting a
large balanced set where the mean-like pool converges to the same place each run.

The gap between test_acc and macro_f1 gives a second, corroborating reading. NCI1: 78.25 vs 78.21, gap
0.04 — the two classes balanced, both metrics measuring the same thing. PROTEINS: gap 2.11. MUTAG:
81.37 − 76.83 = 4.54, the largest. So the acc-minus-F1 gap orders the datasets MUTAG (4.54) >
PROTEINS (2.11) > NCI1 (0.04) — exactly the order of how count-blind a majority-leaning mean would look:
it buys accuracy on the frequent class while the minority F1 sags, loudest on the most imbalanced set.
MUTAG's macro_f1 seeds {83.25, 72.97, 74.26} span 10.28, wider even than its accuracy spread, so the
minority-class performance is what swings most. Both readings point the same way: on the larger balanced
set the pool is stable but merely mean-like; on the small imbalanced set it is a high-variance
majority-leaner.

The diagnosis is clean and it is not a tuning problem. The one idea I kept from DiffPool — a learned soft
clustering pooled as Sᵀ X — needed the auxiliaries the harness would not let me wire in, and without them
it discards counts (it is a convex-combination pool, a weighted mean, as I proved) and, worse, it makes no
use whatsoever of the per-layer node embeddings handed to me in `layer_outputs`. I threw away two
distinct things at once: the *injective* reduction that keeps multiplicities, and the *multi-scale* signal
sitting unused in the layer stack. Any correction that recovers only one of the two leaves points on the
table.

Four candidates fit the post-backbone slot. A plain global *sum* over the last layer recovers counts but
reads a single depth — injectivity fixed, scale untouched. A *learned attention-weighted sum* re-weights
each node but re-introduces the learned softmax I just watched go diffuse in DiffPool, and still reads one
depth. A Set2Set-style recurrent set network is non-destructive but a single sequential, parameter-heavy
aggregation over one layer, again single-scale. Three of the four leave the multi-scale signal on the
floor, and two spend fresh learned capacity on the soft pooling that just failed. Only *sum every layer
and concatenate* repairs both diagnosed failures — counts and scale — while spending no fragile learned
capacity on the pooling. I still have to earn it on the aggregator side, but the elimination points there.

Go back to first principles on the *flat* reduction. Among permutation-invariant reductions over the node
multiset, which keeps the most information? DiffPool became a mean, and a mean is count-blind (it keeps
the *distribution*, invariant to multiset inflation — the defect I already proved). Max is worse: max over
any duplication of a set is unchanged, so it sees only the *support*, losing counts and proportions. The
sum, meanwhile, is *injective* on bounded multisets, which a positional trick makes concrete: map each
integer feature x to 10^{−x} and sum. {1, 1, 2} → 0.21 ("two ones, one two"); {1, 2, 2} → 0.12; and
{1, 2} → 0.11 vs {1, 1, 2} → 0.21 distinguishes what max cannot even at identical support (max gives 2 for
both). So there is a strict pecking order in discriminative power — sum ⊐ mean ⊐ max — and DiffPool, by
drifting to a mean, sat one step below the top.

This is not an abstract preference; it is the whole reason the backbone in front of me exists. The
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
here I can fix that without a clustering at all. `layer_outputs` holds the per-layer node embeddings —
one `[N_total, hidden_dim]` tensor per GIN layer — and DiffPool used *none* of it. Why does it matter? A
node's embedding after k rounds is a learned summary of its rooted subtree of height k: height-1 sees
immediate neighbors, height-5 sees five hops. Deeper is more global and more discriminative, so I want
depth for power — but the deepest features are also the most specialized, and GINs over-smooth: after
several rounds embeddings drift toward a common value as neighborhoods overlap, and on small sets like
MUTAG the shallower, local features often generalize better than a washed-out last layer. Reading only
the last layer — the `x` argument — gambles that one depth is right for tiny MUTAG molecules and
hundred-node NCI1 compounds at once. Rather than gamble, use *all* of them. The multi-scale signal
DiffPool wanted from a learned hierarchy is already present, for free, in the layer stack — I do not have
to *learn* a hierarchy, only *read* the one message passing already computed.

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

Check the budget on that 320 width. The fixed classifier's first layer is `Linear(output_dim, 64)`: at
output_dim = 64 it had 4,160 params, at 320 it becomes 320·64 + 64 = 20,544, up 16,384; the per-layer
batch-norms add 5 × 2·64 = 640. So ~17,000 extra parameters, comfortably inside the 10·H² + 9·H = 41,536
headroom — about forty percent of the allowance. The width is free; a projection bottleneck would only
throw away the side-by-side separation of depths the concat exists to preserve.

One wrinkle is the single place this departs from a textbook JK-sum, and it needs reasoning, not copying.
Concatenating per-layer *sum* pools concatenates vectors at very *different scales*. The dominant source
is graph size: a sum grows linearly with node count, so a ~100-node NCI1 molecule gives a readout ~6×
the magnitude of a ~17-node MUTAG one before any content enters — and PROTEINS spreads that factor wider
*within a single batch*. On top, deeper GIN layers (more rounds of neighbor summation, the (1+ε)
self-weighting, per-layer BatchNorm→ReLU) carry systematically different magnitudes than shallow ones.
Fed straight into the classifier, the large-magnitude coordinates dominate the first linear layer's
gradient, the small ones are ignored, and some folds fail to converge — the optimization stalls on the
scale mismatch, not the graphs. The fix that keeps injectivity intact is to **batch-normalize each
layer's graph-level embedding before concatenating**: a `BatchNorm1d(hidden_dim)` per layer on the
`[B, hidden_dim]` pooled vectors, subtracting a per-channel mean and dividing by a per-channel std over
the batch — removing the O(N) size scaling and per-depth drift. It is a strictly monotone affine
rescaling, so it does *not* collapse the multiset distinctions the sum encoded (a monotone affine map of
an injective code is still injective); it only conditions the input so every fold trains. This graph-BN
is the one piece beyond "sum each layer and concatenate," and it is there for stability, not expressivity
— I should not credit it with any accuracy the readout earns.

Permutation invariance is inherited for free: each per-layer sum is node-order-invariant, BatchNorm acts
per graph-level channel, and a concatenation of invariants is invariant. I read from `layer_outputs` and
ignore `x` (which is just `layer_outputs[-1]`, already included); no `edge_index`, no dense batching, no
clusters — the whole readout is a per-layer sum, a per-layer BN, and a concat, `output_dim = 320`. And it
spends *no* learned capacity on the pooling itself, the exact opposite of DiffPool: the only learned
parameters are the conditioning BNs.

Where is this falsifiable against DiffPool's numbers? PROTEINS is the cleanest test: its diffuse pool
flattened to 70.98 with a near-zero seed spread, so if the diagnosis holds — a mean discards counts, a
single depth misses scale — summing every layer should lift PROTEINS clearly above 71 and stay tight now
that the readout is deterministic. NCI1, where the injective-sum-feeds-WL argument bites hardest, should
also rise above 78.25, and with its near-zero acc-minus-F1 gap any gain shows on both metrics at once.
MUTAG I am least sure about: the deterministic readout removes the coin-flip, so its seed variance should
shrink markedly off DiffPool's 6.93-point range, but on 188 graphs the mean could land near or modestly
above 81.37; I would also watch whether the injective sum narrows the 4.54-point acc-minus-F1 gap by
serving the minority class better than a majority-leaning mean did. The load-bearing claim is the
ranking: beat DiffPool on the aggregate, most clearly on PROTEINS and NCI1. If PROTEINS does *not* move
off 70.98, the whole "DiffPool became a mean" story is wrong.
