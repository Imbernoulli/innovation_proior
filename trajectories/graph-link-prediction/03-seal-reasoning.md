The GCN+MLP decoder did almost exactly what I predicted, and the way it did it tells me where the next
gain has to come from. Removing the variational noise tightened everything: the seed-123/456 AUC
collapses from VGAE are gone (Cora seeds now 93.1 / 90.5 / 90.3, CiteSeer 93.1 / 90.5 / 88.8), and the
learned decoder lifted the ranking metrics the way I claimed — Cora MRR went 20.0→31.2, Hits@20
49.3→70.3; CiteSeer MRR 27.1→40.8, Hits@20 53.5→73.7. The cleanest verdict was the large graph:
ogbl-collab Hits@50 jumped 31.77→**53.74**, a full twenty-two points, confirming that a learned
pairwise decoder beats a bare inner product at large-pool ranking. AUC moved up modestly (87→91 on
both) but did not jump — exactly the "AUC was never the failing metric" pattern I expected. So the
decoder *was* the bottleneck, and the MLP fixed the part of it that was about parametric capacity.

But the per-seed ranking numbers expose the limit, and I want to read them closely because the shape of
the residual instability is what picks the next move. Cora MRR has a violent seed spread — 39.45, 16.10,
38.00 — that one collapsed seed dragging the mean down to 31.18. Look at what else that seed 123 did: its
AUC is a perfectly healthy 90.47 and its Hits@20 a healthy 68.50, yet its MRR is 16.10. Translate the
numbers: MRR 16.1 is a mean reciprocal rank near $0.16$, so the true edge sits on average around rank 6;
the good seeds' MRR of ${\sim}39$ puts it around rank 2.5. So on the bad seed separation is fine and the
top-20 is fine, but the *very front* of the list is wrong — the true edge is buried two to three slots
deeper than it should be. That is a pure top-of-list-precision failure, isolated to one seed. And it is
Cora-specific: CiteSeer's MRR seeds are 39.40 / 36.46 / 46.67, all healthy, no collapse at all. I cannot
fully pin why Cora is fragile where CiteSeer is not, but I have a plausible read: Cora is the denser graph
(about $5.3\text{k}$ undirected edges over $2.7\text{k}$ nodes, average degree ${\sim}3.9$, versus
CiteSeer's ${\sim}2.7$) and carries far fewer features per node (1,433 vs 3,703), so its embeddings must
lean harder on structure and less on the feature signal — which makes the learned geometry more sensitive
to initialization. That is a hypothesis, not a proof; what I *can* say for certain is that the failure is
top-of-list and seed-dependent, and that is enough to diagnose the mechanism.

Why would one collapsed seed wreck the front of the list while leaving AUC and Hits@20 intact? Because
everything the gcn_dot decoder knows about a pair, it knows through two things only:
$z_{\text{src}}\odot z_{\text{dst}}$ (the interaction) and the two raw embeddings (the identities). All
three are *learned* quantities — they depend on the GCN converging to a good geometry — and when the
GCN's geometry is a little off (different seed), the decoder has nothing solid to fall back on. AUC and
Hits@20 forgive that, because they only need coarse separation and a roughly-right top-20; MRR does not,
because it demands the single true edge above every near-tied negative, and near ties are exactly where a
slightly-off learned geometry fails. The decoder has no *direct* measurement of the one feature that
classical link prediction has leaned on for twenty years and that is robust to embedding quality: how the
two nodes' neighborhoods relate. Two papers that share many citations are likely to cite each other
regardless of where the GCN happened to place them in latent space. The gcn_dot decoder can only
approximate that overlap *indirectly*, through whatever the message passing folded into the embeddings —
the very indirection I flagged as the load-bearing hedge last rung — and that approximation is what is
failing on the bad seeds. So the next move is to give the decoder a richer, more *structural* view of each
pair, so it stops depending entirely on the learned embedding alignment.

This is the SEAL idea, and I have to be careful here because the canonical version of SEAL is heavy
machinery this harness does not let me build, so I will derive what the *interface actually supports*
and name what it omits. The original SEAL formulation reframes link prediction as subgraph
classification: for each candidate pair $(i,j)$, extract the $k$-hop enclosing subgraph around the
two nodes, label every node in that subgraph by its *double-radius* — its distance to $i$ and its
distance to $j$ via the Double-Radius Node Labeling (DRNL) trick, which marks the two target nodes and
encodes each other node's structural role relative to them — and then run a graph-level GNN with
pooling over the labeled subgraph to classify whether the link exists. The theory behind it (the
labeling trick) is that a GNN over a properly *labeled* enclosing subgraph can in principle learn *any*
neighborhood-overlap heuristic — common neighbors, Adamic–Adar, Katz, the whole family — rather than
being handed one. That expressive power is the appeal.

But none of that fits this scaffold's contract, and it is worth being precise about the cost, not just
the shape, of why. The interface is `encode(x, edge_index) -> z` once over the *whole* graph, then
`decode(edge_label_index, z, edge_index)` per pair. There is no per-edge subgraph extraction loop, no
place to build and pool over thousands of small labeled subgraphs. And the expense is not marginal: on
ogbl-collab the evaluation touches on the order of $10^5$ candidate pairs, each needing a $k$-hop
enclosing subgraph that, on a graph of average degree ${\sim}11$, blows up to hundreds or thousands of
nodes at $k=2$; extracting, labeling, and running a pooling GNN over that many subgraphs per evaluation
is orders of magnitude more compute than the whole-graph encode the harness budgets for, and it would
also blow the startup parameter check with a subgraph-GNN stack. DRNL itself needs per-pair
shortest-path distances computed inside each enclosing subgraph; the harness gives me a single global
embedding table and the raw `edge_index`, not a subgraph machine. So I cannot implement DRNL-labelled
subgraph classification. What I *can* do is keep SEAL's load-bearing insight — *the decoder should see
structural/positional information about the pair, not just embedding alignment* — and realize it within
the full-graph encode/decode interface. The honest framing is: this is a SEAL-inspired predictor that
approximates the subgraph information through richer pairwise features at decode time, deliberately
dropping the subgraph-extraction and DRNL machinery the interface cannot host.

So how do I enrich the decoder's view of the pair without subgraph extraction? Let me lay out the real
options and pick on information content rather than instinct. The gcn_dot decoder saw
$[z_{\text{src}}\,\|\,z_{\text{dst}}\,\|\,z_{\text{src}}\odot z_{\text{dst}}]$. The Hadamard product
captures *agreement* — it is large where both embeddings are large and aligned — but it is symmetric and
sign-coupled, and it misses *dissimilarity*, which is structurally informative in the opposite direction.
Four candidate extra features present themselves. I could add the squared difference
$(z_{\text{src}}-z_{\text{dst}})^2$; I could add the absolute difference
$|z_{\text{src}}-z_{\text{dst}}|$; I could append a single cosine-similarity scalar; or I could reach past
the embeddings entirely and compute explicit shared-neighbor counts against the adjacency. Take them in
turn. The squared difference is almost useless as a *new* input, and the algebra says why:
$(z_i-z_j)^2 = z_i^2 + z_j^2 - 2\,z_i\odot z_j$, so it is a linear combination of the squared identities
and the Hadamard block the decoder already holds — the first MLP layer plus a rectifier can already form
those squares from $z_i,z_j$, so appending $(z_i-z_j)^2$ hands the network a quantity it can essentially
already build. The absolute difference is different in kind: $|z_i-z_j|$ is *not* a polynomial of
$\{z_i, z_j, z_i\odot z_j\}$ — the absolute value is non-smooth and genuinely new, giving the decoder the
elementwise *magnitude of separation* directly, a feature it cannot cheaply synthesize from what it has.
So among the embedding-derived options, $|z_i-z_j|$ is the one that adds the most information per
dimension. I should be honest that, strictly, *every* one of these is a function of $z_i$ and $z_j$,
which the decoder already receives in the identity blocks — a wide enough MLP could in principle
synthesize any of them internally. So the real criterion is not formal independence but
*synthesizability*: which features are hardest for the MLP to build itself, and therefore most worth
handing it pre-computed so they are available at the first layer instead of costing depth and capacity to
approximate. By that test the ranking is clear. The Hadamard cross-term is a product, awkward for a ReLU
MLP to form, which is why gcn_dot already supplies it. The squared difference is *mostly* redundant on top
of that, because its only piece not already present is the pure per-dimension squares $z_i^2,z_j^2$, which
are single-argument and comparatively easy for the network to approximate. The absolute value is the
outlier: a non-smooth, non-polynomial, city-block distance that no low-order polynomial of the existing
blocks reproduces and that the MLP would have to spend real capacity to imitate. So $|z_i-z_j|$ is the
feature whose explicit provision buys the most. The cosine scalar I reject as a single number that would be drowned among $3H$ inputs and is in
any case largely recoverable from the summed Hadamard block and the norms. That leaves the explicit
shared-neighbor counts — and those are the genuinely structural move — but I deliberately hold them for
later: they are a *different kind* of change, requiring the decoder to read the live adjacency with sparse
set operations rather than to engineer features on embeddings it already has. Before I reach into the
graph I want to isolate one question cleanly — does a *separation* view of the embeddings, set beside the
*similarity* view, close the gcn_dot ranking gap? — and the SEAL lineage's harness-compatible realization
is exactly this pairwise-feature enrichment. If the distance view is not enough, I will have earned the
right to make the overlap exact.

So I extend the decoder's input to four blocks:
$h=[\,z_{\text{src}}\;\|\;z_{\text{dst}}\;\|\;z_{\text{src}}\odot z_{\text{dst}}\;\|\;
|z_{\text{src}}-z_{\text{dst}}|\,]$, a $4H$-dimensional pair representation, and feed it to the same
shape of MLP ($4H\to H\to H\to 1$). These three interaction blocks — concatenation for identity, product
for similarity, absolute difference for distance — are the standard pairwise basis used across metric
learning and link prediction, and the added difference block gives the MLP a *distance* view of the pair
to set alongside the *similarity* view it already had: a direct structural signal that does not depend on
the embeddings aligning perfectly, which is precisely what should stabilize the seed-to-seed variance I
saw on Cora's MRR.

It is worth spelling out why the separation view should help most on ogbl-collab specifically, because
that is where I am predicting the largest gain. A true collaboration pair tends to have *similar
neighborhoods* — co-authors share collaborators — so the GCN pushes their embeddings close together, and
$|z_i-z_j|$ is small; a random non-edge, drawn from the vast pool of authors who never collaborated, has
embeddings that are typically far apart, so $|z_i-z_j|$ is large. The difference block therefore hands the
MLP a monotone discriminator — small separation votes for an edge, large separation against — that is
computed directly and does not require the embeddings to have converged to a perfectly aligned inner-
product geometry. On the citation graphs the candidate pool is small enough that the similarity block
already captured most of this; on the quarter-million-node collaboration graph, where each true edge must
be pushed above tens of thousands of non-edges, having an explicit, robust "are these two close or far"
signal alongside "do they co-activate" is exactly the extra axis of discrimination the Hits@50 metric
rewards.

There is a second change, and it is small but it matters and it distinguishes this rung's encoder from
gcn_dot's. In the gcn_dot encoder I kept BatchNorm on intermediate layers *only* and left the final
embedding un-normalized, with the original justification (inherited from the dot-product era) that the
decoder needs the final embedding's magnitude. But now the decoder is a four-block MLP whose first
linear layer rescales whatever comes in, and — here is the point — the four blocks live at *different
scales in the embedding magnitude*. Let the typical coordinate magnitude be $s$. Then the identity blocks
are of order $s$, the difference block is of order $s$ (linear), but the Hadamard block is of order $s^2$
(quadratic). If the final embedding is un-normalized and $s$ drifts from seed to seed, the *relative*
weight of the similarity block against the separation block drifts as $s$ — so the very balance between
"co-activated" and "separated" that I am adding the difference feature to exploit would itself wobble with
initialization, feeding right back into the seed instability I am trying to kill. Normalizing the final
embedding fixes this: BatchNorm on *every* encoder layer including the last (a separate
`StructuralEncoder` module) forces $s\approx1$ per dimension, so the quadratic and linear blocks land at
comparable magnitudes and the MLP sees a seed-stable mix. With an MLP decoder this costs nothing — the
MLP can recover any needed scale through its own weights — and it is licensed precisely because the
decoder is no longer a magnitude-sensitive inner product. This is the deliberate inversion of the
VGAE/gcn_dot "no BN on the last layer" rule, and now I can say exactly why the inversion is safe here and
was not there.

I should be honest that this same normalization has a plausible downside, and naming it now is what makes
my later prediction falsifiable rather than a post-hoc excuse. Forcing the final embedding to unit
variance per dimension deletes absolute-magnitude information — the very thing the VGAE and gcn_dot rungs
were careful to preserve for the dot product. On the small citation graphs, the top of the candidate list
is decided by fine magnitude distinctions among a handful of near-tied candidates, and a GCN that was
allowed to make one node's embedding simply *larger* than another's has an easy lever to separate them.
BatchNorm removes that lever. So I am buying seed stability and cross-block scale-matching at the cost of
some raw magnitude resolution, and the trade need not be positive everywhere: on the small graphs, where
the similarity block already did most of the work and magnitude resolution matters most, the difference
feature may not repay what the normalization costs. That is the concrete mechanism behind the
mixed-outcome prediction I will state at the end — AUC and large-graph Hits up, small-graph MRR possibly
flat or down — and if the small-graph ranking instead *improves*, that falsifies this magnitude-cost
story and says the difference feature dominated.

Before I commit I should check the cost, because "add a feature and a BatchNorm" should be cheap and I
want to confirm it. The decoder grows from $3H\to H\to H\to1$ to $4H\to H\to H\to1$; the only change is
the first layer, $3H\!\cdot\!H\to4H\!\cdot\!H$, i.e. an extra $H^2=65{,}536$ parameters on top of
gcn_dot's ${\sim}2.62\times10^5$ decoder. The extra final-layer BatchNorm adds $2H=512$ affine
parameters — negligible. So the whole rung costs about $66\text{k}$ parameters over gcn_dot, well inside
the budget, and the encoder and loop are otherwise untouched. This is a targeted structural probe, not a
capacity increase.

So the step-3 edit lands in the editable region as two pieces: a `StructuralEncoder` (GCN stack with
BatchNorm on all layers, ReLU and dropout on the intermediate ones) and a `LinkPredictor` whose
`decode` builds the four-block $4H$ feature $[z_{\text{src}},z_{\text{dst}},z_{\text{src}}\odot
z_{\text{dst}},|z_{\text{src}}-z_{\text{dst}}|]$ and runs the $4H\to H\to H\to 1$ MLP. A shape pass:
$z\in\mathbb{R}^{N\times H}$, gathered endpoints $\mathbb{R}^{M\times H}$, the four blocks concatenate to
$\mathbb{R}^{M\times4H}$, and the MLP returns the $M$-vector the loop expects. The encoder geometry is
otherwise the gcn_dot encoder; the new content is the difference feature and full-layer normalization.
Everything else — the loop, the BCE, the negative sampling — is untouched. (The full scaffold module is
in the answer.) I want to be explicit that this is *not* the full canonical SEAL: there is no
enclosing-subgraph extraction, no DRNL labeling, no subgraph-level GNN with pooling; the structural
information is approximated by the extra pairwise feature and normalized embeddings, which is what the
encode/decode interface permits.

Now the falsifiable expectations against the gcn_dot numbers, because the gains here should be *narrow
and specific*, not a blowout — I am adding one structural feature and a normalization, not a new model
class. First, the seed-variance claim: the Cora MRR collapse seed (16.1) should come up, tightening the
{39.5, 16.1, 38.0} spread, because the difference feature and the normalization give the decoder a
structural signal that survives a slightly-off embedding geometry. I expect the per-seed AUC to be
*tighter* and slightly higher on average than gcn_dot (the full-layer BN stabilizes training), landing
around 92–93 mean on both citation graphs — a small AUC gain, not a large one. Second, and this is the
clearest verdict, ogbl-collab Hits@50: the extra distance feature plus normalization should push past
gcn_dot's 53.74 into the high fifties, because on the dense collaboration graph the distinction between
"co-activated" and "separated" embeddings is exactly what discriminates true collaborations from the
huge non-edge pool. I expect Hits@50 around 57–58. Where I expect *no* gain, or even a small regression
relative to gcn_dot, is the small-graph ranking metrics MRR and Hits@20: gcn_dot's MLP already extracts
most of what the small Cora/CiteSeer graphs offer through the three-block decoder, and the fourth
feature plus full-layer normalization can trade a little of that top-of-list precision for stability —
so I would not be surprised if Cora MRR/Hits@20 come in at or slightly below gcn_dot's 31.2/70.3 even
as AUC and ogbl-collab improve. I can sharpen that into a per-graph split: CiteSeer's gcn_dot MRR was
already stable across seeds (39.4 / 36.5 / 46.7, no collapse), so the magnitude-cost of full-BN has less
to take away there and I expect CiteSeer MRR to hold near gcn_dot's 40.8; Cora is where the risk
concentrates, because that is where gcn_dot's ranking was fragile and where the normalization's deletion
of magnitude resolution has the most to spoil. If instead Cora's MRR is the one that holds and CiteSeer's
that drops, my magnitude-cost reading of the mechanism is wrong and I would go back to the block-scale
argument. That mixed outcome — SEAL stronger on AUC and the large graph,
gcn_dot still competitive on small-graph MRR/Hits — is the honest prediction, and if it holds, the
diagnosis for going *past* the strongest baseline is already written: the structural signal helps but I
am still *approximating* neighborhood overlap through embeddings and a single difference feature, when
the thing that actually discriminates links on these graphs is the *explicit* count of shared
neighbors. The next move would be to compute common-neighbor / Adamic–Adar / resource-allocation
features directly against the live adjacency and hand those to the decoder — the structural signal SEAL
gestures at, made exact: a discrete, graph-derived count that does not wobble with the embedding geometry
the way this rung's difference feature necessarily does.
