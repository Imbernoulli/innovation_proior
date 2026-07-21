The GCN+MLP decoder did almost exactly what I predicted, and the way it did it tells me where the next
gain has to come from. Removing the variational noise tightened everything: the seed-123/456 AUC
collapses from VGAE are gone (Cora seeds now 93.1 / 90.5 / 90.3, CiteSeer 93.1 / 90.5 / 88.8), and the
learned decoder lifted the ranking metrics — Cora MRR 20.0→31.2, Hits@20 49.3→70.3; CiteSeer MRR
27.1→40.8, Hits@20 53.5→73.7. The cleanest verdict was the large graph: ogbl-collab Hits@50 jumped
31.77→**53.74**, a full twenty-two points, confirming that a learned pairwise decoder beats a bare inner
product at large-pool ranking. AUC moved up modestly (87→91 on both) but did not jump — the "AUC was
never the failing metric" pattern I expected. So the decoder *was* the bottleneck, and the MLP fixed the
part of it that was about parametric capacity.

But the per-seed ranking numbers expose the limit. Cora MRR has a violent seed spread — 39.45, 16.10,
38.00 — that one collapsed seed dragging the mean to 31.18. Look at what else seed 123 did: its AUC is a
healthy 90.47 and its Hits@20 a healthy 68.50, yet its MRR is 16.10. MRR 16.1 is a mean reciprocal rank
near $0.16$, so the true edge sits on average around rank 6; the good seeds' $\sim39$ puts it around rank
2.5. So on the bad seed separation is fine and the top-20 is fine, but the *very front* of the list is
wrong. That is a pure top-of-list-precision failure, isolated to one seed, and it is Cora-specific:
CiteSeer's MRR seeds are 39.40 / 36.46 / 46.67, no collapse. I cannot fully pin why Cora is fragile where
CiteSeer is not, but a plausible read is that Cora is denser (average degree $\sim3.9$ versus CiteSeer's
$\sim2.7$) and carries far fewer features per node (1,433 vs 3,703), so its embeddings must lean harder on
structure and less on features, making the learned geometry more sensitive to initialization. That is a
hypothesis; what I *can* say for certain is that the failure is top-of-list and seed-dependent.

Why would one collapsed seed wreck the front of the list while leaving AUC and Hits@20 intact? Because
everything the gcn_dot decoder knows about a pair, it knows through two learned quantities:
$z_{\text{src}}\odot z_{\text{dst}}$ and the two raw embeddings. Both depend on the GCN converging to a
good geometry, and when that geometry is a little off (a bad seed) the decoder has nothing solid to fall
back on. AUC and Hits@20 forgive that — they need only coarse separation and a roughly-right top-20 — but
MRR demands the single true edge above every near-tied negative, and near ties are exactly where a
slightly-off learned geometry fails. The decoder has no *direct* measurement of the one feature classical
link prediction has leaned on for two decades and that is robust to embedding quality: how the two nodes'
neighborhoods relate. Two papers that share many citations are likely to cite each other regardless of
where the GCN happened to place them. The gcn_dot decoder can only approximate that overlap *indirectly*
— the load-bearing hedge I flagged last step — and that approximation is what fails on the bad seeds. So
the next move is to give the decoder a richer, more *structural* view of each pair.

This is the SEAL idea, but I have to be careful because the canonical version is machinery this harness
does not let me build. Original SEAL reframes link prediction as subgraph classification: for each
candidate $(i,j)$, extract the $k$-hop enclosing subgraph, label every node by its *double-radius* —
distance to $i$ and to $j$ via Double-Radius Node Labeling — and run a graph-level GNN with pooling to
classify whether the link exists. The theory (the labeling trick) is that a GNN over a properly labeled
enclosing subgraph can in principle learn *any* neighborhood-overlap heuristic — common neighbors,
Adamic–Adar, Katz — rather than being handed one. That expressive power is the appeal. But none of it fits
this contract. The interface is `encode(x, edge_index)` once over the *whole* graph, then
`decode(edge_label_index, z, edge_index)` per pair — there is no per-edge subgraph extraction loop. And
the expense is not marginal: on ogbl-collab evaluation touches $\sim10^5$ candidate pairs, each needing a
$k$-hop subgraph that on average degree $\sim11$ blows up to hundreds or thousands of nodes at $k=2$;
extracting, labeling, and pooling over that many subgraphs is orders of magnitude more compute than the
whole-graph encode the harness budgets, and it would blow the startup parameter check. So I cannot
implement DRNL-labelled subgraph classification. What I *can* keep is SEAL's load-bearing insight — *the
decoder should see structural/positional information about the pair, not just embedding alignment* —
realized within the full-graph encode/decode interface: a SEAL-inspired predictor that approximates the
subgraph information through richer pairwise features at decode time.

How do I enrich the decoder's view without subgraph extraction? The gcn_dot decoder saw
$[z_{\text{src}}\,\|\,z_{\text{dst}}\,\|\,z_{\text{src}}\odot z_{\text{dst}}]$. The Hadamard product
captures *agreement* but is symmetric and sign-coupled, and it misses *dissimilarity*, which is
structurally informative in the opposite direction. Every candidate extra feature — squared difference,
absolute difference, a cosine scalar, explicit shared-neighbor counts — is *some* function of $z_i$ and
$z_j$, which the decoder already receives, so the real criterion is not formal independence but
*synthesizability*: which features are hardest for the MLP to build itself, and therefore most worth
handing it pre-computed at the first layer instead of costing depth to approximate. By that test the
ranking is clear. The squared difference is nearly useless as a new input: $(z_i-z_j)^2 = z_i^2 + z_j^2 -
2\,z_i\odot z_j$, so it is a linear combination of squared identities and the Hadamard block already
present, its only novel piece the single-argument squares $z_i^2,z_j^2$ which the network approximates
cheaply. The absolute difference $|z_i-z_j|$ is different in kind: non-smooth, non-polynomial, a
city-block distance no low-order polynomial of the existing blocks reproduces, so the MLP would spend real
capacity to imitate it — it is the one whose explicit provision buys the most. The cosine scalar I reject
as a single number drowned among $3H$ inputs and largely recoverable from the summed Hadamard block and
the norms. The explicit shared-neighbor counts are the genuinely structural move, but I hold them for
later: they are a *different kind* of change, requiring the decoder to read the live adjacency with sparse
set operations. First I want to isolate one question — does a *separation* view of the embeddings, set
beside the *similarity* view, close the gcn_dot ranking gap? If the distance view is not enough, I will
have earned the right to make the overlap exact.

So I extend the decoder's input to four blocks: $h=[\,z_{\text{src}}\;\|\;z_{\text{dst}}\;\|\;
z_{\text{src}}\odot z_{\text{dst}}\;\|\;|z_{\text{src}}-z_{\text{dst}}|\,]$, a $4H$-dimensional pair
representation, fed to the same shape of MLP ($4H\to H\to H\to 1$). These three interaction blocks —
concatenation for identity, product for similarity, absolute difference for distance — are the standard
pairwise basis in metric learning and link prediction, and the difference block gives the MLP a *distance*
view that does not depend on the embeddings aligning perfectly, which is what should stabilize the seed
variance I saw on Cora's MRR.

The separation view should help most on ogbl-collab. A true collaboration pair tends to have *similar
neighborhoods* — co-authors share collaborators — so the GCN pushes their embeddings close together and
$|z_i-z_j|$ is small; a random non-edge from the vast pool of non-collaborators has embeddings typically
far apart, so $|z_i-z_j|$ is large. The difference block hands the MLP a monotone discriminator — small
separation votes for an edge, large against — computed directly, not requiring the embeddings to have
converged to a perfectly aligned inner-product geometry. On the small citation graphs the similarity block
already captured most of this; on the quarter-million-node graph, where each true edge must be pushed
above tens of thousands of non-edges, an explicit robust "close or far" axis alongside "do they
co-activate" is exactly what the Hits@50 metric rewards.

The second change is small but it matters and distinguishes this encoder from gcn_dot's. In gcn_dot I kept
BatchNorm on intermediate layers only, leaving the final embedding un-normalized so the dot product had
its magnitude. But now the four decoder blocks live at *different scales* in the embedding magnitude: with
typical coordinate magnitude $s$, the identity and difference blocks are of order $s$ (linear) but the
Hadamard block is of order $s^2$ (quadratic). If the final embedding is un-normalized and $s$ drifts from
seed to seed, the *relative* weight of the similarity block against the separation block drifts as $s$ —
so the very balance I am adding the difference feature to exploit would itself wobble with initialization,
feeding right back into the seed instability I am trying to kill. BatchNorm on *every* encoder layer
including the last (a separate `StructuralEncoder`) forces $s\approx1$ per dimension, so the quadratic and
linear blocks land at comparable magnitudes. With an MLP decoder this costs nothing — the MLP recovers any
needed scale — and it is licensed precisely because the decoder is no longer a magnitude-sensitive inner
product. It is the deliberate inversion of the earlier "no BN on the last layer" rule, and now I can say
exactly why the inversion is safe here and was not there.

That normalization has a plausible downside. Forcing unit variance per dimension deletes
absolute-magnitude information — the very thing the earlier steps preserved for the dot product. On the small citation graphs
the top of the candidate list is decided by fine magnitude distinctions among a handful of near-tied
candidates, and a GCN allowed to make one node's embedding simply *larger* than another's had an easy
lever to separate them; BatchNorm removes that lever. So I am buying seed stability and cross-block
scale-matching at the cost of some raw magnitude resolution, and the trade need not be positive
everywhere: on the small graphs, where the similarity block already did most of the work, the difference
feature may not repay what the normalization costs. If the small-graph ranking instead *improves*, that
falsifies this magnitude-cost story and says the difference feature dominated.

The cost is negligible: the decoder's first layer grows $3H\!\cdot\!H\to4H\!\cdot\!H$, an extra
$H^2=65{,}536$ parameters on gcn_dot's $\sim2.62\times10^5$ decoder, plus $2H=512$ affine parameters for
the final BatchNorm. A targeted structural probe, not a capacity increase.

So the step-3 edit lands as two pieces: a `StructuralEncoder` (GCN stack with BatchNorm on all layers,
ReLU and dropout on the intermediate ones) and a `LinkPredictor` whose `decode` builds the four-block
$4H$ feature and runs the $4H\to H\to H\to 1$ MLP. Everything else — the loop, the BCE, the negative
sampling — is untouched; the structural signal here is still the *approximation* the interface permits
(difference feature plus normalized embeddings), not the subgraph machine. (The full module is in the
answer.)

The gains here should be *narrow and specific*, not a blowout — one structural feature and a
normalization. The difference feature and the normalization give a signal that survives a slightly-off
embedding geometry, so the Cora MRR collapse seed (16.1) should come up and tighten the {39.5, 16.1,
38.0} spread, with per-seed AUC tighter and holding in the low nineties. The clearest verdict is
ogbl-collab Hits@50: the extra distance feature plus normalization should push past gcn_dot's 53.74,
because on the dense collaboration graph the distinction between "co-activated" and "separated"
embeddings is exactly what discriminates true collaborations from the huge non-edge pool. Where I expect
*no* gain, or even a small regression, is the small-graph MRR and Hits@20: gcn_dot's MLP already extracts
most of what Cora/CiteSeer offer, and the magnitude-cost of full-BN can trade a little top-of-list
precision for stability, concentrated on Cora. If that mixed outcome holds — SEAL stronger on AUC and the
large graph, gcn_dot still competitive on small-graph ranking — the diagnosis for going past it is already
written: I am still *approximating* neighborhood overlap through embeddings and a single difference
feature, when the thing that actually discriminates links is the *explicit* count of shared neighbors,
computed directly against the live adjacency — a discrete, graph-derived count that does not wobble with
the embedding geometry the way this step's difference feature necessarily does.
