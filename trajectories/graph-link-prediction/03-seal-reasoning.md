The GCN+MLP decoder did almost exactly what I predicted, and the way it did it tells me where the next
gain has to come from. Removing the variational noise tightened everything: the seed-123/456 AUC
collapses from VGAE are gone (Cora seeds now 93.1 / 90.5 / 90.3, CiteSeer 93.1 / 90.5 / 88.8), and the
learned decoder lifted the ranking metrics the way I claimed — Cora MRR went 20.0→31.2, Hits@20
49.3→70.3; CiteSeer MRR 27.1→40.8, Hits@20 53.5→73.7. The cleanest verdict was the large graph:
ogbl-collab Hits@50 jumped 31.77→**53.74**, a full twenty-two points, confirming that a learned
pairwise decoder beats a bare inner product at large-pool ranking. AUC moved up modestly (87→91 on
both) but did not jump — exactly the "AUC was never the failing metric" pattern I expected. So the
decoder *was* the bottleneck, and the MLP fixed the part of it that was about parametric capacity.

But the per-seed ranking numbers expose the limit. Cora MRR has a violent seed spread — 39.5, 16.1,
38.0 — that one collapsed seed (16.1) dragging the mean down. CiteSeer is steadier but Cora's
instability says the decoder's *signal* about a pair is fragile: on the seed where the embeddings land
slightly differently, the MLP loses the ability to rank the true edge near the front. Why would that
be? Because everything the gcn_dot decoder knows about a pair, it knows through two things only:
$z_{\text{src}}\odot z_{\text{dst}}$ (the interaction) and the two raw embeddings (the identities). All
three are *learned* quantities — they depend on the GCN converging to a good geometry — and when the
GCN's geometry is a little off (different seed), the decoder has nothing solid to fall back on. It has
no *direct* measurement of the one feature that classical link prediction has leaned on for twenty
years and that is robust to embedding quality: how the two nodes' neighborhoods relate. Two papers that
share many citations are likely to cite each other regardless of where the GCN happened to place them
in latent space. The gcn_dot decoder can only approximate that overlap *indirectly*, through whatever
the message passing folded into the embeddings, and that approximation is what's failing on the bad
seeds. So the next move is to give the decoder a richer, more *structural* view of each pair — to
augment what it sees with explicit pairwise geometry, so it stops depending entirely on the learned
embedding alignment.

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

But none of that fits this scaffold's contract. The interface is `encode(x, edge_index) -> z` once over
the *whole* graph, then `decode(edge_label_index, z, edge_index)` per pair. There is no per-edge
subgraph extraction loop, no place to build and pool over thousands of small labeled subgraphs (that
would be ruinously expensive on ogbl-collab's 235k nodes anyway, and would blow the parameter and time
budget). DRNL needs per-pair shortest-path distances computed inside each enclosing subgraph; the
harness gives me a single global embedding table and the raw `edge_index`, not a subgraph machine. So I
cannot implement DRNL-labelled subgraph classification. What I *can* do is keep SEAL's load-bearing
insight — *the decoder should see structural/positional information about the pair, not just embedding
alignment* — and realize it within the full-graph encode/decode interface. The honest framing is: this
is a SEAL-inspired predictor that approximates the subgraph information through richer pairwise features
at decode time, deliberately dropping the subgraph-extraction and DRNL machinery the interface cannot
host.

So how do I enrich the decoder's view of the pair without subgraph extraction? The gcn_dot decoder saw
$[z_{\text{src}}\,\|\,z_{\text{dst}}\,\|\,z_{\text{src}}\odot z_{\text{dst}}]$. The Hadamard product
captures *agreement* (it's large where both embeddings are large and aligned), but it is symmetric and
sign-coupled and it misses *dissimilarity* — and dissimilarity is structurally informative. Two nodes
that are far apart in embedding space along some dimension are structurally different in a way that
predicts non-edges. The cleanest extra feature that captures this is the absolute difference
$|z_{\text{src}}-z_{\text{dst}}|$, the elementwise $L_1$ gap. It is the natural complement to the
Hadamard product: where $z\odot z$ measures co-activation, $|z-z|$ measures separation, and together
they span the standard pairwise interaction basis used across metric learning and link prediction
(concat for identity, product for similarity, absolute difference for distance). So I extend the
decoder's input to four blocks:
$h=[\,z_{\text{src}}\;\|\;z_{\text{dst}}\;\|\;z_{\text{src}}\odot z_{\text{dst}}\;\|\;
|z_{\text{src}}-z_{\text{dst}}|\,]$, a $4H$-dimensional pair representation, and feed it to the same
shape of MLP ($4H\to H\to H\to 1$). The added difference block gives the MLP a *distance* view of the
pair to set alongside the *similarity* view it already had — a direct structural signal that does not
depend on the embeddings aligning perfectly, which is precisely what should stabilize the seed-to-seed
variance I saw on Cora's MRR.

There is a second change, and it is small but it matters and it distinguishes this rung's encoder from
gcn_dot's. In the gcn_dot encoder I kept BatchNorm on intermediate layers *only* and left the final
embedding un-normalized, with the original justification (inherited from the dot-product era) that the
decoder needs the final embedding's magnitude. But now the decoder is a four-block MLP whose first
linear layer rescales whatever comes in, and more importantly, the four pairwise features mix raw
embeddings, products, and differences at very different scales — the product block is quadratic in the
embedding magnitude, the difference block linear. If the final embedding is un-normalized, those scale
differences across blocks make the MLP's job harder and add to the seed instability. So I put BatchNorm
on *every* encoder layer including the last (a separate `StructuralEncoder` module), normalizing the
final embedding before it enters the four-feature decoder. With an MLP decoder this costs nothing — the
MLP can recover any needed scale — and it controls the feature magnitudes so the product and difference
blocks are comparable, which should further damp the variance. This is the deliberate inversion of the
VGAE/gcn_dot "no BN on the last layer" rule, and it is licensed precisely because the decoder is no
longer a magnitude-sensitive inner product.

So the step-3 edit lands in the editable region as two pieces: a `StructuralEncoder` (GCN stack with
BatchNorm on all layers, ReLU and dropout on the intermediate ones) and a `LinkPredictor` whose
`decode` builds the four-block $4H$ feature $[z_{\text{src}},z_{\text{dst}},z_{\text{src}}\odot
z_{\text{dst}},|z_{\text{src}}-z_{\text{dst}}|]$ and runs the $4H\to H\to H\to 1$ MLP. The encoder
geometry is otherwise the gcn_dot encoder; the new content is the difference feature and full-layer
normalization. Everything else — the loop, the BCE, the negative sampling — is untouched. (The full
scaffold module is in the answer.) I want to be explicit that this is *not* the full canonical SEAL: there is
no enclosing-subgraph extraction, no DRNL labeling, no subgraph-level GNN with pooling; the structural
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
as AUC and ogbl-collab improve. That mixed outcome — SEAL stronger on AUC and the large graph,
gcn_dot still competitive on small-graph MRR/Hits — is the honest prediction, and if it holds, the
diagnosis for going *past* the strongest baseline is already written: the structural signal helps but I
am still *approximating* neighborhood overlap through embeddings and a single difference feature, when
the thing that actually discriminates links on these graphs is the *explicit* count of shared
neighbors. The next move would be to compute common-neighbor / Adamic–Adar / resource-allocation
features directly against the live adjacency and hand those to the decoder — the structural signal SEAL
gestures at, made exact.
