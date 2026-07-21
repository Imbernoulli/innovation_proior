The SEAL step landed almost exactly where I predicted, and the precise shape of its result tells me the
next move. The full-layer normalization and the distance feature stabilized Cora's AUC (seeds now 92.2 /
92.3 / 93.0, mean 92.5 — tighter and higher than gcn_dot's 91.3), and ogbl-collab Hits@50 climbed to
57.88, past gcn_dot's 53.74, confirming that giving the decoder a *separation* view alongside the
*similarity* view helps most on the dense large graph. But the small-graph ranking metrics did not
improve — they regressed: Cora MRR 31.18→27.49, Hits@20 70.27→61.16; CiteSeer MRR 40.84→35.22, Hits@20
73.70→72.09.

The per-seed numbers say whether the *mechanism* I predicted is the one that fired. On Cora the gcn_dot
MRR seeds were {39.45, 16.10, 38.00} — one collapsed floor at 16.10 — and SEAL's are {23.46, 21.49,
37.52}. Read the two changes separately. The floor came *up*: the worst seed went 16.10→21.49, the spread
tightened from a range of 23.4 to 16.0 — exactly the stabilization the difference feature and full-BN
were supposed to buy. But the *ceiling fell harder*: gcn_dot had two strong seeds near 38–39, SEAL only
one (37.52) with two dragged into the low twenties, so the mean dropped to 27.49. That is precisely the
magnitude-cost mechanism I named introducing the full-layer BatchNorm — normalizing the final embedding
to unit variance deletes the absolute-magnitude lever the GCN used to separate near-tied candidates at
the very top of the list, and on the small graphs that lever mattered more than the stability was worth.
Hits@20 tells the same story louder: Cora fell nine points while AUC rose. And the large graph shows the
same fingerprint from the other side: ogbl-collab Hits@50 rose 53.74→57.88, yet its MRR *fell*
13.79→10.58 — SEAL pushed more true edges into the top-50 bucket but made the very top of the ordering
worse, and since Hits@50 is a coarser cutoff than MRR, a change that helps bulk top-50 membership while
hurting the razor's-edge ordering registers as exactly this split. So the trade was real and the
mechanism was the predicted one. SEAL is still *approximating* neighborhood overlap, and when I asked it
to discriminate the hardest pairs its "structural" signal, $|z_{\text{src}}-z_{\text{dst}}|$, is a
function of the *learned embeddings*, not of the graph. It had no measurement of the one quantity that
actually decides links: how many neighbors do $u$ and $v$ literally share. So I am not going to add
another *learned* pairwise feature. I am going to compute the structural signal *exactly*, against the
live adjacency, and hand it to the decoder.

This is a real expressiveness ceiling, not a tuning issue, and the smallest example shows it. A
message-passing GNN produces one embedding per node as a function of that node's rooted neighborhood, so
two destinations with the *same* rooted neighborhood receive the *same* embedding and the *same* score
against any fixed source. Take a source $u$ with neighbors $\{a,b\}$ and two candidates: $v_1$ with
neighbors $\{a,b\}$, $v_2$ with neighbors $\{c,d\}$, where $a,b,c,d$ carry the same features and the same
degree. Then $\mathrm{CN}(u,v_1)=2$ while $\mathrm{CN}(u,v_2)=0$ — $v_1$ is a strong candidate, $v_2$
weak. But a GCN embeds each node as the normalized aggregate of its neighbors' features, and since $v_1$
and $v_2$ both aggregate two same-feature, same-degree neighbors, $z_{v_1}=z_{v_2}$ exactly. Every
decoder I have built so far scores $(u,v)$ as a function of $(z_u,z_v)$ alone — dot product, MLP on the
Hadamard block, MLP on the four-block SEAL features — so all of them return
$\text{score}(u,v_1)=\text{score}(u,v_2)$ and *cannot* rank the true neighbor above the non-neighbor. It
is a representational failure, the same fact as the well-known result that a plain GNN cannot count
triangles — a triangle through $u,v,w$ is exactly a common neighbor $w$ — so the common-neighbor /
Adamic–Adar / resource-allocation family, the signal that has dominated link prediction on these graphs
for two decades, is precisely what every step so far has been *structurally unable to represent*. The
explicit $\mathrm{CN}$ feature ($2$ versus $0$) breaks the tie instantly.

The toy is exact only because I made the neighbors feature-identical; on real Cora and CiteSeer, where
nodes carry distinct features, two candidates rarely collide *perfectly*, so the failure is a soft tie,
not a hard one — and that strengthens the case. As two destinations become close in the joint
feature-and-structure space the GCN reads, their embeddings become close *continuously*, and the
decoder's margin for separating a true neighbor from a look-alike non-neighbor shrinks toward zero — which
is precisely the near-tie regime MRR lives in, and precisely where my models' ranking has been fragile.
The exact $\mathrm{CN}$ count does not degrade with embedding similarity: it stays $2$ versus $0$ no
matter how close the two embeddings drift. So injecting it hands the decoder a separator that is robust in
exactly the soft-collision region where the learned geometry is weakest.

So the insight the SEAL lineage was reaching for but its harness-friendly form could not deliver: score a
pair by a *learned function of explicit neighborhood-overlap features*, fused with the GNN embeddings.
For each candidate $(u,v)$ compute the three canonical overlap heuristics directly from the adjacency:
common neighbors $\mathrm{CN}(u,v)=|N(u)\cap N(v)|$, Adamic–Adar $\mathrm{AA}=\sum_{w\in N(u)\cap N(v)}
1/\log\deg(w)$, resource allocation $\mathrm{RA}=\sum_{w\in N(u)\cap N(v)}1/\deg(w)$. $\mathrm{CN}$ is the
raw shared-neighbor count the GNN cannot produce; $\mathrm{AA}$ and $\mathrm{RA}$ are degree-discounted
variants, and the discounting is the reason to carry all three. A shared neighbor that is a low-degree,
specific node is strong evidence of a link; a high-degree hub cited by everyone is weak evidence. Take two
shared neighbors of degree $2$ and $1000$: to $\mathrm{CN}$ each adds $1$; to $\mathrm{AA}$ the hub is
$1/\log1000\approx0.145$ against $\sim1$ for the rare one, a sevenfold down-weight; to $\mathrm{RA}$ the
hub is $1/1000=0.001$ against $0.5$, a five-hundredfold down-weight. So the three span a spectrum of
degree-sensitivity from none through mild to aggressive, and handing all three to the decoder lets the MLP
learn *which* discounting the data prefers rather than committing to one formula. One implementation
nuance in the $\mathrm{AA}$ weight: I compute it as $1/\max(\log\deg,1.0)$, clipping the log at $1.0$ to
guard the degenerate $1/\log1$; since a common neighbor necessarily has degree $\ge2$, in practice this
only flattens the discount for degree-2 shared neighbors ($\log2\approx0.693$ rounded up to $1.0$). A
deliberate, mild approximation of the exact $1/\log\deg$, not a change to the signal's meaning. These
features are pair-relative — they depend on the joint neighborhood geometry, not either node alone — so
they separate exactly the automorphic-node links the embedding-only decoders could not.

The degree-discounting sharpens the ogbl-collab prediction. On the citation graphs shared neighbors are
mostly specific papers, so raw $\mathrm{CN}$ already discriminates well. On the collaboration graph
prolific authors are hubs co-appearing in enormous numbers of pairs, so raw $\mathrm{CN}$ is inflated by
"both once co-authored with the field's superstar" — a weak signal wearing a big number. $\mathrm{AA}$ and
$\mathrm{RA}$ are exactly the correction, down-weighting the superstar and letting a shared *specific*
collaborator count for more. So I expect the MLP to lean hardest on the discounted variants precisely on
ogbl-collab — a second reason, beyond $\mathrm{CN}$ being unrepresentable by the GNN at all, that the
large-graph Hits@50 should be where the exact-overlap features pay off most.

Now the realization within this contract, and I have to be honest about what the interface lets me build
versus the fullest version of the idea. The fullest version computes, per pair, the entire distance-label
count table $A[d_u,d_v]$ over a $k$-hop window and, because exact intersection cardinalities over huge
neighborhoods are expensive, estimates them with set sketches — MinHash for the Jaccard, HyperLogLog for
the cardinality — propagated node-wise so the per-edge cost is independent of graph size. That machinery
is what makes the idea scale to millions of nodes. But the arithmetic says I do not need it here. The
interface gives me `decode(edge_label_index, z, edge_index)` — the *original* node indices and the *live*
adjacency — on graphs of at most $\sim236$k nodes with average degree of order ten. Build the CSR
adjacency from `edge_index` (sparse, $|E|$ nonzeros, never the dense $N\times N$), slice the rows for the
batch's sources and destinations, take the elementwise product of those sparse rows to get the per-pair
common-neighbor indicator, then sum it ($\mathrm{CN}$), sum it weighted by $1/\log\deg$ ($\mathrm{AA}$) and
by $1/\deg$ ($\mathrm{RA}$). Each intersection touches only the $\sim10$ nonzeros of the two rows, so the
whole batch costs $O(M\cdot\overline{\deg})$ and working memory is the sliced rows' nonzeros — nothing
near the $236\text{k}^2\approx5.6\times10^{10}$ entries a dense adjacency would demand. At this scale I
compute $\mathrm{CN}/\mathrm{AA}/\mathrm{RA}$ *exactly*, which is strictly tighter than a sketch estimate.
So the deliberate substitution is exact scipy-sparse counts in place of sketched multi-distance counts,
keeping the load-bearing insight (explicit pair-relative overlap counts as direct decoder inputs) and
dropping the sketching the harness neither needs at this scale nor exposes a hook for. The harness passes
the correct adjacency per phase — train-only during validation, train+val at test, as OGB prescribes — so
the counts are computed against the right graph automatically.

I should close off pushing the counting *into the encoder*, the obvious "more principled" move, because it
fails on arithmetic. A plain message-passing stack cannot count triangles — the ceiling I diagnosed — so I
would need a higher-order architecture: a 2-FWL GNN with $O(N^2)$ state and $O(N^3)$ update, or a subgraph
GNN per rooted subgraph. On $236$k nodes an $O(N^2)$ state is $5.6\times10^{10}$ entries — the same wall —
and neither is expressible through the `GCNConv`-only encode contract or inside the parameter budget.
Injecting the exact count at decode is cheap and *provably sufficient*: it hands the decoder the exact
quantity the higher-order GNN would laboriously recompute, which is why the substitution is a
simplification rather than a compromise.

The fusion has a scale subtlety. The fullest version combines the counts with the Hadamard product of
propagated features, edge-pooling style. The cleaner thing inside this decoder, given I already have a good
encoder, is to project the three counts up to the hidden width and *concatenate* them with the two raw
embeddings: $h=[z_{\text{src}}\,\|\,z_{\text{dst}}\,\|\,\text{proj}(\mathrm{CN},\mathrm{AA},\mathrm{RA})]$,
then an MLP $3H\to H\to H\to 1$. Why project $3\to H$ rather than concatenate three raw scalars onto the
$2H=512$-dimensional pair? Because three raw columns among $515$ would be drowned: at initialization the
first MLP layer weights every input comparably, so the structural signal would enter at $\sim0.6\%$ of the
input variance and the optimizer would fight uphill to attend to it. Lifting $[\mathrm{CN},\mathrm{AA},
\mathrm{RA}]$ through a learned $3\to256$ projection gives it a full third of the $768$-dimensional input,
so the MLP can model the interaction between "who the nodes are" and "how much they overlap."
Concatenation rather than Hadamard is deliberate: the counts are a *different kind* of quantity —
integer-ish overlap magnitudes, not learned coordinates — so forcing a product with the embeddings would
be semantically odd, whereas concatenation lets the MLP condition freely on both. The overlap computation
runs under `no_grad` — fixed structural measurements, deterministic functions of an adjacency that is not
a learned parameter, so there is no gradient to flow and wrapping them saves the memory of tracking them —
while the projection and MLP train end-to-end. The encoder stays the GCN stack with BatchNorm on all
layers, since the embeddings now sit beside projected counts and benefit from the same scale control.

So the finale edit lands as a `StructuralFeatureComputer` (a `no_grad` scipy-sparse routine returning the
$[\mathrm{CN},\mathrm{AA},\mathrm{RA}]$ per-pair feature) plus a `LinkPredictor` whose `encode` is the GCN
stack and whose `decode` computes the structural features against the passed `edge_index`, projects them,
concatenates with the two embeddings, and runs the MLP. One correctness point is load-bearing: `decode`
uses the `edge_index` the loop *hands it*, falling back to the cached encode-time adjacency only when none
is passed. That ordering keeps the counts phase-correct — at validation the loop passes the train-only
graph, at test the train+val graph — so a test edge never contributes to its own $\mathrm{CN}$ feature. If
I had instead always used the cached encode-time adjacency, the validation-time counts could include edges
the model is being asked to predict, a leak that would inflate the metrics and invalidate the comparison
to the baselines. Honoring the passed `edge_index` over the cache is the small line that prevents it.
Nothing else changes — same loop, same BCE, same negative sampling.

The bar is the strongest baseline, SEAL — Cora (AUC 92.5 / MRR 27.5 / Hits@20 61.2), CiteSeer (92.9 / 35.2
/ 72.1), ogbl-collab (Hits@50 57.88 / MRR 10.58). The sharpest claim: the small-graph *ranking* metrics
SEAL regressed should recover and exceed, because the explicit overlap counts are exactly the top-of-list
signal SEAL lacked — Cora and CiteSeer MRR and Hits@20 climbing back above gcn_dot's 31.2 / 70.3, clear of
SEAL's 27.5 / 61.2. On ogbl-collab, $\mathrm{CN}/\mathrm{AA}/\mathrm{RA}$ are *the* dominant signal on a
dense collaboration graph and SEAL only approximated them, so Hits@50 should pass 57.88 — the cleanest
single-number verdict. AUC should hold (never the failing metric); the way this could fail is the exact
counts dominating the MLP so strongly it ignores the node features, collapsing toward a pure heuristic and
losing the feature-driven gains on the feature-rich citation graphs — the warning sign would be AUC
*dropping* on Cora/CiteSeer, saying the fusion needs the structural features down-weighted.
