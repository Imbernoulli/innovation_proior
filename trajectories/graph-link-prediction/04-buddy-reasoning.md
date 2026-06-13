The SEAL rung landed almost exactly where I predicted, and the precise shape of its result is what
tells me the next move. The full-layer normalization and the distance feature did stabilize Cora's AUC
(seeds now 92.2 / 92.3 / 93.0, mean 92.5 — tighter and higher than gcn_dot's 91.3), and ogbl-collab
Hits@50 climbed to 57.88, past gcn_dot's 53.74, confirming that giving the decoder a *separation* view
of the pair alongside the *similarity* view helps most on the dense large graph. But the small-graph
ranking metrics did *not* improve — they regressed: Cora MRR fell from gcn_dot's 31.18 to 27.49, Hits@20
from 70.27 to 61.16; CiteSeer MRR 40.84→35.22, Hits@20 73.70→72.09. That is the tell. SEAL bought
AUC and large-graph Hits by trading away top-of-list precision on the small graphs, and the reason is
exactly what I flagged closing the last rung: SEAL is still *approximating* neighborhood overlap. Its
"structural" signal is the absolute-difference feature $|z_{\text{src}}-z_{\text{dst}}|$ — a function of
the *learned embeddings*, not of the graph. When I asked it to discriminate the hardest pairs (the top
of the MRR list), it had no measurement of the one quantity that actually decides links on citation and
collaboration graphs: how many neighbors do u and v literally share. It can only infer that overlap
through whatever message passing folded into the embeddings, and on the hardest pairs that inference is
too soft. So I am not going to add another *learned* pairwise feature. I am going to compute the
structural signal *exactly*, against the live adjacency, and hand it to the decoder.

Let me be precise about the diagnosis before I commit, because this is a real expressiveness ceiling,
not a tuning issue. A message-passing GNN produces one embedding per node, so two structurally-identical
(automorphic) destinations get the *same* embedding and therefore the *same* score against a fixed
source — even when one is a true neighbor and the other is not. Worse, a plain GNN cannot count
triangles, and a triangle through u, v, w is exactly a common neighbor w. So the common-neighbor /
Adamic–Adar / resource-allocation family — the signal that has dominated link prediction on these graphs
for two decades — is precisely what every rung so far has been *structurally unable to represent*. VGAE,
gcn_dot, and SEAL all score a pair only through the geometry of two independently-encoded points; none
of them ever counts a shared neighbor. That is why SEAL's MRR stalled. The fix is to break the
all-learned scoring and inject the explicit overlap counts.

So the structural insight I want is the one the SEAL lineage was reaching for but its harness-friendly
form could not deliver: score a pair by a *learned function of explicit neighborhood-overlap features*,
fused with the GNN embeddings. Concretely, for each candidate pair (u, v) compute the three canonical
overlap heuristics directly from the adjacency: common neighbors CN(u,v) = |N(u) ∩ N(v)|, Adamic–Adar
AA(u,v) = Σ_{w ∈ N(u)∩N(v)} 1/log deg(w), and resource allocation RA(u,v) = Σ_{w} 1/deg(w). CN is the
raw triangle count — the A[1,1] distance-label count, the backbone signal a GNN cannot produce. AA and
RA are degree-discounted variants: a shared neighbor that is a high-degree hub (cited by everyone) is
weak evidence of a link, so AA down-weights it by 1/log deg and RA by 1/deg. Handing all three to the
decoder lets the MLP learn *which* discounting the data prefers rather than committing to one formula —
the learned-heuristic move. These three features are pair-relative: they depend on the joint
neighborhood geometry of u and v, not on either node alone, so they separate exactly the
automorphic-node links the embedding-only decoders could not.

Now the realization within this task's contract, and here I have to be honest about what the interface
lets me build versus the fullest version of this idea. The fullest version computes, for each pair, the
entire distance-label count table A[d_u, d_v] over a k-hop window — A[1,1] is CN, A[1,2]/A[2,1]/A[2,2]
the multi-hop overlaps — and, because computing those intersection cardinalities exactly over huge
neighborhoods is expensive, estimates them with set sketches: MinHash for the Jaccard (intersection
shape) and HyperLogLog for the cardinality (union size), propagated node-wise by elementwise min/max so
the per-edge cost is independent of graph size. That sketch machinery is what makes the idea scale to
millions of nodes. But this scaffold's interface gives me `decode(edge_label_index, z, edge_index)` —
the *original* node indices and the *live* adjacency — at decode time, on graphs of at most ~236k nodes,
and the parameter budget is checked at startup. At this scale I do not need the sketches: I can compute
CN/AA/RA *exactly* with a sparse-matrix routine. Build the CSR adjacency from `edge_index`, slice the
rows for the batch's sources and destinations, take the elementwise product of those sparse rows to get
the per-pair common-neighbor indicator, then sum it (CN), sum it weighted by 1/log deg (AA), and by
1/deg (RA). The whole thing stays sparse — no dense N×N materialization — so it is memory-feasible on
ogbl-collab. So the deliberate substitution is: *exact* CN/AA/RA via scipy sparse, in place of
*sketched* multi-distance counts. I am keeping the load-bearing insight (explicit, pair-relative
overlap counts as direct decoder inputs) and dropping the sketching and the multi-hop count table that
the harness neither needs at this scale nor exposes a clean hook for. The harness already passes the
correct adjacency at each phase — train-only during validation, train+val at test, exactly as the OGB
protocol prescribes — so the overlap counts are computed against the right graph automatically.

The fusion is the other design point. The fullest version combines the structural counts with the
*Hadamard product* of the propagated node features, edge-pooling style. The cleaner thing to do inside
this decoder, given I already have a good GNN encoder, is to project the three overlap counts up to the
hidden width with a linear layer (so they enter the MLP on the same scale as the embeddings) and
*concatenate* them with the two raw node embeddings: $h = [z_{\text{src}}\,\|\,z_{\text{dst}}\,\|\,
\text{proj}(\text{CN}, \text{AA}, \text{RA})]$, then an MLP $3H\to H\to H\to 1$. Concatenation rather
than Hadamard here is a deliberate, harness-matched choice: the structural counts are a *different kind*
of quantity than the embeddings (integer-ish overlap magnitudes, not learned coordinates), so giving the
MLP all of $z_{\text{src}}$, $z_{\text{dst}}$, and a learned embedding of the counts lets it model the
interaction between "who the nodes are" and "how much they overlap" without forcing the counts through a
product with the embeddings. The overlap computation runs under `no_grad` — these are fixed structural
measurements, not learned features, so no gradient needs to flow through the sparse adjacency ops — while
the projection and MLP are trained end-to-end. The encoder stays the GCN stack with BatchNorm on all
layers (as in SEAL), since the embeddings now sit beside projected counts and benefit from the same
scale control.

So the finale edit lands in the editable region as a `StructuralFeatureComputer` (a `no_grad` scipy-
sparse routine returning the [CN, AA, RA] per-pair feature) plus a `LinkPredictor` whose `encode` is the
GCN stack and whose `decode` computes the structural features against the passed `edge_index`, projects
them, concatenates with the two embeddings, and runs the MLP. The encoder caches the encode-time
`edge_index` and node count so `decode` has a sensible default adjacency when the loop does not pass one
explicitly. Nothing else changes — same loop, same BCE, same negative sampling. I am explicit that this
is the exact-overlap, sketch-free realization: no MinHash, no HyperLogLog, no multi-distance count
table — the scale here makes exact CN/AA/RA both feasible and tighter than estimates.

What is the bar, and what would I validate? There is no leaderboard row for this method on this task;
the bar is the strongest baseline, SEAL — Cora (AUC 92.5 / MRR 27.5 / Hits@20 61.2), CiteSeer (92.9 /
35.2 / 72.1), ogbl-collab (Hits@50 57.88 / MRR 10.58). The falsifiable claims follow directly from the
diagnosis. First and sharpest: the small-graph *ranking* metrics that SEAL regressed should recover and
exceed, because the explicit overlap counts are exactly the top-of-list signal SEAL lacked — I expect
Cora MRR and Hits@20 to climb back above gcn_dot's 31.2 / 70.3 (so clear of SEAL's 27.5 / 61.2), and
CiteSeer likewise. Second: ogbl-collab Hits@50 should pass SEAL's 57.88, into the low-to-mid sixties,
because CN/AA/RA are *the* dominant signal on a dense collaboration graph and SEAL only approximated
them — this is the cleanest single-number verdict. Third, the consistency check: AUC should hold around
the low-to-mid nineties (it was never the failing metric), and if it holds *while* MRR/Hits@20/Hits@50
all rise, the whole trajectory's thesis is confirmed — that the ceiling on every embedding-only rung was
the inability to count shared neighbors, and that handing the decoder the exact overlap removes it. The
way this could fail: if the exact CN/AA/RA features dominate the MLP so strongly that it ignores the
node features, the model could collapse toward a pure heuristic and lose the feature-driven gains on the
feature-rich citation graphs — I would watch for AUC *dropping* on Cora/CiteSeer as the warning sign,
which would say the fusion (concat + projection) needs the structural features down-weighted relative to
the embeddings.
