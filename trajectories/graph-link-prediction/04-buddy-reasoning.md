The SEAL rung landed almost exactly where I predicted, and the precise shape of its result is what
tells me the next move. The full-layer normalization and the distance feature did stabilize Cora's AUC
(seeds now 92.2 / 92.3 / 93.0, mean 92.5 — tighter and higher than gcn_dot's 91.3), and ogbl-collab
Hits@50 climbed to 57.88, past gcn_dot's 53.74, confirming that giving the decoder a *separation* view
of the pair alongside the *similarity* view helps most on the dense large graph. But the small-graph
ranking metrics did *not* improve — they regressed: Cora MRR fell from gcn_dot's 31.18 to 27.49, Hits@20
from 70.27 to 61.16; CiteSeer MRR 40.84→35.22, Hits@20 73.70→72.09.

I want to autopsy that regression carefully, because SEAL predicted this trade and the per-seed numbers
say whether the predicted *mechanism* is the one that fired. On Cora the gcn_dot MRR seeds were
{39.45, 16.10, 38.00} — a wide spread with one collapsed floor at 16.10 — and SEAL's are
{23.46, 21.49, 37.52}. Read the two changes separately. The floor came *up*: the worst seed went from
16.10 to 21.49, and the spread tightened from a range of 23.4 to a range of 16.0. That is exactly the
stabilization the difference feature and full-BN were supposed to buy, and it happened. But the *ceiling
fell harder*: gcn_dot had two strong seeds near 38–39, SEAL has only one (37.52) and two dragged into the
low twenties, so the mean dropped from 31.18 to 27.49. That is precisely the magnitude-cost mechanism I
named when I introduced the full-layer BatchNorm — normalizing the final embedding to unit variance
deletes the absolute-magnitude lever the GCN was using to separate near-tied candidates at the very top
of the list, and on the small graphs that lever mattered more than the stability was worth. Hits@20 tells
the same story even louder: Cora fell nine points, 70.27→61.16, a top-of-list metric collapsing while AUC
rose. And the large graph shows the same fingerprint from the other side: ogbl-collab Hits@50 rose
53.74→57.88, yet its MRR actually *fell*, 13.79→10.58. SEAL pushed more true edges into the top-50 bucket
but made the very top of the ordering worse — Hits@50 is a coarser, more forgiving cutoff than MRR, so a
change that helps bulk top-50 membership while hurting the razor's-edge ordering registers as exactly this
split, up on Hits@50 and down on MRR. It is the magnitude-cost mechanism once more, now caught on the
metric most sensitive to it. So the trade was real and the mechanism was the predicted one: I bought floor stability and
cross-block scale-matching with magnitude resolution, and on the small graphs the bill came due at the
front of the ranking. SEAL is still *approximating* neighborhood overlap, and when I asked it to
discriminate the hardest pairs — the top of the MRR list — its "structural" signal, the
absolute-difference feature $|z_{\text{src}}-z_{\text{dst}}|$, is a function of the *learned embeddings*,
not of the graph. It had no measurement of the one quantity that actually decides links on citation and
collaboration graphs: how many neighbors do $u$ and $v$ literally share. So I am not going to add another
*learned* pairwise feature. I am going to compute the structural signal *exactly*, against the live
adjacency, and hand it to the decoder.

Let me be precise about the diagnosis before I commit, because this is a real expressiveness ceiling,
not a tuning issue. A message-passing GNN produces one embedding per node as a function of that node's
rooted neighborhood, so two destinations with the *same* rooted neighborhood receive the *same*
embedding and therefore the *same* score against any fixed source — even when one is a true neighbor and
the other is not. Let me make that concrete with the smallest example that shows it. Take a source $u$
whose neighbors are $\{a,b\}$, and two candidate destinations: $v_1$ with neighbors $\{a,b\}$ and $v_2$
with neighbors $\{c,d\}$, where $a,b,c,d$ all carry the same node features and the same degree. Then
$\mathrm{CN}(u,v_1)=|\{a,b\}\cap\{a,b\}|=2$ while $\mathrm{CN}(u,v_2)=|\{a,b\}\cap\{c,d\}|=0$ — $v_1$ is a
strong candidate, $v_2$ a weak one. But a GCN layer embeds each node as the normalized aggregate of its
neighbors' features, and since $v_1$ and $v_2$ both aggregate two same-feature, same-degree neighbors,
$z_{v_1}=z_{v_2}$ exactly. Every decoder in the ladder so far scores $(u,v)$ as a function of $(z_u,z_v)$
alone — dot product, MLP on the Hadamard block, MLP on the four-block SEAL features — so all of them
return $\text{score}(u,v_1)=\text{score}(u,v_2)$ and *cannot* rank the true neighbor above the
non-neighbor. This is not a training failure; it is a representational one. It is the same fact as the
well-known result that a plain GNN cannot count triangles — a triangle through $u,v,w$ is exactly a
common neighbor $w$ — so the common-neighbor / Adamic–Adar / resource-allocation family, the signal that
has dominated link prediction on these graphs for two decades, is precisely what every rung so far has
been *structurally unable to represent*. VGAE, gcn_dot, and SEAL all score a pair only through the
geometry of two independently-encoded points; none of them ever counts a shared neighbor. That is why
SEAL's MRR stalled. The explicit $\mathrm{CN}$ feature ($2$ versus $0$ in my toy) breaks the tie
instantly, which is the whole reason to inject it.

The toy is exact only because I made the neighbors feature-identical; on real Cora and CiteSeer, where
nodes carry distinct features, two candidate destinations rarely collide *perfectly*, so the failure is
not a hard tie but a soft one. That actually strengthens the case rather than weakening it. As two
destinations become close in the joint feature-and-structure space the GCN reads, their embeddings become
close *continuously*, and the decoder's margin for separating a true neighbor from a look-alike non-
neighbor shrinks toward zero — which is precisely the near-tie regime the MRR metric lives in, and
precisely where the ladder's ranking numbers have been fragile. The exact $\mathrm{CN}$ count does not
degrade continuously with embedding similarity: it is a discrete, graph-derived integer that stays $2$
versus $0$ no matter how close the two embeddings drift. So injecting it does not just fix a measure-zero
automorphic corner case; it hands the decoder a separator that is robust in exactly the soft-collision
region where the learned geometry is weakest.

So the structural insight I want is the one the SEAL lineage was reaching for but its harness-friendly
form could not deliver: score a pair by a *learned function of explicit neighborhood-overlap features*,
fused with the GNN embeddings. Concretely, for each candidate pair $(u, v)$ compute the three canonical
overlap heuristics directly from the adjacency: common neighbors $\mathrm{CN}(u,v)=|N(u)\cap N(v)|$,
Adamic–Adar $\mathrm{AA}(u,v)=\sum_{w\in N(u)\cap N(v)}1/\log\deg(w)$, and resource allocation
$\mathrm{RA}(u,v)=\sum_{w\in N(u)\cap N(v)}1/\deg(w)$. $\mathrm{CN}$ is the raw triangle count — the
$A[1,1]$ distance-label count, the backbone signal a GNN cannot produce. $\mathrm{AA}$ and $\mathrm{RA}$
are degree-discounted variants, and the discounting is worth making numerical because it is the reason to
carry all three. A shared neighbor that is a low-degree, specific node is strong evidence of a link; a
shared neighbor that is a high-degree hub (cited by everyone) is weak evidence. Take two shared
neighbors, one of degree $2$ and one of degree $1000$. To $\mathrm{CN}$ they are identical: each adds
$1$. To $\mathrm{AA}$ the hub is discounted to $1/\log 1000\approx0.145$ against roughly $1$ for the rare
one — about a sevenfold down-weight. To $\mathrm{RA}$ the hub is discounted to $1/1000=0.001$ against
$0.5$ — a five-hundredfold down-weight. So the three features span a spectrum of degree-sensitivity from
none ($\mathrm{CN}$) through mild ($\mathrm{AA}$) to aggressive ($\mathrm{RA}$), and handing all three to
the decoder lets the MLP learn *which* discounting the data prefers rather than committing to one formula
— the learned-heuristic move. There is one implementation nuance in the $\mathrm{AA}$ weight I should be
honest about: I compute it as $1/\max(\log\deg,1.0)$, clipping the log at $1.0$. The clip guards the
degenerate $1/\log 1=1/0$ and, since a common neighbor necessarily has degree $\ge 2$, in practice it
only flattens the discount for degree-2 shared neighbors ($\log 2\approx0.693$ rounded up to $1.0$, so
their weight becomes $1.0$ rather than $1.44$). It is a deliberate, mild harness-side approximation of the
exact $1/\log\deg$, not a change to the signal's meaning. These three features are pair-relative: they
depend on the joint neighborhood geometry of $u$ and $v$, not on either node alone, so they separate
exactly the automorphic-node links the embedding-only decoders could not.

That degree-discounting is not equally important everywhere, and thinking about where it bites sharpens
the ogbl-collab prediction. On the citation graphs, shared neighbors are mostly specific papers, and raw
$\mathrm{CN}$ already discriminates well because sharing a citation is a fairly specific event. On the
collaboration graph the story is different: prolific authors are hubs co-appearing in enormous numbers of
pairs, so raw $\mathrm{CN}$ is inflated by "both of these people once co-authored with the field's
superstar" — a weak signal wearing a strong number. $\mathrm{AA}$ and $\mathrm{RA}$ are exactly the
correction: they down-weight the superstar (by my arithmetic above, RA by a factor of hundreds) and let a
shared *specific* collaborator count for more. So I expect the MLP to lean hardest on the discounted
variants precisely on ogbl-collab, which is a second reason — beyond $\mathrm{CN}$ being unrepresentable
by the GNN at all — that the large-graph Hits@50 should be where the exact-overlap features pay off most.

Now the realization within this task's contract, and here I have to be honest about what the interface
lets me build versus the fullest version of this idea. The fullest version computes, for each pair, the
entire distance-label count table $A[d_u, d_v]$ over a $k$-hop window — $A[1,1]$ is $\mathrm{CN}$,
$A[1,2]/A[2,1]/A[2,2]$ the multi-hop overlaps — and, because computing those intersection cardinalities
exactly over huge neighborhoods is expensive, estimates them with set sketches: MinHash for the Jaccard
(intersection shape) and HyperLogLog for the cardinality (union size), propagated node-wise by
elementwise min/max so the per-edge cost is independent of graph size. That sketch machinery is what makes
the idea scale to millions of nodes. But I should reckon whether I actually *need* it here, and the
arithmetic says clearly not. This scaffold's interface gives me `decode(edge_label_index, z, edge_index)`
— the *original* node indices and the *live* adjacency — at decode time, on graphs of at most
${\sim}236\text{k}$ nodes with an average degree of order ten. Build the CSR adjacency from `edge_index`
(a sparse matrix with $|E|$ nonzeros, never the dense $N\times N$), slice the rows for the batch's
sources and destinations, take the elementwise product of those sparse rows to get the per-pair
common-neighbor indicator, then sum it ($\mathrm{CN}$), sum it weighted by $1/\log\deg$ ($\mathrm{AA}$),
and by $1/\deg$ ($\mathrm{RA}$). Each such per-pair intersection touches only the ${\sim}10$ nonzeros of
the two rows, so the whole batch costs $O(M\cdot\overline{\deg})$ and the working memory is the sum of the
sliced rows' nonzeros — nothing near the $236\text{k}^2\approx5.6\times10^{10}$ entries a dense adjacency
would demand. At this scale I do not need the sketches at all: I can compute $\mathrm{CN}/\mathrm{AA}/
\mathrm{RA}$ *exactly*, and exact is strictly tighter than a MinHash/HLL estimate. The sketches exist in
the full method to make the *same* counts survive when even a sparse exact intersection is too costly —
millions of nodes, dense neighborhoods — which is not the regime I am in. So the deliberate substitution
is: *exact* $\mathrm{CN}/\mathrm{AA}/\mathrm{RA}$ via scipy sparse, in place of *sketched* multi-distance
counts, keeping the load-bearing insight (explicit, pair-relative overlap counts as direct decoder
inputs) and dropping the sketching and the multi-hop count table that the harness neither needs at this
scale nor exposes a clean hook for. The harness already passes the correct adjacency at each phase —
train-only during validation, train+val at test, exactly as the OGB protocol prescribes — so the overlap
counts are computed against the right graph automatically, including folding the validation edges into
the test-time adjacency on ogbl-collab.

I should also close off the alternative of pushing the counting *into the encoder*, because it is the
obvious "more principled" move and it fails on arithmetic. If I wanted the GNN itself to count triangles,
a plain message-passing stack cannot — that is the very ceiling I diagnosed — so I would need a
higher-order architecture: a 2-FWL-style GNN with $O(N^2)$ state and $O(N^3)$ update, or a subgraph GNN
running a network per rooted subgraph. On $236\text{k}$ nodes an $O(N^2)$ state is $5.6\times10^{10}$
entries — the same wall the dense adjacency hit — and neither construction is even expressible through the
`GCNConv`-only encode contract, let alone inside the startup parameter budget. So the encoder route is
both infeasible and out of contract, while injecting the exact count at decode is cheap and *provably
sufficient*: it simply hands the decoder the exact quantity the higher-order GNN would laboriously
recompute. The exactness makes the expensive machinery pointless, which is the whole reason the
substitution is a simplification rather than a compromise.

The fusion is the other design point, and it has a scale subtlety worth deriving. The fullest version
combines the structural counts with the *Hadamard product* of the propagated node features, edge-pooling
style. The cleaner thing to do inside this decoder, given I already have a good GNN encoder, is to project
the three overlap counts up to the hidden width with a linear layer and *concatenate* them with the two
raw node embeddings: $h = [z_{\text{src}}\,\|\,z_{\text{dst}}\,\|\,\text{proj}(\mathrm{CN},\mathrm{AA},
\mathrm{RA})]$, then an MLP $3H\to H\to H\to 1$. Why project $3\to H$ rather than concatenate three raw
scalars onto the $2H=512$-dimensional embedding pair? Because three raw columns among $515$ would be
drowned: at initialization the first MLP layer weights every input dimension comparably, so the structural
signal would enter at ${\sim}3/515\approx0.6\%$ of the input variance and the optimizer would have to
fight uphill to attend to it. Lifting $[\mathrm{CN},\mathrm{AA},\mathrm{RA}]$ through a learned
$3\to256$ projection gives it $256$ of the $768$ input dimensions — a full third of the pair
representation — so the MLP can actually model the interaction between "who the nodes are" and "how much
they overlap." Concatenation rather than Hadamard here is a deliberate, harness-matched choice: the
structural counts are a *different kind* of quantity than the embeddings — integer-ish overlap magnitudes,
not learned coordinates — so forcing them through a product with the embeddings would be semantically
odd, whereas concatenation lets the MLP condition freely on both. The overlap computation runs under
`no_grad` — these are fixed structural measurements, deterministic functions of an adjacency that is not a
learned parameter, so there is no gradient to flow through the sparse ops and wrapping them saves the
memory of tracking them — while the projection and MLP are trained end-to-end. The encoder stays the GCN
stack with BatchNorm on all layers (as in SEAL), since the embeddings now sit beside projected counts and
benefit from the same scale control.

So the finale edit lands in the editable region as a `StructuralFeatureComputer` (a `no_grad` scipy-
sparse routine returning the $[\mathrm{CN},\mathrm{AA},\mathrm{RA}]$ per-pair feature) plus a
`LinkPredictor` whose `encode` is the GCN stack and whose `decode` computes the structural features
against the passed `edge_index`, projects them, concatenates with the two embeddings, and runs the MLP.
A shape pass: the sparse routine returns $\mathbb{R}^{M\times3}$, `struct_proj` lifts it to
$\mathbb{R}^{M\times H}$, the gathered endpoints are $\mathbb{R}^{M\times H}$ each, the concatenation is
$\mathbb{R}^{M\times3H}$, and the MLP returns the $M$-vector the loop's BCE consumes. The encoder caches
the encode-time `edge_index` and node count so `decode` has a sensible default adjacency when the loop
does not pass one explicitly. One correctness point in that resolution is load-bearing and I want to check
it, because it is exactly the kind of subtlety that silently leaks labels. `decode` uses the `edge_index`
the loop *hands it*, falling back to the cached encode-time adjacency only when none is passed. That
ordering is what keeps the overlap counts phase-correct: at validation the loop passes the train-only
graph, at test the train+val graph, so the counts are always taken against the adjacency the OGB protocol
specifies, and a test edge never contributes to its own $\mathrm{CN}$ feature. If I had instead always
used the cached encode-time adjacency, the validation-time counts could include edges the model is being
asked to predict — a leak that would inflate the metrics and invalidate the comparison to the baselines.
Honoring the passed `edge_index` over the cache is the small line that prevents it. Nothing else changes —
same loop, same BCE, same negative sampling. I am
explicit that this is the exact-overlap, sketch-free realization: no MinHash, no HyperLogLog, no
multi-distance count table — the scale here makes exact $\mathrm{CN}/\mathrm{AA}/\mathrm{RA}$ both
feasible and tighter than estimates.

What is the bar, and what would I validate? There is no leaderboard row for this method on this task;
the bar is the strongest baseline, SEAL — Cora (AUC 92.5 / MRR 27.5 / Hits@20 61.2), CiteSeer (92.9 /
35.2 / 72.1), ogbl-collab (Hits@50 57.88 / MRR 10.58). The falsifiable claims follow directly from the
diagnosis. First and sharpest: the small-graph *ranking* metrics that SEAL regressed should recover and
exceed, because the explicit overlap counts are exactly the top-of-list signal SEAL lacked — I expect
Cora MRR and Hits@20 to climb back above gcn_dot's 31.2 / 70.3 (so clear of SEAL's 27.5 / 61.2), and
CiteSeer likewise. Second: ogbl-collab Hits@50 should pass SEAL's 57.88, into the low-to-mid sixties,
because $\mathrm{CN}/\mathrm{AA}/\mathrm{RA}$ are *the* dominant signal on a dense collaboration graph and
SEAL only approximated them — this is the cleanest single-number verdict. And since ogbl-collab MRR is the
metric that SEAL's magnitude-cost actually *hurt* (13.79→10.58), the exact top-of-list signal should pull
it back up too; I expect ogbl MRR to recover past gcn_dot's 13.79, though as the secondary metric there I
weight the Hits@50 verdict more heavily. Third, the consistency check:
AUC should hold around the low-to-mid nineties (it was never the failing metric), and if it holds *while*
MRR/Hits@20/Hits@50 all rise, the whole trajectory's thesis is confirmed — that the ceiling on every
embedding-only rung was the inability to count shared neighbors, and that handing the decoder the exact
overlap removes it. The way this could fail: if the exact $\mathrm{CN}/\mathrm{AA}/\mathrm{RA}$ features
dominate the MLP so strongly that it ignores the node features, the model could collapse toward a pure
heuristic and lose the feature-driven gains on the feature-rich citation graphs — I would watch for AUC
*dropping* on Cora/CiteSeer as the warning sign, which would say the fusion (concat + projection) needs
the structural features down-weighted relative to the embeddings.
