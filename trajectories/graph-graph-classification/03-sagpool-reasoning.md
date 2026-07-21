The injective-sum readout came back and confirmed the diagnosis cleanly, which tells me both what it
fixed and what it left on the table. The *size* of each move against DiffPool is the mechanism talking.
PROTEINS moved from 70.98 to 74.54, a lift of 3.56 — the largest of the three, and exactly the dataset I
singled out as the diffuse-mean victim, larger graphs where a real hierarchy would help most and where
DiffPool got the least. So the biggest gain landed where the biggest loss had been. NCI1 rose from 78.25
to 79.52, a smaller lift of 1.27, and *robustly* — {78.13, 80.49, 79.93}, tight — fitting my reading that
DiffPool was already converging stably on the large balanced set, just to a mean-like answer, so the
injective sum bought less headroom but bought it without cost. MUTAG came in at 84.02, up 2.65, its seed
spread tightened to {82.98, 86.14, 82.95} — range 3.19 against DiffPool's 6.93, essentially halved: the
determinism removed the coin-flip, so the residual variance is now just the irreducible noise of 188
graphs.

The two-metric picture corroborates all of this. DiffPool's acc-minus-F1 gaps were ordered
MUTAG 4.54 > PROTEINS 2.11 > NCI1 0.04; gin-sum's are MUTAG 84.02 − 81.91 = 2.11, PROTEINS 74.54 − 73.03 =
1.51, NCI1 79.52 − 79.39 = 0.13. The gap narrowed on the two imbalanced sets — MUTAG from 4.54 down to
2.11, more than halved — which is the injective sum serving the minority class better than a majority-leaning
mean did: keeping multiplicities lets the classifier see the rare-class node-counts a mean had washed out.
So every prediction I made last time landed: PROTEINS cleared 70.98 decisively, NCI1 cleared
78.25 robustly, MUTAG variance tightened off its six-point range, and the F1 gap closed on the imbalanced
sets. gin-sum is a genuinely solid, robust readout — it wins all three datasets over DiffPool and is steady
across seeds.

I want the bar precise before I try to beat it. The NCI1 seeds span just 2.36 points with macro_f1
tracking almost exactly ({77.97, 80.36, 79.83}) — on the largest, hardest set the readout is essentially
seed-independent. Its cross-dataset spread (best minus worst dataset mean) is 84.02 − 74.54 = 9.48, and
that spread comes from the datasets genuinely differing in difficulty, not from any one being a
liability: a readout with no soft spot, not spectacular anywhere but not falling down anywhere. So the
honest framing is not "gin-sum is broken" but "gin-sum is uniform to a fault" — it applies the identical
equal-weight rule to a tiny molecule whose label is one nitro group and to a large compound whose label
is its whole scaffold, and on the former that uniformity is leaving accuracy on the table.

But "solid and robust" is also its ceiling, and I can see the exact shape of that ceiling in the numbers.
The readout sums *every* node with *equal weight* at every layer. Look at where it is weakest: MUTAG at
84.02, the set where the discriminative substructure — a mutagenic functional group, a particular
nitro-aromatic ring — is small and local, a handful of atoms out of eighteen. A uniform sum over all
eighteen atoms *dilutes* that handful across the whole molecule; the few decisive nodes are drowned in the
many irrelevant ones, and the readout has no mechanism to say "these three atoms are the answer and the
other fifteen are scaffolding." The same dilution logic applies to PROTEINS, where the enzyme/non-enzyme
label often turns on a specific binding motif rather than the bulk of the residues. NCI1 is the one place
this is *not* obviously the bottleneck — there the decision is more distributed across the molecule, so a
uniform sum's "count everything equally" is closer to the right thing, which is consistent with NCI1 being
gin-sum's *most* robust win and the place the injective-sum argument bit cleanest. So the ceiling is
specifically a *motif-dilution* ceiling: gin-sum leaves the per-node-importance signal unused, and it costs
the most on the motif-driven sets where the label is carried by a few substructures.

So the next move is to make the readout *selective* — let the model decide which nodes carry the label
and pool preferentially over those. Two families do this. One keeps every node but *weights* it (soft
attention, a learned weighted sum): safer, nothing discarded, but it only re-weights the dilution, since
a graph dominated by uninformative nodes can still have its decisive signal diluted. The other *drops*
the unimportant nodes outright and reads out over the survivors — hierarchical, hard top-k. This actually
changes the regime: by removing nodes it builds a true hierarchy, the structured pooling DiffPool reached
for but realized through *selection* rather than the soft assignment that left DiffPool diffuse, and a
top-k can be sharp without an entropy auxiliary because it is one-hot by construction. That is the more
ambitious correction, and its failure would teach me something the soft route cannot, so I derive the
hard route.

The mechanism I need has two pieces: a way to *score* each node's importance, and a way to *select and
coarsen* using those scores. Start with scoring. The score must depend on both a node's features and its
position in the graph, because importance is structural, not just featural — a node is decisive partly
because of what it is connected to, and two atoms with identical features can matter differently depending
on their neighborhoods. The natural object that turns (features, adjacency) into a per-node scalar is a
single graph-convolution layer that outputs *one* channel: Z = σ(GNN(X, A)), one self-attention score per
node, computed from the node and its neighborhood. Using a graph convolution rather than a plain MLP is the
load-bearing choice — it is what lets the score "consider both node features and graph topology," so the
importance of a node is informed by who it is wired to, not by its features in isolation. This is exactly
the structural awareness DiffPool's adjacency-blind assignment MLP lacked, and part of why DiffPool stayed
feature-arbitrary and diffuse; here the adjacency enters the score directly. The score is squashed by a
bounded nonlinearity so it acts as a gate.

Now selection. Given a score per node, keep the top fraction: with retain ratio k, select the ⌈k·N⌉
highest-scoring nodes and discard the rest — a hard top-k. This is where the method differs sharply from
both predecessors, and the difference is worth stating precisely. DiffPool kept *all* nodes softly
redistributed into K fixed clusters — a learned assignment matrix, K a fixed hyperparameter independent of
graph size, which is exactly what mis-sized on MUTAG when K = 25 exceeded the node count. Top-k instead
keeps a *size-proportional* subset of the *original* nodes: no new cluster-nodes, the survivors are real
atoms, and the coarsened graph scales with the input, so there is no N_max padding and no fixed cluster
count to mis-size against a tiny graph. That alone fixes one of DiffPool's structural mistakes for free.
The selection is differentiable in a specific way, and getting it wrong would leave the score net
untrained. The top-k *index* is a discrete argmax with no gradient — merely gathering the survivors'
features would cut the score convolution off from the loss. The trick is to *gate* the survivors by their
own scores: the retained features become X_idx ⊙ Z_idx, each scaled by its own attention value. Why that
one Hadamard product rescues the gradient: suppose two nodes survive with scores z₁ and z₂ and embeddings
x₁, x₂, and the readout at this level is a sum, so its contribution is z₁x₁ + z₂x₂. The derivative of the
loss with respect to z₁ is L'·x₁ — nonzero, flowing straight back into the score convolution that produced
z₁. Contrast the un-gated version, where the contribution would be x₁ + x₂ with the scores appearing *only*
inside the top-k comparison that chose the survivors; there the loss has no continuous dependence on z₁ at
all — a small nudge to z₁ leaves the same two nodes selected and the same x₁ + x₂ summed — so ∂L/∂z₁ = 0 and
the score net never learns which nodes to keep. So the gating is the only thing making "this node
mattered" differentiable while *which* nodes survive stays a discrete choice. After selecting, filter the
edges to the induced subgraph on the survivors — keep (u, v) only when both endpoints survive, relabeled
into the compacted indexing. On a 4-node path 1–2–3–4 with node 3 dropped, the subgraph on {1, 2, 4}
keeps (1,2), drops (2,3) and (3,4), leaving node 4 isolated but still visible to the next level's readout.

That gives one coarsening level; stacking it and reading out at each level lets the graph vector see every
scale — the same multi-scale instinct gin-sum got from jumping knowledge across *layers*, here across
*coarsening levels*. Three scales: level 0 the original graph, level 1 after one top-k coarsening, level 2
after a second. At ratio 0.5 each level keeps ⌈N/2⌉ nodes, so N → ⌈N/2⌉ → ⌈N/4⌉: a MUTAG molecule of ~18
atoms becomes ~9 then ~5; a ~30-atom NCI1 compound becomes ~15 then ~8. By level 2 only about a quarter
survive — a lot of irreversible discarding, and this is where I expect the method to break.

At each level I summarize the coarsened node set with a *concatenation of sum-pool and mean-pool*. Sum
keeps the count information the injective argument says matters — I do not want to throw the multiplicities
away at the readout. But sum alone across levels has a scale problem the hierarchy itself creates: level 0
sums over the full ~30 atoms of an NCI1 compound, level 2 over ~8 survivors, so the two sum channels
differ in magnitude by ~4× purely from node count — the same O(N) mismatch that forced the per-layer
BatchNorm before, now across coarsening levels. The mean is the scale-stable companion: it divides the
count out, comparable at every level. So sum carries counts, mean a level-invariant summary, and the
classifier decides per level which it trusts. Each level gives [Σ, μ] of width 2H; three levels give
6H = 384, and a single `Linear(6H, H)` projection compresses back to hidden_dim.

Budget: the projection `Linear(6H, H)` is 384·64 + 64 = 24,640 params; the two score convolutions
(single graph-conv, one output channel, ~2H each) add only a few hundred. So ≈ 25,000 total, inside the
10·H² + 9·H = 41,536 headroom, with the classifier's first layer staying at its cheap 4,160 since the
output is only hidden_dim.

Unlike DiffPool, this fits the surface without stripping: `GraphReadout` receives `edge_index` and
`batch`, exactly what a top-k pool needs (the score conv needs the adjacency, the pooling needs `batch`).
Two top-k modules at ratio 0.5; `forward` reads out level 0 on `x`, pools to (x1, edge_index1, batch1),
reads level 1, pools again, reads level 2, concatenates the three [Σ, μ] blocks, projects. The score conv
operates on the final-layer embeddings `x`, not raw atom types — arguably better, since the GIN embeddings
already encode K-hop structure, so scoring them asks "given everything message passing computed, which
node is decisive." Permutation invariance holds: the score conv is node-equivariant, top-k-by-score is
equivariant, the [Σ, μ] readouts are invariant.

Now what should this do, falsifiably in advance? Where the label is a small local substructure — MUTAG's
mutagenic group, a PROTEINS binding motif — selection should *help*: dropping the irrelevant nodes
concentrates the readout on the decisive ones and undoes the dilution that capped MUTAG at 84.02. So I
expect MUTAG and PROTEINS to rise above gin-sum's 84.02 and 74.54, possibly by a lot on MUTAG. But the
same hard selection should *hurt* where the decision is *distributed*. NCI1 is the worst case: activity
often depends on the whole molecular context, and ratio-0.5-twice keeps only a quarter of the atoms; a
mis-calibrated single-conv score, with no auxiliary to guarantee the right nodes survive, can discard
exactly the atoms that mattered, irreversibly. gin-sum won NCI1 at 79.52 *because* it threw nothing away;
this throws half away, twice. The damage should show not just in the mean but in the variance and the
acc-minus-F1 gap — scattered NCI1 seeds if the score keeps the right quarter only sometimes, and macro_f1
sagging below accuracy if the discard drops minority-class atoms, where gin-sum's gap was near-zero 0.13.

So the honest prediction is a *split*: beat gin-sum on the motif-driven MUTAG and PROTEINS, regress
sharply and with high seed variance on the distributed NCI1. If NCI1 instead holds at or above 79.5, the
"hard selection destroys distributed signal" story is wrong and selection is simply better everywhere. But
I expect the split — and a severe NCI1 regression names the next problem: a readout that gains selectivity
on local motifs but loses robustness on graphs whose decision is spread across the node set, because it
commits to an irreversible discard. The aggregate may still edge above gin-sum on a big MUTAG jump, but it
would be a worse worst-case dressed up as a better average.
