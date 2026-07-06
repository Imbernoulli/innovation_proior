The readout is the whole job, and the floor I have to climb off is the simplest thing that turns node
vectors into a graph vector and works at all. The scaffold default does that with one line —
`global_add_pool(x, batch)` — and I could just run it. But before I accept the flattest possible readout
as my starting rung, let me ask whether there is a more *structured* default that is still simple enough
to be a baseline, because the structure I add at the bottom is the structure every later rung will
either keep or argue against. The thing that nags at me about a single global sum is that it is *flat*.
Stare at what message passing actually does: it moves information along edges, K hops out, and stops.
Every node ends up holding a summary of its K-hop neighborhood, and then a global sum dumps all N of
those summaries into one pot and stirs. At no point does the model ever represent anything at a scale
*between* "one node's neighborhood" and "the entire graph." And the graphs in front of me are not flat:
a molecule is atoms → bonds → functional groups → the whole molecule, and the label I want — is this
compound mutagenic, is this protein an enzyme — often lives at an intermediate, functional-group scale
that a sum over atom embeddings has thrown away before the classifier ever sees it.

Where has this been solved before? Image CNNs. What makes a deep CNN powerful is not just convolution
but convolution *interleaved with pooling*: conv, downsample to a coarser grid, conv on the coarser
grid, downsample again. Each pooling step shrinks resolution so deeper layers see more global structure
with larger receptive fields, while early layers keep fine detail. That alternation is what builds a
hierarchy. My GIN backbone has the convolution half — message passing *is* graph convolution — but the
readout, as a single global sum, has no pooling half at all. So the structured default I want, stated
abstractly, is a *graph analogue of spatial pooling*: a module that takes a graph and produces a
*smaller* graph — fewer nodes, a coarser adjacency — so that, in spirit, GNN-then-pool can be stacked
and the readout can see bigger structures than a single node neighborhood.

Let me try the naive transcription of CNN pooling and watch it break, because the way it breaks tells me
what the real difficulty is. In a CNN, pooling takes a 2×2 patch and outputs one pixel; the patch is
well-defined because the image is a grid — every pixel has the same neighborhood and "2×2 block" means
the same thing everywhere. So I want to pool "a patch" of the graph. But there is no grid. Node 5 has
three neighbors arranged one way, node 6 has eleven arranged another; there is no canonical local
window, no "the block to the upper-left." If I tried to fix that by ordering the nodes into a sequence
and pooling consecutive runs — the sort-then-1D-conv route — I would need a canonical, structure-
preserving node ordering, and finding one is essentially graph isomorphism; nodes far apart in the graph
would land adjacent in my sequence and I would be pooling together things that have nothing to do with
each other. Dead end. The spatial locality CNN pooling leans on simply does not exist on a graph. And
there is a second constraint the CNN analogy hides: my graphs have different numbers of nodes, so the
pooling operator cannot be tied to a fixed grid size — it has to be one rule that applies uniformly
whether a molecule has seventeen atoms or a hundred and ten.

So what *is* a graph pooling operator, mechanically, once I drop the patch idea? The honest abstraction:
I have N nodes with embeddings `x` and an adjacency, and I want to output a coarser graph with K < N
nodes — cluster-node features and, in the full version, an adjacency between clusters. The new "nodes"
are *clusters* of old nodes. So pooling on a graph reduces to "cluster the nodes, treat each cluster as
a coarse node, aggregate within each cluster." The whole problem is how to cluster and how to aggregate.

There is an off-the-shelf answer to "cluster the nodes": run a deterministic graph-clustering algorithm
— spectral clustering, a coarsening heuristic — get a partition, pool over it. So why not? Two problems
that are really one. First, the clustering is computed *before* any gradient flows, by a subroutine that
knows nothing about whether grouping these particular atoms helps predict the molecule's label; the
partition that minimizes a spectral cut is not the partition that minimizes my classification loss.
Second, it is a per-graph black box, so there is no *learned, shared* pooling strategy that transfers
across the thousands of graphs in NCI1 the way the backbone weights transfer. What I want is the
clustering itself to be *learned*, end-to-end, by shared parameters that apply across graphs.

Learned clustering, differentiable. The instant I say "differentiable" the obstacle appears: a
clustering is a *hard* assignment — node i goes to cluster j, a discrete choice — and discrete choices
have no gradient. I cannot backprop through "argmax which cluster." The standard escape is to *soften*
it: instead of assigning each node to one cluster, give it a *distribution over clusters*. Define an
assignment matrix S whose entries are the soft probabilities relating nodes to clusters, made a genuine
distribution by a softmax. Now the assignment is a smooth function of parameters and gradients flow.
Given S and node features Z, the embedding of cluster j is the soft collection of the nodes assigned to
it, node i contributing in proportion to S_{ij} — a weighted sum, which stacked over clusters is exactly
the matrix product **X' = Sᵀ Z**. Row j of X' is cluster j's pooled embedding. That is the coarse-graph
readout, and it falls right out of treating S as a soft node↔cluster indicator.

I should check this respects permutations, because it would be embarrassing to break the one invariant I
must keep. Relabel the original nodes by a permutation P: features ↦ PZ and, if the assignment network
is equivariant (it is a per-node map, so it commutes with row permutation), scores and hence S ↦ PS.
Then X' = (PS)ᵀ(PZ) = Sᵀ PᵀP Z = Sᵀ Z, because P is a permutation so PᵀP = I. The coarsened embedding is
invariant to node numbering. Good — that is the property a graph-level answer needs to be well-defined,
and it holds regardless of *which* axis of S I choose to normalize, as long as the normalization is
itself node-equivariant.

That choice of axis is not cosmetic, and I want to make it deliberately rather than inherit it, because
it decides the whole character of this bottom rung. There are two ways to softmax S. I can normalize
over *clusters* — each node's row sums to one, so every node distributes its unit of mass across the K
clusters, the canonical row-stochastic assignment. Or I can normalize over *nodes* — each cluster's
column sums to one, so every cluster is a normalized mixture of the real nodes, a proper weighted
average of node embeddings. The second reading is the cleaner one to write against the dense tensor I
will build (`softmax` over the node axis of an `[B, N_max, K]` score block), and it has a consequence I
should trace all the way out before I commit, because it will tell me exactly how much this rung can
possibly deliver.

Take the node-normalized version and work a tiny case by hand: two nodes with embeddings x₁, x₂, two
clusters, score matrix with columns softmaxed over the two nodes. Column j gives weights S_{1j}, S_{2j}
with S_{1j}+S_{2j}=1, so cluster j's pooled embedding is X'_j = S_{1j}x₁ + S_{2j}x₂ — a convex
combination of the two nodes, literally a weighted average. Now the last step of the readout collapses
the K cluster-nodes to one graph vector by a plain **mean over clusters**. So the output is
(1/K)·Σ_j X'_j = (1/K)·Σ_j Σ_i S_{ij} x_i = Σ_i ( (1/K)·Σ_j S_{ij} ) x_i = Σ_i w_i x_i with
w_i = (1/K)·Σ_j S_{ij}. And those node-weights sum to exactly one: Σ_i w_i = (1/K)·Σ_j Σ_i S_{ij} =
(1/K)·Σ_j 1 = 1, since each column summed to one. So the entire pool, after all the machinery, is
*provably a convex combination of the node embeddings* — a generalized, data-dependent **weighted global
mean**, never anything else. When the assignment softmax stays diffuse (every entry ≈ 1/N) the weights
flatten to w_i ≈ 1/N and it is exactly the plain global mean. This is not a hand-wave that it is
"mean-like"; it is an identity, and it means the ceiling of this readout is a weighted mean.

A weighted mean has the same fatal blind spot as a plain mean, and I can verify that too. Inflate the
multiset — duplicate every node k times, feeding identical copies with identical scores. Each duplicate
gets weight proportional to its softmax score, and because the copies share a score the total weight
landing on any distinct value stays exactly what it was before duplication (the normalizer grows by the
same factor k it multiplies in). So the weighted-mean output is *invariant to multiset inflation*: it
cannot distinguish {a, b} from {a, a, b, b}. It captures proportions and throws counts away. That is the
precise defect I am building in at the bottom, on purpose — and it is exactly what a plain *sum* would
have kept, since a sum over the inflated set is k times the original.

I should convince myself the *other* axis is not secretly better, because it would be lazy to pick the
node-normalized softmax without checking what row-normalization would have done. Row-normalize (each node
distributes its unit of mass over the K clusters, Σ_j S_{ij} = 1), then run the same pool-then-mean.
The output is (1/K)·Σ_j Σ_i S_{ij} x_i = (1/K)·Σ_i (Σ_j S_{ij}) x_i = (1/K)·Σ_i x_i, because each row of S
sums to one — a *constant multiple of the global sum*, and, tellingly, *completely independent of S*. The
assignment matrix cancels out of the output entirely; its gradient through this readout is zero, so the
MLP that produces it would never train. I checked this on a random 6-node case and swapping in a totally
different assignment left the output bit-for-bit identical. So row-softmax gives me a scaled sum with a
dead, un-learnable clustering — a degenerate readout that only *pretends* to pool. Node-softmax gives me
a live, learned convex combination that at least exercises the assignment, at the cost of being a
weighted mean. Neither is good, but only one keeps the learned pooling actually learning, so I take the
node axis with eyes open: I want the ambitious idea (a *learned* soft clustering) to be genuinely under
test at this rung, not silently short-circuited into a constant.

Where does S come from? A small network over the node embeddings: a per-node output of width K, softmaxed
over the node axis. Here I make the one decision that separates *this task's* DiffPool from the canonical
one, and I make it by reading the constraints I actually face. The full method generates the assignment
with its own *GNN over the current adjacency* (so a node's cluster is informed by its neighborhood),
generates the embeddings with a *second, separate* GNN, and — crucially — adds two auxiliary losses to
train the pooling: a link-prediction term ‖A − S Sᵀ‖_F that forces connected nodes into the same cluster,
and an entropy term that forces each assignment row toward one-hot so clusters are crisp. Those
auxiliaries exist because the task gradient alone, routed back through the coarsening and a non-convex
factorization-like clustering, is too weak to find good clusters on its own. But my editable surface is
*only* the `GraphReadout` class: it receives node embeddings from a fixed GIN backbone and returns one
graph vector. It is downstream of the message passing — I cannot insert a clustering GNN into the
backbone, I cannot feed a coarsened adjacency back into more convolution, and I have no clean hook to add
auxiliary loss terms to the fixed training step, which only backpropagates the classification
cross-entropy. So the version that fits here is the *stripped* one: a single coarsening level, the
assignment produced by a plain MLP over node features (not a GNN over the adjacency), and **no link or
entropy auxiliary** — only the classification gradient shapes S. I am keeping DiffPool's load-bearing
idea (a learned soft assignment that pools X' = Sᵀ Z) and dropping the machinery the harness will not let
me wire in.

Before I fix sizes, let me spend the parameter budget honestly, because I want to know whether the
constraint binds. The allowance is roughly 10·H² + 9·H extra parameters on top of the fixed
backbone/classifier, and at H = 64 that is 10·4096 + 576 = 41,536 parameters of headroom. My assignment
MLP is `Linear(D, D) → ReLU → Linear(D, K)`: the first layer is 64·64 + 64 = 4,160, the second is
64·25 + 25 = 1,625, so 5,785 parameters total — about one-seventh of the headroom. The budget does not
bind here at all; if anything it tells me the *stripped* readout is under-spending, which is consistent
with its being the weakest rung. Output width is `hidden_dim`, so the fixed classifier head — whose first
layer is `Linear(output_dim, 64)` — consumes it unchanged with no extra cost from a wide readout.

Now the two sizes I have to pick, K clusters and N_max padding. I fix K = 25 clusters and pad to
N_max = 150, sizes comfortably above the per-graph node counts in these datasets, so no real graph
overflows the padding and every graph gets the same number of cluster slots regardless of its size. But
K = 25 is worth a second look against the smallest set, because it exposes how little this can do there.
MUTAG's molecules are tiny — on the order of a dozen-and-a-half atoms each, often fewer than K = 25
nodes. When the cluster count *exceeds* the node count, the column-softmax over ~18 nodes has to fill 25
cluster slots, several of which are near-redundant averages of the same handful of atoms; the assignment
is forced diffuse by sheer arithmetic before training ever gets a say. So the very degeneration I proved
above — the pool collapsing toward a plain mean — is *most* forced exactly on MUTAG, where K > N makes
crisp, well-separated clusters impossible. On PROTEINS and NCI1, with larger graphs, K = 25 is a genuine
compression and the softmax at least has room to be selective, but with no structural prior it has no
reason to be.

Made concrete in the scaffold's vocabulary: the readout converts the batch's stacked node embeddings to
dense, padded form with `to_dense_batch(x, batch)` → `[B, N_max, D]` plus a validity mask, computes raw
assignment scores with the 2-layer MLP → `[B, N_max, K]`, masks padded nodes to −∞ before the softmax
(so phantom padding nodes get zero assignment), softmaxes *over the nodes* so each cluster is a
normalized mixture of real nodes, re-zeroes the padded rows, and pools `X' = Sᵀ X_dense` → `[B, K, D]`.
The last collapse to one graph vector is the mean over the K clusters, `[B, D]`. One honest note about
the code: I also materialize the dense adjacency with `to_dense_adj` and could form the coarsened
adjacency `Sᵀ A S`, exactly as the canonical method does before its *next* convolution — but there is no
next convolution here, the readout returns a graph vector, so that adjacency is computed and never
consumed. I leave the line in for parity with the full method's coarsening step while being clear that
only the embedding pool `Sᵀ X` feeds the output. The dense pipeline's cost is modest: with N_max = 150
the adjacency is 150² ≈ 22.5k entries per graph, trivial at these batch sizes, and the assignment and
pool are two small batched matmuls. The full scaffold module is in the answer.

Now let me reason about what this stripped DiffPool must do, because that is the whole point of starting
here, and I want my expectations falsifiable. I have already proved the ceiling — a convex-combination
pool, a learned weighted mean, count-blind by construction — and the single thing I kept is also the
single thing the auxiliaries were there to *protect*: I removed the protection. Without the link loss
there is nothing telling connected nodes to share a cluster; the MLP assigns from node *features alone*,
blind to the adjacency, so "clusters" need not correspond to any graph structure at all. Without the
entropy loss there is nothing stopping the softmax from staying diffuse, in which case the weighted mean
slides all the way to the plain global mean I derived as its diffuse limit. So my honest prediction is
that this readout behaves, at best, like a *learned, noisier global mean* — and a mean, as the readout
lineage already warned, throws away counts and is strictly weaker than a sum on the multiset of node
features. The risk is not that it crashes; it is that the learned soft clustering, pulled only by a
distant classification gradient through a non-convex assignment, settles into something diffuse or
feature-arbitrary and pools *less* informatively than the trivial sum the scaffold ships with. That is
exactly why it is the bottom rung: it carries the most ambitious idea on the ladder — a *learned*
hierarchy — but in the one form the harness permits, shorn of the auxiliaries that make the idea pay, so
it should be the most likely to underperform a simple injective reduction.

Where is this falsifiable against the three datasets, and on which of the two metrics? On tiny MUTAG the
K > N arithmetic and the 188-graph sample together predict a *coin flip*: the soft assignment has almost
no data to fit and is forced diffuse, so I expect the cross-validation test_acc to swing several points
across seeds {42, 123, 456} — this is variance, not signal. On PROTEINS and NCI1, with more graphs, the
MLP gets more gradient, and here I expect the opposite tell: because the pool collapses to nearly the
same diffuse weighted mean every run, the *seed spread should be tight* — a low-variance band is the
fingerprint of a readout that is not learning a seed-distinguishing partition but converging to one flat
answer. PROTEINS, with the larger graphs where a real hierarchy would help most, is the cleanest place to
read this: if the stripped pool gets the *least* out of the graphs that should reward hierarchy the most,
that is the diagnosis confirming itself. The macro_f1 column is the honest cross-check on the imbalanced
sets: a count-blind mean that leans on the majority class should show macro_f1 sitting below test_acc by
a few points on the more skewed datasets, while on a balanced set the two should nearly coincide.
Whatever the precise split, the diagnosis is already pointed at the next rung: the weakness here is that
I *discarded structure and counts* — a provably-convex-combination pool with no injective reduction and
no use of the per-layer (`layer_outputs`) information the scaffold hands me. The cleanest next move is to
stop trying to learn a hierarchy with no support for it and instead take the most expressive *flat*
reduction there is — a sum, which keeps the counts a mean loses — and read it from *every* layer rather
than one, turning the unused `layer_outputs` into the multi-scale signal this hierarchy attempt was
reaching for but could not reach.

The causal chain in one breath: a single global sum is flat and represents no scale between a node's
neighborhood and the whole graph, so the structured floor is a graph analogue of CNN pooling → that
means clustering nodes into a coarser graph, hard clustering kills the gradient, so soften it into a
node-normalized assignment S = softmax(scores) and pool X' = Sᵀ Z (permutation-invariant via PᵀP = I) →
but that node-normalized pool followed by a mean over clusters is provably a convex combination of the
nodes, Σᵢ wᵢ xᵢ with Σwᵢ = 1, i.e. a learned weighted mean that is invariant to multiset inflation and so
count-blind → and my editable slot is only the post-backbone readout, so I cannot wire in the canonical
method's pooling-GNN or its link/entropy auxiliaries; I keep the learned soft assignment (a plain MLP,
5,785 params, well inside the 41,536 headroom, one level, mean over clusters) and drop the protection →
which means, with no structural prior and no anti-diffuse term, and with K = 25 exceeding the node count
on tiny MUTAG, the pool likely behaves like a learned, noisier global mean that discards counts → so
this is the bottom rung by construction, and its failure points straight at the next: recover the counts
with an injective *sum* and read it from every layer.
