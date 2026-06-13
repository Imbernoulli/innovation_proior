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
whether a molecule has 17 atoms or 110.

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
assignment matrix S whose row i is node i and whose entries are the soft probabilities that node i
belongs to each cluster, made a genuine distribution by a row-wise softmax. Now the assignment is a
smooth function of parameters and gradients flow. Given S and node features Z, the embedding of cluster
j is the soft collection of the nodes assigned to it, node i contributing in proportion to S_{ij} — a
weighted sum, which stacked over clusters is exactly the matrix product **X' = Sᵀ Z**. Row j of X' is
cluster j's pooled embedding. That is the coarse-graph readout, and it falls right out of treating S as
a soft node→cluster indicator.

I should check this respects permutations, because it would be embarrassing to break the one invariant I
must keep. Relabel the original nodes by a permutation P: features ↦ PZ and, if the assignment network
is equivariant, scores and hence S ↦ PS. Then X' = (PS)ᵀ(PZ) = Sᵀ PᵀP Z = Sᵀ Z, because P is a
permutation so PᵀP = I. The coarsened embedding is invariant to node numbering. Good — that is the
property a graph-level answer needs to be well-defined.

Where does S come from? It should depend on each node's features and its position in the graph, which is
exactly what a small network over the node embeddings computes: a per-node output of width K (the number
of clusters), softmaxed over clusters. Here I make the one decision that separates *this task's* DiffPool
from the canonical one, and I make it deliberately, reading the constraints I actually face. The full
method generates the assignment with its own *GNN* over the current adjacency (so a node's cluster is
informed by its neighborhood), generates the embeddings with a *second, separate* GNN, and — crucially —
adds two auxiliary losses to train the pooling: a link-prediction term ‖A − S Sᵀ‖_F that forces
connected nodes into the same cluster, and an entropy term that forces each assignment row toward
one-hot so clusters are crisp. Those auxiliaries exist because the task gradient alone, routed back
through the coarsening and a non-convex factorization-like clustering, is too weak to find good clusters
on its own. But my editable surface is *only* the `GraphReadout` class: it receives node embeddings from
a fixed GIN backbone and returns one graph vector. It is downstream of the message passing — I cannot
insert a clustering GNN into the backbone, I cannot feed a coarsened adjacency back into more
convolution, and I have no clean hook to add auxiliary loss terms to the fixed training step, which only
backpropagates the classification cross-entropy. So the version that fits here is the *stripped* one: a
single coarsening level, the assignment produced by a plain MLP over node features (not a GNN over the
adjacency), and **no link or entropy auxiliary** — only the classification gradient shapes S. I am
keeping DiffPool's load-bearing idea (a learned soft assignment that pools X' = Sᵀ Z) and dropping the
machinery the harness will not let me wire in.

Made concrete in the scaffold's vocabulary: the readout converts the batch's stacked node embeddings to
dense, padded form with `to_dense_batch(x, batch)` → `[B, N_max, D]` plus a validity mask, computes raw
assignment scores with a 2-layer MLP `Linear(D,D) → ReLU → Linear(D,K)` → `[B, N_max, K]`, masks padded
nodes to −∞ before the softmax (so phantom padding nodes get zero assignment), softmaxes *over the
nodes* so each cluster is a normalized mixture of real nodes, re-zeroes the padded rows, and pools
`X' = Sᵀ X_dense` → `[B, K, D]`. The last collapse to one graph vector is a plain **mean over the K
clusters**, giving `[B, D]`. I fix K = 25 clusters and pad to N_max = 150 nodes, sizes comfortably above
the per-graph node counts in these datasets. The output width is `hidden_dim`, so the fixed classifier
head consumes it unchanged. The full scaffold module is in the answer.

Now let me reason about what this stripped DiffPool must do, because that is the whole point of starting
here, and I want my expectations falsifiable. The single thing I have kept — a learned soft assignment
pooled as Sᵀ X — is also the single thing the auxiliaries were there to *protect*, and I have removed
the protection. Without the link loss there is nothing telling connected nodes to share a cluster; the
MLP assigns from node *features alone*, blind to the adjacency, so "clusters" need not correspond to any
graph structure at all. Without the entropy loss there is nothing stopping the softmax from staying
diffuse — each node spread ≈1/K across every cluster — in which case every cluster-node is a faint
average of the whole graph and Sᵀ X degenerates toward "the same global average, copied K times," whose
mean over clusters is just a global mean pool of the node embeddings, scaled. So my honest prediction is
that this readout behaves, at best, like a *learned, noisier global mean* — and a mean, as the readout
lineage already warned, throws away counts and is strictly weaker than a sum on the multiset of node
features. The risk is not that it crashes; it is that the learned soft clustering, pulled only by a
distant classification gradient through a non-convex assignment, settles into something diffuse or
feature-arbitrary and pools *less* informatively than the trivial sum the scaffold ships with. That is
exactly why it is the bottom rung: it carries the most ambitious idea on the ladder — a *learned*
hierarchy — but in the one form the harness permits, shorn of the auxiliaries that make the idea pay,
so it should be the most likely to underperform a simple injective reduction.

I expect the three datasets to split on how much the missing auxiliaries hurt. On tiny MUTAG (188
graphs) the soft assignment has very little data to learn a sensible partition and the cross-validation
mean will be high-variance seed to seed — a coin flip more than a method. On PROTEINS and NCI1, with
more graphs, the MLP assignment gets more gradient, but with no structural prior I expect it to land
near, and plausibly a touch below, a plain global pool — good enough to be a baseline, not good enough
to beat a well-read sum. Whatever the precise split, the diagnosis is already pointed at the next rung:
the weakness here is that I *discarded structure and counts* — a soft mean-like pool with no injective
reduction and no use of the per-layer (`layer_outputs`) information the scaffold hands me. The cleanest
next move is to stop trying to learn a hierarchy with no support for it and instead take the most
expressive *flat* reduction there is — a sum, which keeps the counts a mean loses — and read it from
*every* layer rather than one, turning the unused `layer_outputs` into the multi-scale signal this
hierarchy attempt was reaching for but could not reach.

The causal chain in one breath: a single global sum is flat and represents no scale between a node's
neighborhood and the whole graph, so the structured floor is a graph analogue of CNN pooling → that
means clustering nodes into a coarser graph, hard clustering kills the gradient, so soften it into a
row-stochastic assignment S = softmax(scores) and pool X' = Sᵀ Z (permutation-invariant via PᵀP = I) →
but my editable slot is only the post-backbone readout, so I cannot wire in the canonical method's
pooling-GNN or its link/entropy auxiliaries; I keep the learned soft assignment (a plain MLP over node
features, one level, mean over clusters) and drop the protection → which means, with no structural prior
and no anti-diffuse term, the pool likely behaves like a learned, noisier global mean that discards
counts → so this is the bottom rung by construction, and its failure points straight at the next:
recover the counts with an injective *sum* and read it from every layer.
