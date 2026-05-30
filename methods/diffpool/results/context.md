# Context: hierarchical pooling for graph-level prediction with graph neural networks

## Research question

Graph neural networks have become the standard tool for learning on graph-structured data: a node's
features and its neighborhood are distilled, through repeated local message passing, into a dense
embedding that downstream models consume. For node-level tasks — node classification, link
prediction — this works cleanly, because the unit of prediction is a node and the GNN already
produces one vector per node.

But many of the most important problems are *graph-level*: predict a property of an entire molecule
from its atom-bond graph, classify a protein's function from its structure graph, label a social
network by community type. Here the unit of prediction is the whole graph, and a GNN that emits one
vector per node leaves an unsolved last step: how to go from a variable-sized bag of node
embeddings to a single fixed-dimensional vector for the graph — one that can be fed to an ordinary
classifier.

The precise problem: given a graph G = (A, F) with adjacency A and node features F, produce a
fixed-dimensional embedding of the *whole graph* that supports graph classification, in a way that
(i) is learned end-to-end with the task rather than hand-engineered, (ii) respects and exploits the
*hierarchical* structure real graphs have — atoms group into functional units, people into
communities — rather than treating all nodes as a flat set, (iii) is permutation-invariant (the
answer cannot depend on how the nodes happened to be numbered), and (iv) generalizes across graphs
of different sizes and shapes, since a graph classifier sees many distinct graphs at train and test
time.

## Background

**Message-passing GNNs.** The dominant framework (Gilmer et al. 2017) views a graph as a
computation graph: node representations are computed iteratively from the features of neighboring
nodes through a differentiable aggregation. One writes H^(k) = M(A, H^(k-1); θ^(k)), where H^(k) ∈
R^{n×d} are the node embeddings after k rounds of propagation, M is the propagation function, and
H^(0) = F. After K rounds (typically K in 2–6) the final node embeddings Z = H^(K) capture each
node's K-hop neighborhood. A canonical instance is the graph convolutional network (Kipf & Welling
2017), which implements one round as

  H^(k) = ReLU( D̃^{-1/2} Ã D̃^{-1/2} H^(k-1) W^(k-1) ),  Ã = A + I,  D̃ = diag(Σ_j Ã_{ij}),

a symmetric-normalized neighborhood average followed by a learned linear map and a nonlinearity.
Other members of the family — neural fingerprints (Duvenaud et al. 2015), gated graph networks (Li
et al. 2016), GraphSAGE (Hamilton et al. 2017), structure2vec (Dai et al. 2016), attention-based
variants (Veličković et al. 2018) — differ in the aggregator but share the message-passing shape.
A property that matters for graph-level use: when the aggregation is symmetric over neighbors, the
whole GNN is permutation-equivariant — relabel the nodes by a permutation P and the output node
embeddings are relabeled by the same P.

**The flat-pooling gap.** Message passing only ever moves information *along edges*; the
representation stays at the granularity of individual nodes. To produce a single graph vector the
standard recipe is a *global* readout in one shot: sum or average all node embeddings (Duvenaud et
al. 2015), introduce a virtual node connected to everything (Li et al. 2016), or run a
set-aggregation network over the node embeddings (Gilmer et al. 2017, set2set). Each of these
collapses the entire graph in a single flat operation. The diagnostic problem with all of them is
that they discard hierarchy: a molecule's atoms first form bonds, then functional groups, then the
whole molecule, and a flat sum over atom embeddings never represents the intermediate
functional-group scale. By analogy, image CNNs owe much of their power to *spatial pooling* — they
interleave convolution with downsampling so deeper layers see coarser, more global structure with
larger receptive fields. GNNs as described have convolution but no analogue of that pooling step.

**Why a graph pooling operator is hard.** The naive transcription of CNN pooling fails immediately.
Images have spatial locality: an m×m patch is well defined and the same everywhere, so pooling is a
fixed deterministic stride. Graphs have no such locality — there is no canonical "patch", because the
neighborhood structure is irregular and differs node to node, and solving for a canonical node
ordering is as hard as graph isomorphism (this is what sinks approaches that linearize the graph to
a sequence and apply a 1-D CNN, e.g. Niepert et al. 2016, Zhang et al. 2018). Moreover graphs in a
dataset have different numbers of nodes and edges, so any pooling operator must be defined uniformly
across sizes.

**Two-stage clustering approaches.** One line builds hierarchy by first running a *deterministic*
graph-coarsening or clustering algorithm — spectral clustering (Defferrard et al. 2016), graph
coarsening heuristics (Simonovsky & Komodakis 2017, Fey et al. 2018) — to group nodes, then running
a GNN over the coarsened graph. The diagnostic limitation: the clustering is fixed in advance, not
learned, so it cannot adapt to the prediction task, and it is computed per-graph by a separate
subroutine rather than as a reusable strategy that generalizes across graphs.

**Low-rank factorization view.** Approximating an adjacency matrix A by S Sᵀ for a tall thin S is a
low-rank / soft-clustering factorization, related to well-separated pair decomposition: S Sᵀ tries
to reconstruct which nodes are connected, so a good S groups connected nodes together. This
factorization is non-convex with many local minima, which is the central reason a fixed two-step
"factorize then classify" pipeline can underperform something trained jointly with the task.

**Weisfeiler–Lehman and structural features.** Beyond raw node features, structural descriptors —
node degree, local clustering coefficient — are cheap, permutation-respecting summaries of a node's
role, and the Weisfeiler–Lehman refinement test (iterated hashing of neighbor multisets) is the
classical lens on what neighbor-aggregation can and cannot distinguish.

## Baselines

The natural points of comparison are the existing recipes for turning a GNN's node embeddings into a
graph-level representation:

- **Global mean / sum pooling (Duvenaud et al. 2015).** Run a GNN, then average or sum all final node
  embeddings into one vector for the classifier. Core idea: a permutation-invariant readout by a
  symmetric reduction. Gap: a single flat aggregation over all nodes; no intermediate scales, so
  community / functional-group structure is invisible to the classifier.

- **Virtual-node / gated readout (Li et al. 2016).** Add a node connected to every other node so that,
  after message passing, the virtual node's embedding summarizes the graph; gated recurrent updates
  refine node states. Gap: still a single global summary; the hierarchy is not represented, and the
  virtual node creates an artificial global hub.

- **Set-aggregation readout (Gilmer et al. 2017, set2set).** Feed the set of node embeddings to a
  permutation-invariant set network (e.g. an attention-based set2set readout) to produce the graph
  vector. Core idea: a more expressive symmetric pooling than a plain sum. Gap: more capacity in the
  readout, but it is still one flat operation over all nodes at once — no coarsening, no hierarchy.

- **Sort-and-CNN / canonical-ordering readouts (Niepert et al. 2016; Zhang et al. 2018, SortPool).**
  Impose an ordering on nodes (via WL-style coloring or learned sorting), then apply an ordinary 1-D
  CNN over the ordered node embeddings. Core idea: borrow CNN pooling by sequentializing the graph.
  Gap: choosing a structure-preserving canonical order is essentially graph isomorphism; nodes far
  apart in the graph can land adjacent in the sequence, so the imposed locality is artificial.

- **Two-stage deterministic clustering + GNN (Defferrard et al. 2016; Simonovsky & Komodakis 2017;
  Fey et al. 2018).** Coarsen the graph with a fixed clustering/coarsening algorithm, then run a GNN
  on the coarsened graph; repeat to build hierarchy. Core idea: real coarsening, real hierarchy. Gap:
  the coarsening is deterministic and task-agnostic — fixed before any gradient flows — so it cannot
  specialize to the classification objective, and it is a per-graph subroutine rather than a learned,
  transferable pooling strategy.

## Evaluation settings

The natural yardstick is graph classification on standard benchmark collections of many small-to-
medium graphs, each with a single label:

- **Bioinformatics / molecule graphs with node labels:** D&D (protein structures, ~1178 graphs, avg
  ~284 nodes), PROTEINS (~1113 graphs, avg ~39 nodes), ENZYMES (600 graphs, 6 classes, avg ~33
  nodes). Nodes carry discrete labels (e.g. amino-acid / atom type).
- **Social / collaboration graphs without node labels:** COLLAB (5000 graphs, 3 classes, avg ~74
  nodes, very dense — avg ~2458 edges), REDDIT-MULTI-12K (~11929 graphs, 11 classes, avg ~391 nodes).
  When nodes lack features, structural descriptors (degree, clustering coefficient) serve as inputs.

Protocol: a GNN backbone produces node embeddings; a readout produces the graph vector; an MLP +
softmax classifier maps it to class probabilities; cross-entropy loss; accuracy under cross-validation
the metric. Standard graph-kernel methods (Weisfeiler–Lehman subtree kernel, graphlet kernels) are
the classical non-neural reference on these same datasets.

## Code framework

The primitives that already exist: a message-passing GNN layer that maps (A, X) to node embeddings,
the GCN propagation rule, an Adam optimizer, cross-entropy loss, and a training loop over a dataset
of (adjacency, features, label) graphs. What is missing is the operator that turns per-node
embeddings into a graph-level prediction while respecting hierarchy. We lay out the scaffold with that
operator left as an empty slot.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN(nn.Module):
    """K rounds of message passing: (A, X) -> node embeddings Z. Permutation-equivariant."""
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        # stacked GCN-style layers with per-layer weight matrices
        # TODO: build the message-passing stack
        pass

    def forward(self, A, X):
        # H^(0) = X ; H^(k) = ReLU( normalized(A) H^(k-1) W^(k-1) )
        # TODO: return final node embeddings Z (and possibly per-layer reps)
        pass

def coarsen(A, Z, assignment):
    """Take node embeddings + a node-grouping and emit a smaller (A', X') graph.
    This is the operator we are missing. TODO."""
    pass

class GraphLevelModel(nn.Module):
    """Backbone GNN(s) + the hierarchy/readout module + classifier."""
    def __init__(self, in_dim, hidden_dim, num_classes, max_nodes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        # TODO: the modules that build and consume the coarsened hierarchy

    def forward(self, A, X):
        # repeatedly: embed nodes -> group them -> coarsen -> recurse,
        # until a single graph vector remains; then classify.
        # TODO
        graph_vec = None
        return self.classifier(graph_vec)

def aux_losses(A, assignment):
    """Extra training signal to shape the grouping. TODO."""
    pass

def train_step(model, batch, opt):
    A, X, y = batch
    logits = model(A, X)
    loss = F.cross_entropy(logits, y)
    # loss = loss + (auxiliary terms once defined)
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    opt.step()
    return loss.item()
```
