# Context: link prediction in networks (circa 2017-2018)

## Research question

Given a partially observed network `G = (V, E)` — an undirected graph with adjacency matrix `A`
where `A_{i,j} = 1` iff `(i,j)` is an observed edge — decide, for a pair of nodes `(x, y)` with
no observed edge, how likely it is that the edge exists (is missing) or will form. This is link
prediction, and it underpins friend recommendation in social networks, paper recommendation in
citation networks, knowledge-graph completion, and protein-interaction prediction. The practical
demand is a single predictor that works *across* networks of very different character — dense
social graphs, sparse power grids, biological interaction maps — without the practitioner having
to know in advance which structural signal drives link formation in each one.

The difficulty is precisely that the signal differs by domain. The best-known predictors are
fixed scoring functions, each of which silently assumes one mechanism of link formation. When
the assumption matches the network they are excellent and nearly free; when it does not they are
no better than chance. A solution would have to (1) not commit to a single hand-chosen
assumption, but adapt the structural signal it uses to the network at hand; (2) remain
computationally affordable — the strongest fixed scores are the ones that read the *entire*
network, and any method that literally needed the whole graph for every candidate pair would not
scale; (3) be able to fold in not just raw topology but also learned node representations and any
side information attached to nodes; and (4) work for graphs of arbitrary, varying size without
assuming a fixed feature dimension or node count. No method on the table below achieves all four.

## Background

A link-prediction score is a function of the observed graph structure around `(x, y)`. The
standard taxonomy of these *graph structure features* is by **order**: the maximum hop of
neighborhood needed to compute them. Let `Γ(x)` denote the one-hop neighbors of `x` and `d(x,y)`
the shortest-path distance.

- **First-order** scores use only one-hop neighbors. Common Neighbors `|Γ(x) ∩ Γ(y)|`; Jaccard
  `|Γ(x) ∩ Γ(y)| / |Γ(x) ∪ Γ(y)|`; Preferential Attachment `|Γ(x)|·|Γ(y)|`.
- **Second-order** scores use up to two-hop neighbors. Adamic-Adar `Σ_{z∈Γ(x)∩Γ(y)} 1/log|Γ(z)|`
  and Resource Allocation `Σ_{z∈Γ(x)∩Γ(y)} 1/|Γ(z)|` down-weight common neighbors that are
  themselves high-degree.
- **High-order** scores require, in their definition, the *whole* network: Katz, rooted
  PageRank, SimRank. These typically outperform local scores (Lü & Zhou 2011), at the cost of
  reading the entire graph.

A useful fact for what follows: a score's order is exactly the radius of neighborhood it reads,
so any score of order `h` is, by definition, computable from the subgraph induced by all nodes
within `h` hops of `x` or `y`. The standard high-order scores are infinite series over walks:

- **Katz index**: `Katz(x,y) = Σ_{l=1}^∞ β^l |walks^{(l)}(x,y)| = Σ_{l=1}^∞ β^l [A^l]_{x,y}`, a
  damped (`0 < β < 1`) sum over all walks of every length between `x` and `y`, shorter walks
  weighted more. In practice `β` is set tiny, e.g. `5e-4` (Liben-Nowell & Kleinberg 2007).
- **rooted PageRank**: the stationary distribution `π_x` of a random walker that at each step
  moves to a random neighbor with probability `α` and teleports back to `x` with probability
  `1-α`, i.e. `π_x = αPπ_x + (1-α)e_x` where `P_{i,j} = 1/|Γ(v_j)|` on edges. The score is
  `[π_x]_y` (or `[π_x]_y + [π_y]_x` for symmetry). The *inverse P-distance* identity (Jeh &
  Widom 2003) re-expresses it as a damped sum over walks:
  `[π_x]_y = (1-α) Σ_{w: x⇝y} P[w] α^{len(w)}`, with `P[w] = Π_i 1/|Γ(v_i)|` the probability of
  traversing walk `w`.
- **SimRank** (Jeh & Widom 2002): "two nodes are similar if their neighbors are similar",
  `s(x,y) = γ · (Σ_{a∈Γ(x)} Σ_{b∈Γ(y)} s(a,b)) / (|Γ(x)|·|Γ(y)|)` with `s(x,x)=1`. It has an
  equivalent expansion over *simultaneous* walks that start at `x` and `y` and first meet at a
  common node `z`: `s(x,y) = Σ_{w:(x,y)⊸(z,z)} P[w] γ^{len(w)}`.

Critically, each fixed score embeds one assumption about *when* links form. Common Neighbors
assumes more shared friends ⇒ more likely to link. That holds in social networks but is observed
to *invert* in protein-protein interaction networks: two proteins that share many interaction
partners are actually *less* likely to interact directly (Kovács et al. 2018). A score whose
assumption is wrong for the network is useless on it, and on networks like Power grids and
router graphs most classic scores perform near chance.

Two other families of features exist and are largely orthogonal to topology scores.
**Latent-feature methods** factorize a matrix representation of the network to learn a
low-dimensional embedding per node — matrix factorization, the stochastic block model, and the
network-embedding methods DeepWalk, LINE, and node2vec (all shown to implicitly factorize some
network matrix, Qiu et al. 2017). Latent features capture global, long-range structure but are
*transductive* (a change in the graph forces re-training), cannot capture structural similarity
between distant nodes (Ribeiro et al. 2017), and can need very large dimension to express even a
simple topology score (Nickel et al. 2014). **Explicit features** are node attributes (e.g.
word distributions on document nodes). Combining the three families is known to help (Nickel et
al. 2014; Zhao et al. 2017), but doing so in one principled model was unsolved.

On the learning side, **graph neural networks** had recently matured. A GNN consists of *graph
convolution* layers that update each node from its neighbors, plus, for graph-level tasks, an
*aggregation* layer that pools node features into one graph vector. The renormalized
first-order spectral convolution `H^{(l+1)} = σ(D̃^{-1/2} Ã D̃^{-1/2} H^{(l)} W^{(l)})` with
`Ã = A + I` (Kipf & Welling 2017) propagates information one hop per layer and scales linearly
in edges. A closely related spatial variant `Z = f(D̃^{-1} Ã X W)` uses a random-walk
normalization; both can be read as a differentiable, trainable relaxation of the
Weisfeiler-Lehman color-refinement procedure. For *graph classification*, a sorting-based
aggregation — sort nodes into a consistent order by their final convolutional features, then
truncate/pad to a fixed length `k` and run an ordinary 1-D CNN on the sequence — gives a
permutation-invariant, end-to-end-trainable readout that keeps more information than simply
summing node features (Zhang et al. 2018, DGCNN/SortPooling).

## Baselines

**Predefined heuristics** (CN, Jaccard, PA, AA, RA, Katz, rooted PageRank, SimRank). Each is a
closed-form score of the observed structure, requiring no training; the high-order ones read the
whole graph. **Gap:** every one of them hard-codes a single hypothesis about link formation, so
each is excellent on the networks whose mechanism it matches and worthless on the rest — there
is no single heuristic that is good everywhere, and on networks with an unusual formation
mechanism (or where shared neighbors *anti*-correlate with linking) none of them works.

**Latent-feature / network-embedding methods** (matrix factorization, SBM, DeepWalk, LINE,
node2vec; and the GNN-based VGAE, a node-level encoder with an inner-product decoder). These
learn a vector per node from the global network and score a pair by aggregating the two node
vectors — e.g. node2vec runs a biased random walk (return parameter `p`, in-out parameter `q`)
through a skip-gram objective and scores a candidate edge by the Hadamard product of the two
endpoint embeddings. **Gap:** they are transductive and operate at the node level, so the
representation of a pair is just a function of two independently learned points; they capture
global proximity but miss the fine-grained structural pattern *between* the two nodes, need
re-training when the graph changes, and can require a very large embedding dimension to even
reproduce a simple topology score.

**Supervised structure-feature learning from local subgraphs (WLNM)** (Zhang & Chen 2017). The
direct predecessor. For each candidate pair `(x, y)` it grows the neighborhood — add the 1-hop,
then 2-hop, … neighbors of `x` and `y` — until the induced *enclosing subgraph* has more than `K`
vertices; it then runs a hashing-based Weisfeiler-Lehman procedure to assign each vertex a
position so subgraphs can be read in a consistent order, **truncates** the subgraph to exactly
`K` vertices by deleting the last-ordered ones, and feeds the resulting fixed-size `K×K`
adjacency matrix to a fully-connected neural network that classifies it as link / no-link
(`K=10` works best). This already *learns* a network-specific structural predictor rather than
assuming one, and beats the hand-crafted scores. **Gap:** the fully-connected network demands a
fixed-size input, which forces the truncation step — so the model cannot consistently read each
pair's full `h`-hop neighborhood and discards structure; the adjacency-matrix-only
representation gives it no way to ingest latent or explicit node features; and there is no
account of *how much* a local subgraph can possibly tell you about the global high-order scores,
so the choice of neighborhood radius is unguided.

**Graph classifiers as a component** (DGCNN with SortPooling; the GCN convolution). These accept
graphs of *arbitrary* size and a continuous per-node feature matrix `X`, stack graph convolutions
to extract multi-hop substructure features, and pool to a graph-level vector for classification.
They were built and evaluated for molecule/protein graph classification, not link prediction.
**Gap:** out of the box, a graph classifier treats all nodes symmetrically and pools them into
one vector, so applied naively to a neighborhood it has no notion that two particular nodes are
the pair whose link is in question — it cannot tell the target nodes apart from the rest, which
is exactly the information a link predictor needs.

## Evaluation settings

The natural yardsticks already in use for link prediction at the time:

- **Datasets** spanning regimes where different mechanisms dominate: USAir (airline network),
  NS (collaboration), PB (political blogs), Yeast and E.coli (biological interaction), C.ele
  (neural), Power (electrical grid), Router (Internet topology) — node counts from a few hundred
  to a few thousand, average degree from ~2.5 (Power, Router) to ~27 (PB). Larger graphs (arXiv,
  Facebook, BlogCatalog, Wikipedia, PPI) for scalability; citation graphs such as Cora and
  CiteSeer add node attributes to the same link-prediction setting.
- **Protocol**: randomly hold out a fraction (e.g. 10%, or 50% for a harder split) of observed
  edges as positive test links; sample an equal number of unconnected node pairs as negative
  test links; train on the remaining observed edges plus an equal number of sampled negative
  (non-)edges. A validation split is used to set hyperparameters.
- **Metrics** (higher is better): area under the ROC curve (AUC) and average precision (AP) for
  balanced held-out positives and negatives; when the evaluation is phrased as ranking a true
  edge against many negatives, mean reciprocal rank (MRR) and Hits@K are the natural summaries.

## Code framework

The pieces that already exist are: a data pipeline that yields the training graph as
`edge_index` (a `[2, E]` COO tensor of undirected edges) and a node-feature matrix `x`
(`[N, in_channels]`, with `in_channels` varying per dataset); a library of graph-neural-network
primitives (graph-convolution layers such as `GCNConv`/`SAGEConv`, message passing, global
pooling, and utilities like `negative_sampling`, `to_undirected`, `degree`); an
automatic-differentiation engine and the Adam optimizer; a binary cross-entropy loss; and an
evaluation routine that, given scores for positive and sampled negative candidate edges, returns
AUC / MRR / Hits@K. What is **not** settled — and is exactly the contribution to design — is the
predictor that turns the observed graph and a batch of candidate edges into edge scores. It is
left as one empty class with the standard `encode` / `decode` / `forward` interface; everything
around it already exists.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling


class LinkPredictor(nn.Module):
    """Turns the observed training graph and a batch of candidate edges into
    edge scores. The internal mechanism — how structure (and any node
    information) is turned into a score for a pair — is the contribution and is
    left empty here."""

    def __init__(self, in_channels, hidden_channels, num_layers, dropout):
        super().__init__()
        # TODO: the predictor we will design.
        pass

    def encode(self, x, edge_index):
        # TODO: produce whatever per-node / per-pair representation the method needs.
        pass

    def decode(self, *args, **kwargs):
        # TODO: turn the representation of a candidate edge into a score.
        pass

    def forward(self, x, edge_index, edge_label_index):
        # TODO: end-to-end -- candidate edges in edge_label_index -> [num_edges] scores.
        pass


def train(model, x, edge_index, optimizer):
    model.train()
    optimizer.zero_grad()
    # sample negative candidate edges to balance the observed positives
    pos = edge_index
    neg = negative_sampling(edge_index, num_nodes=x.size(0),
                            num_neg_samples=pos.size(1))
    edge_label_index = torch.cat([pos, neg], dim=1)
    edge_label = torch.cat([torch.ones(pos.size(1)),
                            torch.zeros(neg.size(1))])
    scores = model(x, edge_index, edge_label_index)      # the slot above
    loss = F.binary_cross_entropy_with_logits(scores, edge_label)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def evaluate(model, x, edge_index, pos_edge_index, neg_edge_index):
    """Score positive and negative candidate edges, return AUC / MRR / Hits@K.
    Already available."""
    ...
```
