# Context: graph-level pooling/readout for graph classification (circa 2018-2019)

## Research question

Graph classification asks for a single label per graph — is this protein an enzyme, is this
molecule a mutagen — from a graph whose node count and connectivity vary from one example to
the next. A message-passing graph network already gives us a good per-node embedding: stacking
graph-convolution layers, each node mixes its features with its neighbors', and after a few
layers every node carries a representation of its local structural neighborhood. But a
classifier needs one fixed-size vector per *graph*, not a variable-size bag of node vectors.
The operation that turns the node set into that fixed vector — the graph-level **readout**, or
**pooling** — is where the difficulty lives, and it is far less settled than graph convolution
itself.

The precise goal is a pooling mechanism that simultaneously: (1) maps a graph of any size to a
fixed-size embedding and is invariant to the (arbitrary) ordering of its nodes; (2) uses
**both** the node features *and* the graph topology to decide what to keep, rather than
treating the nodes as an unordered feature bag; (3) builds a **hierarchical**, multi-scale
summary — coarsen, then coarsen again — the way CNN pooling builds a pyramid, because flat
one-shot aggregation throws away the structural composition that distinguishes graphs; (4) is
trainable end-to-end with the rest of the network; (5) costs a *reasonable* amount of time and
memory — ideally staying sparse, linear in nodes-plus-edges rather than quadratic in nodes;
and (6) has a parameter count that does **not** grow with the size of the graphs, so the same
module works on a 30-node molecule and a 5000-node protein. Each existing pooling method below
hits a subset of these; none hits all six. Closing that gap is the problem.

## Background

By this time, generalizing CNNs to non-Euclidean data — graphs of social networks, molecules,
proteins, knowledge bases — is a fast-moving field (Bronstein et al. 2017). A CNN has two
ingredients: convolution and downsampling (pooling). The convolution side is well developed.
Spectral methods (Bruna et al. 2014; Defferrard et al. 2016) define convolution in the graph
Fourier domain via the graph Laplacian; Kipf & Welling (2016) collapse that to a cheap
first-order rule

```
h^{(l+1)} = sigma( D̃^{-1/2} Ã D̃^{-1/2} h^{(l)} Θ ),   Ã = A + I_N,   D̃ = degree matrix of Ã,
```

where each node averages a normalized combination of its own and its neighbors' features and
then applies a learned linear map Θ ∈ R^{F×F'} and a nonlinearity. The symmetric normalization
`D̃^{-1/2} Ã D̃^{-1/2}` is the first-order approximation of the graph Laplacian; it is the
operator that makes a node's output depend on its neighborhood. Non-spectral methods reach the
same place from the message-passing side: GraphSAGE (Hamilton et al. 2017) aggregates a
sampled fixed-size neighborhood; GAT (Veličković et al. 2018) weights neighbors by learned
attention coefficients (Bahdanau et al. 2014). Gilmer et al. (2017) unify these as
neural message passing. The recurring fact to hold onto: the normalized adjacency term is
already computed inside every convolution layer, and it is precisely what injects topology
into a per-node quantity.

The pooling side is thinner, and the field sorts the attempts into three families. **Topology-
based** coarsening (Graclus, Dhillon et al. 2007, used as the pooling module in Defferrard
et al. 2016) clusters nodes using the equivalence between a spectral-clustering objective and
weighted kernel k-means, avoiding eigendecomposition — but it looks only at graph structure
and ignores node features, and it is a fixed pre-processing step, not learned for the task.
**Global** pooling collapses all node embeddings at once into a graph vector: Set2Set (Vinyals
et al. 2015) uses an LSTM with attention to produce an order-invariant set embedding; SortPool
(Zhang et al. 2018) sorts nodes by their structural role (Weisfeiler-Lehman-style colors) and
feeds the sorted sequence to a 1-D convolution. **Hierarchical** pooling — the newest family —
coarsens the graph in stages so the network learns multi-scale structure, the way CNN pooling
does. The motivating observation across this literature is empirical and structural: flat
global readouts "do not learn hierarchical representations which are crucial for capturing
structural information of graphs," and a single mean/max/sum over all nodes weights an
uninformative leaf node exactly like a hub that determines the graph's class.

Two background facts about *learned selection* are load-bearing. First, the **attention /
self-attention** idea (Bahdanau et al. 2014; Vaswani et al. 2017): produce a per-element
importance score and let it decide how much each element matters; "self-attention" (intra-
attention) is the case where the input itself supplies the criteria for its own scoring.
Second, a sharp **differentiability fact** about hard selection: ranking nodes and keeping the
top-k is a discrete `argsort` — a piecewise-constant function of the scores — so it carries no
gradient back to whatever produced the scores. Any method that selects nodes by a learned
score must therefore arrange a separate, continuous path for that gradient, or the scoring
parameters never train.

## Baselines

These are the prior pooling methods a new one would be measured against and would react to.

**DiffPool (Ying et al., NeurIPS 2018).** The first end-to-end *learned hierarchical* pooler.
At layer `l` a GNN produces a soft cluster-assignment matrix and uses it to coarsen both
features and adjacency:

```
S^{(l)} = softmax( GNN_l(A^{(l)}, X^{(l)}) ) ∈ R^{n_l × n_{l+1}},
A^{(l+1)} = S^{(l)T} A^{(l)} S^{(l)},   X^{(l+1)} = S^{(l)T} Z^{(l)},
```

so every node is *softly* assigned to all `n_{l+1}` clusters and the next adjacency is a dense
contraction of the current one. **Gaps:** the assignment matrix is dense, and `S^T A S`
produces a *dense* coarsened adjacency, so storage is quadratic, O(k|V|^2). The number of next-
layer clusters `n_{l+1}` is fixed when the model is built, which makes the parameter count
depend on the (maximum) number of nodes in the dataset, and forces one cluster size onto
graphs whose sizes span orders of magnitude — on a dataset where node counts run from 30 to
5748, a fixed 10% cluster size that suits the median graph blows most graphs *up* rather than
down. Out-of-memory failures appear once the pooling ratio exceeds 0.5.

**gPool / Graph U-Nets (Gao & Ji, ICML 2019), and the sparse hierarchical classifier of
Cangea et al. (2018).** A *sparse* hierarchical pooler that drops the dense assignment matrix
in favor of hard top-k node selection driven by a single learnable projection vector `p`:

```
y   = X^{(l)} p^{(l)} / ||p^{(l)}||,          # scalar projection of each node's features onto p
idx = rank(y, k),                              # indices of the k largest scores
ỹ   = sigmoid( y(idx) ),                        # gate on the kept scores
X̃   = X^{(l)}(idx, :),
A^{(l+1)} = A^{(l)}(idx, idx),                  # index the sparse adjacency, keep it sparse
X^{(l+1)} = X̃ ⊙ (ỹ 1_C^T).                      # gate the kept features by the sigmoid score
```

This fixes DiffPool's complexity: indexing the adjacency keeps it sparse, storage is
O(|V|+|E|), and because the only learned object is the fixed-length vector `p`, the parameter
count is independent of graph size. The gate `sigmoid(y(idx))` multiplied into the kept
features is what lets gradient reach `p` at all — the hard `rank`/`idx` step is non-
differentiable, and without the multiplicative gate `p` produces purely discrete outputs and
never trains. **Gap:** the score `y_i = x_i · p / ||p||` is a function of node `i`'s own
feature vector and nothing else — the adjacency matrix never enters the score. Two nodes with
identical features but completely different roles in the graph receive identical scores; the
graph topology does not affect which nodes are kept. A pooler advertised as a graph operation
decides what to keep while structurally blind to the graph.

**SortPool (Zhang et al., AAAI 2018).** Sorts nodes by Weisfeiler-Lehman-style structural
roles into a canonical order, truncates/pads to a fixed length `K`, and applies a 1-D
convolution over the sorted node sequence. **Gap:** a single flat global readout — it imposes
one linear order and pools once, with no hierarchical coarsening, so structural composition
above the level of the sort key is not represented.

**Set2Set (Vinyals et al., 2015), as used in the MPNN framework (Gilmer et al. 2017).** An
LSTM with content-based attention reads the node-embedding set over a fixed number of
processing steps and emits an order-invariant graph embedding. **Gap:** also a single global
readout with no coarsening; it summarizes the whole node set at once and cannot build a multi-
scale hierarchy.

**Topology-only coarsening (Graclus, Dhillon et al. 2007).** Clusters nodes from structure
alone, no eigenvectors, used as a fixed pooling module. **Gap:** ignores node features
entirely and is not learned for the downstream task.

## Evaluation settings

The natural yardsticks already in use for graph classification:

- Benchmark graph-classification datasets from the TU Dortmund collection (Kersting et al.
  2016): **D&D** (1178 protein graphs, enzyme vs non-enzyme, ~284 nodes/graph), **PROTEINS**
  (1113 protein graphs, ~39 nodes/graph), **NCI1** and **NCI109** (~4110/4127 chemical-compound
  graphs, anticancer activity, ~30 nodes/graph), and **FRANKENSTEIN** (4337 molecular graphs
  with continuous node features, mutagen vs non-mutagen). The spread of average graph sizes —
  from ~17 to ~284 nodes — is itself part of the test, since a pooler must cope with both small
  and large graphs.
- Two reference architectures, applied identically to every pooling method so the comparison
  isolates the pooling operator: a **global-pooling architecture** (three graph-convolution
  layers, outputs concatenated, one readout, then a linear classifier; following Zhang et al.
  2018) and a **hierarchical-pooling architecture** (three blocks, each a graph-convolution
  layer followed by a pooling layer, with a per-block readout whose outputs are summed before
  the linear classifier; following Cangea et al. 2018). Graph convolution is fixed to the Kipf
  & Welling rule for all models; ReLU activations.
- Readout per block following the Jumping-Knowledge idea (Xu et al. 2018):
  `s = (1/N) Σ_i x_i  ||  max_i x_i`, the concatenation of mean and max over the (current,
  possibly coarsened) node set, giving a fixed-size vector regardless of node count.
- Metric: classification accuracy (and macro F1), higher is better, averaged over many
  random seeds with k-fold cross-validation and early stopping; hyperparameters (learning
  rate, hidden size, weight decay, pooling ratio ∈ {1/2, 1/4}) chosen by grid search. Adam
  optimizer.

## Code framework

The pooling/readout module plugs into a fixed message-passing backbone: a stack of graph-
convolution layers has already produced node embeddings, and the module must turn the node
set of each graph in the batch into one fixed-size graph vector. Batches are handled the
PyTorch-Geometric way — node features `x` are stacked across all graphs in the batch into one
`[N_total, F]` matrix, `edge_index` is the global `[2, E_total]` edge list, and a `batch`
vector `[N_total]` tags each node with its graph id so that batched, per-graph reductions
(`global_add_pool`, `global_mean_pool`, `global_max_pool`) are available. The pieces that
already exist are the convolution operator, these batched global reductions, and the top-k /
adjacency-indexing utilities for sparse selection; the loss and training loop are unchanged.
What is *not* settled is the readout itself — how to decide what to keep and how to build the
graph vector — so that is the one empty slot.

```python
import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool


class GraphReadout(nn.Module):
    """Turns batched node embeddings into one fixed-size vector per graph.

    Inputs (PyG batched form):
      x            [N_total, hidden_dim]  node embeddings from the fixed backbone
      edge_index   [2, E_total]           global edge list
      batch        [N_total]              graph id of each node
      layer_outputs list of [N_total, hidden_dim]  per-conv-layer node embeddings
    Output:
      [B, output_dim]                     one vector per graph (set self.output_dim)
    """

    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # TODO: the readout/pooling module we will design, and its output_dim.
        self.output_dim = hidden_dim

    def forward(self, x, edge_index, batch, layer_outputs):
        # TODO: aggregate the (possibly coarsened) node set into one vector per graph.
        #       Decide what to keep using whatever signal we choose; must stay
        #       permutation-invariant at the graph level and handle variable sizes.
        raise NotImplementedError
```

The backbone supplies node embeddings and the batch tagging; `forward` is where the readout —
and, if we choose, the coarsening that precedes it — will live.
