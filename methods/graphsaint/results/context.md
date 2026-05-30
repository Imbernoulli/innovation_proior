# Context: scalable minibatch training of deep graph convolutional networks

## Research question

Graph convolutional networks learn representations of an attributed graph G(V, E): each node v has a
feature vector x_v, and a GCN layer mixes a node's representation with its neighbors',

  x_v^(ℓ+1) = σ( Σ_{u∈V} Â_{v,u} (W^(ℓ))ᵀ x_u^(ℓ) ),   Â = D^{-1}A,

so after L layers each node has gathered information from its L-hop neighborhood. This is powerful,
but the standard training recipe is *full-batch*: hold the entire graph and all activations in
memory and take one gradient step over all nodes. That does not scale — neither to graphs with
hundreds of thousands or millions of nodes, nor to deep GCNs.

The precise problem: train a GCN by *minibatches* on a large graph, efficiently and accurately,
including when the network is *deep* (more than two layers) and in the *inductive* setting (test
nodes' features and connections are absent during training, so the model must generalize to unseen
graph structure). A solution must (i) avoid the explosive growth of the support set with depth, (ii)
produce minibatches whose gradient is a faithful (unbiased, low-variance) estimate of the full-graph
gradient, and (iii) keep the per-minibatch computation and any preprocessing cheap.

## Background

**The GCN propagation rule and full-batch training.** Spectral graph convolution, made practical by
localized Chebyshev filters (Defferrard et al. 2016) and simplified to the first-order rule (Kipf &
Welling 2017), gives the layer above: a (normalized) neighbor aggregation, a linear map W^(ℓ), and a
nonlinearity. These works target small graphs and train in full batch, recomputing every node's
representation at every layer each step.

**Neighbor explosion — the core diagnostic obstacle.** Consider computing the output of a single
node v under an L-layer GCN. Its layer-L value needs its neighbors' layer-(L−1) values; each of
those needs *their* neighbors' layer-(L−2) values; and so on. The set of "support" nodes required to
produce one output grows multiplicatively with depth — for a graph with average degree d, on the
order of d^L nodes. So even a single-node minibatch can pull in a large fraction of the graph, and
both memory and compute per step blow up with depth. This is the central reason naive minibatching
fails on GCNs and the obstacle every scalable method must defeat.

**Bias and variance of a minibatch estimator.** A minibatch gradient is only useful if it estimates
the full-batch gradient well. Two properties matter. *Bias*: if some nodes or edges are included in
minibatches more often than others — which happens the moment sampling is non-uniform — the
estimator systematically over-weights frequently-sampled (typically high-degree, high-centrality)
nodes, and the model learns their features preferentially. *Variance*: even an unbiased estimator is
useless if it is too noisy; the sampling distribution should be chosen to keep the estimator's
variance small so that SGD converges quickly. The classic tools here are inverse-probability
reweighting (an event included with probability p contributes with weight 1/p to stay unbiased) and
importance sampling (choose the sampling probabilities to minimize variance, typically proportional
to the magnitude of each term's contribution; the Cauchy–Schwarz inequality gives the optimum).

**Connectivity within a minibatch.** For a GCN specifically, a node's representation is only as good
as its access to its neighbors during propagation. If a minibatch contains a node but few or none of
the neighbors it needs at the previous layer, that node's computed representation is impoverished —
the minibatch is "too sparse," and accuracy suffers. So *inter-layer connectivity* inside a
minibatch is itself a quantity to protect.

## Baselines

The prior scalable-GCN methods all share a *layer-sampling* meta-structure: build a complete GCN on
the full training graph, then sample nodes or edges *at each layer* to bound the support, and
propagate through the sampled GCN.

- **GraphSAGE (Hamilton et al. 2017).** For each node, uniformly sample a fixed budget of neighbors
  (typically 2–50) at each layer. Core idea: cap the per-node fanout. Gap: the fanout still
  *multiplies* across layers — an L-layer net pulls in roughly budget^L support nodes per output —
  so cost still grows with depth.

- **Importance-weighted layer sampling (Chen et al. 2018, "adaptive sampling" precursor) / weighted
  neighbor sampling.** Assign each neighbor an importance score and sample by it, hoping to lose less
  information than uniform sampling. Gap: still per-node, per-layer, so the multiplicative blow-up
  remains.

- **VR-GCN / S-GCN (Chen, Zhu & Song 2018).** Restrict to as few as two support nodes per layer by
  reusing *historical* (stale) activations to correct for the missing neighbors. Core idea: variance
  reduction via control variates. Gap: even at fanout 2 the cost is high for deep nets, and it must
  store and maintain historical activations for every node.

- **FastGCN (Chen, Ma & Xiao 2018).** Sample nodes *independently* at each layer by importance
  sampling (probability ∝ ‖Â_{:,u}‖²), giving a *constant* sample size per layer — fanout 1, no
  depth blow-up. Core idea: treat each layer's nodes as i.i.d. samples and reweight. Gap: because the
  layers are sampled independently, a node chosen at layer ℓ+1 may have *no* sampled neighbor at
  layer ℓ — the minibatch becomes too sparse, inter-layer connectivity breaks, and accuracy drops
  (the probability an input node "survives" L independent samplers decays as (1−(1−p)^d)^{L−1}).

- **AS-GCN (Huang et al. 2018).** Add a *learned* sampling network that samples each layer
  conditioned on the nodes already chosen in the next layer, restoring connectivity and accuracy. Gap:
  the adaptive sampler is expensive and introduces extra parameters to train.

- **ClusterGCN (Chiang et al. 2019).** Skip layer sampling entirely: partition the graph into dense
  clusters in preprocessing, and form each minibatch from a random set of clusters (keeping
  intra-cluster edges). Core idea: a cluster is a self-contained subgraph, so no neighbor explosion.
  Gap: the clustering is a heuristic, and because clusters are fixed and selected non-uniformly the
  minibatch is a *biased* estimator of the full-batch loss, with no correction.

## Evaluation settings

The natural yardstick is inductive node classification on large graphs where full-batch training is
infeasible, measuring both accuracy and training time:

- **PPI** (protein–protein interaction; multi-label node classification, multiple graphs — a standard
  inductive benchmark where test graphs are entirely unseen).
- **Reddit** (posts as nodes, edges if the same user comments; large, single graph).
- **Flickr** (images as nodes from NUS-WIDE; edges if images share properties; 500-dim bag-of-words
  features; 7 classes).
- **Yelp** (users as nodes, friendship edges; node features from Word2Vec over review words; multi-label
  business categories).
- **Amazon** (products as nodes, co-purchase edges; features from SVD of 4-gram review text; product
  categories).

Metric: micro-averaged F1 for node classification (multi-class or multi-label). Protocol: a GCN
produces node embeddings, an output layer predicts labels, train by SGD/Adam; select the model with
best validation F1 and report test F1; evaluation is full-batch (no sampling). Wall-clock training
time to convergence is reported alongside accuracy, since efficiency is half the point. Architecture
variants used as backbones include deeper GCNs, jumping-knowledge networks (Xu et al. 2018), and
graph attention networks (Veličković et al. 2018).

## Code framework

The primitives that exist: a GCN layer doing normalized neighbor aggregation, an Adam optimizer, a
cross-entropy / multi-label loss, and sparse adjacency in CSR form. What is missing is the *minibatch
construction* — how to carve the large graph into manageable pieces and how to make the resulting
gradient faithful. The scaffold leaves the sampler and the reweighting as empty slots.

```python
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize_adj(adj):
    """Â = D^{-1} A  (row-normalized)."""
    deg = np.asarray(adj.sum(1)).flatten()
    dinv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg != 0)
    return sp.diags(dinv).dot(adj).tocsr()

class GCN(nn.Module):
    """A complete L-layer GCN on whatever (adj, x) it is given."""
    def __init__(self, in_dim, hidden, num_classes, num_layers):
        super().__init__()
        dims = [in_dim] + [hidden] * (num_layers - 1)
        self.weights = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.out = nn.Linear(dims[-1], num_classes)

    def forward(self, adj_norm, x):
        h = x
        for W in self.weights:
            h = F.relu(torch.sparse.mm(adj_norm, W(h)))   # σ( Â · (W h) )
        return self.out(h)

def sample_minibatch(adj_train):
    """Produce one minibatch from the training graph. THE missing piece. TODO."""
    pass

def reweight(loss_per_node, aggregation):
    """Correct the minibatch estimate so it matches the full-graph quantity. TODO."""
    pass

def train(adj_train, features, labels, model, opt, num_steps):
    # preprocessing: set up the sampler and any normalization coefficients
    # TODO: prepare_minibatching(adj_train)
    for _ in range(num_steps):
        # batch = sample_minibatch(adj_train)        # the carved-out piece
        # logits = model(batch.adj_norm, features[batch.nodes])
        # loss   = weighted_loss(logits, labels[batch.nodes])   # TODO: reweight
        # opt.zero_grad(); loss.backward(); opt.step()
        pass
```
