# Context: scalable minibatch training of deep graph convolutional networks

## Research question

Graph convolutional networks learn representations of an attributed graph G(V, E): each node v has a
feature vector x_v, and a GCN layer mixes a node's representation with its neighbors',

  x_v^(ℓ+1) = σ( Σ_{u∈V} Â_{v,u} (W^(ℓ))ᵀ x_u^(ℓ) ),   Â = D^{-1}A,

so after L layers each node has gathered information from its L-hop neighborhood. The standard
training recipe is *full-batch*: hold the entire graph and all activations in memory and take one
gradient step over all nodes.

The question: how to train a GCN by *minibatches* on a large graph, including when the network is
*deep* (more than two layers) and in the *inductive* setting (test nodes' features and connections
are absent during training, so the model must generalize to unseen graph structure).

## Background

**The GCN propagation rule and full-batch training.** Spectral graph convolution, made practical by
localized Chebyshev filters (Defferrard et al. 2016) and simplified to the first-order rule (Kipf &
Welling 2017), gives the layer above: a (normalized) neighbor aggregation, a linear map W^(ℓ), and a
nonlinearity. These works target small graphs and train in full batch, recomputing every node's
representation at every layer each step.

**Support sets and depth.** Computing the output of a single node v under an L-layer GCN uses its
neighbors' layer-(L−1) values; each of those uses *their* neighbors' layer-(L−2) values; and so on.
The set of "support" nodes required to produce one output grows multiplicatively with depth — for a
graph with average degree d, on the order of d^L nodes.

**Bias and variance of a minibatch estimator.** A minibatch gradient is an estimate of the
full-batch gradient, and two properties characterize it. *Bias*: if some nodes or edges are included
in minibatches more often than others — which happens when sampling is non-uniform — the estimator
weights frequently-sampled nodes more heavily. *Variance*: the sampling distribution sets the
estimator's noise level, which governs how quickly SGD converges. The classic tools are
inverse-probability reweighting (an event included with probability p contributes with weight 1/p to
stay unbiased) and importance sampling (choose the sampling probabilities to minimize variance,
typically proportional to the magnitude of each term's contribution; the Cauchy–Schwarz inequality
gives the optimum).

**Connectivity within a minibatch.** For a GCN, a node's representation depends on its access to its
neighbors during propagation: the neighbors a node needs at the previous layer must be present in
the minibatch for its computed representation to draw on them.

## Baselines

The prior scalable-GCN methods share a *layer-sampling* meta-structure: build a complete GCN on the
full training graph, then sample nodes or edges *at each layer* to bound the support, and propagate
through the sampled GCN.

- **GraphSAGE (Hamilton et al. 2017).** For each node, uniformly sample a fixed budget of neighbors
  (typically 2–50) at each layer, capping the per-node fanout.

- **Importance-weighted layer sampling (Chen et al. 2018) / weighted neighbor sampling.** Assign
  each neighbor an importance score and sample by it, per node and per layer.

- **VR-GCN / S-GCN (Chen, Zhu & Song 2018).** Restrict to as few as two support nodes per layer by
  reusing *historical* (stale) activations to correct for the missing neighbors — variance reduction
  via control variates, storing and maintaining historical activations for every node.

- **FastGCN (Chen, Ma & Xiao 2018).** Sample nodes *independently* at each layer by importance
  sampling (probability ∝ ‖Â_{:,u}‖²), giving a *constant* sample size per layer — fanout 1, no
  depth blow-up. Each layer's nodes are treated as i.i.d. samples and reweighted.

- **AS-GCN (Huang et al. 2018).** Add a *learned* sampling network that samples each layer
  conditioned on the nodes already chosen in the next layer, with extra parameters to train.

- **ClusterGCN (Chiang et al. 2019).** Skip layer sampling: partition the graph into dense clusters
  in preprocessing, and form each minibatch from a random set of clusters (keeping intra-cluster
  edges), so each minibatch is a self-contained subgraph.

## Evaluation settings

The yardstick is inductive node classification on large graphs where full-batch training is
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
time to convergence is reported alongside accuracy. Architecture variants used as backbones include
deeper GCNs, jumping-knowledge networks (Xu et al. 2018), and graph attention networks (Veličković
et al. 2018).

## Code framework

The primitives that exist: a GCN layer doing normalized neighbor aggregation, an Adam optimizer, a
cross-entropy / multi-label loss, and sparse adjacency in CSR form. The *minibatch construction* for
training a GCN on a large graph is left as an empty slot.

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

def make_minibatch(adj_train):
    """Construct one minibatch for GCN training on the large graph. TODO."""
    pass

def train(adj_train, features, labels, model, opt, num_steps):
    for _ in range(num_steps):
        # batch = make_minibatch(adj_train)
        # logits = model(batch.adj_norm, features[batch.nodes])
        # loss   = loss_fn(logits, labels[batch.nodes])
        # opt.zero_grad(); loss.backward(); opt.step()
        pass
```
