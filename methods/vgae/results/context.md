# Context: Unsupervised Link Prediction On Attributed Graphs

## Research Problem

We have one undirected, unweighted graph rather than a collection of independent examples. Its
nodes may be documents, users, proteins, or entities; its observed edges are an incomplete view of
which node pairs are related; and each node may also carry a feature vector such as a bag-of-words
description. The practical task is link prediction: remove a small set of true edges, sample the
same number of non-edges, train only on the remaining graph, and rank the held-out true edges above
the sampled false ones.

The training signal is awkward. There are no external labels for links beyond the adjacency itself,
and in a sparse graph the `N^2` possible node pairs are overwhelmingly zeros. A model trained as an
ordinary binary classifier over all pairs can minimize loss by predicting "no edge" almost
everywhere. A useful solution must therefore learn a per-node representation and a pair score from
the graph's own connectivity while keeping rare positive edges from being drowned by easy negatives.

The harder modeling question is how to use both kinds of evidence. Structure-only methods can learn
communities and proximity from edges, but ignore node attributes. Feature-only methods can compare
node descriptions, but ignore the relation graph. The open slot is an end-to-end unsupervised model
that maps `(X, A_train)` to node representations and uses them to score candidate pairs.

## Available Ingredients

Spectral graph methods embed nodes through eigenvectors of a graph matrix. With the symmetric
normalized Laplacian `L = I - D^{-1/2} A D^{-1/2}`, the eigenbasis captures large-scale graph
structure, and a low-dimensional projection can be used as node features for downstream scoring.
This is a clean global summary of topology, but it is fixed, linear, expensive to compute at scale,
and normally disconnected from node attributes.

Random-walk embedding methods take a different route. A method such as DeepWalk generates truncated
walks from the graph, treats each walk like a sentence, and trains a SkipGram model so nodes that
co-occur in walks receive nearby vectors. This gives useful unsupervised node embeddings and avoids
an eigendecomposition, but it is still a structure-only pipeline: walk generation and word2vec-style
training optimize a proxy objective over node IDs, not a single feature-aware graph model.

A newer graph neural layer can combine structure and features directly. Starting from spectral
graph convolution, a first-order approximation gives a local propagation rule. In its normalized
form, after adding self-loops, one layer is

```text
H^{l+1} = sigma(D_tilde^{-1/2} A_tilde D_tilde^{-1/2} H^l W^l),     H^0 = X.
```

Each node receives a normalized mixture of its own features and its neighbors' features, then a
learned linear map and nonlinearity. Stacking two layers gives access to a two-hop neighborhood.
The computation is sparse-dense multiplication and scales linearly in the number of edges for fixed
feature widths. In its original use, however, this encoder is trained with supervised node labels.

Variational auto-encoding supplies a separate ingredient: a way to fit latent-variable models when
the true posterior is intractable. For a latent `z`, prior `p(z)`, approximate posterior `q(z|x)`,
and likelihood `p(x|z)`, the lower bound is

```text
L = E_q[log p(x|z)] - KL(q(z|x) || p(z)).
```

The first term rewards reconstruction; the second regularizes the approximate posterior toward the
prior. For a diagonal Gaussian posterior, the sampling step can be written as
`z = mu + sigma * eps`, `eps ~ N(0,I)`, so gradients pass through `mu` and `sigma`. The Gaussian
KL against a standard normal has a closed form:

```text
-KL = 0.5 * sum_j(1 + log sigma_j^2 - mu_j^2 - sigma_j^2).
```

This variational machinery is normally written for independent datapoints. A graph setting still
has to decide what the datapoints, latent variables, and reconstruction target should be.

## Prior Baselines

Spectral clustering and related matrix-factor methods provide a topology-only embedding matrix
`Z`. A candidate link can be scored by an affinity such as `z_i^T z_j`, but the embedding is not
learned from node features and does not form an end-to-end neural encoder.

DeepWalk and later random-walk methods also produce one vector per node from graph structure. They
capture neighborhood statistics through co-occurrence in walks, but they leave feature vectors
outside the model and require a multi-stage training pipeline.

The supervised graph-convolutional classifier consumes both `X` and `A`, but its objective is node
label cross-entropy. In link prediction there may be no node labels, and a representation optimized
for a label taxonomy is not automatically optimized to explain missing edges.

The important gap is therefore not the absence of any node embedding method. It is the absence, in
this setup, of a single unsupervised feature-and-structure model whose training objective is the
graph's connectivity.

## Evaluation Setting

The common benchmark uses citation networks such as Cora, Citeseer, and Pubmed. Nodes are
documents, edges are citation links treated as undirected, and features are sparse bag-of-words
vectors. A validation set of about five percent of edges and a test set of about ten percent of
edges are removed from the graph. For validation and testing, the same number of unconnected node
pairs is sampled as negatives.

The model trains on the incomplete graph and all node features. It then scores held-out positive
and negative pairs. The reported metrics are AUC and average precision. Featureless variants are
also meaningful: replace the feature matrix with the identity so each node is represented by its
own one-hot indicator and all usable signal comes from structure.

Because the graphs are modest citation networks and the graph encoder uses sparse operations,
full-batch training is feasible. Hyperparameter selection is done on validation links, not on test
links.

## Scaffold Slot

The implementation harness expects a module with an encoder that builds node representations from
the training graph, a decoder that scores candidate pairs, and a `forward` method that connects the
two. The training loop supplies node features `x`, observed training edges `edge_index`, candidate
pairs `edge_label_index`, and binary labels for those candidate pairs. The scaffold has not chosen
the representation form, pair-scoring rule, or any regularizer.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinkPredictor(nn.Module):
    """Encode nodes from the training graph and score candidate node pairs."""

    def __init__(self, in_channels, hidden_channels, embedding_channels, dropout):
        super().__init__()
        self.dropout = dropout
        # open slot: representation builder, pair scorer, and any regularizer

    def encode(self, x, edge_index):
        raise NotImplementedError

    def decode(self, z_src, z_dst):
        raise NotImplementedError

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z[edge_label_index[0]], z[edge_label_index[1]])


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index, data.edge_label_index)
    recon = F.binary_cross_entropy_with_logits(logits, data.edge_label)
    loss = recon
    loss.backward()
    optimizer.step()
    return float(loss)
```
