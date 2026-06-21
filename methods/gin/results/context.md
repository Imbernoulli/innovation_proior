# Context

## Research question

Graphs are everywhere — molecules, social and biological networks, citation graphs, knowledge bases — and the central modeling task is to turn the *structure* of a graph (and whatever features sit on its nodes) into a vector that a downstream classifier can use, either per-node (node classification, link prediction) or for the whole graph (graph classification).

A family of neural networks does this by *neighborhood aggregation* (equivalently, *message passing*): each node repeatedly pulls in the feature vectors of its neighbors, mixes them with its own, and updates its representation; after k rounds a node's vector summarizes its k-hop surrounding structure, and a graph-level vector is obtained by pooling all node vectors. These models work well empirically and many variants exist, each picking an aggregator (mean, max, an attention-weighted average, an LSTM), a combine step, and a pooling, then validating on benchmarks.

The question this raises is one of *representational power*: given two non-isomorphic graphs, when does a neighborhood-aggregation network produce different embeddings for them? Is there an upper bound on what any model of this family can distinguish, independent of width, depth, or training? And does the choice of aggregator (mean vs max vs sum) and of per-layer transform (a single linear+nonlinearity vs a deeper net) change what the model can tell apart, not just how it optimizes? This calls for a yardstick for "discriminative power" and an account of which graph structures the popular aggregators can and cannot distinguish.

## Background

**The message-passing / neighborhood-aggregation template.** A graph is G=(V,E) with node features X_v. Modern graph networks share a recursive form. Writing h_v^{(k)} for node v's representation after k layers (h_v^{(0)}=X_v) and N(v) for v's neighbors:

```
a_v^{(k)} = AGGREGATE^{(k)}( { h_u^{(k-1)} : u ∈ N(v) } )
h_v^{(k)} = COMBINE^{(k)}( h_v^{(k-1)}, a_v^{(k)} )
```

For graph-level tasks a permutation-invariant READOUT collapses the final node features into one graph vector h_G = READOUT({ h_v^{(K)} : v ∈ G }) (e.g. summing or averaging all node vectors). The whole design space lives in the choices of AGGREGATE, COMBINE, and READOUT. The intuition is that after k iterations h_v encodes the rooted subtree of height k around v.

**The Weisfeiler-Lehman (WL) graph isomorphism test.** Deciding whether two graphs are isomorphic is a notoriously hard combinatorial problem with no known polynomial-time algorithm in general. The 1-dimensional WL test ("naïve vertex refinement", Weisfeiler & Lehman 1968) is a fast, powerful heuristic that distinguishes a very broad class of graphs (Babai & Kucera 1979). It runs *color refinement*: every node starts with a label (color); each round, a node's new label is an injective hash of the pair (its own current label, the multiset of its neighbors' current labels),

```
l_v^{(k)} = HASH( l_v^{(k-1)}, {{ l_u^{(k-1)} : u ∈ N(v) }} ).
```

Two graphs are declared non-isomorphic the moment their multisets of node labels differ at some iteration. The test is not complete — it fails on certain symmetric corner cases such as regular graphs (Cai-Fürer-Immerman constructions) — but it is correct whenever it does declare non-isomorphism. Like neighborhood aggregation, each round it processes a node from "its own label plus the multiset of its neighbors' labels"; its HASH is injective, so distinct (label, neighbor-multiset) pairs are assigned distinct new labels.

**The WL subtree kernel** (Shervashidze et al. 2011) turns WL into a graph similarity: a node's label at iteration k is exactly a height-k rooted subtree pattern, and the kernel's feature vector for a graph is the histogram of these subtree-labels over all iterations. So WL labels correspond to counts of rooted subtrees.

**Multisets.** A node's neighbors carry feature vectors that can *repeat* — two neighbors may be identical. The abstraction for "the bag of neighbor features" is therefore a multiset, a set with multiplicities: X=(S,m) where S is the underlying set of distinct elements and m: S→ℕ≥1 gives each element's multiplicity. (Throughout, node input features are assumed to come from a countable universe; for finite graphs the deeper-layer features stay countable too.)

**Permutation-invariant set functions.** Zaheer et al. (2017) ("Deep Sets") showed that any permutation-invariant function on a *set* can be written ρ(Σ_{x∈S} f(x)) — sum-decomposition with learned f and ρ. This is the template for "aggregate a bag, then transform"; it is stated for *sets*, where there are no repeated elements.

**Universal approximation.** A multilayer perceptron with at least one hidden layer can approximate any continuous function on a compact domain to arbitrary accuracy (Hornik et al. 1989, 1991). This is what licenses replacing an abstract function "f" or "ρ" in a decomposition with a trainable network.

**The aggregators already on the table, and their observed behavior.** Empirical reports describe distinct behaviors for the common poolings. Max-pooling is robust to outliers and tends to pick out a "skeleton" (Qi et al. 2017 show this for 3D point clouds), while mean-aggregation models are strong on node-classification tasks with rich node features.

## Baselines

**Graph Convolutional Network — GCN (Kipf & Welling 2017).** Merges aggregate and combine into one mean over the closed neighborhood, followed by a linear map and a ReLU:

```
h_v^{(k)} = ReLU( W · MEAN{ h_u^{(k-1)} : u ∈ N(v) ∪ {v} } ).
```

Two features define it: the aggregator is a **mean** (over self and neighbors), and the per-layer transform is a **single linear layer + nonlinearity** (a 1-layer perceptron). It is validated on benchmarks.

**GraphSAGE (Hamilton et al. 2017).** Separates aggregate and combine. Its pooling variant is

```
a_v^{(k)} = MAX( { ReLU( W · h_u^{(k-1)} ) : u ∈ N(v) } ),
h_v^{(k)} = W' · [ h_v^{(k-1)} , a_v^{(k)} ]   (concat then linear).
```

It also offers mean and LSTM aggregators. The max-pooling aggregator is the simplest to reason about: an element-wise max over the neighbors. Like GCN it is justified empirically.

**WL subtree kernel + SVM (Shervashidze et al. 2011).** Not a neural model but the natural yardstick for "how much graph structure can you capture": run WL, histogram the subtree-labels, feed to an SVM. The labels are one-hot (discrete hashes), and the kernel measures structural overlap. Strong on graph classification.

**Other graph-classification baselines of the time.** Diffusion-convolutional networks (Atwood & Towsley 2016), PATCHY-SAN (Niepert et al. 2016), Deep Graph CNN / DGCNN (Zhang et al. 2018), and Anonymous Walk Embeddings (Ivanov & Burnaev 2018) are alternative graph-classification architectures used as comparison points; each commits to a specific architecture.

## Evaluation settings

The natural testbed is graph classification on standard benchmarks (Yanardag & Vishwanathan 2015):

- **Bioinformatics datasets** with categorical node labels: MUTAG (188 mutagenic compounds, 7 node labels), PTC (344 compounds, carcinogenicity, 19 labels), NCI1 (4110 compounds screened against tumor cell lines, 37 labels), PROTEINS (nodes are secondary-structure elements; 3 labels).
- **Social-network datasets** without node features: IMDB-BINARY, IMDB-MULTI (actor ego-networks, genre classification), REDDIT-BINARY, REDDIT-MULTI5K (discussion-thread graphs, subreddit/community classification), COLLAB (collaboration ego-networks, field classification). Since these have no intrinsic node features, a feature must be supplied — either a constant (uninformative) feature, or a one-hot encoding of node degree.

The protocol is 10-fold cross-validation with an SVM as the downstream classifier for kernels, reporting mean ± std accuracy across folds. A diagnostic separate from test accuracy is **training-set accuracy**: how well a model can *fit* the training graphs, which probes the ability to represent them independent of generalization. The WL subtree kernel's training fit is a natural reference point for this diagnostic. Standard training machinery of the era applies: Adam optimizer, batch normalization on hidden layers, dropout, learning-rate decay; node features one-hot for categorical labels.

## Code framework

The scaffold is a generic message-passing graph classifier: a data pipeline that batches variable-size graphs by block-diagonal stacking, a stack of neighborhood-aggregation layers built from linear layers, ReLU, batch norm, a permutation-invariant graph pooling matrix, and a standard training loop. The open slots are the per-layer update and the graph readout.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """A stack of Linear, BatchNorm, and ReLU layers; num_layers==1 is a Linear map."""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super().__init__()
        # TODO: choose the capacity of the per-layer transform.
        pass

    def forward(self, x):
        pass

class GraphCNN(nn.Module):
    """Neighborhood-aggregation graph classifier with open update and readout slots."""
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, device):
        super().__init__()
        self.num_layers = num_layers
        self.final_dropout = final_dropout
        self.device = device
        # TODO: instantiate per-layer transforms, normalization, and prediction heads.
        pass

    def preprocess_neighbors(self, batch_graph):
        # TODO: build the batched neighbor structure used by the aggregation step.
        pass

    def preprocess_graph_pool(self, batch_graph):
        # TODO: build a sparse graph-to-node pooling matrix.
        pass

    def next_layer(self, h, layer, neighbor_struct=None):
        # TODO: one round of neighborhood aggregation and node update.
        pass

    def forward(self, batch_graph):
        # TODO: stack node features, build batched neighbor structures,
        # run num_layers-1 rounds of message passing, then turn the node
        # representations into class scores.
        pass
```
