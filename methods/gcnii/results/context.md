# Context: deep message passing for semi-supervised node classification

## Research question

On a single citation graph — nodes are documents with sparse bag-of-words features, edges are
citations, and only a handful of nodes per class are labeled — I want to classify the rest. The
established way to do this is to make the predictor a function of both features and the adjacency,
stacking message-passing layers so that each node mixes in its neighbors. The best-performing such
models are *shallow*: they typically use two layers, and a two-layer model reaches a node's two-hop
neighborhood. The question is how to design the per-layer propagation-and-transform update, and the
model that stacks it, for this semi-supervised setting.

## Background

A node-classification layer aggregates each node's neighbors and updates its representation. The
renormalized graph convolution fixes the propagation operator to
$\tilde{\mathbf P} = \tilde{\mathbf D}^{-1/2}\tilde{\mathbf A}\tilde{\mathbf D}^{-1/2}$, the symmetric
normalized adjacency of the self-looped graph ($\tilde{\mathbf A}=\mathbf A+\mathbf I$,
$\tilde{\mathbf D}=\mathbf D+\mathbf I$), and a layer is
$\mathbf H^{(\ell+1)}=\sigma(\tilde{\mathbf P}\,\mathbf H^{(\ell)}\mathbf W^{(\ell)})$.

A property that frames the analysis here is *over-smoothing* (observed by Li et al. 2018): as the
number of such layers grows, node representations converge toward each other and become
indistinguishable. Two prior results characterize the mechanism. Wu et al. 2019 (SGC) show that
stacking $K$ convolution layers is, up to the weight matrices, a fixed order-$K$ polynomial filter
$\tilde{\mathbf P}^{K}\mathbf x$ with predetermined coefficients. Wang et al. 2019 observe that the
residual-augmented operator $(\mathbf I + \tilde{\mathbf P})/2$ is a *lazy random walk*, which
converges to its stationary distribution; in the limit the representation depends only on node degrees
and a single global inner product with the features. Independently, Oono & Suzuki 2020 prove that
$K$-layer GCN features converge to a low-dimensional subspace at a rate governed by $s^{K}$, where $s$
is the largest singular value of the weight matrices.

Several established results sit nearby. From computer vision, He et al. 2016 (ResNet) made very deep
networks trainable with *residual connections* — a layer learns a correction added to its input rather
than a full remap — and Hardt & Ma 2017 prove that for a *linear* residual network
$\mathbf H^{(\ell+1)}=\mathbf H^{(\ell)}(\mathbf W^{(\ell)}+\mathbf I)$ the optimal weights have small
norm and the only critical point is the global minimum. From the graph side, several methods *decouple*
propagation from transformation. SGC collapses many hops into one linear layer. PPNP/APPNP (Klicpera et
al. 2019) replace the fixed power filter with a personalized-PageRank operator and, in the truncated
form, iterate
$\mathbf H^{(\ell+1)}=(1-\alpha)\tilde{\mathbf P}\mathbf H^{(\ell)}+\alpha\mathbf H^{(0)}$ with a
*teleport* back to the transformed input $\mathbf H^{(0)}=f_\theta(\mathbf X)$ — propagating many hops
without deepening the network; APPNP's own finding is that repeated nonlinearities overfit on these
small-label datasets. GDC (Klicpera et al. 2019) generalizes the diffusion. JKNet (Xu et al. 2018)
keeps every layer's output and combines them at the end.

## Baselines

- **Renormalized graph convolution (Kipf & Welling 2017).**
  $\mathbf H^{(\ell+1)}=\sigma(\tilde{\mathbf P}\mathbf H^{(\ell)}\mathbf W^{(\ell)})$. Cheap, linear in
  edges, strong at two layers. $K$ layers realize the fixed-coefficient order-$K$ filter
  $\tilde{\mathbf P}^{K}\mathbf x$ that converges to a degree-only stationary state.
- **Graph attention (Veličković et al. 2018).** Replaces the fixed degree-based edge weight with a
  learned attention coefficient $\alpha_{ij}=\mathrm{softmax}_j(\mathrm{LeakyReLU}(\vec a^\top[\mathbf W
  h_i\Vert \mathbf W h_j]))$. Learns *which* neighbor matters; best at two layers.
- **Mean-aggregation message passing (Hamilton et al. 2017, GraphSAGE).** Separates self and neighbor
  ($\mathbf H^{(\ell+1)}=\sigma(\mathbf W_{\text{self}}\mathbf H^{(\ell)} + \mathbf W_{\text{neigh}}\,
  \mathrm{mean}_{j\in N}\mathbf H^{(\ell)}_j)$), inductive. The neighbor pool is a fixed linear mean.
- **Decoupled propagation (APPNP, Klicpera et al. 2019).** $\mathbf H^{(\ell+1)}=(1-\alpha)\tilde{\mathbf
  P}\mathbf H^{(\ell)}+\alpha\mathbf H^{(0)}$ with $\mathbf H^{(0)}=f_\theta(\mathbf X)$. Reaches many
  hops with a teleport that keeps a fraction $\alpha$ of the input; the per-layer step is linear, with
  no weight matrix or nonlinearity per hop.
- **ResNet-style residual GCN (Kipf & Welling 2017).** $\mathbf H^{(\ell+1)}=\sigma((\tilde{\mathbf
  P}\mathbf H^{(\ell)}+\mathbf H^{(\ell)})\mathbf W^{(\ell)})$, a skip to the *previous* layer.
- **JKNet (Xu et al. 2018).** Concatenates/max-pools all layer outputs at the end.

## Evaluation settings

Three citation networks with the standard Planetoid semi-supervised splits — Cora (2,708 nodes, 7
classes, 1,433 features), CiteSeer (3,327 nodes, 6 classes, 3,703 features), PubMed (19,717 nodes, 3
classes, 500 features) — 20 labeled nodes per class, 500 validation nodes, and 1,000 test nodes.
The metric for these transductive citation tasks is classification accuracy. Full-batch training uses
Adam at learning rate 0.01, validation-based early stopping with patience 100, and separate weight
decay for propagation-layer weights and dense input/output weights. The split, loss, and protocol are
fixed; the open design choice is the deep propagation layer.

## Code framework

The normalized sparse adjacency and the shallow input/output classifier are already standard; the open
slot is the *layer update rule* and the *model that stacks it deep*.

```python
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    """One sparse-adjacency graph-convolution layer. The deep update rule is the open slot."""

    def __init__(self, channels):
        super().__init__()
        # TODO: per-layer parameters for the deep update rule.
        self.weight = Parameter(torch.FloatTensor(channels, channels))
        pass

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, adj, layer_index, **kwargs):
        propagated = torch.spmm(adj, x)   # P̃ H
        # TODO: the deep update rule.
        raise NotImplementedError


class NodeClassifier(nn.Module):
    """Input map -> deep graph-convolution stack -> output classifier."""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers, dropout):
        super().__init__()
        self.dropout = dropout
        self.input_layer = nn.Linear(in_channels, hidden_channels)
        # TODO: `num_layers` graph-convolution layers using the deep update rule.
        self.output_layer = nn.Linear(hidden_channels, out_channels)
        pass

    def forward(self, x, adj):
        # TODO: build the first hidden representation, run the deep stack, and classify.
        raise NotImplementedError
```
