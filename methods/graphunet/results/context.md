# Context: Encoder-Decoder Representation Learning on Graphs

## Research question

Many node-level prediction problems on graphs - node classification, link prediction, and network embedding - have the same shape as pixel-wise prediction on images: every input unit needs its own output, and a good prediction for one unit may depend on context several hops away. On images, the encoder-decoder with skip connections gives this multi-scale behavior. The contracting path repeatedly applies convolution and pooling so deep features summarize larger regions; the expansive path restores resolution; the skip connections copy high-resolution features across the bottleneck so localization is not lost.

The question is how to build the same kind of high-level-encode / decode-to-resolution machine for graph data. Graph neural networks already give a convolution-like primitive: aggregate a node's neighborhood, transform the result, and repeat. What is missing is the rest of the U-shaped recipe: a way to down-sample a graph to a smaller graph, and a matching way to up-sample back to the original node set. The difficulty is intrinsic. A graph has no fixed spatial locality and no canonical ordering of its nodes, so grid pooling operations such as fixed-window max pooling do not transfer directly. A useful solution has to decide which nodes survive a down-sampling step, preserve enough connectivity among them for later message passing, and remember how to restore features to the original node identities.

## Background

**Graph convolution.** The dominant graph-convolution layer is the GCN of Kipf & Welling. It comes from a first-order approximation of spectral graph convolution. A spectral filter is $g_\theta \star x = U g_\theta(\Lambda) U^\top x$ with $U$ the eigenvectors of the normalized graph Laplacian $L = I_N - D^{-1/2}AD^{-1/2}$, but the eigendecomposition is expensive and global. Truncating to a first-order localized filter and applying the renormalization trick gives
$$X_{\ell+1} = \sigma\!\left(\hat D^{-1/2}\hat A\hat D^{-1/2}X_\ell W_\ell\right),\qquad \hat A = A + I,$$
where the self-loop lets a node keep its own feature and $\hat D$ is the degree matrix of $\hat A$. Each layer mixes a node with its one-hop neighbors; stacked layers increase the hop radius, but the node set and graph resolution stay fixed.

**Inductive aggregation.** GraphSAGE recasts message passing as sample-and-aggregate so the learned aggregator can be applied to unseen nodes or graphs:
$$h_v^k = \sigma\!\left(W^k\cdot\mathrm{CONCAT}\big(h_v^{k-1},\ \mathrm{AGG}\{h_u^{k-1}:u\in\mathcal N(v)\}\big)\right),$$
with optional normalization after each layer. The mean aggregator is close in spirit to a GCN layer, while concatenating a node's previous representation with aggregated neighbor information gives the layer an explicit self/neighbor split. It is a natural backbone when the graph is large, inductive, or changes between layers.

**The U-Net template.** The image U-Net repeats convolution and pooling in a contracting path, then mirrors the process with up-sampling in an expansive path. The skip connections are the load-bearing detail: pooling throws away precise spatial information that the decoder needs, so encoder feature maps at matching resolutions are copied forward and fused back in during up-sampling. The template depends on two grid facts that graphs do not have: a regular ordering and local pooling windows.

**Why deeper flat GNNs are not enough.** A flat GNN only propagates over the existing graph at a fixed resolution. More layers increase the hop radius, but they do not create a coarser graph on which a feature can represent a larger region as a single unit. Deep GCN stacks also tend to over-smooth node representations, pushing neighboring features toward similar values. The missing operation is genuine graph down-sampling paired with graph up-sampling.

**Pooling and up-sampling before a graph encoder-decoder.** Several down-sampling ideas existed. Spectral or clustering coarsening fixes a hierarchy before training, so the hierarchy cannot adapt to task features. Global pooling collapses all nodes into one graph-level vector, which is too coarse for a decoder that must return node-level outputs. $k$-max pooling over feature channels selects units rather than whole nodes, so selected activations may come from different nodes and no coherent induced graph remains. Soft-assignment pooling methods learn a cluster-assignment matrix $S$ and form $X' = S^\top Z$, $A' = S^\top A S$, but this introduces a dense assignment matrix, a second pooling network, fixed cluster counts, and often auxiliary objectives to make the assignments behave. On the up-sampling side, image decoders have deconvolution and unpooling layers; graphs had no standard operation that restores a coarsened graph to the original node identities so skip connections can align row-for-row.

## Baselines

**GCN-only deep network.** Stack GCN layers, optionally with residual or skip connections, at the original graph resolution. This tests whether depth alone can provide the needed context. It has no learned coarsening and no inverse operation.

**Flat GraphSAGE or GAT backbones.** Replace the local convolution primitive with sampled aggregation or attention over neighbors. These strengthen each message-passing layer but keep the same node set at every layer.

**Learned soft-assignment pooling.** Use a pooling GNN to produce a soft cluster-assignment matrix and an embedding GNN to produce node embeddings, then pool with $S^\top Z$ and $S^\top A S$. This is learned and permutation-invariant, but the dense assignment and auxiliary losses make it heavier than a node-selection layer.

**Fixed graph coarsening plus a decoder.** Precompute a graph hierarchy with clustering or spectral coarsening and run an encoder-decoder on that hierarchy. The limitation is that the coarsening is detached from the task loss and the decoder must still solve the alignment problem between coarse nodes and original nodes.

## Evaluation settings

**Transductive node classification.** Citation networks such as Cora, Citeseer, and Pubmed use document nodes, citation edges, sparse bag-of-words features, and a small labeled training set with the full graph visible during training. The metric is node-classification accuracy.

**Inductive graph classification.** Protein and collaboration datasets contain many graphs whose sizes vary. Testing graphs are unseen during training, so a graph-level readout over node embeddings is needed after the encoder-decoder body. Standard evaluation uses cross-validation and graph-classification accuracy.

**Regularization and implementation conventions.** Models of this period commonly use $L_2$ weight decay, dropout on features or adjacency entries, and identity or ReLU activations after graph-convolution layers. Dense educational implementations often row-normalize an adjacency before multiplication; the method equation for GCN uses symmetric normalization with self-loops.

## Code framework

The existing pieces are a graph-convolution layer, adjacency preparation, and task heads. The empty slot is a multi-resolution body that takes a normalized graph and node features, returns same-resolution node features or a list of node-feature states, and preserves node order for any downstream node-level head.

```python
import torch
import torch.nn as nn


def add_self_loops(g):
    g = g.clone()
    idx = torch.arange(g.size(0), device=g.device)
    g[idx, idx] = 1.0
    return g


def norm_g(g):
    degrees = torch.sum(g, 1, keepdim=True).clamp_min(1.0)
    return g / degrees


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, act, p):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = act
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()

    def forward(self, g, h):
        h = self.drop(h)
        h = torch.matmul(g, h)
        h = self.proj(h)
        return self.act(h)


class MultiResolutionBody(nn.Module):
    # TODO: fill the learned graph down/up path while preserving row alignment.
    def __init__(self, *args):
        super().__init__()
        raise NotImplementedError

    def forward(self, g, h):
        raise NotImplementedError


class GraphNet(nn.Module):
    def __init__(self, in_dim, out_dim, dim, act, drop_p):
        super().__init__()
        self.embed = GCN(in_dim, dim, act, drop_p)
        self.body = MultiResolutionBody(dim, act, drop_p)
        self.head = GCN(dim, out_dim, act, drop_p)

    def forward(self, g, h):
        g = norm_g(add_self_loops(g))
        h = self.embed(g, h)
        h = self.body(g, h)
        return self.head(g, h)
```
