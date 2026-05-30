# Context

## Research question

We want a single neural architecture that consumes a graph — a set of nodes, each with a feature vector, joined by edges that may themselves carry features — and produces a vector representation of the *whole graph* suitable for graph-level prediction (e.g. regressing a molecular property from a 2D molecular graph, or classifying whether a molecule is active). The yardstick is the public graph-level leaderboards, where one family of models has held the top spots for years: message-passing graph neural networks (GNNs).

A different family of architectures — the Transformer — has by now displaced recurrent and convolutional models across natural language, speech, and vision. Its self-attention mechanism gives every element a global receptive field in a single layer and scales beautifully with data and compute. The obvious question is whether that same architecture can be turned on graphs and win. So far it has not: on graph-level leaderboards, attempts to use a Transformer have either lost to plain GNNs or have quietly degenerated into a GNN with the neighborhood aggregator swapped for softmax attention. The open problem is therefore concrete: **can a standard Transformer encoder, kept essentially intact, be made competitive on graph representation — and if so, what is the minimal structural information it must be given?**

The crux is a mismatch. Self-attention treats its input as an unordered *set* of vectors and re-weights them purely by feature similarity. A graph is not just a set of node vectors: it carries structure — which nodes are central/important, how far apart two nodes are, and what edges lie between them. None of that structure is visible to raw self-attention. A solution has to inject the graph's structure into the attention computation while preserving the standard Transformer block, so that the resulting model is at least as expressive as the GNNs it must beat.

## Background

**Self-attention is permutation-equivariant and structure-blind.** Given node representations stacked as $H=[h_1^\top,\dots,h_n^\top]^\top\in\mathbb{R}^{n\times d}$, a self-attention layer forms queries, keys, values $Q=HW_Q,\ K=HW_K,\ V=HW_V$, computes the score matrix $A=QK^\top/\sqrt{d_K}$, and outputs $\mathrm{softmax}(A)\,V$. The output for token $i$ is a function only of the *multiset* of input vectors; permute the inputs and the outputs permute identically. For sequences this is fixed by adding positional information (absolute positional encodings, or relative ones that add a learned bias to $A_{ij}$ depending on the offset $i-j$). For a graph there is no canonical ordering of the nodes and no one-dimensional offset between them, so neither absolute nor sequence-relative positional encoding applies directly. As written, an attention layer fed a graph's node features sees no degrees, no distances, and no edges — it ranks neighbors and non-neighbors alike purely by feature similarity.

**Graphs carry structure that a set does not.** Three structural facts are knowable for any graph, independent of any model: (1) *node centrality* — degree centrality (in/out-degree) is a standard measure of how important a node is; (2) *spatial relation* — the shortest-path distance $\phi(v_i,v_j)$ between two nodes, computable for all pairs with Floyd–Warshall, gives a graph-native notion of "how far apart"; (3) *edge features* — edges can carry attributes (a chemical bond type between two atoms), and the edges lying along the path between two nodes describe how they are connected. These are pre-model properties of the data.

**Message-passing GNNs and their ceiling.** Modern GNNs follow an AGGREGATE–COMBINE schema. With $h_i^{(0)}=x_i$,
$$a_i^{(l)}=\mathrm{AGGREGATE}^{(l)}\!\big(\{h_j^{(l-1)}:j\in\mathcal N(v_i)\}\big),\qquad h_i^{(l)}=\mathrm{COMBINE}^{(l)}\!\big(h_i^{(l-1)},a_i^{(l)}\big),$$
and for a graph vector a permutation-invariant READOUT pools the final node states $h_G=\mathrm{READOUT}(\{h_i^{(L)}\})$. Each layer mixes information one hop; the receptive field grows only with depth. This locality is also a limitation: stacking many layers to widen the receptive field tends to over-smooth (node states converge), and the entire family is provably bounded in discriminative power by the one-dimensional Weisfeiler–Lehman (1-WL) color-refinement test, so there exist non-isomorphic graphs no such GNN can ever tell apart.

**The virtual-node / supernode trick.** A known way to give a GNN a global pathway is to add an extra node connected to every node in the graph. It aggregates whole-graph information (much like READOUT) and broadcasts it back to all nodes. It empirically helps on leaderboards, but a naively added supernode can over-smooth the propagation.

**Relative-position bias as a template.** In sequence Transformers, one effective way to encode position is to add a learned scalar bias to the attention score $A_{ij}$ as a function of the relation between $i$ and $j$ (the relative offset), rather than perturbing the inputs. This "bias on the score" mechanism is content-independent and shared across layers, and it is the natural template for injecting a *pairwise* structural quantity into attention.

## Baselines

**Standard Transformer encoder (Vaswani et al., 2017).** A stack of identical blocks, each a multi-head self-attention sublayer and a position-wise feed-forward network (FFN), with residual connections and layer normalization. Multi-head attention runs several attention maps in parallel on projected subspaces and concatenates them; the FFN is two linear layers with a nonlinearity, conventionally widened to $4d$ inside. Layer normalization can be placed after the sublayers (post-LN) or before them (pre-LN); pre-LN is generally easier to optimize for deep stacks. As a graph model the block is structure-blind (see Background): without injected structure it would treat a molecule as a bag of atom embeddings.

**GCN (Kipf & Welling, 2017).** AGGREGATE is a (symmetric-normalized) *mean* over the node and its neighbors; COMBINE is folded in as a single linear map followed by a nonlinearity. Cheap and strong, but mean aggregation cannot count multiplicities (it captures only the *distribution* of neighbor features), the transform is a single linear+nonlinearity rather than a universal approximator, and the receptive field is strictly local.

**GraphSAGE (Hamilton et al., 2017).** Samples a fixed number of neighbors and aggregates them; the pooling variant uses element-wise *max* of a per-neighbor transform, with COMBINE concatenating the center node to the aggregate before a linear map. Max aggregation is robust but captures only the *support* (which distinct neighbor features are present), discarding both counts and proportions; like GCN it is local and bounded by 1-WL.

**GIN (Xu et al., 2019).** AGGREGATE is a *sum* over neighbors, COMBINE is $(1+\epsilon)h_i+\sum_{j\in\mathcal N(i)}h_j$ fed through an MLP. Sum-with-MLP is injective on multisets, so GIN reaches the 1-WL ceiling — the maximally expressive member of this family — but it cannot exceed 1-WL, and its receptive field is still local.

**Edge-feature handling in GNNs.** Two common recipes: add edge features to the incident nodes' features, or fold the incident edges' features into the aggregation at each node. Both only ever move an edge's information to its two endpoints; neither lets a distant node pair "see" the edges that connect them.

**Prior graph-Transformers.** Several lines graft attention onto graphs: restrict attention to a node's neighbors (using the adjacency matrix as an attention mask) to preserve sparsity, and supply node position via spectral features such as Laplacian eigenvectors; others inject Weisfeiler–Lehman labels or hop counts as positional encodings, or add the adjacency / inter-atomic distance matrices directly to the attention probabilities. The common thread — and the gap — is that they either abandon the global receptive field by masking attention to neighbors, or rely on spectral/precomputed positional codes, rather than keeping a full standard Transformer and feeding it a small set of graph-native structural biases.

## Evaluation settings

The natural yardsticks are molecular graph-level benchmarks. **PCQM4M-LSC** (OGB Large-Scale Challenge): ~3.8M molecular graphs, regress the DFT HOMO–LUMO gap from the 2D graph, scored by mean absolute error (MAE); the dataset is large enough to train data-hungry architectures. **OGBG-MolPCBA** (multi-task binary activity classification, scored by average precision) and **OGBG-MolHIV** (binary classification, scored by ROC-AUC), both with the official scaffold split. **ZINC** (Benchmarking-GNN): constrained-solubility regression on a small subset with uniform split, MAE, under a fixed parameter budget (~500K) to compare architectures fairly. Optimization uses AdamW with a warmup-then-linear-decay schedule; graph data augmentation (adversarial feature perturbation) is available to curb overfitting on the smaller datasets. Edge/node attributes are categorical and embedded.

## Code framework

The starting codebase can already provide categorical feature embedding, dense batching for variable-size graphs, a standard Transformer encoder block, a graph-level prediction head, an optimizer, and a training loop. The empty parts are the graph-derived tensors, the way pairwise graph relations enter the encoder, and the readout mechanism.

```python
import torch
import torch.nn as nn

def convert_to_single_emb(x, offset=512):
    # Map columns of categorical features into disjoint id ranges.
    pass

def preprocess_item(item):
    # item has node features x [N, ...], edge_index [2, E], edge_attr [E, ...]
    # TODO: convert categorical features, build graph-native tensors, and
    #       attach them to item for batching.
    pass

def collator(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    # TODO: filter oversized graphs, pad node/pair tensors, reserve zero for
    #       padding, and build masks for padded or overlong relations.
    pass

class NodeFeature(nn.Module):
    """Map raw node attributes to d-dim node embeddings."""
    def __init__(self, num_node_attr, hidden_dim):
        super().__init__()
        self.encoder = nn.Embedding(num_node_attr, hidden_dim, padding_idx=0)
        # TODO: add any graph-derived per-node inputs.

    def forward(self, batched_data):
        x = batched_data["x"]
        node_feature = self.encoder(x).sum(dim=-2)  # [B, N, d]
        # TODO: combine raw node features with graph-derived per-node inputs.
        return node_feature

class PairRelation(nn.Module):
    """Turn graph-derived pair tensors into whatever the encoder consumes."""
    def __init__(self, num_heads, hidden_dim):
        super().__init__()
        # TODO: parameters for pairwise graph relations.

    def forward(self, batched_data):
        # TODO: return pairwise relation data for the encoder.
        pass

class SelfAttention(nn.Module):
    def __init__(self, d, num_heads, dropout=0.1):
        super().__init__()
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pair_relation=None, padding_mask=None):
        # TODO: compute multi-head self-attention; decide how pair_relation
        #       should affect the attention computation.
        pass

class EncoderLayer(nn.Module):
    def __init__(self, d, ffn_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = SelfAttention(d, num_heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d)
        self.ln2 = nn.LayerNorm(d)
        self.act = nn.GELU()

    def forward(self, x, pair_relation=None, padding_mask=None):
        h = self.ln1(x)
        h = self.attn(h, pair_relation=pair_relation, padding_mask=padding_mask)
        x = x + h
        h = self.ln2(x)
        x = x + self.fc2(self.act(self.fc1(h)))
        return x

class GraphModel(nn.Module):
    def __init__(self, num_node_attr, d, ffn_dim, num_heads, n_layers):
        super().__init__()
        self.node_feature = NodeFeature(num_node_attr, d)
        self.pair_relation = PairRelation(num_heads, d)
        self.layers = nn.ModuleList(
            [EncoderLayer(d, ffn_dim, num_heads) for _ in range(n_layers)]
        )
        self.out = nn.Linear(d, 1)
        # TODO: define how the single whole-graph vector is obtained.

    def forward(self, batched_data):
        x = self.node_feature(batched_data)        # [B, N, d]
        pair_relation = self.pair_relation(batched_data)
        padding_mask = batched_data.get("padding_mask")
        x = x.transpose(0, 1)                      # [N, B, d]
        for layer in self.layers:
            x = layer(x, pair_relation=pair_relation, padding_mask=padding_mask)
        # TODO: produce h_G from the final node states (readout)
        pass

def train(model, loader, opt, loss_fn):
    for batch in loader:
        pred = model(batch)
        loss = loss_fn(pred, batch["y"])
        opt.zero_grad(); loss.backward(); opt.step()
```
