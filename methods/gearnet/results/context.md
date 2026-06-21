## Research question

A protein is a chain of amino-acid residues that folds into a 3D conformation, and that
conformation is often what determines biological function. The goal is an encoder: given a
single protein structure as one alpha-carbon coordinate per residue, `x in R^{n x 3}`, plus the
residue identity at each point, produce a vector for every residue and a pooled vector for the
whole protein. Those vectors should be useful to downstream predictors of function and fold.

The encoder operates on the 3D arrangement, because spatial contacts between residues can be far
apart along the sequence but close in the folded structure. It is invariant to translation,
rotation, and reflection: the same protein in a different coordinate frame is the same protein.
It exposes both local structure and longer-range fold information. The question is how to build
such a residue-level encoder from alpha-carbon coordinates and residue identities.

## Background

The usual substrate is a residue graph. Nodes are residues, edges connect residues that are
related in sequence or close in 3D space, and a graph neural network updates each node by
aggregating messages from its neighbors. In the message-passing abstraction of Gilmer et al.
2017, each layer gathers messages from the neighbors of node `i`, sums them, transforms the
result, and repeats; stacking `L` layers lets information travel across `L` graph hops.

Rigid-motion invariance can be handled at the input level. Distances between residues and angles
formed by residues are unchanged by translation, rotation, or reflection. If the graph
construction, edge descriptors, and message rules only use such scalar geometric quantities, the
resulting residue and graph embeddings are invariant without carrying a coordinate frame through
the network.

The graph construction itself shapes what the encoder sees. Sequential edges preserve local
backbone order, while spatial edges capture tertiary contacts. A fixed radius graph connects
residues within a distance cutoff. A k-nearest-neighbor graph connects each residue to its `k`
closest neighbors, keeping a more stable degree. These are the standard ways to turn coordinates
into a graph, and they shape what a practical protein encoder works with.

Distance-only messages summarize local geometry through scalar distances. Two different local
geometries can present similar distance patterns around each residue while differing in
directional arrangement. Small-molecule models and structure-prediction systems use richer
directional geometry such as bond and dihedral angles.

## Baselines

**GCN (Kipf & Welling 2017).** A plain graph convolution applies one shared kernel to every
neighbor before aggregation, for example
`H^{(l+1)} = sigma(D^{-1/2} A D^{-1/2} H^{(l)} W^{(l)})`. It is cheap and works as a generic
residue-graph baseline.

**R-GCN (Schlichtkrull et al. 2018).** Relational graph convolution was built for graphs whose
edges have a discrete type `r`:
`h_i^{(l+1)} = sigma(sum_{r in R} sum_{j in N_r(i)} (1/c_{i,r}) W_r h_j^{(l)} + W_0 h_i^{(l)})`.
Each relation has its own kernel, shared across edges of that relation, so capacity scales with
the number of relation types rather than the number of edges. The relation types are supplied by
the graph design.

**IEConv (Hermosilla et al. 2021).** IEConv is a protein-structure convolution whose per-edge
geometric descriptors are passed through an MLP that outputs a kernel for that edge, so the
transform varies continuously with geometry, together with intrinsic-distance features.

**SchNet (Schutt et al. 2017).** SchNet uses continuous filters generated from interatomic
distances expanded over radial basis functions. Because its geometric inputs are distances, it
is invariant and smooth. It was designed for small molecules.

**EGNN (Satorras, Hoogeboom & Welling 2021).** EGNN updates both node features and coordinates
with equivariant coordinate shifts based on relative positions, preserving 3D symmetry by
carrying coordinates through every layer.

**DimeNet (Klicpera/Gasteiger et al. 2020).** DimeNet incorporates angular information alongside
distances in a continuous-filter model for small molecules. It uses a specialized spherical
Bessel/harmonic basis over distance and angle.

## Evaluation settings

The natural evaluation settings are structure-based protein prediction tasks that existed before
the encoder is designed:

- Enzyme Commission number prediction from structure, as multi-label binary classification or
  single-label multiclass classification depending on the benchmark split.
- Gene Ontology term prediction for biological process, molecular function, and cellular
  component annotations, usually as multi-label binary classification.
- Fold classification over structural fold labels from SCOP/CATH-style hierarchies.
- Reaction classification for enzyme structures.
- Metrics such as protein-centric `F_max`, pair-centric area under the precision-recall curve for
  multi-label EC/GO tasks, and top-1 accuracy for multiclass fold or reaction tasks.
- Standard training machinery: Adam or AdamW for EC/GO-style tasks, SGD or Adam-family optimizers
  for multiclass settings, learning-rate scheduling, and residue identity as the basic node
  feature supplied to the encoder.

## Code framework

The fixed harness provides residue identities, alpha-carbon coordinates, batch membership, a
classifier head, an optimizer, and a loss. The unsettled part is the encoder architecture: how
to construct the graph from coordinates and how to update residue states while preserving
rigid-motion invariance.

```python
import torch
import torch.nn as nn


RESIDUE_NODE_DIM = 21


class ProteinEncoder(nn.Module):
    """Encode alpha-carbon coordinates and residue features into node and graph embeddings."""

    def __init__(self, input_dim: int = RESIDUE_NODE_DIM, hidden_dim: int = 512,
                 num_layers: int = 6, dropout: float = 0.0):
        super().__init__()
        # TODO: the architecture we will design.
        pass

    def forward(self, graph, node_feature):
        # graph supplies coordinates, batch membership, and any edge structures
        # the encoder chooses to build.
        # returns {"node_feature": node_feature, "graph_feature": graph_feature}
        raise NotImplementedError


def train(encoder, head, loss_fn, data_loader, optimizer):
    for graph, node_feature, targets in data_loader:
        optimizer.zero_grad()
        output = encoder(graph, node_feature)
        preds = head(output["graph_feature"])
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
```
