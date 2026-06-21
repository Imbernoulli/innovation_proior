# Context: learned structure-to-sequence design for fixed-backbone proteins (circa 2019–2022)

## Research question

Given the three-dimensional backbone of a protein — the coordinates of the heavy
backbone atoms `N, CA, C, O` for each of `L` residues — recover an amino-acid sequence
that would actually fold into that shape. This is the inverse of protein folding:
folding maps sequence → structure, and here we are handed the structure and must produce
the sequence. It is the central computational step in protein design, because a designer
first sketches or hallucinates a backbone with a desired geometry and then needs a
sequence whose chemistry is compatible with that geometry (good packing, satisfied
hydrogen bonds, no buried unpaired charges).

The map is many-to-one and degenerate: many sequences fold to nearly the same backbone.
Dependencies between residues are long-range in sequence but local in 3D — two residues
far apart in the chain can be in direct contact in space, while two sequence-adjacent
residues may point into opposite environments. And the relevant signal is purely
geometric, which should be invariant to how the structure happens to be placed in the
coordinate frame — rotating or translating the whole molecule cannot change the sequence
it should encode.

How should a learned model read raw 3D coordinates and produce per-residue amino-acid
probabilities over sequences compatible with a given backbone?

## Background

The classical approach to fixed-backbone design is physics-based. **Rosetta** (Leaver-Fay
et al. 2011; Alford et al. 2017) scores a sequence–structure pair with a hand-built
energy function — a weighted sum of van der Waals, solvation, hydrogen-bonding,
electrostatic, and statistical rotamer terms — and searches over amino-acid identities
and side-chain rotamers with Monte-Carlo simulated annealing (the `fixbb` protocol). It
encodes decades of structural-biology knowledge. On native backbones its recovery of the
natural amino acid is roughly a third of positions.

In parallel, deep learning reframed the task as **conditional sequence modeling**: learn
`p(sequence | structure)` directly from the Protein Data Bank, letting the model discover
its own features and its own effective energy instead of hand-coding one. Two ingredients
from the wider field make this feasible.

The first is the **message-passing / graph-neural-network framework** (Gilmer et al.
2017, unifying Duvenaud 2015, Li et al. 2016 gated graph nets, Battaglia 2016 interaction
networks, Kearnes 2016 molecular graph convolutions, and spectral GCNs). A molecule is a
graph with node features `h_v` and edge features `e_vw`; computation runs in rounds, each
round forming a message to each node by summing a learned function over its neighbors and
then updating the node,

```
m_v^{t+1} = sum_{w in N(v)} M_t(h_v^t, h_w^t, e_vw),
h_v^{t+1} = U_t(h_v^t, m_v^{t+1}),
```

with a final permutation-invariant readout. The neighbor sum is symmetric, so the whole
computation is invariant to graph isomorphism — exactly the inductive bias a structural
graph wants. Most prior models update only node states; one of the unified models (Kearnes
et al. 2016 molecular graph convolutions) additionally maintained and updated *edge*
states across rounds, showing that the relational features themselves can be refined, not
just consumed.

The second is **self-attention / the Transformer** (Vaswani et al. 2017), whose
sequence-level relational reasoning had transformed language modeling. Plain attention is
dense — `O(N^2)` memory and compute in the number of positions — which is limiting on a
GPU for long proteins. But a well-evidenced fact in protein science (Marks et al. 2011;
Morcos et al. 2011) is that the contacts that matter are *spatially local*: a residue's
energetically relevant partners are its neighbors in 3D, of which there are a small
constant number regardless of chain length. Restricting attention/messages to a
**`k`-nearest-neighbor graph in space** (`k` around 30) therefore turns the cost linear
in length.

Representing the backbone as an *invariant* graph requires care. Pairwise distances among
backbone atoms are invariant to rigid-body motion by construction. A single representative
atom per residue (the `Cα`) plus the scalar distance to each neighbor is invariant but
*not locally informative*: distances alone cannot tell whether two neighbors sit on the
same side or opposite sides of a residue, so node updates cannot fully reconstruct the
local geometry. Encoding distance as a raw scalar is also a poor input to a linear layer;
the standard fix is a **radial basis expansion** — evaluate the distance against a bank of
Gaussians at fixed centers, turning "distance ≈ d" into a soft one-hot code over distance
bins that a linear layer can read like a lookup table. Since the chain has a direction
(N→C) but no canonical origin, sequence position enters as a **relative** encoding of the
offset `i − j` (keeping its sign), not an absolute index.

A useful piece of geometry: although only backbone atoms are given, the side-chain
direction can be approximated by a **virtual `Cβ`** placed at the ideal tetrahedral
position from `N, CA, C`. With `b = CA − N`, `c = C − CA`, and `a = b × c`,

```
Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA,
```

which gives every residue a side-chain-pointing reference atom from backbone coordinates
alone.

On this data, a *plain multilayer-perceptron* message aggregator `Δh_i = sum_j MLP(h_i,
h_j, e_ij)` was observed to reach *better* held-out perplexity than multi-head attention
over the same graph (Ingraham et al. 2019, Table 3). Local *orientation* information
helped above bare distances, but even distance-only graphs already beat profile-based
neural baselines.

## Baselines

**Rosetta `fixbb` (physics-based; Leaver-Fay et al. 2011; Alford et al. 2017).** Scores a
sequence–structure pair with a fixed, hand-engineered energy function and searches
amino-acid identity plus side-chain rotamers by Monte-Carlo simulated annealing. Strong
priors from structural biology; no learning.

**SPIN2 (O'Connell et al. 2018) and related profile predictors.** Neural networks that
predict a position-specific amino-acid profile from local structural descriptors. Faster
than Rosetta; treat each position's distribution given local context, with limited
coupling between positions.

**Structured Transformer / Struct2Seq (Ingraham, Garg, Barzilay, Jaakkola, 2019).** The
load-bearing prior art: cast design as autoregressive language modeling conditioned on the
structure graph,

```
p(s | x) = prod_i p(s_i | x, s_{<i}),
```

with an encoder–decoder built from sparse, graph-restricted self-attention. The backbone
is a `k`-NN graph (`k = 30`) over `Cα`; **node** features are the backbone dihedrals
`(sin, cos)(φ, ψ, ω)`; **edge** features concatenate a 16-Gaussian distance RBF (0–20 Å),
a relative-orientation encoding via the quaternion of the rotation between local backbone
frames, and a sinusoidal encoding of the sequence offset `i − j`. The encoder runs
multi-head self-attention plus position-wise feed-forward over the spatial neighbors to
build a sequence-independent representation of structure; the decoder predicts each `s_i`
left-to-right (N→C), masking so position `i` sees the full structure but only the already
emitted amino acids `s_{<i}`. Three encoder and three decoder layers, hidden dimension
128, trained with the Transformer (warmup) learning-rate schedule, dropout 0.1, and label
smoothing 0.1. It improved markedly over profile baselines and beat Rosetta on both
recovery and speed.

**GVP-GNN (Jing, Eismann, Suriana, Townshend, Dror, 2021).** An SE(3)-equivariant
alternative on the same task and the same CATH split. Node and edge features carry both
**scalar** and **vector** channels; the geometric vector perceptron replaces dense layers
with a module that mixes scalars and vectors while preserving rotation equivariance for
vectors and invariance for scalars. Node features include `(sin, cos)(φ, ψ, ω)`,
forward/reverse `Cα` unit vectors, and the imputed `Cβ − Cα` direction (tetrahedral
geometry); edges carry the `Cα–Cα` unit vector, a 16-Gaussian distance RBF (0–20 Å), and a
sinusoidal `i − j`. It reuses Ingraham's masked autoregressive encoder–decoder (three
graph-propagation steps each).

## Evaluation settings

- **CATH 4.2 single-chain split** (Ingraham et al. 2019): all CATH-4.2 domains at 40%
  non-redundancy, chains up to length ~500, partitioned by CATH (class / architecture /
  topology / homologous-superfamily) so that train, validation, and test share *no* CAT
  topology — a structure-split that forces generalization to unseen folds. Sizes 18024 /
  608 / 1120 chains. Reported on the full test set and on "short" (≤100 residues) and
  "single-chain" subsets.
- **CATH 4.3**: an updated, more diverse CATH release with the same structure-split
  protocol (~21k train / ~1120 test).
- **TS50**: 50 native structures (Li et al. 2014), used as an out-of-distribution test
  for models trained on CATH 4.2 (filtering training/validation to remove sequences
  similar to TS50).
- **Metrics.** *Native sequence recovery*: fraction of positions whose predicted amino
  acid matches the native (higher is better), reported as the median over the per-protein
  averages of multiple samples. *Perplexity*: exponential of the per-residue
  cross-entropy (lower is better). Sampling uses a temperature-adjusted softmax, with low
  temperature (≈ 0.1) for high-recovery samples and higher temperature for diversity.
- **Inputs/targets.** Input backbone coordinates `X` of shape `(B, L, 4, 3)` for atoms
  `[N, CA, C, O]`, a validity `mask (B, L)`; the per-position label is one of 20 standard
  amino acids (21 with a non-canonical/unknown slot). Loss is masked per-residue
  cross-entropy with label smoothing.

## Code framework

The reusable substrate is a generic structure-encoder harness: a featurizer
that turns coordinates into an attributed `k`-NN graph, a stack of identical
graph-propagation layers, an output head producing per-residue amino-acid logits, and a
masked cross-entropy training loop. What the *layer update*, the *feature design*, and the
*conditioning structure of the head* should be is exactly what is to be designed; those
are left as stubs.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_AA = 21  # 20 standard amino acids (+ 1 unknown/non-canonical slot)


# ---- graph/data primitives ----

def gather_nodes(h_V, E_idx):
    """Gather node features h_V (B,L,C) at neighbor indices E_idx (B,L,K) -> (B,L,K,C)."""
    B, L, K = E_idx.shape
    C = h_V.shape[-1]
    idx = E_idx.unsqueeze(-1).expand(-1, -1, -1, C)
    return torch.gather(h_V.unsqueeze(2).expand(-1, -1, K, -1), 1, idx)


def cat_neighbors_nodes(h_nodes, h_edges, E_idx):
    """Concatenate gathered neighbor node features onto edge features."""
    return torch.cat([h_edges, gather_nodes(h_nodes, E_idx)], dim=-1)


def rbf(D, centers, sigma):
    """Expand a distance tensor into a bank of Gaussian radial basis functions."""
    centers = centers.to(D.device).view(*([1] * D.dim()), -1)
    return torch.exp(-((D.unsqueeze(-1) - centers) / sigma) ** 2)


def knn_graph(X_ca, mask, k):
    """Build a k-nearest-neighbor graph over Ca coordinates; returns neighbor dists + indices."""
    mask_2D = mask.unsqueeze(1) * mask.unsqueeze(2)
    dX = X_ca.unsqueeze(1) - X_ca.unsqueeze(2)
    D = mask_2D * torch.sqrt((dX ** 2).sum(-1) + 1e-6)
    D_max, _ = D.max(-1, keepdim=True)
    D = D + (1.0 - mask_2D) * (D_max + 1.0)          # push padded entries to the back
    D_nbr, E_idx = torch.topk(D, min(k, D.shape[-1]), dim=-1, largest=False)
    return D_nbr, E_idx


# ---- the slots to be designed ----

class StructureFeatures(nn.Module):
    """Turn backbone coords into per-edge (and/or per-node) input features on the kNN graph.
    The exact geometric features are the thing to design."""
    def __init__(self, hidden_dim, k_neighbors):
        super().__init__()
        # TODO: the feature design and its embedding into hidden_dim.
        pass

    def forward(self, X, mask, residue_idx=None, chain_encoding=None):
        # X: (B,L,4,3) [N,CA,C,O]; mask: (B,L)
        # TODO: build the graph and the node/edge features the encoder will consume.
        pass


class GraphLayer(nn.Module):
    """One round of graph propagation over the kNN graph."""
    def __init__(self, hidden_dim):
        super().__init__()
        # TODO: the per-layer update we will design.
        pass

    def forward(self, h_V, h_E, E_idx, mask):
        # TODO: produce updated node (and possibly edge) states.
        pass


class InverseFoldingModel(nn.Module):
    """Encode structure, then produce per-residue amino-acid log-probabilities."""
    def __init__(self, hidden_dim=128, num_layers=3, k_neighbors=30):
        super().__init__()
        self.features = StructureFeatures(hidden_dim, k_neighbors)
        self.layers = nn.ModuleList(GraphLayer(hidden_dim) for _ in range(num_layers))
        # TODO: the output head and the way it conditions predictions.
        self.W_out = nn.Linear(hidden_dim, NUM_AA)

    def forward(self, X, mask, residue_idx=None, chain_encoding=None):
        # TODO: encode, then map to log-probs (B,L,NUM_AA).
        pass


def train_step(model, batch, optimizer):
    X, S, mask = batch['X'], batch['S'], batch['mask']     # coords, labels, validity
    residue_idx = batch.get('residue_idx')
    chain_encoding = batch.get('chain_encoding')
    log_probs = model(X, mask, residue_idx, chain_encoding) # (B,L,NUM_AA)
    loss = F.nll_loss(log_probs.reshape(-1, NUM_AA), S.reshape(-1), reduction='none')
    loss = (loss * mask.reshape(-1)).sum() / mask.sum()    # masked per-residue CE
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return loss.item()
```

The open slots are `StructureFeatures` (the geometric features and their graph), the body
of `GraphLayer` (the per-round update), and the head/conditioning inside
`InverseFoldingModel`.
