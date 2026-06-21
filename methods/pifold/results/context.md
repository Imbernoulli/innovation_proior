## Research question

Fixed-backbone protein design asks for an amino-acid sequence for a desired structure. The
input is the backbone geometry of a protein chain: for each residue there are coordinates for
`N`, `CA`, `C`, and `O`, and the side-chain atoms are not given. The output is a residue type
`s_i` in a 20-symbol amino-acid alphabet for each position `i`.

The map from backbone to sequence is not one-to-one. Homologous proteins can preserve the same
fold while changing many residues, so the useful learning target is a conditional distribution
over residue identities rather than a unique inverse. The practical model has to use local
and nonlocal geometric constraints and stay invariant to global rotation and translation.

## Background

A residue-level graph is the standard abstraction. Nodes are residues, and an edge connects a
residue to its spatial neighbors, usually the `k = 30` nearest residues by `CA`-`CA`
distance. This keeps message passing local, with per-layer cost proportional to `n k` rather
than to all residue pairs, while still covering the local contact shell of most residues.

Every useful geometric quantity has to ignore global rotation and translation. A local frame
can be attached to residue `i` using its backbone atoms:

```
u_i = CA_i - N_i
v_i = C_i - CA_i
b_i = (u_i - v_i) / ||u_i - v_i||
n_i = (u_i x v_i) / ||u_i x v_i||
Q_i = [b_i, n_i, b_i x n_i]
```

Once this frame exists, distances are invariant directly, directions become invariant after
projection into the frame, and the relative orientation of residues `i` and `j` can be
encoded from `Q_i^T Q_j`, commonly as a quaternion.

Distance features are normally lifted through radial basis functions rather than passed as
single scalars. The common graph-protein design recipe uses 16 Gaussian RBF channels with
centers evenly spaced over `[0, 20]` angstroms and width `(20 - 0) / 16`. Angle features use
trigonometric encodings so periodic angles do not tear at the wrap point. Dihedral angles
describe the ordered backbone chain, bond angles add local geometry between adjacent backbone
bonds, direction features project normalized atom-to-frame vectors into the local coordinate
system, and edge orientation features encode the frame-to-frame rotation.

There are two broad ways to build rotation-aware protein models. One route constructs
rotation-invariant scalar features first and then uses ordinary graph neural network layers.
The other route keeps geometric vectors as vectors in `R^3` and uses equivariant operators so
the network itself preserves the rotation structure.

Decoding is another key choice. Autoregressive models factorize
`p(S | X) = product_t p(s_t | s_<t, X)`, so each residue prediction can condition on earlier
residue choices. Parallel decoders predict all residues from the structure-conditioned
embeddings at once.

## Baselines

Structured Transformer / GraphTrans establishes the residue-graph template. It builds a
`k`-nearest-neighbor graph, attaches local-frame invariant edge features, uses graph
attention in a structural encoder, and decodes with an autoregressive Transformer. The
attention score is query-key-value attention: a query comes from the center node, keys and
values are built from neighbor and edge information, and the score is a scaled dot product.

MLP and CNN approaches cover the older non-graph baselines. MLP models can be very fast
because each residue is predicted from engineered local descriptors. CNN models can learn
from local 2D or 3D grids around residues.

GVP represents the equivariant-vector route. It keeps scalar and vector channels together
and updates them with geometric vector perceptrons: vector norms can feed scalar channels,
and vector outputs transform predictably under rotation. This gives a principled way to
process geometry without reducing everything to scalar features first.

GCA adds global information to local graph message passing. Local neighborhoods capture
contacts, and some residue preferences depend on the broader protein context. GCA addresses
this with full self-attention over the whole protein.

AlphaDesign contributes a simplified graph transformer and a partially parallel decoder. Its
graph layer scores neighbor importance with an MLP over endpoint and edge representations
instead of only a query-key dot product, and it adds bond-angle node features on top of the
dihedral-angle template.

ProteinMPNN updates both node and edge representations in a message passing encoder. Its
edge features include rich pairwise geometry over backbone atoms and a virtual `CB`
construction, and its decoder is order-agnostic but still autoregressive.

## Evaluation settings

The standard structure-design setting uses CATH topology splits so training and test folds
do not overlap by topology. The CATH 4.2 split contains 18024 training proteins, 608
validation proteins, and 1120 test proteins; CATH 4.3 is a larger successor split. Results
are commonly broken out for all chains, short chains, and single-chain proteins.

Generalization is also checked on TS50 and TS500, two curated native-structure benchmark
sets. Long-chain inference speed can be measured by generating a fixed collection of long
proteins on one GPU.

The two main metrics are:

```
REC  = (1 / |D|) sum_{(X,S) in D} (1 / N) sum_i 1[s_i = argmax_a p(a | X)]
PERP = exp(-(1 / N) sum_i log p(s_i | X))
```

Training uses per-residue cross-entropy, the Adam optimizer, a one-cycle learning-rate
schedule, batch size on the order of 8, and learning rate `1e-3` in the common reproduced
graph-model harness.

## Code framework

The code substrate is a fixed structure-to-sequence training harness. It receives backbone
coordinates `X` with shape `(B, L, 4, 3)`, a residue mask, and native sequence labels, and a
few known geometric primitives: an RBF distance expansion, a per-residue local frame, and a
`k`-nearest-neighbor graph builder. It trains with per-residue negative log likelihood and
calls a model that returns log-probabilities over 20 amino acids. What is not settled — and is
exactly what must be designed — is how to turn the backbone graph into per-residue embeddings
and how to read those embeddings out as amino-acid distributions. Those are the two empty
slots below.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_AA = 20


def rbf(distance, num_rbf=16, d_min=0.0, d_max=20.0):
    centers = torch.linspace(d_min, d_max, num_rbf, device=distance.device)
    width = (d_max - d_min) / num_rbf
    return torch.exp(-((distance.unsqueeze(-1) - centers) / width) ** 2)


def local_frame(N, CA, C, eps=1e-8):
    u = CA - N
    v = C - CA
    b = F.normalize(u - v, dim=-1, eps=eps)
    n = F.normalize(torch.cross(u, v, dim=-1), dim=-1, eps=eps)
    t = torch.cross(b, n, dim=-1)
    return b, n, t


def knn_graph(CA, mask, k=30):
    # k-NN graph on CA coordinates; returns edge_idx (centers, neighbors) and batch_id.
    raise NotImplementedError


class StructureEncoder(nn.Module):
    """Turn backbone coordinates into per-residue embeddings.
    Input: X (B, L, 4, 3) backbone coords [N, CA, C, O]; mask (B, L).
    Output: per-residue embeddings of width hidden_dim."""

    def __init__(self, hidden_dim=128, **kwargs):
        super().__init__()
        # TODO: the featurizer and graph architecture we will design.
        pass

    def forward(self, X, mask):
        # TODO
        raise NotImplementedError


class StructureToSequence(nn.Module):
    def __init__(self, hidden_dim=128, **kwargs):
        super().__init__()
        self.encoder = StructureEncoder(hidden_dim=hidden_dim, **kwargs)
        # TODO: the decoder head we will design.
        pass

    def forward(self, X, mask):
        # Returns log-probabilities over 20 amino acids, shape (B, L, 20).
        raise NotImplementedError


def train_step(model, batch, optimizer):
    X, S, mask = batch
    optimizer.zero_grad()
    log_probs = model(X, mask)
    loss = F.nll_loss(log_probs[mask.bool()], S[mask.bool()])
    loss.backward()
    optimizer.step()
    return loss
```
