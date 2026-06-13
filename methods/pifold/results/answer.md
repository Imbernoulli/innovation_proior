PiFold is a fixed-backbone protein inverse-folding model. Given backbone atom coordinates
`N`, `CA`, `C`, and `O` for each residue, it predicts amino-acid log-probabilities for all
positions in one parallel pass. Its core pieces are a rich invariant residue featurizer with
learnable virtual atoms, a PiGNN encoder with MLP-scored neighbor attention, edge updates,
and a global context gate, and a one-shot non-autoregressive decoder.

## Problem

The task is:

```
X = {(N_i, CA_i, C_i, O_i)}_{i=1}^L
log p_theta(s_i | X),    s_i in {1, ..., 20}
```

Because multiple sequences can fit the same fold, the model learns residue distributions
rather than a unique inverse mapping. It is trained with per-residue cross-entropy:

```
L = -(1 / sum_i mask_i) sum_i mask_i log p_theta(s_i | X)
```

## Featurizer

For each residue `i`, build the local frame from `N_i`, `CA_i`, and `C_i`:

```
u_i = CA_i - N_i
v_i = C_i - CA_i
b_i = (u_i - v_i) / ||u_i - v_i||
n_i = (u_i x v_i) / ||u_i x v_i||
Q_i = [b_i, n_i, b_i x n_i]
```

The `k`-nearest-neighbor graph uses `CA`-`CA` distances with default `k = 30`. Distances are
expanded by 16 Gaussian RBF channels with centers uniformly spaced over `[0, 20]` angstroms:

```
RBF(D)_m = exp(-((D - mu_m) / sigma)^2),    sigma = 20 / 16
```

Learnable virtual atom `k` is placed in the local frame:

```
V_i^k = x_k b_i + y_k n_i + z_k (b_i x n_i) + CA_i
x_k^2 + y_k^2 + z_k^2 = 1
```

The coefficients are shared across residues, so the virtual atoms learn transferable
frame-relative positions.

Default feature dimensions with `num_virtual = 3` and `num_rbf = 16`:

```
node distance blocks = 6 real + 3 * (3 - 1) virtual = 12
node_in = 12 * 16 + 12 angle + 9 direction = 213

edge distance blocks = 16 real + 3 same-index virtual + 3 * (3 - 1) cross virtual = 25
edge_in = 25 * 16 + 4 quaternion + 12 direction = 416
```

The final feature tensor does not include a positional-encoding block.

## PiGNN Layer

For edge `j -> i` at layer `l`, the node update uses MLP-scored neighbor attention:

```
w_ji = AttMLP(h_j^l || e_ji^l || h_i^l) / sqrt(d_head)
a_ji = softmax_{j in N_i}(w_ji)
v_j  = NodeMLP(e_ji^l || h_j^l)
hhat_i = sum_{j in N_i} a_ji v_j
```

The softmax is over neighbors of the same center residue. The sparse implementation stores
`edge_idx[0]` as the center residue and applies `scatter_softmax(..., index=center_id)`.

The edge state updates from the current endpoints and edge:

```
e_ji^{l+1} = EdgeMLP(hhat_j || e_ji^l || hhat_i)
```

Global context is injected with a linear-cost gate:

```
c_i = mean({hhat_k : k in the same protein as i})
h_i^{l+1} = hhat_i * sigmoid(GateMLP(c_i))
```

Residual connections, BatchNorm, dropout, and a `4x` feed-forward layer wrap the node update.
The default encoder uses hidden dimension 128, 4 attention heads, and 10 graph layers.

## Decoder

The decoder is one linear readout plus `log_softmax`:

```
log_probs_i = log_softmax(W h_i)
p(S | X) = product_i p(s_i | X)
```

There is no autoregressive conditioning on previously predicted residues.

## Core Implementation

This is the code-level shape of the canonical sparse implementation: node and edge features
are embedded first, the graph layer concatenates `edge || neighbor` for the value MLP and
`center || edge || neighbor` for the attention-score MLP, and the decoder is a parallel
linear readout.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_softmax, scatter_sum

NUM_AA = 20


def _normalize(x, dim=-1, eps=1e-8):
    return F.normalize(x, dim=dim, eps=eps)


def _rbf(D, num_rbf=16):
    centers = torch.linspace(0.0, 20.0, num_rbf, device=D.device)
    sigma = 20.0 / num_rbf
    return torch.exp(-((D.unsqueeze(-1) - centers) / sigma) ** 2)


def local_frame(N, CA, C):
    u = CA - N
    v = C - CA
    b = _normalize(u - v)
    n = _normalize(torch.cross(u, v, dim=-1))
    return b, n, torch.cross(b, n, dim=-1)


class ResidueFeaturizer(nn.Module):
    def __init__(self, num_virtual=3, num_rbf=16):
        super().__init__()
        self.virtual_atoms = nn.Parameter(torch.rand(num_virtual, 3))
        self.num_virtual = num_virtual
        self.num_rbf = num_rbf
        self.node_in = (6 + num_virtual * (num_virtual - 1)) * num_rbf + 12 + 9
        self.edge_in = (
            (16 + num_virtual + num_virtual * (num_virtual - 1)) * num_rbf
            + 4
            + 12
        )

    def virtual_positions(self, X_flat):
        N, CA, C = X_flat[:, 0], X_flat[:, 1], X_flat[:, 2]
        b, n, t = local_frame(N, CA, C)
        coeff = _normalize(self.virtual_atoms, dim=-1)
        return (
            CA[:, None, :]
            + coeff[None, :, 0, None] * b[:, None, :]
            + coeff[None, :, 1, None] * n[:, None, :]
            + coeff[None, :, 2, None] * t[:, None, :]
        )


class NeighborAttention(nn.Module):
    def __init__(self, hidden_dim=128, num_in=256, num_heads=4, output_mlp=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.output_mlp = output_mlp
        self.W_V = nn.Sequential(
            nn.Linear(num_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.Bias = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_heads),
        )
        self.W_O = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, h_V, h_E_neighbor, center_id):
        E = h_E_neighbor.shape[0]
        d_head = self.hidden_dim // self.num_heads
        logits = self.Bias(torch.cat([h_V[center_id], h_E_neighbor], dim=-1))
        logits = logits.view(E, self.num_heads, 1) / math.sqrt(d_head)
        weights = scatter_softmax(logits, index=center_id, dim=0)
        values = self.W_V(h_E_neighbor).view(E, self.num_heads, d_head)
        update = scatter_sum(weights * values, center_id, dim=0).reshape(-1, self.hidden_dim)
        return self.W_O(update) if self.output_mlp else update


class EdgeMLP(nn.Module):
    def __init__(self, hidden_dim=128, num_in=256, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.W11 = nn.Linear(hidden_dim + num_in, hidden_dim)
        self.W12 = nn.Linear(hidden_dim, hidden_dim)
        self.W13 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h_V, h_E, edge_idx):
        center_id, neighbor_id = edge_idx[0], edge_idx[1]
        h_EV = torch.cat([h_V[center_id], h_E, h_V[neighbor_id]], dim=-1)
        h_message = self.W13(F.gelu(self.W12(F.gelu(self.W11(h_EV)))))
        return self.norm(h_E + self.dropout(h_message))


class Context(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.V_MLP_g = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, h_V, h_E, edge_idx, batch_id):
        c_V = scatter_mean(h_V, batch_id, dim=0)
        return h_V * self.V_MLP_g(c_V[batch_id]), h_E


class PiGNNLayer(nn.Module):
    def __init__(self, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.attention = NeighborAttention(hidden_dim, num_in=hidden_dim * 2, num_heads=4)
        self.edge_update = EdgeMLP(hidden_dim, num_in=hidden_dim * 2, dropout=dropout)
        self.context = Context(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(2)])
        self.dense = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, h_V, h_E, edge_idx, batch_id):
        center_id, neighbor_id = edge_idx[0], edge_idx[1]
        h_E_neighbor = torch.cat([h_E, h_V[neighbor_id]], dim=-1)
        dh = self.attention(h_V, h_E_neighbor, center_id)
        h_V = self.norm[0](h_V + self.dropout(dh))
        h_V = self.norm[1](h_V + self.dropout(self.dense(h_V)))
        h_E = self.edge_update(h_V, h_E, edge_idx)
        return self.context(h_V, h_E, edge_idx, batch_id)


class StructureEncoder(nn.Module):
    def __init__(self, hidden_dim=128, num_encoder_layers=10, dropout=0.1):
        super().__init__()
        self.encoder_layers = nn.ModuleList(
            [PiGNNLayer(hidden_dim, dropout) for _ in range(num_encoder_layers)]
        )

    def forward(self, h_V, h_E, edge_idx, batch_id):
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, edge_idx, batch_id)
        return h_V, h_E


class MLPDecoder(nn.Module):
    def __init__(self, hidden_dim=128, vocab=20):
        super().__init__()
        self.readout = nn.Linear(hidden_dim, vocab)

    def forward(self, h_V, batch_id=None):
        logits = self.readout(h_V)
        return F.log_softmax(logits, dim=-1), logits


class PiFold(nn.Module):
    def __init__(self, node_in=213, edge_in=416, hidden_dim=128,
                 num_encoder_layers=10, dropout=0.1):
        super().__init__()
        self.node_embedding = nn.Linear(node_in, hidden_dim)
        self.edge_embedding = nn.Linear(edge_in, hidden_dim)
        self.norm_nodes = nn.BatchNorm1d(hidden_dim)
        self.norm_edges = nn.BatchNorm1d(hidden_dim)
        self.W_v = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.W_e = nn.Linear(hidden_dim, hidden_dim)
        self.encoder = StructureEncoder(hidden_dim, num_encoder_layers, dropout)
        self.decoder = MLPDecoder(hidden_dim)

    def forward(self, node_feat, edge_feat, edge_idx, batch_id):
        h_V = self.W_v(self.norm_nodes(self.node_embedding(node_feat)))
        h_E = self.W_e(self.norm_edges(self.edge_embedding(edge_feat)))
        h_V, h_E = self.encoder(h_V, h_E, edge_idx, batch_id)
        log_probs, logits = self.decoder(h_V, batch_id)
        return log_probs
```

## Relation to Prior Methods

- Structured Transformer supplies the residue `k`-NN graph, local-frame invariant geometry,
  RBF distances, quaternion orientations, and the autoregressive graph-to-sequence template.
- GVP is the main equivariant-vector alternative; PiFold keeps the invariant-scalar route and
  makes the scalar feature set richer.
- AlphaDesign supplies the MLP-scored neighbor-attention idea and bond-angle features.
- GCA motivates global context; PiFold uses a mean-pooled sigmoid gate instead of full
  quadratic global attention.
- ProteinMPNN motivates iterative edge representations and virtual geometric points; PiFold
  keeps edge updates, makes the virtual atom positions learnable, and decodes all residues in
  one shot.
