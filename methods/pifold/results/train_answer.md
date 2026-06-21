Fixed-backbone protein design hands me the backbone geometry of a chain — for each residue the coordinates of `N`, `CA`, `C`, and `O`, with no side-chain atoms — and asks for an amino-acid identity $s_i \in \{1,\dots,20\}$ at every position. The map from backbone to sequence is not one-to-one: homologous proteins keep the same fold while swapping many residues, so the right learning target is a conditional distribution $p_\theta(s_i \mid X)$, not a deterministic inverse to folding. Three constraints have to hold at once. The model must be invariant to global rotation and translation, because relabeling residues must not depend on how the protein happens to sit in space. It must use both local contacts and broader protein context. And it should not pay a latency cost that grows linearly with chain length at generation time. The standard recipe — build a residue $k$-nearest-neighbor graph, encode the structure, then decode amino acids — has the right shape but two recurring weaknesses. The graph encoders tend to expose thin geometry: a single $CA$–$CA$ distance per pair is a coarse view, and the input never contains the side chain that actually distinguishes many residue identities. And the decoders are usually autoregressive, factorizing $p(S\mid X)=\prod_t p(s_t\mid s_{<t},X)$, which forces $L$ ordered steps for a chain of length $L$. One escape is to keep geometric vectors as vectors and use equivariant operators, as the geometric-vector-perceptron route does, but that demands specialized scalar-vector layers throughout. I want to see how far the simpler invariant-scalar route can go if I stop starving it of geometry, and then make the encoder strong enough that the decoder no longer needs to be sequential.

I propose PiFold. It pairs a rich rotation-invariant residue featurizer that includes learnable virtual atoms with a graph encoder, the PiGNN, whose layers update nodes, edges, and a global context together, feeding a one-shot non-autoregressive readout. The invariance comes first. For residue $i$ I build a local frame from its three backbone atoms around $CA$: with $u_i = CA_i - N_i$ and $v_i = C_i - CA_i$, set $b_i = (u_i - v_i)/\lVert u_i - v_i\rVert$, $n_i = (u_i \times v_i)/\lVert u_i \times v_i\rVert$, and $Q_i = [\,b_i,\ n_i,\ b_i \times n_i\,]$. A vector such as $N_j - CA_i$ becomes an invariant edge direction once projected into $Q_i$, and the relative rotation of two residues, $Q_i^\top Q_j$, is encoded as a quaternion. This keeps me in ordinary scalar-graph territory while losing no geometry. The failure of the scalar route was never that scalars cannot be invariant — it was that they were too sparse, so I enrich them. Instead of one $CA$–$CA$ distance, node features use the six intra-residue real-atom distances ($CA$–$N$, $CA$–$C$, $CA$–$O$, $N$–$C$, $N$–$O$, $O$–$C$) and edge features use the sixteen directed real-atom pair distances between residue $i$ and neighbor $j$. Every scalar distance is lifted through the same sixteen-channel Gaussian RBF over $[0,20]$ angstroms, $\mathrm{RBF}(D)_m = \exp\!\big(-((D-\mu_m)/\sigma)^2\big)$ with centers $\mu_m$ uniformly spaced and $\sigma = 20/16$, because a smooth distance bin is far easier for the network to use than a raw scalar.

Real backbone atoms still leave a blind spot: the side chain, which is exactly the discriminative part, is absent from the input. I cannot use the true side chain, but I can ask for a learnable proxy tied to the local frame. For virtual atom $k$ I choose shared coefficients $(x_k, y_k, z_k)$, normalize them so $x_k^2 + y_k^2 + z_k^2 = 1$, and place it at
$$ V_i^k = x_k\, b_i + y_k\, n_i + z_k\,(b_i \times n_i) + CA_i. $$
Because the point lives in the frame, it rotates and translates with the residue and distances from it stay invariant. Sharing $(x_k, y_k, z_k)$ across all residues is the load-bearing choice: per-residue virtual coordinates would be arbitrary per example, whereas shared coordinates force a transferable, consensus geometric probe that injects side-chain-like information without ever seeing a side chain. With $\text{num\_virtual}=3$ and $\text{num\_rbf}=16$ the dimensions close exactly. Node distances add the $3\cdot(3-1)=6$ ordered distinct virtual-virtual blocks to the six real ones, giving $(6+6)\cdot 16 = 192$; with twelve angle channels — $\sin$ and $\cos$ of the three torsions $(\phi,\psi,\omega)$ and the three bond angles $(\alpha,\beta,\gamma)$, so the periodicity never tears at the wrap point — and nine direction channels from projecting normalized $N$, $C$, $O$ offsets into $Q_i$, the total is $\text{node\_in}=213$. Edge distances add three same-index virtual distances and another $3\cdot(3-1)=6$ cross-virtual distances to the sixteen real ones, giving $(16+3+6)\cdot 16 = 400$; with the four-channel quaternion of $Q_i^\top Q_j$ and the twelve direction channels from $CA_i$ to all four atoms of residue $j$ projected into $Q_i$, the total is $\text{edge\_in}=416$. No positional-encoding block is needed; the counts close without it.

The encoder has to actually use this geometry, and standard graph attention is too restrictive here. Scaled dot-product attention makes a query from node $i$ and a key from neighbor $j$, but in a protein graph the edge is not a small additive bias — it carries the real pairwise geometry — so the edge must participate directly in deciding how much $j$ matters to $i$. I therefore score each edge with an MLP over the concatenation of center node, edge, and neighbor node rather than a query-key dot product:
$$ w_{ji} = \mathrm{AttMLP}\big(h_j^l \,\Vert\, e_{ji}^l \,\Vert\, h_i^l\big)/\sqrt{d_{\text{head}}}, \qquad a_{ji} = \operatorname*{softmax}_{j \in \mathcal N_i} w_{ji}, $$
$$ v_j = \mathrm{NodeMLP}\big(e_{ji}^l \,\Vert\, h_j^l\big), \qquad \hat h_i = \sum_{j \in \mathcal N_i} a_{ji}\, v_j. $$
The $1/\sqrt{d_{\text{head}}}$ scaling matters for the same reason it does in Transformer attention: without it, larger head dimension blows up the logits and saturates the neighbor softmax. That softmax must be taken over the incoming neighbors of one center residue, not over all edges in the batch, which in sparse code means grouping by `center_id` and using a scatter softmax; I store `edge_idx[0]` as the center and `edge_idx[1]` as the neighbor so the grouping is exactly on the center index. A residual connection, batch normalization, and a four-times-wide feed-forward layer wrap the node update. If I stopped there the edges would stay frozen at their input embeddings while the nodes evolve, a bad mismatch when the geometry lives mainly on the edges, so once the endpoints have been refined I refine the edge too:
$$ e_{ji}^{l+1} = \mathrm{EdgeMLP}\big(\hat h_j \,\Vert\, e_{ji}^l \,\Vert\, \hat h_i\big), $$
again with a residual and batch normalization, so node and edge states co-adapt instead of passing messages through static labels. Local neighborhoods are not the whole protein, and full self-attention would give global reach at quadratic cost. I only need a cheap protein-level summary, so I mean-pool the current node embeddings within each protein and let that vector gate each channel:
$$ c_i = \operatorname{mean}\{\hat h_k : k \text{ in the same protein as } i\}, \qquad h_i^{l+1} = \hat h_i \cdot \sigma\big(\mathrm{GateMLP}(c_i)\big). $$
The sigmoid gate is deliberate: addition would overwrite local information, whereas multiplication lets the global summary keep or suppress channels while the local representation stays the carrier of residue-specific geometry — and the cost is linear in the number of residues. One PiGNN block is thus MLP-scored neighbor attention, the wide feed-forward refinement, the edge update, and the context gate; the default stacks ten of them at hidden dimension $128$ with four attention heads.

That strong encoder is what lets me drop the autoregressive decoder. The autoregressive factorization is expressive because it models residue-residue dependencies through previously generated tokens, but many of those dependencies are induced by the shared structure: if two residues pack together, the backbone geometry and the contact graph already tell the encoder so. So I make the encoder carry that burden and decode in one shot,
$$ p(S \mid X) = \prod_i p(s_i \mid X), \qquad \log p(s_i \mid X) = \operatorname{log\,softmax}(W h_i)_{s_i}. $$
This does not claim residues are physically independent; it claims the final node embeddings already hold the structural context each marginal needs. The decoder is a linear readout to twenty logits followed by a log-softmax, training stays per-residue cross-entropy $\mathcal L = -\big(\sum_i \text{mask}_i\big)^{-1}\sum_i \text{mask}_i \log p_\theta(s_i\mid X)$, and inference is a single parallel pass with no length-$L$ decoding loop.

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
