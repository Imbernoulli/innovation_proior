Let me start from the bottleneck I can actually act on. I am given a fixed backbone, one `N`, `CA`, `C`, and `O` coordinate per residue, and I need a residue identity at every position. The answer is not unique, so I should think in terms of `p(s_i | X)` or `p(S | X)`, not a deterministic inverse to folding. The graph models already have the right broad shape: make a residue graph, encode the 3D structure, then decode amino acids. The trouble is that the graph encoder may not expose enough geometry, and the decoder often spends one sequential step per residue. If I want both accurate residue marginals and fast generation, I cannot just bolt a bigger autoregressive decoder onto thin features. The lever I have most control over is the structure encoder, so I will push as much of the work as I can onto it and see how thin a decoder I can get away with.

The first constraint is rigid-motion invariance. If I rotate or translate the whole protein, the residue labels must not change. Distances are safe, but directions and orientations only become safe after I express them in a residue-attached coordinate frame. For residue `i`, I can build the frame directly from the three backbone atoms around `CA`: `u_i = CA_i - N_i`, `v_i = C_i - CA_i`, `b_i = (u_i - v_i) / ||u_i - v_i||`, `n_i = (u_i x v_i) / ||u_i x v_i||`, and the third axis is `b_i x n_i`. So `Q_i = [b_i, n_i, b_i x n_i]`. Now a vector like `N_j - CA_i` becomes an invariant edge direction after projection into this frame, and two frames give a relative rotation `Q_i^T Q_j` that I can encode as a quaternion. This lets me stay in ordinary scalar graph-network territory without losing the geometry.

There is another possible route: keep vectors as vectors and use equivariant layers, as in geometric vector perceptrons. That is attractive because the network carries the rotation structure itself. But it also means every layer has to be a specialized scalar-vector operator. I want to see how far the invariant-scalar route can go if I stop starving it. The failure mode is not that scalar features cannot be invariant; the failure mode is that the scalar features are too sparse.

Distances are the first obvious place to enrich. A `CA`-only distance is a very coarse view of a residue pair. Each residue has `N`, `CA`, `C`, and `O`, so for node features I can use the intra-residue real-atom distances among those four atoms. There are `4 choose 2 = 6` of them: `CA-N`, `CA-C`, `CA-O`, `N-C`, `N-O`, and `O-C`. For edge features I can use directed real-atom pair distances between residue `i` and neighbor residue `j`. If I list the directed pairs that the standard recipe keeps — `CA-CA`, `CA-C`, `C-CA`, `CA-N`, `N-CA`, `CA-O`, `O-CA`, `C-C`, `C-N`, `N-C`, `C-O`, `O-C`, `N-N`, `N-O`, `O-N`, `O-O` — that is 16 directed pairs. Each scalar distance should be expanded with the same 16-channel RBF over `[0, 20]` angstroms, `exp(-((D - mu) / sigma)^2)` with uniformly spaced centers and `sigma = 20 / 16`, because a smooth distance bin is easier for the model to use than one raw number.

But all real backbone atom distances still have a blind spot. The input contains no side-chain atoms, and the side chain is exactly what distinguishes many residue identities. I cannot use the true side chain because it is not part of the fixed-backbone input. Still, I can ask for a learnable proxy point tied to the local backbone frame. The reason this is allowed to exist without breaking invariance is worth checking rather than asserting: if the point is expressed in the frame, it should rotate and translate with the residue, and distances from it should stay invariant. So for virtual atom `k`, I choose shared learnable coefficients `(x_k, y_k, z_k)`, normalize them to `x_k^2 + y_k^2 + z_k^2 = 1`, and place

```
V_i^k = x_k b_i + y_k n_i + z_k (b_i x n_i) + CA_i.
```

Let me actually verify the invariance numerically before I trust it. I take one residue's backbone, a random unit-norm coefficient, build the frame and the virtual atom, then apply a random proper rotation `Q` (`det Q = +1`) and translation `s` to the three backbone atoms and rebuild. Running it: the rotated virtual atom lands at `V@Q^T + s` to within `1e-5`, i.e. it moves exactly as a rigid point of the residue, so the `CA`-to-virtual distance is the same before and after — I get `0.99999994` before and `1.0` after. That is the invariance I wanted, and it is real, not assumed. It also surfaces something I had not thought about: because the coefficient is unit-norm and the frame `(b, n, b x n)` is orthonormal, that distance is exactly `1`. So a single normalized virtual atom sits on the unit sphere around `CA`; what the learning actually places is its *direction* in the frame, and the distinguishing geometry comes from how several such points sit relative to each other and to the real atoms. That is fine — three virtual atoms give three frame-relative directions plus the inter-virtual distances — but it tells me one virtual atom alone carries almost no distance information, which is a reason to use more than one.

Sharing `(x_k, y_k, z_k)` across residues is the other choice to think through. If every residue had separate virtual coordinates, the points would be arbitrary per example and could not transfer. Shared coordinates force the model to learn a single transferable placement in the residue frame, a consensus geometric probe that can add side-chain-like information without ever seeing a side chain.

Now I want to count the feature width this gives, because I would rather have the dimensions fall out of the construction than pick them by hand. Take `num_virtual = 3`. For the node distance block I add, on top of the 6 real intra-residue distances, the distances among the three virtual atoms. If I keep them ordered and distinct, that is `3 * (3 - 1) = 6` blocks, so `6 + 6 = 12` distance blocks, times 16 RBF channels gives `192`. For the edge distance block I add, on top of the 16 directed real pairs, the same-index virtual distances `V_i^k` to `V_j^k` (`3` of them) plus the ordered distinct cross-virtual distances (`3 * (3 - 1) = 6`), so `16 + 3 + 6 = 25` blocks, times 16 gives `400`. Now the angle and direction pieces. For nodes I want `sin` and `cos` for six angles: the three torsions `(phi, psi, omega)` and the three bond angles `(alpha, beta, gamma)`, so periodicity is handled smoothly — `12` channels. For node directions I take the inner atoms `N`, `C`, and `O`, subtract `CA`, normalize, and project into `Q_i`: three atoms times three components is `9`. For edges the quaternion of `Q_i^T Q_j` is `4`, and the directions from `CA_i` to all four atoms of residue `j` projected into `Q_i` are `4 * 3 = 12`. Adding up: `node_in = 192 + 12 + 9 = 213` and `edge_in = 400 + 4 + 12 = 416`. I ran the same counts in a small script to be sure I had not double-counted the virtual blocks, and they come out to `213` and `416`. Those are clean numbers that account for every channel, so I do not need to append a separate positional-encoding block to pad the tensor; the geometry already fills it.

Now the graph encoder has to use these features. Standard graph attention would make a query from node `i`, make a key and value from neighbor information, and score by a dot product divided by `sqrt(d)`. The dot product is the restrictive part. In a protein graph, the edge feature is not a small bias; it is the actual pairwise geometry. I want the edge to participate directly in deciding how much neighbor `j` matters to center residue `i`. So instead of separate query and key projections, I score the edge with an MLP over the concatenated neighbor node, edge, and center node:

```
w_ji = AttMLP(h_j^l || e_ji^l || h_i^l) / sqrt(d_head)
a_ji = softmax over j in N_i of w_ji
v_j  = NodeMLP(e_ji^l || h_j^l)
hhat_i = sum_j a_ji v_j
```

The `1 / sqrt(d_head)` scaling matters for the same reason it matters in Transformer attention: without it, larger head dimensions make logits too large and the neighbor softmax saturates. The subtler point is the scope of the softmax: it has to be over the incoming neighbors of the same center residue, not over all edges in the batch. In sparse code, where edges from every protein are flattened into one long list, that means grouping by `center_id` and using a scatter softmax. It is easy to get this wrong, so let me check it on a tiny graph: three residues, center 0 with neighbors {1,2}, centers 1 and 2 each with neighbor {0}, and some logits. A scatter-softmax grouped on the center index gives, for center 0, weights `0.731` and `0.269` that sum to `1.0`, and `1.0` for each of the single-neighbor centers — proper per-residue distributions. A plain global softmax over the same four logits instead spreads mass across centers: center 0's two edges then sum to only `0.83`, which is not a distribution over its own neighbors. So the grouping is not cosmetic; the wrong reduction would feed each center a mis-normalized convex combination. When I flatten the graph I store `edge_idx[0]` as the center residue and `edge_idx[1]` as the neighbor residue, so the code can group exactly on the center index.

After this node update, a residual connection, batch normalization, and a four-times-wide feed-forward layer give the usual per-node refinement. But if I stop there, the edges remain frozen at their input embeddings while the nodes change layer by layer. That is a bad mismatch, because the geometry lives mainly on the edges. Once the endpoint states have been refined, the edge between them should be refined too. So the edge update is another MLP over the two endpoints and the current edge:

```
e_ji^{l+1} = EdgeMLP(hhat_j || e_ji^l || hhat_i)
```

again with a residual connection and batch normalization. Now the graph is not just passing messages through static geometric labels; node and edge states co-adapt.

Local neighborhoods are not the whole protein. Full global attention would give every residue direct access to every other residue, but that is quadratic in length. I only need a cheap summary of the protein-level context: take the mean of the current node embeddings over residues in the same protein, then let that vector gate each residue channel. For residue `i`,

```
c_i = mean({hhat_k : k belongs to the same protein as i})
h_i^{l+1} = hhat_i * sigmoid(GateMLP(c_i)).
```

I considered just adding `c_i` (or a projection of it) to each node instead of gating. The problem with addition is that it overwrites: the same global vector is broadcast to every residue, so an additive term pushes all residues' embeddings in a shared direction and washes out the per-residue geometry I worked to build. A multiplicative `sigmoid` gate in `[0, 1]` instead lets the global summary keep or suppress channels per residue while the local representation stays the carrier of residue-specific information — at worst it scales a channel toward zero, it never injects a residue-independent offset. This is linear in the number of residues, and it fits naturally after the local node and edge updates. One block is therefore node attention, feed-forward refinement, edge update, and global channel gating. Stack the block enough times and the structural encoder can carry dependencies that an autoregressive decoder would otherwise have to handle.

That brings me to the decoder. The autoregressive factorization `p(S | X) = product_t p(s_t | s_<t, X)` is expressive, but it pays a sequential cost. Why does it help at all? It models residue-residue dependencies through previously generated sequence tokens. But many of those dependencies are induced by the shared structure: if two residues pack together, the backbone geometry and the contact graph are already telling the encoder that. So if the encoder is strong enough to fold those dependencies into each node embedding, the gain from sequential conditioning should shrink. That suggests trying the one-shot factorization:

```
p(S | X) = product_i p(s_i | X)
log p(s_i | X) = log_softmax(W h_i)_s_i.
```

This does not claim the residues are physically independent; it is a bet that the final node embeddings already contain the structural context needed for each marginal prediction. Whether that bet holds is an empirical question I cannot settle on paper — I would want to compare recovery against an autoregressive head on a held-out CATH split before believing the sequential step is redundant. What I can say is what it buys if it holds: the decoder is just a linear readout to 20 amino-acid logits and a log-softmax, training stays per-residue cross-entropy, and inference becomes one parallel forward pass with no length-`L` sequential loop. Given how much I have loaded onto the encoder, this is the version worth trying first.

The code I end up with mirrors the sparse implementation: the feature builder produces 213 node channels and 416 edge channels under the default switches, masked residues and valid neighbor edges are flattened, the node update receives `edge || neighbor` as its value input, the MLP score sees `center || edge || neighbor` in the implementation order, and scatter operations keep every protein and every center residue separated.

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_softmax, scatter_sum

NUM_AA = 20


def normalize(x, dim=-1, eps=1e-8):
    return F.normalize(x, dim=dim, eps=eps)


def rbf(D, num_rbf=16, d_min=0.0, d_max=20.0):
    centers = torch.linspace(d_min, d_max, num_rbf, device=D.device)
    width = (d_max - d_min) / num_rbf
    return torch.exp(-((D.unsqueeze(-1) - centers) / width) ** 2)


def local_frame(N, CA, C):
    u = CA - N
    v = C - CA
    b = normalize(u - v)
    n = normalize(torch.cross(u, v, dim=-1))
    t = torch.cross(b, n, dim=-1)
    return b, n, t


class ResidueFeaturizer(nn.Module):
    def __init__(self, num_virtual=3, num_rbf=16):
        super().__init__()
        self.num_virtual = num_virtual
        self.num_rbf = num_rbf
        self.virtual_atoms = nn.Parameter(torch.rand(num_virtual, 3))
        self.node_in = (6 + num_virtual * (num_virtual - 1)) * num_rbf + 12 + 9
        self.edge_in = (
            (16 + num_virtual + num_virtual * (num_virtual - 1)) * num_rbf
            + 4
            + 12
        )

    def virtual_positions(self, X_flat):
        N, CA, C = X_flat[:, 0], X_flat[:, 1], X_flat[:, 2]
        b, n, t = local_frame(N, CA, C)
        coeff = normalize(self.virtual_atoms, dim=-1)
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
        msg = self.W13(F.gelu(self.W12(F.gelu(self.W11(h_EV)))))
        return self.norm(h_E + self.dropout(msg))


class ContextGate(nn.Module):
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
        self.context = ContextGate(hidden_dim)
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


class ProDesignModel(nn.Module):
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

So I have a coherent final shape: a rich invariant featurizer with shared unit-norm virtual atoms in the residue frame, whose distance invariance I checked numerically; a stack of sparse graph blocks that update nodes by MLP-scored neighbor attention with a per-center softmax I checked on a small graph, update edges from both endpoints, and inject global context by a mean-pooled sigmoid gate; and a one-shot linear decoder trained with per-residue cross-entropy, on the bet — to be confirmed against an autoregressive baseline — that a strong enough encoder makes the sequential step unnecessary.
