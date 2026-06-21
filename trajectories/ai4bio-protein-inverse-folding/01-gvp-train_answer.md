The scaffold default is already a working message-passing encoder — a KNN graph over $C\alpha$, dihedral and orientation scalars on the nodes, a distance-RBF plus a $C\alpha\!\to\!$neighbor direction vector on the edges, three plain MPNN layers feeding an MLP decoder. It runs and it is invariant, but its invariance is bought the wrong way. The instant `F.normalize(X_ca_neighbors - X_ca)` is concatenated onto the edge RBFs and pushed through `edge_embed`, the geometry stops being geometry: the three direction components become entries in a feature vector, and three layers deep the network can only do affine arithmetic on them — it can no longer treat that arrow as something to rotate, project into a frame, or recombine. This is the Ingraham (2019) freeze restated in the scaffold's own code: invariance is purchased by killing the geometry at the door. I want the first rung to keep the relational message passing but stop freezing the geometry — let directional information flow through the graph as honest 3D objects I can keep manipulating at every layer, while the scalar predictions stay invariant.

I propose a **geometric vector perceptron (GVP) encoder**, the minimal change to the default that un-freezes the geometry. The construction is forced once I ask which operations on a vector feature survive a rigid motion. Suppose a node (and an edge) carries, alongside its scalar features $s$, a set of vector features $V\in\mathbb{R}^{\nu\times 3}$. Rotate or reflect the input — right-multiply every vector by a unitary $U$. I need the scalar outputs not to move at all (invariance) and the vector outputs to rotate with $U$ (equivariance). Enumerate the allowed moves. A channel-mixing linear map $WV$ commutes with $U$ because $W$ touches the channel index and $U$ touches the spatial axis: $(WV)U = W(VU)$ — equivariant, but only if it is *bias-free*, since there is no nonzero rotation-invariant constant vector to add. The L2 norm of a row $\|v\|$ is preserved by $U$ ($\|vU\| = \|v\|$), so it is flatly invariant — that norm is my one bridge from a vector into a scalar. And the forbidden move is the dangerous one: a coordinate-wise nonlinearity like $\mathrm{ReLU}(v_x),\mathrm{ReLU}(v_y),\mathrm{ReLU}(v_z)$ scrambles under rotation because the components are defined only relative to my arbitrary axes. But a vector path with no nonlinearity is just stacked linear maps and collapses, so I need *some* nonlinearity. The escape is forced: take the norm of a vector (invariant), pass it through any nonlinearity (it is a scalar now, anything goes), and use the result to scale the vector,
$$v' = \sigma^{+}(\|v\|)\,\cdot\,v,$$
which checks as $\sigma^{+}(\|vU\|)\cdot(vU) = (\sigma^{+}(\|v\|)\cdot v)U$ — equivariant, direction preserved, only the length modulated. So the closed toolkit for vectors is exactly three things: bias-free channel-linear maps, L2 norms, and scale-by-a-function-of-the-norm, with the norm as the one-way door into the scalar path.

Package that toolkit as a perceptron processing a tuple $(s, V)$. The scalar path must *see* the vectors or the geometry never reaches the prediction, so I concatenate the norms of transformed vector channels onto $s$ before the scalar linear and nonlinearity. The vector path channel-mixes and then scales by its own norms. One subtlety would otherwise tie two unrelated widths together: the number of norms I feed the scalars (call it $h$) is about how much invariant geometric summary I want, while the number of output vector channels ($\mu$) is about how rich a directional representation I propagate — if one matrix did both, $h=\mu$ by accident. So I split it: $W_h$ lifts the input vectors to an intermediate $V_h$ with $h=\max(\nu_\text{in},\nu_\text{out})$ channels whose row-norms feed the scalars, and a second bias-free $W_\mu$ produces the output vectors $V_\mu = W_\mu V_h$, scaled by $\sigma^{+}(\|V_\mu\|)$. Now $h$ and $\mu$ are decoupled. This module — scalar path reads vector norms, vector path channel-mixes then scales-by-norm — is the GVP, and its equivariance is just the toolkit: every operation it ever applies to a vector is $W_h$, $W_\mu$, a norm, or a norm-based scaling, all of which commute with $U$. The supporting pieces obey the same constraint. Ordinary LayerNorm would subtract a mean vector and apply per-coordinate scale/shift, both of which break equivariance; the allowed move is a single invariant rescaling that divides the vectors by their root-mean-square norm, with no per-coordinate parameters. Dropout drops whole vector channels (an invariant per-channel 0/1 scaling), never a coordinate.

The encoder runs Gilmer (2017) message passing with GVPs as the learned functions. Per node, the scalar features are the six backbone dihedrals $\{\sin,\cos\}(\phi,\psi,\omega)$ and the vector features are the unit arrows $C\alpha\!\to\!N$, $C\alpha\!\to\!C$, $C\alpha\!\to\!O$ — three directions that pin the residue's orientation, written once per node, absolutely, with no per-neighbor frame redundancy of the kind Ingraham stored. Per edge $(j\!\to\!i)$ the scalar features are the $C\alpha$ distance in 16 RBFs plus a 16-dim sinusoidal encoding of the backbone offset $j-i$, and the vector feature is the single unit direction $C\alpha_j - C\alpha_i$ kept as a vector, so the GVP can take whatever norms and inner products it wants and the direction stays geometric downstream. Input GVPs (identity activations, just to raise dimension) lift these to hidden node tuples $(100,16)$ and edge tuples $(32,1)$, each followed by the equivariant LayerNorm; 100 scalar channels carry residue context while 16 vector channels are a deliberately modest geometric width — each is a full arrow, and the value is in keeping a handful of meaningful directions live rather than in brute width. For each edge I concatenate the source node tuple, the edge tuple, and the target node tuple (scalars to scalars, vectors to vectors, since they live in different spaces), run a short GVP stack whose last layer uses identity activations so the summed message stays expressive, mask out padded neighbors, and aggregate by a degree-stable mean. A residual update with the equivariant LayerNorm follows, then a wide GVP feed-forward (expand, contract, identity-activation last), residual and normed again. Because each GVP updates *both* scalar and vector channels, the vector features at every node get refined as the network deepens — that is the whole point. Three such layers match the scaffold's budget, and a final GVP with a $(\textit{hidden\_dim},0)$ scalar-only output gives the per-residue embedding the scaffold's two-layer MLP decoder consumes. Invariance is then guaranteed by construction: node scalars (dihedrals) and edge scalars (distances, offset) are invariant, node vectors ($C\alpha\!\to\!N/C/O$) and edge vectors ($C\alpha_j - C\alpha_i$) are equivariant, every layer touches vectors only through the GVP toolkit and the equivariant LayerNorm, and the final readout keeps scalars only — so rotating the backbone leaves the predicted log-probabilities fixed, at $O(L\cdot k)$ cost.

The task only wants the *encoder*. The generic GVP design wraps this in a sequence-conditioned autoregressive decoder with a causal mask, but the harness gives `forward(X, mask)` with no sequence input at all and a fixed MLP head — so the autoregressive decoder, the sequence-on-edges, and the `src < dst` masking are all amputated, and what survives is the dense, batched, encoder-only GVP above. I expect this to be the *weakest* rung even so: its per-edge geometry is thin — one $C\alpha\!\to\!C\alpha$ direction and one $C\alpha$-only distance under-determine the relative pose of two residues, which is the signal the prediction most depends on — and its decoder is amputated, hurting a method that expected the decoder to carry the residue-residue couplings. So I anticipate recovery in the low-0.43 range on CATH 4.2 and 4.3 with perplexity well above 5.5, and the diagnosis for the next rung writing itself: not *more* invariance machinery but *richer invariant features* on the edges.

```python
# =====================================================================
# EDITABLE SECTION START — GVP baseline
# =====================================================================

import numpy as np


def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    """L2 norm clamped above eps."""
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


def gather_nodes_gvp(h_V, E_idx):
    B, L, K = E_idx.shape
    D = h_V.shape[-1]
    h_V_expand = h_V.unsqueeze(2).expand(-1, -1, K, -1)
    E_idx_expand = E_idx.unsqueeze(-1).expand(-1, -1, -1, D)
    return torch.gather(h_V_expand, 1, E_idx_expand)


def gather_vectors(V, E_idx):
    """Gather vector features. V: (B, L, n_vec, 3), E_idx: (B, L, K)"""
    B, L, K = E_idx.shape
    nv = V.shape[2]
    E_idx_v = E_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, nv, 3)
    V_expand = V.unsqueeze(2).expand(-1, -1, K, -1, -1)
    return torch.gather(V_expand, 1, E_idx_v)


class GVPModule(nn.Module):
    """Geometric Vector Perceptron — dense batched version.

    Processes tuples of (scalar, vector) features.
    Scalar: (B, L, s_in) -> (B, L, s_out)
    Vector: (B, L, v_in, 3) -> (B, L, v_out, 3)
    """

    def __init__(self, in_dims, out_dims, activations=(F.relu, torch.sigmoid)):
        super().__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.scalar_act, self.vector_act = activations

        if self.vi:
            self.h_dim = max(self.vi, self.vo)
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)

    def forward(self, s, v=None):
        if self.vi and v is not None:
            # v: (*, vi, 3)
            v_t = v.transpose(-1, -2)  # (*, 3, vi)
            vh = self.wh(v_t)  # (*, 3, h_dim)
            vn = _norm_no_nan(vh, axis=-2)  # (*, h_dim)
            s = self.ws(torch.cat([s, vn], -1))
            if self.vo:
                v_out = self.wv(vh).transpose(-1, -2)  # (*, vo, 3)
                if self.scalar_act:
                    gate = self.wsv(self.scalar_act(s))
                else:
                    gate = self.wsv(s)
                v_out = v_out * torch.sigmoid(gate).unsqueeze(-1)
            else:
                v_out = None
        else:
            s = self.ws(s)
            v_out = None

        if self.scalar_act:
            s = self.scalar_act(s)
        return s, v_out


class GVPLayerNorm(nn.Module):
    """LayerNorm for GVP scalar+vector tuples."""
    def __init__(self, dims):
        super().__init__()
        self.s_dim, self.v_dim = dims
        self.norm_s = nn.LayerNorm(self.s_dim)

    def forward(self, s, v=None):
        s = self.norm_s(s)
        if v is not None and self.v_dim > 0:
            vn = _norm_no_nan(v, axis=-1, keepdims=True)
            v = v / vn.clamp(min=1e-5)  # unit vectors, scaled
            v = v * vn  # restore magnitude (still normalized in mean)
        return s, v


class GVPConvLayer(nn.Module):
    """GVP convolution layer — dense batched version.

    Message passing with GVP for both node and edge updates.
    """

    def __init__(self, node_dims, edge_dims, drop_rate=0.1):
        super().__init__()
        self.node_s, self.node_v = node_dims
        self.edge_s, self.edge_v = edge_dims

        # Message function: edge_s + 2*node_s, edge_v + 2*node_v -> node_s, node_v
        msg_in_s = self.edge_s + 2 * self.node_s
        msg_in_v = self.edge_v + 2 * self.node_v
        self.msg_gvp = nn.Sequential(
            GVPModule((msg_in_s, msg_in_v), (self.node_s, self.node_v)),
            GVPModule((self.node_s, self.node_v), (self.node_s, self.node_v),
                      activations=(None, None)),
        )

        # Node update
        self.ff_gvp = nn.Sequential(
            GVPModule((self.node_s, self.node_v), (self.node_s * 4, self.node_v)),
            GVPModule((self.node_s * 4, self.node_v), (self.node_s, self.node_v),
                      activations=(None, None)),
        )

        self.norm1 = GVPLayerNorm(node_dims)
        self.norm2 = GVPLayerNorm(node_dims)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, h_s, h_v, e_s, e_v, E_idx, mask, mask_attend):
        """
        h_s: (B, L, node_s), h_v: (B, L, node_v, 3)
        e_s: (B, L, K, edge_s), e_v: (B, L, K, edge_v, 3)
        E_idx: (B, L, K), mask: (B, L), mask_attend: (B, L, K)
        """
        B, L, K = E_idx.shape

        # Gather neighbor node features
        h_s_j = gather_nodes_gvp(h_s, E_idx)  # (B, L, K, node_s)
        h_s_i = h_s.unsqueeze(2).expand(-1, -1, K, -1)

        # Build message input (scalar)
        msg_s = torch.cat([h_s_i, e_s, h_s_j], dim=-1)  # (B, L, K, msg_in_s)

        # Build message input (vector)
        if h_v is not None:
            h_v_j = gather_vectors(h_v, E_idx)  # (B, L, K, node_v, 3)
            h_v_i = h_v.unsqueeze(2).expand(-1, -1, K, -1, -1)
            if e_v is not None:
                msg_v = torch.cat([h_v_i, e_v, h_v_j], dim=-2)  # (B, L, K, msg_in_v, 3)
            else:
                msg_v = torch.cat([h_v_i, h_v_j], dim=-2)
        else:
            msg_v = e_v

        # Apply message GVP
        for layer in self.msg_gvp:
            msg_s, msg_v = layer(msg_s, msg_v)

        # Mask and aggregate
        mask_expand = mask_attend.unsqueeze(-1)
        msg_s = msg_s * mask_expand
        if msg_v is not None:
            msg_v = msg_v * mask_expand.unsqueeze(-1)

        # Sum aggregation
        num_neighbors = mask_attend.sum(dim=-1, keepdim=True).clamp(min=1)
        agg_s = msg_s.sum(dim=2) / num_neighbors
        if msg_v is not None:
            agg_v = msg_v.sum(dim=2) / num_neighbors.unsqueeze(-1)
        else:
            agg_v = None

        # Residual + norm
        h_s_res, h_v_res = self.norm1(h_s + self.drop(agg_s),
                                        h_v + self.drop(agg_v) if h_v is not None and agg_v is not None else h_v)

        # Feed-forward
        ff_s, ff_v = h_s_res, h_v_res
        for layer in self.ff_gvp:
            ff_s, ff_v = layer(ff_s, ff_v)

        h_s_out, h_v_out = self.norm2(h_s_res + self.drop(ff_s),
                                        h_v_res + self.drop(ff_v) if h_v_res is not None and ff_v is not None else h_v_res)

        # Mask
        h_s_out = h_s_out * mask.unsqueeze(-1)
        if h_v_out is not None:
            h_v_out = h_v_out * mask.unsqueeze(-1).unsqueeze(-1)

        return h_s_out, h_v_out


class StructureEncoder(nn.Module):
    """GVP-based structure encoder.

    Uses geometric vector perceptrons for SE(3)-equivariant message passing.
    Node features: scalar (6) = dihedrals; vector (3) = local frame vectors.
    Edge features: scalar (32) = RBF distances + positional; vector (1) = direction.
    """

    def __init__(self, hidden_dim=128, num_layers=3, k_neighbors=30, dropout=0.1, num_rbf=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        self.num_rbf = num_rbf

        # Dimensions
        self.node_s_in = 6    # dihedral sin/cos
        self.node_v_in = 3    # 3 direction vectors
        self.node_s_h = 100   # hidden scalar dim (GVP default)
        self.node_v_h = 16    # hidden vector dim
        self.edge_s_in = num_rbf + 16  # RBF + positional encoding
        self.edge_v_in = 1    # direction unit vector
        self.edge_s_h = 32    # hidden edge scalar
        self.edge_v_h = 1     # hidden edge vector

        # Input projections
        self.W_v = GVPModule(
            (self.node_s_in, self.node_v_in),
            (self.node_s_h, self.node_v_h),
            activations=(None, None)
        )
        self.norm_v = GVPLayerNorm((self.node_s_h, self.node_v_h))

        self.W_e = GVPModule(
            (self.edge_s_in, self.edge_v_in),
            (self.edge_s_h, self.edge_v_h),
            activations=(None, None)
        )
        self.norm_e = GVPLayerNorm((self.edge_s_h, self.edge_v_h))

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            GVPConvLayer(
                (self.node_s_h, self.node_v_h),
                (self.edge_s_h, self.edge_v_h),
                drop_rate=dropout
            )
            for _ in range(num_layers)
        ])

        # Output projection to scalar hidden_dim
        self.out_proj = nn.Linear(self.node_s_h, hidden_dim)

    def forward(self, X, mask):
        B, L = int(X.shape[0]), int(X.shape[1])
        X_ca = X[:, :, 1, :]

        # Build KNN graph
        E_idx, D_neighbors = knn_graph(X_ca, mask, self.k_neighbors)
        K = int(E_idx.shape[2])

        # Node features
        # Scalar: dihedral angles
        node_s = _dihedrals(X)  # (B, L, 6)

        # Vector: local frame vectors (CA->N, CA->C, CA->O unit vectors)
        N_pos, CA_pos, C_pos, O_pos = X[:, :, 0], X[:, :, 1], X[:, :, 2], X[:, :, 3]
        v_cn = F.normalize(N_pos - CA_pos, dim=-1)   # (B, L, 3)
        v_cc = F.normalize(C_pos - CA_pos, dim=-1)
        v_co = F.normalize(O_pos - CA_pos, dim=-1)
        node_v = torch.stack([v_cn, v_cc, v_co], dim=2)  # (B, L, 3, 3)

        # Edge features
        # Scalar: RBF distances + positional encoding
        rbf = _rbf(D_neighbors, device=X.device)  # (B, L, K, num_rbf)
        residue_idx = torch.arange(L, device=X.device).unsqueeze(0).expand(B, -1)
        offset = residue_idx.unsqueeze(2) - torch.gather(
            residue_idx.unsqueeze(2).expand(-1, -1, K), 1,
            E_idx.clamp(0, L - 1)
        )
        pe_dim = 16
        freq = torch.exp(torch.arange(0, pe_dim, 2, dtype=torch.float32, device=X.device) * -(np.log(10000.0) / pe_dim))
        angles = offset.unsqueeze(-1).float() * freq
        pos_enc = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        edge_s = torch.cat([rbf, pos_enc], dim=-1)  # (B, L, K, num_rbf+16)

        # Vector: direction to neighbors
        CA_neighbors = gather_nodes_gvp(CA_pos, E_idx)  # (B, L, K, 3)
        edge_dir = F.normalize(CA_neighbors - CA_pos.unsqueeze(2), dim=-1)  # (B, L, K, 3)
        edge_v = edge_dir.unsqueeze(3)  # (B, L, K, 1, 3)

        # Project inputs
        h_s, h_v = self.W_v(node_s, node_v)
        h_s, h_v = self.norm_v(h_s, h_v)

        e_s, e_v = self.W_e(edge_s, edge_v)
        e_s, e_v = self.norm_e(e_s, e_v)

        # Attention mask
        mask_attend = torch.gather(mask.unsqueeze(2).expand(-1, -1, K), 1,
                                    E_idx.clamp(0, L - 1))
        mask_attend = mask.unsqueeze(-1) * mask_attend

        # Message passing
        for layer in self.encoder_layers:
            h_s, h_v = layer(h_s, h_v, e_s, e_v, E_idx, mask, mask_attend)

        # Project to output dim
        h_V = self.out_proj(h_s)
        return h_V


class InverseFoldingModel(nn.Module):
    """GVP inverse folding model."""

    def __init__(self, hidden_dim=128, num_encoder_layers=3, k_neighbors=30,
                 dropout=0.1, num_rbf=16):
        super().__init__()
        self.encoder = StructureEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_encoder_layers,
            k_neighbors=k_neighbors,
            dropout=dropout,
            num_rbf=num_rbf,
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_AA),
        )

    def forward(self, X, mask):
        h_V = self.encoder(X, mask)
        logits = self.decoder(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs
```
