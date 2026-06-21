I am handed a protein backbone — for each of $L$ residues the coordinates of the heavy atoms $N, C_\alpha, C, O$ — and asked to produce an amino-acid sequence that would fold into that shape. Folding maps sequence to structure; this is the inverse, structure to sequence, the central computational step in protein design. The honest difficulties have to be named before any machinery is chosen. The map is degenerate: many sequences fold to nearly the same backbone, so there is no single correct answer, and regressing to one would be wrong in principle — what I actually want is a distribution $p(s \mid x)$ over compatible sequences. The couplings are awkward: residues far apart along the chain can be in direct contact in 3D and must be chosen jointly, while sequence-adjacent residues may point into opposite environments. And the only legitimate determinant of the sequence is the geometry, so whatever I compute must be invariant to rigid-body motion — rotating or translating the whole molecule cannot change the answer. The target is therefore an invariant joint distribution over all $L$ positions, on proteins of hundreds of residues, fitting on one GPU.

The physics route is Rosetta: hand-build an energy function (van der Waals, solvation, hydrogen bonds, electrostatics, rotamer statistics) and Monte-Carlo over amino-acid identities and side-chain rotamers until the energy is low. It encodes real structural-biology knowledge, but the energy is approximate and manually tuned, the rotamer-packing search is slow (seconds to minutes per design), and it recovers the native amino acid only about a third of the time. I would rather learn the energy from the Protein Data Bank directly. The strongest learned prior is the Structured Transformer, which casts design as autoregressive language modeling conditioned on the structure graph, $p(s \mid x) = \prod_i p(s_i \mid x, s_{<i})$, with a sparse graph-restricted encoder–decoder over a $k$-nearest-neighbor graph; it beats the profile baselines and beats Rosetta on recovery and speed. But tracing it reveals four things left on the table. Its encoder refines only the node embeddings while the geometric edge features — which carry the inter-residue signal — are computed once and frozen forever. Its decoding is strictly left-to-right ($N \to C$), so at inference a position can only condition on lower-index residues, forbidding the use of arbitrary fixed positions as context for partial redesign, symmetry, or complexes. It carries multi-head attention and explicit orientation/quaternion edge features even though the ablation on this very data showed a plain MLP-over-neighbors message generalizing better and distance-only edges already beating the profile baselines. And it trains only on crystal-exact coordinates, so it has never seen the imperfect predicted or hallucinated backbones it must handle at deployment.

I propose ProteinMPNN, a message-passing graph neural network that fixes each of these four points while keeping the autoregressive graph-conditioned skeleton. The backbone becomes a $k$-nearest-neighbor graph over $C_\alpha$ atoms ($k$ set by the caller; the training class defaults to $32$), and computation runs as message passing: each round forms a message into every node by summing a learned function over its neighbors and then updates the node, $m_i = \sum_{j \in N(i)} M(h_i, h_j, e_{ij})$, $h_i' = U(h_i, m_i)$. The symmetric neighbor sum gives permutation invariance for free, the $k$-NN restriction makes everything $O(L \cdot k)$ — linear in length — and the contacts that matter are spatially local anyway, so almost nothing is lost.

The first design decision is to get invariance purely from distances. Pairwise inter-atomic distances are invariant to rotation and translation by construction, so a graph whose edges carry distance-derived features is automatically invariant with no equivariant vector bookkeeping. But a single $C_\alpha$–$C_\alpha$ distance per edge is not locally informative — two neighbors at the same distance could sit on the same side or opposite sides of a residue. The resolution is to use enough distances: for each edge $(i,j)$ I take the pairwise distances among a small set of backbone atoms on both residues, which pins down the relative rigid-body pose as a little distance-geometry problem and encodes orientation implicitly through the pattern of which atoms are near which. Side-chain identity is what I am predicting, so I have no side-chain atoms, but the direction a side chain would point is a strong environmental cue, and I can recover a proxy for free by placing a virtual $C_\beta$ at the ideal tetrahedral position from the backbone: with $b = C_\alpha - N$, $c = C - C_\alpha$, $a = b \times c$,
$$C_\beta = -0.58273431\,a + 0.56802827\,b - 0.54067466\,c + C_\alpha,$$
where the constants are just the tetrahedral geometry. With five atoms $\{N, C_\alpha, C, O, C_\beta\}$ per residue I take $25$ ordered atom-pair distances per edge — the five same-atom pairs plus cross pairs in both orientations, since keeping both directions helps the network read the relative pose. A raw distance is a poor input to a linear layer, so each is expanded into a bank of $16$ Gaussian radial basis functions,
$$\mathrm{RBF}(D)_n = \exp\!\left(-\left((D - \mu_n)/\sigma\right)^2\right),\qquad \mu_n = \mathrm{linspace}(2, 22, 16)\ \text{\AA},\quad \sigma = (22-2)/16,$$
turning "distance $\approx \mu_n$" into a near-one-hot soft code the first linear layer can read like a lookup table. That gives $25 \cdot 16 = 400$ distance features per edge. Sequence position enters as a relative encoding: I take the signed offset $i - j$, clip it to a window $[-32, 32]$, reserve one extra bucket for cross-chain neighbors, one-hot it, and learn a $16$-dimensional positional embedding, concatenated onto the $400$ distance features and embedded to the hidden width with a linear map and LayerNorm.

The second decision pushes the "geometry lives in the edges" principle to its limit: the node states are initialized to zero, carrying no geometry at all, so every structural signal must enter through the edges and propagate into the nodes by message passing. The dihedral features the Structured Transformer put in its nodes are a weaker, redundant summary of the same backbone and risk encoding fold-specific quirks; forcing all information through the invariant relational channel is exactly what should transfer to unseen folds.

The third decision is the encoder update, which now refreshes edges as well as nodes. For the node message I gather, per edge, the triple $[h_i, h_j, e_{ij}]$ and run it through a three-linear MLP $W_3 \circ g \circ W_2 \circ g \circ W_1$ with $g = \mathrm{GELU}$. The neighbor sum has a scale subtlety: summing literally makes the message magnitude grow with the (variable, masked) neighbor count, which would fight LayerNorm, so I divide by a fixed constant $\mathrm{scale} = 30$ of the order of the neighborhood size to get a degree-stable approximate mean,
$$\Delta h_i = \frac{1}{\mathrm{scale}}\sum_{j \in N(i)} m_{ij},\qquad h_i \leftarrow \mathrm{LayerNorm}(h_i + \mathrm{dropout}(\Delta h_i)),\qquad h_i \leftarrow \mathrm{LayerNorm}(h_i + \mathrm{dropout}(\mathrm{FFN}(h_i))),$$
the Transformer block shape with attention replaced by the symmetric MLP-over-neighbors the ablation favored, followed by a $4\times$-wide position-wise feed-forward. Then, with the refreshed node states, the edge is re-derived through a second three-linear MLP and residual-added,
$$e_{ij} \leftarrow \mathrm{LayerNorm}\big(e_{ij} + \mathrm{dropout}(W_{13}(g(W_{12}(g(W_{11}([h_i, h_j, e_{ij}]))))))\big),$$
so each contact's meaning is refined through depth rather than frozen at featurization. Three encoder layers reach roughly three-hop spatial neighborhoods — enough for the dominant local geometry — at hidden dimension $128$.

The fourth decision is order-agnostic decoding. Any permutation $o$ of positions is a valid factorization of the same joint, $p(s \mid x) = \prod_t p(s_{o_t} \mid x, s_{o_{<t}})$, so locking in $N \to C$ is an arbitrary restriction. I train over random orders: each example samples an order and predicts each residue from the structure plus the amino acids preceding it in that order. This lets a designer fix arbitrary positions and design the rest, gives a denser BERT-like training signal since each residue is predicted from many context subsets, and is a consistent objective because every order is a correct factorization. Implementing the causal mask: a per-position flag $\mathrm{chain\_M}$ is $1$ for designed positions and $0$ for fixed ones, and sorting by $(\mathrm{chain\_M} + 10^{-4})\,|\mathrm{randn}|$ places fixed positions first (tiny keys) and designed positions in random order, so $\mathrm{decoding\_order} = \mathrm{argsort}((\mathrm{chain\_M} + 10^{-4})\,|\mathrm{randn}|)$. Let $P$ be the one-hot permutation matrix of this order and $\hat{L} = 1 - \mathrm{triu}(\mathbf{1}_{L\times L})$ the strictly-lower-triangular matrix in order space, with $\hat{L}_{ij} = 1$ iff decode step $j$ precedes step $i$. Conjugating through $P$ carries this from order space back to position space,
$$\mathrm{order\_mask}_{b,q,p} = \sum_{i,j} \hat{L}_{ij}\,P_{b,i,q}\,P_{b,j,p} = \mathrm{einsum}('ij,biq,bjp\!\to\! bqp',\,\hat{L},P,P),$$
which is $1$ exactly when $p$ precedes $q$ in the order — the already-decoded-predecessors indicator I wanted. Gathering it along the neighbor axis splits each position's neighbors into backward (decoded) and forward (undecoded): an already-decoded neighbor contributes its amino-acid embedding $W_s(s_j)$ on top of its structure, a not-yet-decoded neighbor contributes structure only with its sequence zeroed, via $\mathrm{mask\_bw} = \mathrm{mask}\cdot\mathrm{mask\_attend}$ and $\mathrm{mask\_fw} = \mathrm{mask}\cdot(1-\mathrm{mask\_attend})$. The three decoder layers are the same message-plus-feed-forward block as the encoder but without the edge update (the edges are done being refined), and a final linear maps each node to the $21$ amino-acid logits followed by log-softmax. Training uses masked cross-entropy with label smoothing $0.1$ — appropriate for a degenerate target with many acceptable answers — the smoothed objective normalized by a fixed $2000.0$ in the implementation.

The last decision addresses the deployment distribution shift. Training on crystal-exact coordinates but deploying on imperfect backbones means the model would overfit to sub-Ångström detail that simply will not be present. The fix is input-space augmentation: during training only, and only inside featurization so the labels are untouched, perturb the coordinates $X \leftarrow X + \mathrm{augment\_eps}\cdot N(0,1)$. This destroys the fine detail and forces reliance on the coarser geometry that survives prediction or hallucination. The noise level is a robustness-versus-fidelity knob swept over $\{0.02, 0.10, 0.20, 0.30\}$ Å and chosen by the backbone source. Training is Adam with $\beta = (0.9, 0.98)$, $\epsilon = 10^{-9}$ under the Noam warmup schedule $\mathrm{lr} = \mathrm{factor}\cdot d^{-0.5}\cdot \min(\mathrm{step}^{-0.5}, \mathrm{step}\cdot \mathrm{warmup}^{-1.5})$ with $\mathrm{factor} = 2$, $\mathrm{warmup} = 4000$, dropout $0.1$ throughout, and sampling from a temperature-adjusted softmax ($T \approx 0.1$ for high recovery, higher for diversity). Checking invariance end to end: zero node init is invariant, edges are functions of inter-atomic distances and relative offsets (invariant and frame-independent), every message is a symmetric MLP-over-neighbors sum, and the decoder's sequence embeddings are frame-independent — so the predicted distribution is invariant to rigid-body motion by construction, with cost linear in length.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

NUM_AA = 21  # 20 standard amino acids + 1 unknown/non-canonical slot


def gather_edges(edges, E_idx):
    nbr = E_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    return torch.gather(edges, 2, nbr)


def gather_nodes(h_V, E_idx):
    B, L, K = E_idx.shape
    flat = E_idx.reshape(B, L * K, 1).expand(-1, -1, h_V.size(2))
    return torch.gather(h_V, 1, flat).reshape(B, L, K, -1)


def cat_neighbors_nodes(h_nodes, h_edges, E_idx):
    return torch.cat([h_edges, gather_nodes(h_nodes, E_idx)], dim=-1)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super().__init__()
        self.W_in = nn.Linear(num_hidden, num_ff)
        self.W_out = nn.Linear(num_ff, num_hidden)
        self.act = nn.GELU()

    def forward(self, h):
        return self.W_out(self.act(self.W_in(h)))


class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2 * max_relative_feature + 2, num_embeddings)

    def forward(self, offset, same_chain):
        d = torch.clamp(offset + self.max_relative_feature, 0, 2 * self.max_relative_feature)
        d = d * same_chain + (1 - same_chain) * (2 * self.max_relative_feature + 1)
        d_onehot = F.one_hot(d.long(), 2 * self.max_relative_feature + 2)
        return self.linear(d_onehot.float())


class ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16, num_rbf=16,
                 top_k=30, augment_eps=0.0):
        super().__init__()
        self.edge_features, self.node_features = edge_features, node_features
        self.top_k, self.augment_eps, self.num_rbf = top_k, augment_eps, num_rbf
        self.embeddings = PositionalEncodings(num_positional_embeddings)
        self.edge_embedding = nn.Linear(num_positional_embeddings + num_rbf * 25, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(self, X, mask, eps=1e-6):
        mask_2D = mask.unsqueeze(1) * mask.unsqueeze(2)
        dX = X.unsqueeze(1) - X.unsqueeze(2)
        D = mask_2D * torch.sqrt((dX ** 2).sum(3) + eps)
        D_max, _ = D.max(-1, keepdim=True)
        D = D + (1.0 - mask_2D) * D_max
        D_nbr, E_idx = torch.topk(D, min(self.top_k, X.shape[1]), dim=-1, largest=False)
        return D_nbr, E_idx

    def _rbf(self, D):
        mu = torch.linspace(2.0, 22.0, self.num_rbf, device=D.device).view(1, 1, 1, -1)
        sigma = (22.0 - 2.0) / self.num_rbf
        return torch.exp(-((D.unsqueeze(-1) - mu) / sigma) ** 2)

    def _get_rbf(self, A, B, E_idx):
        D = torch.sqrt(((A[:, :, None, :] - B[:, None, :, :]) ** 2).sum(-1) + 1e-6)
        D_nbr = gather_edges(D[:, :, :, None], E_idx)[:, :, :, 0]
        return self._rbf(D_nbr)

    def forward(self, X, mask, residue_idx, chain_labels):
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
        N, Ca, C, O = X[:, :, 0, :], X[:, :, 1, :], X[:, :, 2, :], X[:, :, 3, :]
        b = Ca - N
        c = C - Ca
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + Ca
        D_nbr, E_idx = self._dist(Ca, mask)
        atoms = [Ca, N, C, O, Cb]
        pairs = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4),
                 (0, 1), (0, 2), (0, 3), (0, 4),
                 (1, 2), (1, 3), (1, 4), (4, 2), (4, 3), (3, 2),
                 (1, 0), (2, 0), (3, 0), (4, 0),
                 (2, 1), (3, 1), (4, 1), (2, 4), (3, 4), (2, 3)]
        RBF_all = torch.cat([self._get_rbf(atoms[i], atoms[j], E_idx) for i, j in pairs], dim=-1)
        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        offset = gather_edges(offset[:, :, :, None], E_idx)[:, :, :, 0]
        same_chain = ((chain_labels[:, :, None] - chain_labels[:, None, :]) == 0).long()
        E_chains = gather_edges(same_chain[:, :, :, None], E_idx)[:, :, :, 0]
        E = torch.cat([self.embeddings(offset.long(), E_chains), RBF_all], -1)
        return self.norm_edges(self.edge_embedding(E)), E_idx


class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30):
        super().__init__()
        self.scale = scale
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(num_hidden) for _ in range(3))
        self.drop1, self.drop2, self.drop3 = (nn.Dropout(dropout) for _ in range(3))
        self.W1 = nn.Linear(num_hidden + num_in, num_hidden)
        self.W2 = nn.Linear(num_hidden, num_hidden)
        self.W3 = nn.Linear(num_hidden, num_hidden)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden)
        self.W12 = nn.Linear(num_hidden, num_hidden)
        self.W13 = nn.Linear(num_hidden, num_hidden)
        self.act = nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask, mask_attend):
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_i = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_i, h_EV], -1)
        m = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            m = mask_attend.unsqueeze(-1) * m
        h_V = self.norm1(h_V + self.drop1(m.sum(-2) / self.scale))
        h_V = self.norm2(h_V + self.drop2(self.dense(h_V)))
        if mask is not None:
            h_V = mask.unsqueeze(-1) * h_V
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_i = h_V.unsqueeze(-2).expand(-1, -1, h_EV.size(-2), -1)
        h_EV = torch.cat([h_i, h_EV], -1)
        m = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.drop3(m))
        return h_V, h_E


class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, scale=30):
        super().__init__()
        self.scale = scale
        self.norm1, self.norm2 = nn.LayerNorm(num_hidden), nn.LayerNorm(num_hidden)
        self.drop1, self.drop2 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.W1 = nn.Linear(num_hidden + num_in, num_hidden)
        self.W2 = nn.Linear(num_hidden, num_hidden)
        self.W3 = nn.Linear(num_hidden, num_hidden)
        self.act = nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask, mask_attend=None):
        h_i = h_V.unsqueeze(-2).expand(-1, -1, h_E.size(-2), -1)
        h_EV = torch.cat([h_i, h_E], -1)
        m = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            m = mask_attend.unsqueeze(-1) * m
        h_V = self.norm1(h_V + self.drop1(m.sum(-2) / self.scale))
        h_V = self.norm2(h_V + self.drop2(self.dense(h_V)))
        if mask is not None:
            h_V = mask.unsqueeze(-1) * h_V
        return h_V


class ProteinMPNN(nn.Module):
    def __init__(self, num_letters=NUM_AA, node_features=128, edge_features=128,
                 hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
                 vocab=NUM_AA, k_neighbors=32, augment_eps=0.1, dropout=0.1):
        super().__init__()
        self.features = ProteinFeatures(edge_features=edge_features, node_features=node_features,
                                        top_k=k_neighbors, augment_eps=augment_eps)
        self.W_e = nn.Linear(edge_features, hidden_dim)
        self.W_s = nn.Embedding(vocab, hidden_dim)
        self.encoder_layers = nn.ModuleList(
            EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout) for _ in range(num_encoder_layers))
        self.decoder_layers = nn.ModuleList(
            DecLayer(hidden_dim, hidden_dim * 3, dropout=dropout) for _ in range(num_decoder_layers))
        self.W_out = nn.Linear(hidden_dim, num_letters)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        device = X.device
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros(E.shape[0], E.shape[1], E.shape[-1], device=device)
        h_E = self.W_e(E)
        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            h_V, h_E = torch.utils.checkpoint.checkpoint(layer, h_V, h_E, E_idx, mask, mask_attend)

        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
        h_EX = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_enc = cat_neighbors_nodes(h_V, h_EX, E_idx)

        chain_M = chain_M * mask
        randn = torch.randn(chain_M.shape, device=device)
        decoding_order = torch.argsort((chain_M + 1e-4) * torch.abs(randn))
        L = E_idx.shape[1]
        P = F.one_hot(decoding_order, num_classes=L).float()
        Lhat = 1.0 - torch.triu(torch.ones(L, L, device=device))
        order_mask = torch.einsum('ij,biq,bjp->bqp', Lhat, P, P)   # 1 iff p precedes q in the order
        mask_attend = torch.gather(order_mask, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view(mask.size(0), mask.size(1), 1, 1)
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1.0 - mask_attend)

        h_EXV_enc_fw = mask_fw * h_EXV_enc
        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_enc_fw
            h_V = torch.utils.checkpoint.checkpoint(layer, h_V, h_ESV, mask)
        return F.log_softmax(self.W_out(h_V), dim=-1)


def loss_smoothed(S, log_probs, mask, weight=0.1):
    S_onehot = F.one_hot(S, 21).float()
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)
    loss = -(S_onehot * log_probs).sum(-1)
    return loss, (loss * mask).sum() / 2000.0


class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer, step=0):
        self.optimizer, self.factor, self.warmup, self.model_size = optimizer, factor, warmup, model_size
        self._step = step
        self._rate = 0

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** -0.5) * min(step ** -0.5, step * self.warmup ** -1.5)

    def step(self):
        self._step += 1
        rate = self.rate()
        for g in self.optimizer.param_groups:
            g['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()


def get_std_opt(parameters, d_model, step=0):
    return NoamOpt(d_model, 2, 4000,
                   torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9), step)
```
