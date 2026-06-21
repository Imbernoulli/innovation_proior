The architecture is the whole contribution on this task — graph construction, featurization, splits, optimizer, and loss are frozen — so before reaching for anything clever I want a principled floor that takes the binding problem seriously, something the later rungs can beat and diagnose against. The scaffold default is not that floor: it runs two independent encoders over the ligand and the pocket, mean-pools each, concatenates, and regresses, and it *ignores the inter-molecular edges entirely*. That is physically a non-starter. A binding affinity is a free energy, and the free energy of binding lives almost entirely in the non-covalent contacts *across the interface* — hydrogen bonds, salt bridges, van der Waals packing, $\pi$-stacking — each between a particular ligand atom and a particular pocket atom. The covalent skeletons inside each molecule exist whether or not the ligand is bound; they set conformation but they are not the binding. A model that encodes the two molecules separately and never lets a ligand atom see the pocket atom it touches has thrown away the one thing that sets the answer, and can only guess $-\log K_d/K_i$ from the marginal shapes of the two molecules — exactly the dataset shortcut to distrust. My floor must at least let the interface edges into the message passing.

There is a second non-negotiable, and it is physics, not taste. The affinity is a scalar property of the *arrangement* of atoms, so it cannot change under a rigid motion: send every position $r \mapsto Q r + t$ and the answer must be byte-for-byte identical. I refuse to learn that invariance from augmentation. What survives a rigid motion is the interatomic *distance*: a raw coordinate scrambles, a difference $(Q r_i + t) - (Q r_j + t) = Q(r_i - r_j)$ still rotates, but its length is fixed, since $\|Q(r_i - r_j)\| = \sqrt{(r_i - r_j)^\top Q^\top Q (r_i - r_j)} = \|r_i - r_j\|$ because $Q^\top Q = I$. Build everything out of distances and the predictor is invariant by construction; let one raw coordinate or bare difference vector leak in and it breaks. The harness makes this easy and also constrains it: the model is not handed coordinates at all, only a `PLABatch` whose edges carry precomputed rigid-motion-invariant geometry. The last geometric channel of each edge is the L2 distance scaled by $0.1$, so a layer that wants a scalar distance in angstroms reads `edge_attr[:, -1:] * 10`. That single channel is the only handle on geometry, and it is enough to build a distance-driven message.

I propose **SchNet** — Schütt et al.'s continuous-filter convolution (2017) — slotted into a heterogeneous interface skeleton. The defining move is to make the message depend on distance through a learned radial filter rather than feeding the distance into an MLP raw. First I expand the scalar distance $d$ in a bank of Gaussian radial basis functions on a grid of centers $\mu_k$,

$$\text{RBF}(d)_k = \exp\!\Big(-\tfrac{1}{2}\big((d - \mu_k)/\sigma\big)^2\Big),$$

with centers laid from $0$ to $6$ Å at $0.1$ Å spacing (so $60$ channels) and width $\sigma$ equal to the gap. The reason for the expansion is concrete: a fresh MLP is nearly linear at initialization, so feeding it one scalar makes every output channel come out as the same near-linear ramp in $d$ — the filter channels are correlated, there is no diversity, and training stalls on a plateau because the filter is effectively one-dimensional. A Gaussian bank fixes this by decorrelating the input: a short distance lights up the near centers, a long distance the far ones, so the filter starts diverse. The grid resolves a bond length ($\sim 1.5$ Å) from a contact reaching to the $5$ Å cutoff, and $6$ Å of range covers the whole contact shell. The RBF is then mapped to a per-edge filter $W = \text{filter\_net}(\text{RBF}(d))$ by a two-layer MLP with a Softplus in the middle, and the message into a destination atom is the projected neighbour gated *elementwise* by that filter, summed over neighbours, with the destination's own feature added back as a residual and a Softplus output MLP:

$$h_{\text{dst}} \leftarrow h_{\text{dst}} + \text{output}\!\Big(\sum_{s \to \text{dst}} \text{node\_proj}(h_s) \odot W_{s\to\text{dst}}\Big).$$

Softplus rather than ReLU is the canonical choice for this filter family — it is the smooth cousin of ReLU, and a response to a continuous physical distance ought to be smooth, with no kinks. Because only the invariant distance enters, every block is invariant and the whole predictor is invariant by construction, with zero augmentation.

The original continuous-filter convolution was built for a *single homogeneous molecule* where every edge is the same kind, processed by one shared filter. My complex is not homogeneous: it is two molecules joined by an interface, with two physically distinct edge types — stiff covalent bonds around a bond length apart, and soft non-covalent contacts reaching to $5$ Å. Forcing both through one filter asks a single distance-to-filter map to model two regimes at once, which was the homogeneous net's whole limitation. So I keep one node set per molecule but run *four* separate continuous-filter convolutions per layer — covalent on the ligand, covalent on the pocket, non-covalent ligand$\to$pocket, non-covalent pocket$\to$ligand — each with its own filter map, all computed in parallel from the same input features, then summed per destination node type (the HeteroGraphConv pattern). A pocket atom receives its own covalent update plus the non-covalent update from ligand contacts pointing into it; a ligand atom its covalent update plus the pocket contacts. Sum, because covalent and non-covalent influences on an atom are additive, and sharing the input features cleanly separates "what my own molecule tells me" from "what my binding partner tells me." Three layers — the contact graph is a local $5$ Å shell, and three rounds carry a protein atom's influence a few bonds into the ligand and back without oversmoothing the complex into mush. Hidden width $256$.

For the readout I deliberately do not pool-and-regress; I read the affinity off the interface, which is the part that makes this floor meaningful. For each non-covalent contact I score a per-contact affinity from a low-rank triple product of the projected source atom, the projected destination atom, and the projected RBF edge geometry, $e_{\text{proj}} \odot \text{src}_{\text{proj}} \odot \text{dst}_{\text{proj}}$, collapsed to a scalar by a final linear, and I sum those scalars over a complex's contacts in both directions. A raw sum over a variable number of contacts carries a size-dependent offset — bigger complexes simply have more contacts in the shell — so per direction I subtract an attention-weighted bias correction whose softmax over a complex's contacts normalizes that offset away, then average the two directional estimates. This is the shared readout the whole ladder uses; here it sits on continuous-filter message passing and the edge geometry it scores is the same RBF expansion. The SchNet fill predicts a single `forward` output, so I keep the harness's plain MSE loss and expose no `compute_loss` hook.

One honest scoping note: this is the canonical continuous-filter *geometric core* — RBF, Softplus filter, elementwise gating — but not SchNet end-to-end. There are no forces to train, since the target is a single scalar and not an energy whose gradient is a force, so the second-derivative machinery and energy-plus-force loss are irrelevant. And I drop the cosine cutoff: contacts are already capped at $5$ Å by graph construction and the RBF bank only reaches $6$ Å, so no neighbour drifts across a boundary mid-training. What survives is the move that matters — distance $\to$ Gaussian expansion $\to$ learned filter $\to$ elementwise gate — slotted into the heterogeneous interface skeleton.

I expect this to be a competent but *under-informed* predictor, and that expectation is the whole point of running it. Its only window onto geometry is the scalar distance through the RBF; the $11$ geometric numbers per edge — angles, triangle areas, neighbour-distance statistics — never enter the message. Two contacts at the same distance but different orientation are indistinguishable to it, and binding cares about orientation. So I expect SchNet to beat the interface-blind default comfortably yet land as the weakest geometric fill — worst on the harder benchmarks where held-out chemistry rewards richer geometry — and the next rung's gain to come precisely from letting more than the scalar distance into the message.

```python
# EDITABLE SECTION START — SchNet: RBF Distance-based Heterogeneous GNN

class RBFExpansion(nn.Module):
    """Radial basis function expansion of distances."""
    def __init__(self, low=0.0, high=6.0, gap=0.1):
        super().__init__()
        centers = torch.arange(low, high, gap)
        self.register_buffer('centers', centers)
        self.register_buffer('width', torch.tensor(gap))

    @property
    def num_features(self):
        return self.centers.size(0)

    def forward(self, dist):
        return torch.exp(-0.5 * ((dist - self.centers) / self.width) ** 2)


class CFConv(nn.Module):
    """Continuous-filter convolution (SchNet interaction block).
    filter_net(rbf) * node_proj(src), sum aggregation, residual, output MLP.
    """
    def __init__(self, node_dim, rbf_dim, hidden_dim):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(rbf_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x_src, x_dst, edge_index, rbf_feat, num_dst):
        src, dst = edge_index
        W = self.filter_net(rbf_feat)
        msg = self.node_proj(x_src[src]) * W
        agg = torch.zeros(num_dst, msg.size(-1), device=msg.device)
        agg.index_add_(0, dst, msg)
        return x_dst + self.output(agg)


class FC(nn.Module):
    """Fully connected prediction head."""
    def __init__(self, d_in, d_hidden, n_layers, dropout, n_out):
        super().__init__()
        layers = []
        for j in range(n_layers):
            if j == 0:
                layers += [nn.Linear(d_in, d_hidden), nn.Dropout(dropout),
                           nn.LeakyReLU(), nn.BatchNorm1d(d_hidden)]
            if j == n_layers - 1:
                layers.append(nn.Linear(d_hidden, n_out))
            else:
                layers += [nn.Linear(d_hidden, d_hidden), nn.Dropout(dropout),
                           nn.LeakyReLU(), nn.BatchNorm1d(d_hidden)]
        self.layers = nn.ModuleList(layers)

    def forward(self, h):
        for layer in self.layers:
            h = layer(h)
        return h


class AffinityModel(nn.Module):
    """SchNet-based heterogeneous GNN for binding affinity.

    Uses RBF distance expansion and continuous-filter convolution for all edge types.
    HeteroGraphConv pattern: parallel compute, sum aggregate per node type.
    Dual bidirectional prediction with attention-based bias correction.
    """
    def __init__(self, lig_dim, poc_dim, intra_edge_dim, inter_edge_dim):
        super().__init__()
        H = 256
        num_layers = 3
        self.rbf = RBFExpansion(high=6.0, gap=0.1)
        rbf_dim = self.rbf.num_features

        self.lin_node_l = nn.Linear(lig_dim, H)
        self.lin_node_p = nn.Linear(poc_dim, H)

        self.cf_l = nn.ModuleList([CFConv(H, rbf_dim, H) for _ in range(num_layers)])
        self.cf_p = nn.ModuleList([CFConv(H, rbf_dim, H) for _ in range(num_layers)])
        self.cf_lp = nn.ModuleList([CFConv(H, rbf_dim, H) for _ in range(num_layers)])
        self.cf_pl = nn.ModuleList([CFConv(H, rbf_dim, H) for _ in range(num_layers)])

        # Readout via inter-molecular interaction scoring
        self.prj_lp_src = nn.Linear(H, H)
        self.prj_lp_dst = nn.Linear(H, H)
        self.prj_lp_edge = nn.Linear(rbf_dim, H)
        self.fc_lp = nn.Linear(H, 1)
        self.prj_pl_src = nn.Linear(H, H)
        self.prj_pl_dst = nn.Linear(H, H)
        self.prj_pl_edge = nn.Linear(rbf_dim, H)
        self.fc_pl = nn.Linear(H, 1)

        # Bias correction (L->P) with attention
        self.bc_lp_prj_src = nn.Linear(H, H)
        self.bc_lp_prj_dst = nn.Linear(H, H)
        self.bc_lp_prj_edge = nn.Linear(rbf_dim, H)
        self.bc_lp_att = nn.Sequential(nn.PReLU(), nn.Linear(H, 1))
        self.bc_lp_w_src = nn.Linear(H, H)
        self.bc_lp_w_dst = nn.Linear(H, H)
        self.bc_lp_w_edge = nn.Linear(rbf_dim, H)
        self.bc_lp_fc = FC(H, 200, 2, 0.1, 1)

        # Bias correction (P->L) with attention
        self.bc_pl_prj_src = nn.Linear(H, H)
        self.bc_pl_prj_dst = nn.Linear(H, H)
        self.bc_pl_prj_edge = nn.Linear(rbf_dim, H)
        self.bc_pl_att = nn.Sequential(nn.PReLU(), nn.Linear(H, 1))
        self.bc_pl_w_src = nn.Linear(H, H)
        self.bc_pl_w_dst = nn.Linear(H, H)
        self.bc_pl_w_edge = nn.Linear(rbf_dim, H)
        self.bc_pl_fc = FC(H, 200, 2, 0.1, 1)

    def _get_rbf(self, edge_attr):
        dist = edge_attr[:, -1:] * 10
        return self.rbf(dist)

    def _edge_softmax(self, scores, batch_idx, num_graphs):
        max_scores = torch.zeros(num_graphs, 1, device=scores.device).fill_(-1e9)
        max_scores.index_reduce_(0, batch_idx, scores, 'amax', include_self=True)
        exp_scores = torch.exp(scores - max_scores[batch_idx])
        sum_exp = torch.zeros(num_graphs, 1, device=scores.device)
        sum_exp.index_add_(0, batch_idx, exp_scores)
        return exp_scores / sum_exp[batch_idx].clamp(min=1e-8)

    def forward(self, batch: PLABatch) -> torch.Tensor:
        B = batch.labels.size(0)
        lig_h = self.lin_node_l(batch.lig_x)
        poc_h = self.lin_node_p(batch.poc_x)

        lig_rbf = self._get_rbf(batch.lig_edge_attr)
        poc_rbf = self._get_rbf(batch.poc_edge_attr)
        lp_rbf = self._get_rbf(batch.l2p_edge_attr) if batch.l2p_edge_attr.size(0) > 0 else None
        pl_rbf = self._get_rbf(batch.p2l_edge_attr) if batch.p2l_edge_attr.size(0) > 0 else None

        # HeteroGraphConv pattern: parallel compute, sum aggregate
        for i in range(len(self.cf_l)):
            lig_in, poc_in = lig_h, poc_h

            lig_intra = self.cf_l[i](lig_in, lig_in, batch.lig_edge_index, lig_rbf, lig_in.size(0))
            poc_intra = self.cf_p[i](poc_in, poc_in, batch.poc_edge_index, poc_rbf, poc_in.size(0))

            lig_inter = torch.zeros_like(lig_in)
            poc_inter = torch.zeros_like(poc_in)
            if lp_rbf is not None and batch.l2p_edge_index.size(1) > 0:
                poc_inter = self.cf_lp[i](lig_in, poc_in, batch.l2p_edge_index, lp_rbf, poc_in.size(0))
            if pl_rbf is not None and batch.p2l_edge_index.size(1) > 0:
                lig_inter = self.cf_pl[i](poc_in, lig_in, batch.p2l_edge_index, pl_rbf, lig_in.size(0))

            lig_h = lig_intra + lig_inter
            poc_h = poc_intra + poc_inter

        # Scoring (L->P)
        l2p_src, l2p_dst = batch.l2p_edge_index
        i_lp = self.prj_lp_edge(lp_rbf) * self.prj_lp_src(lig_h)[l2p_src] * self.prj_lp_dst(poc_h)[l2p_dst]
        logit_lp = self.fc_lp(i_lp)
        pred_lp = torch.zeros(B, 1, device=logit_lp.device)
        pred_lp.index_add_(0, batch.inter_batch, logit_lp)

        # Scoring (P->L)
        p2l_src, p2l_dst = batch.p2l_edge_index
        p2l_batch = batch.lig_batch[p2l_dst]
        i_pl = self.prj_pl_edge(pl_rbf) * self.prj_pl_src(poc_h)[p2l_src] * self.prj_pl_dst(lig_h)[p2l_dst]
        logit_pl = self.fc_pl(i_pl)
        pred_pl = torch.zeros(B, 1, device=logit_pl.device)
        pred_pl.index_add_(0, p2l_batch, logit_pl)

        # Bias correction (L->P) with attention
        w_lp = self.bc_lp_prj_src(lig_h)[l2p_src] + self.bc_lp_prj_dst(poc_h)[l2p_dst] + self.bc_lp_prj_edge(lp_rbf)
        a_lp = self._edge_softmax(self.bc_lp_att(w_lp), batch.inter_batch, B)
        s_lp = a_lp * self.bc_lp_w_edge(lp_rbf) * self.bc_lp_w_src(lig_h)[l2p_src] * self.bc_lp_w_dst(poc_h)[l2p_dst]
        bias_lp_agg = torch.zeros(B, s_lp.size(-1), device=s_lp.device)
        bias_lp_agg.index_add_(0, batch.inter_batch, s_lp)
        bias_lp = self.bc_lp_fc(bias_lp_agg)

        # Bias correction (P->L) with attention
        w_pl = self.bc_pl_prj_src(poc_h)[p2l_src] + self.bc_pl_prj_dst(lig_h)[p2l_dst] + self.bc_pl_prj_edge(pl_rbf)
        a_pl = self._edge_softmax(self.bc_pl_att(w_pl), p2l_batch, B)
        s_pl = a_pl * self.bc_pl_w_edge(pl_rbf) * self.bc_pl_w_src(poc_h)[p2l_src] * self.bc_pl_w_dst(lig_h)[p2l_dst]
        bias_pl_agg = torch.zeros(B, s_pl.size(-1), device=s_pl.device)
        bias_pl_agg.index_add_(0, p2l_batch, s_pl)
        bias_pl = self.bc_pl_fc(bias_pl_agg)

        pred = ((pred_lp - bias_lp) + (pred_pl - bias_pl)) / 2
        return pred.squeeze(-1)

# EDITABLE SECTION END
```
