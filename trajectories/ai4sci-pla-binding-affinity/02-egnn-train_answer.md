SchNet came in exactly as the under-informed floor I expected: RMSE $1.4765$ / Rp $0.792$ on the 2013 core set, $1.3702$ / $0.777$ on 2016, and on the 2019 temporal holdout — the largest test set and the one whose chemistry is most removed from training — its correlation collapsed to Rp $0.570$ at RMSE $1.5624$, the worst it posts anywhere. That last number is the tell. The distance-only continuous filter generalizes *least* exactly where held-out structures demand the most from the geometry. On near-training complexes the model can lean on memorized motifs to post a respectable $0.777$, but on the temporally distant set, where it must actually reason about contact geometry it has not seen, it has nothing extra to reason with. The bottleneck is not the skeleton — the heterogeneous four-convolution structure and the dual interface readout are doing their job — it is what the message is *allowed to depend on*. SchNet's filter is a function of $d$ alone, so two contacts at the same distance but different orientation are identical to it, and binding is exquisitely orientation-dependent. I need to widen what a message can condition on.

I propose **EGNN** — Satorras, Hoogeboom & Welling's E($n$)-equivariant message passing (2021) — but, deliberately, only its *invariant* half. Re-deriving from the symmetry is what makes it both safe and exactly the right size for this edit surface. The affinity is an invariant scalar, so I want every message invariant: rotate or translate the complex and the message is unchanged. Under $r \mapsto Q r + t$, a raw coordinate scrambles and a difference $r_i - r_j \mapsto Q(r_i - r_j)$ still rotates, but the *squared* distance $\|r_i - r_j\|^2$ is fixed, since $Q^\top Q = I$ and $t$ cancels in the difference. So the canonical equivariant edge function feeds itself that invariant squared distance alongside the node features,

$$m_{ij} = \phi_e\big(h_i,\, h_j,\, \|r_i - r_j\|^2,\, a_{ij}\big).$$

Where this differs from SchNet, and where its extra power lives, is that the message is a full MLP of *both* endpoint feature vectors and the distance — not a distance-derived filter multiplying one neighbour. Two contacts at the same distance can now produce different messages because the *atoms* differ, and the MLP can carve the distance dependence per atom-pair type instead of through one shared radial filter. That is the widening I want.

The famous second half of the equivariant layer is the coordinate update — move each point along a weighted sum of relative-difference vectors, $x_i \leftarrow x_i + C \sum_j (x_i - x_j)\,\phi_x(m_{ij})$, where the weight $\phi_x(m_{ij})$ is an invariant scalar, so $Q$ factors out of the differences and the update is equivariant. That update is what lets the method emit a *vector* target, but I do not need it here, for two clean reasons. First, my target is a single invariant scalar, $-\log K_d/K_i$; there is no vector to emit, so the equivariant coordinate channel has nothing to do — and this is no loss, because for a fixed node indexing the pairwise distance matrix already determines the geometry up to a rigid motion, so for an invariant target the distances carry all the geometric information the difference *vectors* would. The coordinate update buys expressiveness only when the *output* must be equivariant. Second, and decisively, the harness hands me no coordinates: the `PLABatch` geometry is precomputed into invariant edge features, there is no `pos` to update and no relative-difference vector to form. The coordinate update is not merely unnecessary here — it is *unimplementable*. I keep the half I can use and that the target wants: the invariant equivariant-style message, summed into the node update.

Concretely, I read the distance the only way the harness exposes it, `edge_attr[:, -1:] * 10`, a $1$-dim scalar in angstroms, and that single scalar is the geometric input to every message. I structure the edge function as three additive SiLU-MLP terms so the source atom, the destination atom, and the edge distance each get their own learned transform before they combine,

$$m = \text{mlp}_u(x_{\text{src}}) + \text{mlp}_v(x_{\text{dst}}) + \text{mlp}_e(d),$$

summed over neighbours, then a node MLP on the concatenation of the destination's own feature and the aggregate, $\text{node\_mlp}([x_{\text{dst}}, \text{agg}])$. SiLU throughout because, like SchNet's Softplus, it is the smooth activation the equivariant layer uses on its invariant channels — and here every channel is invariant, since I never touch a coordinate, so no equivariance is endangered by a pointwise nonlinearity. This is strictly richer than SchNet's message on the node side: SchNet gated one projected neighbour by a distance filter, whereas here both endpoints *and* the distance pass through their own MLPs and add, so the message can express "this kind of ligand atom meeting that kind of pocket atom at this separation" in a way one radial filter cannot.

Everything around the message stays the heterogeneous interface skeleton, since SchNet already showed the skeleton is not the bottleneck — I swap only the convolution. Four EGNN convolutions per layer (covalent-ligand, covalent-pocket, non-covalent ligand$\to$pocket, non-covalent pocket$\to$ligand), computed in parallel from the same input features and summed per destination node type: a pocket atom gets its covalent update plus the non-covalent update from ligand contacts pointing into it, a ligand atom its covalent update plus the pocket contacts. Three layers, hidden width $256$, the non-covalent edges feeding the same $1$-dim distance. The readout is the shared dual bidirectional interface scorer — a per-contact triple product of projected source atom, projected destination atom, and a projection of the contact distance, summed over a complex's contacts in both directions, each corrected by an attention-normalized bias term whose softmax over the complex's contacts kills the size-dependent offset of a raw sum, then averaged. As with SchNet this fill produces one prediction, so I keep the harness's plain MSE on the single `forward` output and expose no `compute_loss` hook.

I want to be honest about the one place EGNN here is *not* richer than SchNet, because it sets up the next rung. EGNN's geometric input is still only the scalar distance — the same number SchNet used, just consumed through additive endpoint MLPs instead of a radial filter. The extra expressiveness is all on the node side; the geometry side is no richer, and certainly no richer than the full $11$-dim edge feature sitting unused in the batch. SchNet expanded that one distance into $60$ RBF channels; EGNN feeds it raw as a single channel through $\text{mlp}_e$. So on pure geometric resolution EGNN may be *thinner* than SchNet — it trades distance-resolution for node-pair expressiveness. I expect that trade to pay where the chemistry of the atom pair carries signal a distance filter misses, which should show up first on the temporal holdout: I expect EGNN to lift 2019 the most, its Rp clearing $0.60$ and its RMSE dropping below SchNet's $1.5624$, with a clear gain on 2013 too. The risk is 2016: SchNet's $60$-channel RBF gave fine distance resolution on the near-training core set where memorized motifs pay off, and EGNN's single raw channel is coarser there, so EGNN may actually *lose* on 2016 RMSE even while winning overall. If that split appears — ahead on 2013 and 2019, behind on 2016 RMSE — it is not noise; it is the precise statement that the next rung must stop discarding the $11$-dim edge geometry and feed the full angle/area/distance description into the message, recovering SchNet's resolution while keeping EGNN's node-pair expressiveness.

```python
# EDITABLE SECTION START — EGNN: Equivariant Graph Neural Network

class EGNNConv(nn.Module):
    """E(n)-equivariant message passing layer using distance as edge feature.
    Message: mlp_u(src) + mlp_v(dst) + mlp_e(dist), sum aggregation,
    then node_mlp(cat[dst, agg]).
    """
    def __init__(self, input_dim, hidden_dim, edge_dim=1):
        super().__init__()
        self.edge_mlp_u = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.edge_mlp_v = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.edge_mlp_e = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x_src, x_dst, edge_index, edge_feat, num_dst):
        src, dst = edge_index
        msg = self.edge_mlp_u(x_src[src]) + self.edge_mlp_v(x_dst[dst]) + self.edge_mlp_e(edge_feat)
        agg = torch.zeros(num_dst, msg.size(-1), device=msg.device)
        agg.index_add_(0, dst, msg)
        return self.node_mlp(torch.cat([x_dst, agg], dim=-1))


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
    """EGNN-based heterogeneous model for binding affinity.

    Uses E(n)-equivariant message passing with distance as scalar edge feature.
    HeteroGraphConv pattern: parallel compute, sum aggregate per node type.
    Dual bidirectional prediction with attention-based bias correction.
    """
    def __init__(self, lig_dim, poc_dim, intra_edge_dim, inter_edge_dim):
        super().__init__()
        H = 256
        num_layers = 3

        self.lin_node_l = nn.Linear(lig_dim, H)
        self.lin_node_p = nn.Linear(poc_dim, H)

        # EGNN layers for all 4 edge types (using distance as 1-dim edge feat)
        self.egnn_l = nn.ModuleList([EGNNConv(H, H, edge_dim=1) for _ in range(num_layers)])
        self.egnn_p = nn.ModuleList([EGNNConv(H, H, edge_dim=1) for _ in range(num_layers)])
        self.egnn_lp = nn.ModuleList([EGNNConv(H, H, edge_dim=1) for _ in range(num_layers)])
        self.egnn_pl = nn.ModuleList([EGNNConv(H, H, edge_dim=1) for _ in range(num_layers)])

        # Interaction scoring (with 1-dim distance edge features)
        self.prj_lp_src = nn.Linear(H, H)
        self.prj_lp_dst = nn.Linear(H, H)
        self.prj_lp_edge = nn.Linear(1, H)
        self.fc_lp = nn.Linear(H, 1)
        self.prj_pl_src = nn.Linear(H, H)
        self.prj_pl_dst = nn.Linear(H, H)
        self.prj_pl_edge = nn.Linear(1, H)
        self.fc_pl = nn.Linear(H, 1)

        # Bias correction (L->P)
        self.bc_lp_prj_src = nn.Linear(H, H)
        self.bc_lp_prj_dst = nn.Linear(H, H)
        self.bc_lp_prj_edge = nn.Linear(1, H)
        self.bc_lp_att = nn.Sequential(nn.PReLU(), nn.Linear(H, 1))
        self.bc_lp_w_src = nn.Linear(H, H)
        self.bc_lp_w_dst = nn.Linear(H, H)
        self.bc_lp_w_edge = nn.Linear(1, H)
        self.bc_lp_fc = FC(H, 200, 2, 0.1, 1)

        # Bias correction (P->L)
        self.bc_pl_prj_src = nn.Linear(H, H)
        self.bc_pl_prj_dst = nn.Linear(H, H)
        self.bc_pl_prj_edge = nn.Linear(1, H)
        self.bc_pl_att = nn.Sequential(nn.PReLU(), nn.Linear(H, 1))
        self.bc_pl_w_src = nn.Linear(H, H)
        self.bc_pl_w_dst = nn.Linear(H, H)
        self.bc_pl_w_edge = nn.Linear(1, H)
        self.bc_pl_fc = FC(H, 200, 2, 0.1, 1)

    def _get_dist(self, edge_attr):
        # Last dim is L2 distance * 0.1, rescale to angstroms
        return edge_attr[:, -1:] * 10

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

        lig_dist = self._get_dist(batch.lig_edge_attr)
        poc_dist = self._get_dist(batch.poc_edge_attr)
        lp_dist = self._get_dist(batch.l2p_edge_attr) if batch.l2p_edge_attr.size(0) > 0 else None
        pl_dist = self._get_dist(batch.p2l_edge_attr) if batch.p2l_edge_attr.size(0) > 0 else None

        # HeteroGraphConv pattern: parallel compute, sum aggregate
        for i in range(len(self.egnn_l)):
            lig_in, poc_in = lig_h, poc_h

            lig_intra = self.egnn_l[i](lig_in, lig_in, batch.lig_edge_index, lig_dist, lig_in.size(0))
            poc_intra = self.egnn_p[i](poc_in, poc_in, batch.poc_edge_index, poc_dist, poc_in.size(0))

            lig_inter = torch.zeros_like(lig_in)
            poc_inter = torch.zeros_like(poc_in)
            if lp_dist is not None and batch.l2p_edge_index.size(1) > 0:
                poc_inter = self.egnn_lp[i](lig_in, poc_in, batch.l2p_edge_index, lp_dist, poc_in.size(0))
            if pl_dist is not None and batch.p2l_edge_index.size(1) > 0:
                lig_inter = self.egnn_pl[i](poc_in, lig_in, batch.p2l_edge_index, pl_dist, lig_in.size(0))

            lig_h = lig_intra + lig_inter
            poc_h = poc_intra + poc_inter

        # Atom-atom affinities (L->P) with edge features
        l2p_src, l2p_dst = batch.l2p_edge_index
        i_lp = self.prj_lp_edge(lp_dist) * self.prj_lp_src(lig_h)[l2p_src] * self.prj_lp_dst(poc_h)[l2p_dst]
        logit_lp = self.fc_lp(i_lp)
        pred_lp = torch.zeros(B, 1, device=logit_lp.device)
        pred_lp.index_add_(0, batch.inter_batch, logit_lp)

        # Atom-atom affinities (P->L) with edge features
        p2l_src, p2l_dst = batch.p2l_edge_index
        p2l_batch = batch.lig_batch[p2l_dst]
        i_pl = self.prj_pl_edge(pl_dist) * self.prj_pl_src(poc_h)[p2l_src] * self.prj_pl_dst(lig_h)[p2l_dst]
        logit_pl = self.fc_pl(i_pl)
        pred_pl = torch.zeros(B, 1, device=logit_pl.device)
        pred_pl.index_add_(0, p2l_batch, logit_pl)

        # Bias correction (L->P) with attention
        w_lp = self.bc_lp_prj_src(lig_h)[l2p_src] + self.bc_lp_prj_dst(poc_h)[l2p_dst] + self.bc_lp_prj_edge(lp_dist)
        a_lp = self._edge_softmax(self.bc_lp_att(w_lp), batch.inter_batch, B)
        s_lp = a_lp * self.bc_lp_w_edge(lp_dist) * self.bc_lp_w_src(lig_h)[l2p_src] * self.bc_lp_w_dst(poc_h)[l2p_dst]
        bias_lp_agg = torch.zeros(B, s_lp.size(-1), device=s_lp.device)
        bias_lp_agg.index_add_(0, batch.inter_batch, s_lp)
        bias_lp = self.bc_lp_fc(bias_lp_agg)

        # Bias correction (P->L) with attention
        w_pl = self.bc_pl_prj_src(poc_h)[p2l_src] + self.bc_pl_prj_dst(lig_h)[p2l_dst] + self.bc_pl_prj_edge(pl_dist)
        a_pl = self._edge_softmax(self.bc_pl_att(w_pl), p2l_batch, B)
        s_pl = a_pl * self.bc_pl_w_edge(pl_dist) * self.bc_pl_w_src(poc_h)[p2l_src] * self.bc_pl_w_dst(lig_h)[p2l_dst]
        bias_pl_agg = torch.zeros(B, s_pl.size(-1), device=s_pl.device)
        bias_pl_agg.index_add_(0, p2l_batch, s_pl)
        bias_pl = self.bc_pl_fc(bias_pl_agg)

        pred = ((pred_lp - bias_lp) + (pred_pl - bias_pl)) / 2
        return pred.squeeze(-1)

# EDITABLE SECTION END
```
