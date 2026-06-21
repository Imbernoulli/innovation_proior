Binding affinity prediction asks for a single scalar, `-logKd/Ki`, from the 3D structure of a protein-ligand complex. The real difficulty is not fitting the training set but generalizing to genuinely new complexes and doing so in a way a medicinal chemist can trust. Physically, binding free energy arises from non-covalent contacts across the interface — hydrogen bonds, salt bridges, van der Waals packing, pi-stacking — each between a specific ligand atom and a specific pocket atom at a specific geometry. Covalent bonds only set the conformation that places those contacting atoms, so the complex carries two relational structures that a good model should treat differently.

Existing approaches fall short in two ways. SchNet and EGNN give rigid-motion invariance by feeding only invariant geometric quantities into the network, but they treat the complex as a homogeneous graph and use only a scalar distance inside each message. That loses orientation information, so two contacts at the same distance but different local shape are indistinguishable, and covalent bonds are processed by the same convolution as non-covalent contacts. GIGN splits covalent and non-covalent edges in a heterogeneous interaction layer, but it still relies on an RBF-expanded distance in its messages and pools all node embeddings into one vector before regression. That readout is a black box: the model can fit affinity from any pooled node statistics that correlate with the label, hiding each contact's contribution.

I propose EHIGN, the Heterogeneous Interaction Graph Neural network. It is built on two inductive biases. First, the complex is a heterogeneous graph with distinct relations: covalent edges within the ligand, covalent edges within the pocket, and non-covalent contacts across the interface, each processed by its own convolution. Second, affinity is represented as a sum of pairwise atom-atom affinities over the non-covalent interface contacts. This means the output form itself reflects the physics of binding, and every contact contributes a named, inspectable scalar.

The input is already frame-independent. Atom nodes carry chemical one-hot features, and every edge carries only rigid-motion invariants. For each edge `i->j` we precompute the max, sum, and mean of neighbor angles, triangle areas, and neighbor distances, plus the direct L1 and L2 distance. Covalent edges add bond-type, conjugation, and ring-membership features. Because no raw coordinates enter the network, the entire model is translation- and rotation-invariant by construction, without augmentation.

Message passing uses two distinct convolutions. For covalent edges I use CIGConv: the projected edge feature is added to the source node feature, ReLU is applied, the messages are summed over neighbors, a residual connection keeps the atom's identity, and a small post-MLP with dropout, LeakyReLU, and batch normalization stabilizes training across three layers. Sum aggregation is appropriate for covalent graphs because degree is small and chemically meaningful. For non-covalent contacts I use NIGConv: the projected contact geometry gates the source feature elementwise, messages are mean-aggregated, and a self term plus bias completes the update. Mean aggregation matters here because contact degree varies widely; summing would let incidental weak contacts dominate purely by count. Each layer runs the four convolutions in parallel and sums the contributions per node type. Three layers give enough receptive field around the 5-angstrom interface shell without oversmoothing on small graphs.

The readout enforces the second inductive bias. After message passing, each ligand and pocket atom has a learned representation. For every non-covalent contact I compute a scalar affinity from an elementwise triple product of the projected source atom, destination atom, and contact geometry, followed by a linear map to one dimension. These per-contact scalars are summed over the complex. A raw sum carries a size-dependent offset, since larger complexes simply have more atom pairs inside the cutoff. To remove this I add an attention-based bias correction: a per-contact attention logit is computed from the source, destination, and edge features, then softmax-normalized over each complex's contacts so the weights sum to one. This weighted aggregate is mapped through a small fully-connected head and subtracted from the contact sum. The softmax makes the correction scale-stable regardless of complex size.

The scoring is computed in both directions, ligand-to-pocket and pocket-to-ligand, with separate weights. The final prediction is the average of the two corrected estimates. During training I use a three-term loss: each directional head is fit to the affinity label, and a consistency term forces the two heads to agree. At inference only the averaged prediction is returned.

The result is a model that is invariant by construction, restricts its function class to binding-consistent functions, and produces an interpretable prediction in which each contact contributes a named scalar.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CIGConv(nn.Module):
    """Covalent Interaction Graph Convolution (intra-molecular).
    Message ReLU(src + edge_feat), sum aggregation, residual, MLP."""
    def __init__(self, input_dim, output_dim, drop=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Dropout(drop),
            nn.LeakyReLU(),
            nn.BatchNorm1d(output_dim),
        )

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        msg = F.relu(x[src] + edge_attr)
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, msg)
        rst = x + agg
        return self.mlp(rst)


class NIGConv(nn.Module):
    """Non-covalent Interaction Graph Convolution (inter-molecular).
    Edge weights gate source features, mean aggregation, self term, and bias."""
    def __init__(self, in_feats, out_feats, feat_drop=0.0):
        super().__init__()
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_neigh = nn.Linear(in_feats, out_feats, bias=False)
        self.fc_self = nn.Linear(in_feats, out_feats, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_feats))
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def forward(self, x_src, x_dst, edge_index, edge_weight, num_dst):
        x_src = self.feat_drop(x_src)
        x_dst = self.feat_drop(x_dst)
        src, dst = edge_index
        msg = x_src[src] * edge_weight
        agg = torch.zeros(num_dst, msg.size(-1), device=msg.device)
        count = torch.zeros(num_dst, 1, device=msg.device)
        agg.index_add_(0, dst, msg)
        count.index_add_(0, dst, torch.ones(src.size(0), 1, device=src.device))
        h_neigh = self.fc_neigh(agg / count.clamp(min=1))
        return self.fc_self(x_dst) + h_neigh + self.bias


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
    """EHIGN: Heterogeneous Interaction Graph Network.

    Uses CIGConv for intra-molecular edges and NIGConv for inter-molecular
    contacts. Reads out as a sum of per-contact atom-atom affinities with an
    attention-normalized bias correction, averaged over ligand->pocket and
    pocket->ligand directional views.
    """
    def __init__(self, lig_dim, poc_dim, intra_edge_dim, inter_edge_dim):
        super().__init__()
        H = 256
        num_layers = 3
        self.lin_node_l = nn.Linear(lig_dim, H)
        self.lin_node_p = nn.Linear(poc_dim, H)
        self.lin_edge_ll = nn.Linear(intra_edge_dim, H)
        self.lin_edge_pp = nn.Linear(intra_edge_dim, H)
        self.lin_edge_lp = nn.Linear(inter_edge_dim, H)
        self.lin_edge_pl = nn.Linear(inter_edge_dim, H)

        self.cig_l = nn.ModuleList([CIGConv(H, H) for _ in range(num_layers)])
        self.cig_p = nn.ModuleList([CIGConv(H, H) for _ in range(num_layers)])
        self.nig_lp = nn.ModuleList([NIGConv(H, H, 0.1) for _ in range(num_layers)])
        self.nig_pl = nn.ModuleList([NIGConv(H, H, 0.1) for _ in range(num_layers)])

        # atom-atom affinity heads
        self.prj_lp_src = nn.Linear(H, H)
        self.prj_lp_dst = nn.Linear(H, H)
        self.prj_lp_edge = nn.Linear(H, H)
        self.fc_lp = nn.Linear(H, 1)
        self.prj_pl_src = nn.Linear(H, H)
        self.prj_pl_dst = nn.Linear(H, H)
        self.prj_pl_edge = nn.Linear(H, H)
        self.fc_pl = nn.Linear(H, 1)

        # bias correction (L->P)
        self.bc_lp_prj_src = nn.Linear(H, H)
        self.bc_lp_prj_dst = nn.Linear(H, H)
        self.bc_lp_prj_edge = nn.Linear(H, H)
        self.bc_lp_att = nn.Sequential(nn.PReLU(), nn.Linear(H, 1))
        self.bc_lp_w_src = nn.Linear(H, H)
        self.bc_lp_w_dst = nn.Linear(H, H)
        self.bc_lp_w_edge = nn.Linear(H, H)
        self.bc_lp_fc = FC(H, 200, 2, 0.1, 1)

        # bias correction (P->L)
        self.bc_pl_prj_src = nn.Linear(H, H)
        self.bc_pl_prj_dst = nn.Linear(H, H)
        self.bc_pl_prj_edge = nn.Linear(H, H)
        self.bc_pl_att = nn.Sequential(nn.PReLU(), nn.Linear(H, 1))
        self.bc_pl_w_src = nn.Linear(H, H)
        self.bc_pl_w_dst = nn.Linear(H, H)
        self.bc_pl_w_edge = nn.Linear(H, H)
        self.bc_pl_fc = FC(H, 200, 2, 0.1, 1)

    def _edge_softmax(self, scores, batch_idx, num_graphs):
        max_scores = torch.full((num_graphs, 1), -1e9, device=scores.device)
        max_scores.index_reduce_(0, batch_idx, scores, 'amax', include_self=True)
        exp_scores = torch.exp(scores - max_scores[batch_idx])
        sum_exp = torch.zeros(num_graphs, 1, device=scores.device)
        sum_exp.index_add_(0, batch_idx, exp_scores)
        return exp_scores / sum_exp[batch_idx].clamp(min=1e-8)

    def _forward_heads(self, batch):
        B = batch.labels.size(0)
        lig_h = self.lin_node_l(batch.lig_x)
        poc_h = self.lin_node_p(batch.poc_x)
        lig_e = self.lin_edge_ll(batch.lig_edge_attr)
        poc_e = self.lin_edge_pp(batch.poc_edge_attr)
        lp_e = self.lin_edge_lp(batch.l2p_edge_attr)
        pl_e = self.lin_edge_pl(batch.p2l_edge_attr)

        for i in range(len(self.cig_l)):
            lig_in, poc_in = lig_h, poc_h
            lig_intra = self.cig_l[i](lig_in, batch.lig_edge_index, lig_e)
            poc_intra = self.cig_p[i](poc_in, batch.poc_edge_index, poc_e)
            lig_inter = torch.zeros_like(lig_in)
            poc_inter = torch.zeros_like(poc_in)
            if batch.l2p_edge_index.size(1) > 0:
                poc_inter = self.nig_lp[i](lig_in, poc_in, batch.l2p_edge_index, lp_e, poc_in.size(0))
            if batch.p2l_edge_index.size(1) > 0:
                lig_inter = self.nig_pl[i](poc_in, lig_in, batch.p2l_edge_index, pl_e, lig_in.size(0))
            lig_h = lig_intra + lig_inter
            poc_h = poc_intra + poc_inter

        l2p_src, l2p_dst = batch.l2p_edge_index
        i_lp = self.prj_lp_edge(lp_e) * self.prj_lp_src(lig_h)[l2p_src] * self.prj_lp_dst(poc_h)[l2p_dst]
        logit_lp = self.fc_lp(i_lp)
        pred_lp = torch.zeros(B, 1, device=logit_lp.device)
        pred_lp.index_add_(0, batch.inter_batch, logit_lp)

        p2l_src, p2l_dst = batch.p2l_edge_index
        p2l_batch = batch.lig_batch[p2l_dst]
        i_pl = self.prj_pl_edge(pl_e) * self.prj_pl_dst(poc_h)[p2l_src] * self.prj_pl_src(lig_h)[p2l_dst]
        logit_pl = self.fc_pl(i_pl)
        pred_pl = torch.zeros(B, 1, device=logit_pl.device)
        pred_pl.index_add_(0, p2l_batch, logit_pl)

        w_lp = self.bc_lp_prj_src(lig_h)[l2p_src] + self.bc_lp_prj_dst(poc_h)[l2p_dst] + self.bc_lp_prj_edge(lp_e)
        a_lp = self._edge_softmax(self.bc_lp_att(w_lp), batch.inter_batch, B)
        s_lp = a_lp * self.bc_lp_w_edge(lp_e) * self.bc_lp_w_src(lig_h)[l2p_src] * self.bc_lp_w_dst(poc_h)[l2p_dst]
        bias_lp_agg = torch.zeros(B, s_lp.size(-1), device=s_lp.device)
        bias_lp_agg.index_add_(0, batch.inter_batch, s_lp)
        bias_lp = self.bc_lp_fc(bias_lp_agg)

        w_pl = self.bc_pl_prj_dst(poc_h)[p2l_src] + self.bc_pl_prj_src(lig_h)[p2l_dst] + self.bc_pl_prj_edge(pl_e)
        a_pl = self._edge_softmax(self.bc_pl_att(w_pl), p2l_batch, B)
        s_pl = a_pl * self.bc_pl_w_edge(pl_e) * self.bc_pl_w_dst(poc_h)[p2l_src] * self.bc_pl_w_src(lig_h)[p2l_dst]
        bias_pl_agg = torch.zeros(B, s_pl.size(-1), device=s_pl.device)
        bias_pl_agg.index_add_(0, p2l_batch, s_pl)
        bias_pl = self.bc_pl_fc(bias_pl_agg)

        return (pred_lp - bias_lp).squeeze(-1), (pred_pl - bias_pl).squeeze(-1)

    def forward(self, batch):
        pred_lp, pred_pl = self._forward_heads(batch)
        return (pred_lp + pred_pl) / 2

    def compute_loss(self, batch):
        pred_lp, pred_pl = self._forward_heads(batch)
        labels = batch.labels
        return (F.mse_loss(pred_lp, labels) + F.mse_loss(pred_pl, labels)
                + F.mse_loss(pred_lp, pred_pl)) / 3
```
