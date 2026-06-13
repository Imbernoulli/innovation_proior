EHIGN (Heterogeneous Interaction Graph Neural network) predicts protein-ligand binding affinity
`-logKd/Ki` from a 3D complex by encoding an *interaction-based inductive bias*: it represents the
complex as a heterogeneous graph with covalent (intra-molecular) and non-covalent (inter-molecular)
edges, and produces its prediction as a *sum of pairwise atom-atom affinities* over the non-covalent
interface contacts. Distinct convolutions handle the two interaction types; the interface is scored
from both directions (ligand->pocket and pocket->ligand); each directional sum is corrected by an
attention-pooled bias term; and the two views are averaged and trained to agree.

## Problem it solves

Structure-based binding-affinity prediction with an architecture whose function class is restricted
to binding-relevant functions, for better generalization and built-in interpretability. The input
is a protein-ligand complex (ligand atoms + pocket atoms within a 5 angstrom shell); the output is a
scalar affinity per complex.

## Key ideas

1. **Heterogeneous graph (assumption 1).** Two node types (ligand, pocket); four edge relations —
   covalent ligand, covalent pocket, non-covalent ligand->pocket, non-covalent pocket->ligand —
   each with its own convolution. Covalent bonds set conformation; non-covalent contacts carry the
   binding signal, so they are processed differently rather than by one shared convolution.

2. **Invariance for free.** Inputs are rigid-motion invariants only — atom chemistry plus, per edge,
   eleven geometric numbers: max/sum/mean of (angle, triangle area, neighbor distance) over the
   edge's neighbors, plus the L1 and L2 endpoint distance (max/mean/endpoints scaled by 0.1 and
   sums by 0.01 for stable magnitudes).
   Covalent edges prepend six bond-chemistry features (17-dim total); non-covalent edges carry the
   eleven geometric numbers alone. No coordinates enter, so the network is translation/rotation
   invariant by construction — no augmentation.

3. **Affinity = sum of pairwise atom-atom affinities (assumption 2).** Instead of pooling node
   embeddings, each interface contact gets a scalar affinity from a low-rank triple product of the
   projected source atom, destination atom, and contact-geometry feature, collapsed by a linear
   layer; the per-contact scalars are summed over the complex. The output's *form* encodes the
   physics, so each contact's contribution is inspectable.

4. **Bias correction.** A bare sum over contacts carries a size- and cutoff-dependent offset (more
   atoms -> more shell contacts -> drift independent of true affinity). An attention module computes
   a per-contact weight, softmax-normalized over each complex's contacts (so weights sum to one,
   making the term scale-stable), forms a weighted triple-product aggregate, maps it through a small
   FC head to a scalar, and *subtracts* it from the contact sum.

5. **Bidirectional views + consistency.** The interface is scored ligand->pocket and pocket->ligand
   with separate weights, giving `pred_lp` and `pred_pl`. Inference averages them. Training uses a
   three-term loss — fit each to the label, plus a consistency term forcing the two views to agree.

## Architecture (hidden width H=256, 3 layers)

- **Project:** node features (35-dim) -> H for ligand and pocket; covalent edge (17-dim) -> H;
  non-covalent edge (11-dim) -> H. Separate weights per relation.
- **CIGConv (covalent):** message `ReLU(h_src + e)`, **sum** over neighbors, residual `h + agg`,
  then `Linear -> Dropout(0.1) -> LeakyReLU -> BatchNorm1d`.
- **NIGConv (non-covalent):** edge-gated message `h_src ⊙ e`, **mean** aggregation, `fc_neigh`
  applied after aggregation (since in==out), plus `fc_self(h_dst)` and a bias; Xavier-uniform init,
  feature dropout 0.1.
- **Layer:** run all four convolutions from the same inputs; **sum** per destination node type
  (`lig_out = CIG_l(lig) + NIG_{p->l}`, `poc_out = CIG_p(poc) + NIG_{l->p}`). Stack 3.
- **Atom-atom affinity readout (per direction):** `i = e_proj ⊙ src_proj ⊙ dst_proj`,
  `logit = Linear(i)`, sum logits over the direction's contacts -> `atompairs`.
- **Bias correction (per direction):** `w = Linear(PReLU(prj_src + prj_dst + prj_edge))`;
  `a = softmax_over_complex_contacts(w)`; `l = a ⊙ (w_edge ⊙ w_src ⊙ w_dst)`; sum -> FC
  (`H -> 200 -> 200 -> 1`, with dropout/LeakyReLU/BatchNorm on the two hidden blocks) -> `bias`.
- **Output:** `pred_dir = atompairs_dir - bias_dir`; final `pred = (pred_lp + pred_pl)/2`.
- **Loss:** `(MSE(pred_lp, y) + MSE(pred_pl, y) + MSE(pred_lp, pred_pl)) / 3`.
- **Training:** Adam, lr 1e-4, weight decay 1e-6; PDBbind (general+refined); test on CASF-2013 (107),
  CASF-2016 (285), 2019 holdout; metrics RMSE and Pearson Rp.

## Working code

Filling the message-passing and readout slots of the heterogeneous PLA harness (`PLABatch` carries
COO edge indices and per-edge features; this mirrors the heterogeneous-graph implementation):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CIGConv(nn.Module):
    """Covalent Interaction Graph Convolution (intra-molecular).
    Message ReLU(src + edge), sum aggregation, residual, MLP."""
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
    Edge weights gate the source features, mean aggregation; fc_neigh after
    aggregation when in_feats == out_feats; plus a self term and bias."""
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
    """EHIGN: heterogeneous interaction graph network with a sum-of-atom-pair
    affinity readout, attention bias correction, and bidirectional prediction."""
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
        self.prj_lp_src = nn.Linear(H, H); self.prj_lp_dst = nn.Linear(H, H)
        self.prj_lp_edge = nn.Linear(H, H); self.fc_lp = nn.Linear(H, 1)
        self.prj_pl_src = nn.Linear(H, H); self.prj_pl_dst = nn.Linear(H, H)
        self.prj_pl_edge = nn.Linear(H, H); self.fc_pl = nn.Linear(H, 1)

        # bias correction (L->P)
        self.bc_lp_prj_src = nn.Linear(H, H); self.bc_lp_prj_dst = nn.Linear(H, H)
        self.bc_lp_prj_edge = nn.Linear(H, H)
        self.bc_lp_att = nn.Sequential(nn.PReLU(), nn.Linear(H, 1))
        self.bc_lp_w_src = nn.Linear(H, H); self.bc_lp_w_dst = nn.Linear(H, H)
        self.bc_lp_w_edge = nn.Linear(H, H); self.bc_lp_fc = FC(H, 200, 2, 0.1, 1)
        # bias correction (P->L)
        self.bc_pl_prj_src = nn.Linear(H, H); self.bc_pl_prj_dst = nn.Linear(H, H)
        self.bc_pl_prj_edge = nn.Linear(H, H)
        self.bc_pl_att = nn.Sequential(nn.PReLU(), nn.Linear(H, 1))
        self.bc_pl_w_src = nn.Linear(H, H); self.bc_pl_w_dst = nn.Linear(H, H)
        self.bc_pl_w_edge = nn.Linear(H, H); self.bc_pl_fc = FC(H, 200, 2, 0.1, 1)

    def _edge_softmax(self, scores, batch_idx, num_graphs):
        max_scores = torch.full((num_graphs, 1), -1e9, device=scores.device)
        max_scores.index_reduce_(0, batch_idx, scores, 'amax', include_self=True)
        exp_scores = torch.exp(scores - max_scores[batch_idx])
        sum_exp = torch.zeros(num_graphs, 1, device=scores.device)
        sum_exp.index_add_(0, batch_idx, exp_scores)
        return exp_scores / sum_exp[batch_idx].clamp(min=1e-8)

    def _forward_heads(self, batch):
        B = batch.labels.size(0)
        lig_h = self.lin_node_l(batch.lig_x); poc_h = self.lin_node_p(batch.poc_x)
        lig_e = self.lin_edge_ll(batch.lig_edge_attr); poc_e = self.lin_edge_pp(batch.poc_edge_attr)
        lp_e = self.lin_edge_lp(batch.l2p_edge_attr); pl_e = self.lin_edge_pl(batch.p2l_edge_attr)

        for i in range(len(self.cig_l)):
            lig_in, poc_in = lig_h, poc_h
            lig_intra = self.cig_l[i](lig_in, batch.lig_edge_index, lig_e)
            poc_intra = self.cig_p[i](poc_in, batch.poc_edge_index, poc_e)
            lig_inter = torch.zeros_like(lig_in); poc_inter = torch.zeros_like(poc_in)
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

        # P->L edges use pocket as source, while projections keep ligand/pocket roles.
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

    def compute_loss(self, batch, labels):
        pred_lp, pred_pl = self._forward_heads(batch)
        return (F.mse_loss(pred_lp, labels) + F.mse_loss(pred_pl, labels)
                + F.mse_loss(pred_lp, pred_pl)) / 3
```

## Relation to prior methods

- **SchNet / EGNN:** homogeneous graph, message uses only a scalar distance, pooled-node readout.
  EHIGN separates covalent/non-covalent relations, uses richer per-edge geometric invariants (angles,
  areas, multiple distances), and reads out as a sum of per-contact affinities.
- **The heterogeneous-interaction-layer predecessor (GIGN):** also splits covalent/non-covalent in
  message passing and is invariant, but its message uses only an RBF of the distance and it reads out
  with `global_add_pool` + MLP. EHIGN injects the full geometric edge feature into the messages
  (edge-enhanced covalent conv; edge-gated non-covalent conv) and replaces the pooled readout with
  the explicit sum-of-atom-pair-affinities, the attention bias correction, and the bidirectional
  consistency objective — encoding "affinity = sum of pairwise atom-atom affinities" structurally.
