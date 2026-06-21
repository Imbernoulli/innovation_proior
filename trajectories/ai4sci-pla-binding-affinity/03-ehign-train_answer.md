EGNN landed exactly on the split I predicted, and the split is the whole instruction for this rung. Against SchNet it lifted 2019 the most — RMSE $1.4422$ (down from $1.5624$), Rp $0.6175$ (up from $0.5699$) — and improved 2013 across the board ($1.4114$ / $0.8093$ vs $1.4765$ / $0.7921$), confirming that the node-pair-aware message buys real generalization where reasoning beats memorization. But on 2016 it did what I feared: RMSE rose to $1.4213$ (from SchNet's $1.3702$) and Rp fell to $0.7646$ (from $0.7767$), because EGNN fed the distance raw as one channel where SchNet expanded it into $60$. So two diagnoses now stack and point at the same culprit from opposite sides: SchNet was geometry-resolved but node-poor and orientation-blind; EGNN is node-rich but geometry-thin. Both still feed their messages essentially one scalar — the contact distance — while the batch carries an $11$-dim geometric description of every edge (angle max/sum/mean, triangle-area max/sum/mean, neighbour-distance max/sum/mean, pairwise L1 and L2) and a $17$-dim description of every covalent edge that adds bond type, conjugation, and ring membership. Both rungs threw almost all of it away.

There is a sharper second diagnosis hiding in how all three fills so far read out the affinity: they pool an interface representation and regress it. But the free energy of binding is, to first order, a *sum* of pairwise contributions — each non-covalent contact between a ligand atom and a pocket atom contributes some favourable or unfavourable amount, and the total is their sum. None of the fills enforce that *form*; they let the affinity be any learned function of pooled statistics. If I instead make the output literally a sum over interface contacts of a per-contact atom-atom affinity, I restrict the model to functions that look like the physics — which should generalize better — and I get interpretability for free.

I propose **EHIGN**, an edge-enhanced heterogeneous interaction graph network in the Yang/Zhong interaction-graph lineage (2023–2024). It commits to two inductive biases at once: a genuinely heterogeneous covalent/non-covalent graph with the *full* edge features in the messages, and an output that is a literal sum of pairwise atom-atom affinities. I treat the two edge types with two purpose-built convolutions. The covalent edges inside each molecule are stiff, low-degree, and chemically meaningful, so in `CIGConv` I project each covalent edge's $17$ features to the hidden width and inject it into the message *additively before the nonlinearity*,

$$m_{ij} = \text{ReLU}(h_{\text{src}} + e_{ij}),$$

so that a double bond in a ring sends a different message than a single bond and a sharp angle differs from a straight one — the geometry SchNet and EGNN discarded is now in the message. I aggregate by *sum*, not mean, because covalent degree is small and chemically real: a carbon with four bonds genuinely carries more incident structure than one with two, and averaging would wash that away. Then a residual $\text{rst} = h + \text{agg}$ keeps the atom's identity so the conv only learns the update, followed by a post-MLP $\text{Linear} \to \text{Dropout}(0.1) \to \text{LeakyReLU} \to \text{BatchNorm1d}$ — the BatchNorm doing real work, since residual message passing lets activation magnitudes drift across three layers and normalizing after each conv keeps the scale controlled. Ligand and pocket covalent graphs get separate weights, because drug-like and protein-pocket chemistry are not identical.

The non-covalent contacts are different in character: a pocket atom can contact many ligand atoms, the degree varies a lot, and a contact's strength should depend on its geometry. So in `NIGConv` I do not sum raw source features — I let the projected contact geometry *gate* the message multiplicatively, $m = h_{\text{src}} \odot e$, so a close, well-oriented contact passes more of the source atom's signal and a marginal one passes less. I aggregate by *mean*, precisely because the degree is so variable: summing would let an atom with twenty weak contacts swamp one with three strong ones by count alone, whereas the mean asks "what is the typical gated message into this atom." The update is a linear map of the mean aggregate plus a separate linear map of the destination's own feature and a bias, $\text{rst} = \text{fc\_self}(h_{\text{dst}}) + \text{fc\_neigh}(\text{mean}) + \text{bias}$, with `fc_neigh` applied *after* the mean (cheaper, and equivalent when in- and out-widths match) and Xavier-uniform init on both maps. I run it in both directions, ligand$\to$pocket and pocket$\to$ligand, with separate weights, since a contact summarized from each side is a different view. Each layer runs all four convolutions in parallel from the same inputs and sums per destination type — $\text{lig\_out} = \text{CIG}_{\text{lig}}(\text{lig}) + \text{NIG}_{p\to l}(\text{poc}, \text{lig})$, $\text{poc\_out} = \text{CIG}_{\text{poc}}(\text{poc}) + \text{NIG}_{l\to p}(\text{lig}, \text{poc})$ — because covalent and non-covalent influences are additive. Three layers, hidden width $256$.

The readout *is* the second inductive bias. After three layers I score each non-covalent contact with a low-rank triple product of the projected source atom, the projected destination atom, and the projected contact geometry, $i_{lp} = \text{prj\_edge}(e) \odot \text{prj\_src}(\text{lig\_h})[\text{src}] \odot \text{prj\_dst}(\text{poc\_h})[\text{dst}]$, collapsed by a final $\text{Linear}(H,1)$ to one scalar per contact, summed over a complex's contacts via `inter_batch`. The elementwise triple product is a low-rank bilinear scorer that fires where source atom, destination atom, and contact geometry all agree — exactly "this kind of atom meeting that kind of atom at this geometry is favourable." I do it in both directions for two estimates per complex. But a raw, unweighted sum over every contact inside the $5$ Å cutoff carries a systematic offset: bigger complexes simply have more contacts in the shell, many incidental rather than favourable, so the sum drifts with size independent of true binding strength. This is the price the additive form always pays — interpretability buys in a size-dependent nuisance. The fix is a learned, complex-specific bias correction that knows not all contacts deserve equal weight: I compute a logit per contact from its source, destination, and edge projections, softmax it *over each complex's contacts* (so the weights sum to one and the correction is scale-stable regardless of contact count, killing exactly the size-dependence), form an attention-weighted triple-product aggregate, push it through a $2$-layer $\text{FC}(H, 200)$, and subtract: $\text{pred}_{lp} = \text{atompairs}_{lp} - \text{bias}_{lp}$, likewise $\text{pred}_{pl}$, per direction with separate weights.

That leaves the move that distinguishes this rung's *training*. The two corrected views — ligand$\to$pocket and pocket$\to$ligand — summarize the *same* interface from opposite molecules, so they ought to agree, and a disagreement is a signal the model is being inconsistent. Beyond fitting each to the label, I add a term that drives the two toward each other. The loss is three MSE terms averaged,

$$\mathcal{L} = \tfrac{1}{3}\Big(\text{MSE}(\text{pred}_{lp}, y) + \text{MSE}(\text{pred}_{pl}, y) + \text{MSE}(\text{pred}_{lp}, \text{pred}_{pl})\Big),$$

the third term costing nothing at inference and acting as multi-view distillation, pushing both heads to encode a consistent interface. This is the one rung where the harness's plain-MSE default is not enough — a single `forward` output cannot express a two-head, three-term objective — and the harness anticipates exactly this: it calls `compute_loss(batch, labels)` if the model exposes it, falling back to plain MSE otherwise. So I expose `compute_loss`, compute both heads, and return that average; at inference `forward` averages the two heads. Invariance survives end to end as before — every input is frame-independent, so every message, aggregate, triple product, attention, and bias is invariant, and coordinates never appear.

My expectation against EGNN's numbers: the full $17$/$11$-dim edge geometry in the messages should recover the distance resolution EGNN lost on 2016 while keeping its node-pair richness, so I expect 2016 to be where this rung gains *most* — RMSE well below EGNN's $1.4213$ and Rp well above $0.7646$, plausibly past SchNet's $0.7767$, since I now have both resolution and node expressiveness; that should be the headline. On 2013 I expect a modest gain over $1.4114$ / $0.8093$. The open question is 2019: the consistency-regularized sum-of-contacts form should generalize at least as well as EGNN's $0.6175$, but the richer covalent-edge features (bond type, ring membership) may help less on the temporally distant set than on the near-training core, so 2019 may move only a little and could even sit a touch below EGNN if the extra covalent capacity overfits training-era chemistry. If that happens — strong 2016/2013 gains but flat or slightly worse 2019 — it says the contribution is real but concentrated where chemistry is familiar, and it sets up the final rung's question of whether all this dual-head, bias-corrected, consistency-trained machinery is actually necessary.

```python
# EDITABLE SECTION START — EHIGN: Heterogeneous Interaction Graph Network

class CIGConv(nn.Module):
    """Covalent Interaction Graph Convolution (intra-molecular).
    Message: ReLU(src + edge_feat), sum aggregation, residual, MLP.
    """
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
        rst = x + agg  # residual
        return self.mlp(rst)


class NIGConv(nn.Module):
    """Non-covalent Interaction Graph Convolution (inter-molecular).
    Uses edge weights as multiplicative gates on source features, mean aggregation.
    Matches original: when in_feats == out_feats, fc_neigh applied AFTER aggregation.
    """
    def __init__(self, in_feats, out_feats, feat_drop=0.0):
        super().__init__()
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_neigh = nn.Linear(in_feats, out_feats, bias=False)
        self.fc_self = nn.Linear(in_feats, out_feats, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_feats))
        nn.init.xavier_uniform_(self.fc_self.weight)
        nn.init.xavier_uniform_(self.fc_neigh.weight)

    def forward(self, x_src, x_dst, edge_index, edge_weight, num_dst):
        x_src = self.feat_drop(x_src)
        x_dst = self.feat_drop(x_dst)
        src, dst = edge_index
        # Edge-weighted messages: src_feat * edge_weight (element-wise)
        msg = x_src[src] * edge_weight
        # Mean aggregation
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
    """EHIGN: Edge-enhanced Heterogeneous Interaction Graph Network.

    Uses CIGConv for intra-molecular and NIGConv for inter-molecular message passing.
    HeteroGraphConv pattern: all edge types computed in parallel, outputs summed per node type.
    Dual bidirectional prediction with attention-based bias correction.
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

        # Atom-atom affinity heads
        self.prj_lp_src = nn.Linear(H, H)
        self.prj_lp_dst = nn.Linear(H, H)
        self.prj_lp_edge = nn.Linear(H, H)
        self.fc_lp = nn.Linear(H, 1)
        self.prj_pl_src = nn.Linear(H, H)
        self.prj_pl_dst = nn.Linear(H, H)
        self.prj_pl_edge = nn.Linear(H, H)
        self.fc_pl = nn.Linear(H, 1)

        # Bias correction (L->P direction)
        self.bc_lp_prj_src = nn.Linear(H, H)
        self.bc_lp_prj_dst = nn.Linear(H, H)
        self.bc_lp_prj_edge = nn.Linear(H, H)
        self.bc_lp_att = nn.Sequential(nn.PReLU(), nn.Linear(H, 1))
        self.bc_lp_w_src = nn.Linear(H, H)
        self.bc_lp_w_dst = nn.Linear(H, H)
        self.bc_lp_w_edge = nn.Linear(H, H)
        self.bc_lp_fc = FC(H, 200, 2, 0.1, 1)

        # Bias correction (P->L direction)
        self.bc_pl_prj_src = nn.Linear(H, H)
        self.bc_pl_prj_dst = nn.Linear(H, H)
        self.bc_pl_prj_edge = nn.Linear(H, H)
        self.bc_pl_att = nn.Sequential(nn.PReLU(), nn.Linear(H, 1))
        self.bc_pl_w_src = nn.Linear(H, H)
        self.bc_pl_w_dst = nn.Linear(H, H)
        self.bc_pl_w_edge = nn.Linear(H, H)
        self.bc_pl_fc = FC(H, 200, 2, 0.1, 1)

    def _edge_softmax(self, scores, batch_idx, num_graphs):
        max_scores = torch.zeros(num_graphs, 1, device=scores.device).fill_(-1e9)
        max_scores.index_reduce_(0, batch_idx, scores, 'amax', include_self=True)
        exp_scores = torch.exp(scores - max_scores[batch_idx])
        sum_exp = torch.zeros(num_graphs, 1, device=scores.device)
        sum_exp.index_add_(0, batch_idx, exp_scores)
        return exp_scores / sum_exp[batch_idx].clamp(min=1e-8)

    def _forward_heads(self, batch: PLABatch):
        """Compute both dual prediction heads. Returns (pred_lp, pred_pl) each [B]."""
        B = batch.labels.size(0)
        # Project features
        lig_h = self.lin_node_l(batch.lig_x)
        poc_h = self.lin_node_p(batch.poc_x)
        lig_e = self.lin_edge_ll(batch.lig_edge_attr)
        poc_e = self.lin_edge_pp(batch.poc_edge_attr)
        lp_e = self.lin_edge_lp(batch.l2p_edge_attr)
        pl_e = self.lin_edge_pl(batch.p2l_edge_attr)

        # Message passing: HeteroGraphConv pattern — parallel compute, sum aggregate
        for i in range(len(self.cig_l)):
            # Save inputs (all convs use same input features)
            lig_in, poc_in = lig_h, poc_h

            # Intra-molecular (CIGConv has internal residual)
            lig_intra = self.cig_l[i](lig_in, batch.lig_edge_index, lig_e)
            poc_intra = self.cig_p[i](poc_in, batch.poc_edge_index, poc_e)

            # Inter-molecular (NIGConv with edge weights)
            lig_inter = torch.zeros_like(lig_in)
            poc_inter = torch.zeros_like(poc_in)
            if batch.l2p_edge_index.size(1) > 0:
                poc_inter = self.nig_lp[i](lig_in, poc_in, batch.l2p_edge_index, lp_e, poc_in.size(0))
            if batch.p2l_edge_index.size(1) > 0:
                lig_inter = self.nig_pl[i](poc_in, lig_in, batch.p2l_edge_index, pl_e, lig_in.size(0))

            # Sum aggregation per destination node type
            lig_h = lig_intra + lig_inter
            poc_h = poc_intra + poc_inter

        # Atom-atom affinities (L->P)
        l2p_src, l2p_dst = batch.l2p_edge_index
        i_lp = self.prj_lp_edge(lp_e) * self.prj_lp_src(lig_h)[l2p_src] * self.prj_lp_dst(poc_h)[l2p_dst]
        logit_lp = self.fc_lp(i_lp)
        pred_lp = torch.zeros(B, 1, device=logit_lp.device)
        pred_lp.index_add_(0, batch.inter_batch, logit_lp)

        # Atom-atom affinities (P->L)
        p2l_src, p2l_dst = batch.p2l_edge_index
        p2l_batch = batch.lig_batch[p2l_dst]
        i_pl = self.prj_pl_edge(pl_e) * self.prj_pl_src(poc_h)[p2l_src] * self.prj_pl_dst(lig_h)[p2l_dst]
        logit_pl = self.fc_pl(i_pl)
        pred_pl = torch.zeros(B, 1, device=logit_pl.device)
        pred_pl.index_add_(0, p2l_batch, logit_pl)

        # Bias correction (L->P)
        w_lp = self.bc_lp_prj_src(lig_h)[l2p_src] + self.bc_lp_prj_dst(poc_h)[l2p_dst] + self.bc_lp_prj_edge(lp_e)
        a_lp = self._edge_softmax(self.bc_lp_att(w_lp), batch.inter_batch, B)
        s_lp = a_lp * self.bc_lp_w_edge(lp_e) * self.bc_lp_w_src(lig_h)[l2p_src] * self.bc_lp_w_dst(poc_h)[l2p_dst]
        bias_lp_agg = torch.zeros(B, s_lp.size(-1), device=s_lp.device)
        bias_lp_agg.index_add_(0, batch.inter_batch, s_lp)
        bias_lp = self.bc_lp_fc(bias_lp_agg)

        # Bias correction (P->L)
        w_pl = self.bc_pl_prj_src(poc_h)[p2l_src] + self.bc_pl_prj_dst(lig_h)[p2l_dst] + self.bc_pl_prj_edge(pl_e)
        a_pl = self._edge_softmax(self.bc_pl_att(w_pl), p2l_batch, B)
        s_pl = a_pl * self.bc_pl_w_edge(pl_e) * self.bc_pl_w_src(poc_h)[p2l_src] * self.bc_pl_w_dst(lig_h)[p2l_dst]
        bias_pl_agg = torch.zeros(B, s_pl.size(-1), device=s_pl.device)
        bias_pl_agg.index_add_(0, p2l_batch, s_pl)
        bias_pl = self.bc_pl_fc(bias_pl_agg)

        pred_lp_final = (pred_lp - bias_lp).squeeze(-1)
        pred_pl_final = (pred_pl - bias_pl).squeeze(-1)
        return pred_lp_final, pred_pl_final

    def forward(self, batch: PLABatch) -> torch.Tensor:
        pred_lp, pred_pl = self._forward_heads(batch)
        return (pred_lp + pred_pl) / 2

    def compute_loss(self, batch: PLABatch, labels: torch.Tensor) -> torch.Tensor:
        """EHIGN 3-term dual-head loss (paper: guaguabujianle/EHIGN_PLA train.py#L852):
            loss = (MSE(pred_lp, y) + MSE(pred_pl, y) + MSE(pred_lp, pred_pl)) / 3
        The third term is a consistency regularizer between the two bidirectional heads.
        """
        pred_lp, pred_pl = self._forward_heads(batch)
        loss = (F.mse_loss(pred_lp, labels)
                + F.mse_loss(pred_pl, labels)
                + F.mse_loss(pred_lp, pred_pl)) / 3
        return loss

# EDITABLE SECTION END
```
