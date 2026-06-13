Let me start from what binding affinity actually *is*, because I think the way I represent the problem is going to decide everything downstream. I have a protein-ligand complex: a small molecule sitting in a pocket, all atoms with 3D coordinates, and I want one number, `-logKd/Ki`, larger meaning tighter binding. Physically that number is a free energy, and the free energy of binding comes overwhelmingly from the *non-covalent contacts* across the interface — hydrogen bonds, salt bridges, van der Waals packing, pi-stacking — each one between a particular ligand atom and a particular pocket atom, at a particular distance and orientation. The covalent bonds inside the ligand and inside the protein don't cross the interface and don't directly contribute binding energy; what they do is set the *conformation*, the geometry that decides where the contacting atoms end up. So already there are two completely different kinds of relationship in this object, and they mean different things for binding. If I throw them into one undifferentiated graph and run a generic convolution, I'm asking the network to rediscover, from data, that covalent and non-covalent edges play different roles. That's capacity I'd rather not spend, and it's exactly the kind of thing a dataset shortcut can fake on the training distribution and then fail out of sample. I'd rather restrict the function class up front to functions that respect the physics.

So what assumptions do I actually want to commit to? Two, I think, and I want to be honest that they *are* assumptions — they're the inductive bias. First: a complex is a heterogeneous graph, with covalent (intramolecular) edges and non-covalent (intermolecular) edges treated as genuinely different relations. Second, and this is the sharper one: the predicted affinity is the *sum of pairwise atom-atom affinities determined by the non-covalent interactions*. That second one is a strong claim about the *form* of the output, not just the input. It says: don't pool everything into a graph vector and regress; instead score each interface contact and add them up. If that's true, the model's output is interpretable by construction — I can read off which contacts contributed — and it's restricted to functions that look like a sum over the physics, which should generalize better. Let me hold onto both and see if I can build an architecture that bakes them in.

Before architecture, the invariance. The affinity can't change if I translate or rotate the whole complex; it's a property of the arrangement, not the coordinate frame. If I feed raw coordinates to the network it isn't invariant and I'd have to learn invariance by augmenting with random rotations, which is wasteful. The clean way is to only ever feed the network quantities that are *already* invariant to rigid motion. Interatomic distances are invariant. Angles between bond vectors are invariant. The area of the triangle three atoms span is invariant. So instead of coordinates, let me precompute, for each edge `i->j`, a bag of these invariants. Walk the neighbors `k` of `j`: the angle between `g_j - g_i` and `g_k - g_i`, the triangle area `0.5||(g_j - g_i) x (g_k - g_i)||`, the distance `||g_i - g_k||`. There can be several neighbors, so summarize each by max, sum, and mean — that's nine numbers, a fixed-size description of the local geometry around the edge regardless of how many neighbors there are. Add the direct `i->j` distance in L1 and L2 norm and I have eleven invariant geometric numbers per edge. I'll scale them by small constants (angles and the means and maxes by 0.1, the sums by 0.01) just to keep the magnitudes in a comfortable numeric range so training is stable — a sum of angles over many neighbors can otherwise get large relative to a single distance. For a covalent edge I also know the bond chemistry, so I'll prepend the bond type (single/double/triple/aromatic), conjugation, and ring membership — six more, seventeen total. A non-covalent contact has no covalent bond, so it carries the eleven geometric numbers alone. And the atom nodes get the usual chemical one-hots: element, degree, implicit valence, hybridization, aromatic, hydrogen count, thirty-five dimensions. Good — every input is invariant, so whatever I build on top is invariant for free, no augmentation. The contacts themselves I'll create between every ligand-pocket atom pair within 5 angstroms; beyond that the non-covalent interaction is negligible, and the cutoff keeps the contact graph sparse.

Now, who has done something like this, and where do they stop? The continuous-filter convolution idea — SchNet — updates each atom by `x'_i = sum_j x_j ⊙ h(RBF(d_ij))`, where the scalar distance is expanded onto a bank of Gaussians and an MLP turns that into a filter that gates the neighbor's features, with a smooth cutoff so far neighbors fade out. It's elegant and invariant, but the only geometry it ever sees is the *scalar distance*. All those angles and triangle areas I just computed — gone. And it's a homogeneous graph, one filter for every edge, so covalent and non-covalent get the same treatment, which violates my first assumption. And the readout sums atom embeddings into one vector and regresses — no sum over contacts, so my second assumption isn't expressible either. EGNN is in the same family: its message is an MLP of the endpoint features and `||x_i - x_j||^2`, again a single scalar invariant, one convolution for all edges, pooled readout. So both of these are invariant the way I want but throw away most of the geometry and don't separate the interaction types or decompose the output over contacts.

The closest predecessor splits the interaction types. Its heterogeneous interaction layer keeps one node set but two edge index sets, covalent and non-covalent, and runs two message passes in parallel: a learned MLP of the RBF-expanded distance, `radial = phi(RBF(d))`, the message `x_j ⊙ radial`, summed separately over covalent neighbors into `agg_intra` and over non-covalent neighbors into `agg_inter`, and the node update `x'_i = mlp_cov(x_i + agg_intra_i) + mlp_ncov(x_i + agg_inter_i)`. Three layers, then `global_add_pool` and an MLP. That nails my first assumption — covalent and non-covalent are processed by separate convolutions and separate update MLPs — and it's invariant. But two things still bother me. The geometry it uses is *still just the scalar distance through the RBF*; my eleven-number invariant description, the angles and areas, never enters the message. And the readout is still `global_add_pool` of node embeddings — the affinity is read off a pooled vector, *not* written as a sum over the non-covalent atom pairs. So my second, sharper assumption — affinity is the sum of pairwise atom-atom affinities — is not enforced anywhere; the model is free to fit affinity from whatever pooled node statistics correlate with the label, and the per-contact interpretability is only available afterward, by squinting at embeddings. That's the gap I want to close. I'll keep the heterogeneous split, but I want richer edge geometry *inside* the messages, and I want the readout itself to be the sum-over-contacts.

Let me design the message passing first, relation by relation, since I've decided I want a distinct convolution per relation. Take the covalent edges inside a molecule — call this the covalent interaction graph convolution. I have node features `h` and, now, a *projected edge feature* `e` for each covalent edge, carrying the bond chemistry and the eleven geometric invariants, both mapped into the hidden width by a linear layer. I don't want to discard `e` the way distance-only convolutions effectively do; I want the edge geometry to shape the message. The simplest faithful way to inject it: form the message from the source node and the edge together before the nonlinearity, `m_ij = ReLU(h_src_j + e_ij)`. Adding the edge feature to the source feature and passing through ReLU means the message a node sends along an edge is modulated by that edge's bond type and geometry — a double bond in a ring sends a different message than a single bond, an edge with a sharp angle differs from a straight one. Then aggregate by sum over the neighbors, `neigh_i = sum_j m_ij`. Sum, not mean, because for covalent edges the degree is small and chemically meaningful — a carbon with four bonds genuinely has more incident structure than one with two, and I don't want to average that signal away. Then a residual, `rst_i = h_i + neigh_i`, so the node keeps its own identity and the convolution only has to learn the *update*, which keeps deeper stacks stable. And a small post-MLP: `Linear -> Dropout(0.1) -> LeakyReLU -> BatchNorm1d`. The dropout regularizes, LeakyReLU avoids dead units, and the BatchNorm is doing real work here — message passing with residuals can let activation magnitudes drift across layers, and normalizing after each conv keeps the scale controlled so three layers train stably. That's the covalent conv. I run it separately on the ligand's covalent graph and on the pocket's, with separate weights, because the chemistry of a drug-like ligand and a protein pocket isn't identical.

Now the non-covalent edges across the interface — the non-covalent interaction graph convolution. These are different in character: a pocket atom can be in contact with many ligand atoms and vice versa, the contact degree varies a lot, and a contact's *strength* should depend on its geometry. So here I don't want plain summation of raw source features. I want the edge — the geometric description of the contact — to *gate* the message. Treat the projected contact feature as an edge weight and multiply it into the source feature elementwise: `m = h_src ⊙ edge_weight`. A close, well-oriented contact (its geometric feature large in the right components) passes more of the source atom's signal; a marginal contact passes less. Then aggregate by *mean*, not sum — because the contact degree is so variable, summing would make an atom with twenty weak contacts swamp one with three strong ones purely by count, whereas the mean asks "what's the typical gated message into this atom." This is a SAGE-style neighborhood: `h_neigh_i = mean_j (h_src_j ⊙ e_ij)`, then a linear map of the aggregate, plus a separate linear map of the destination atom's own feature (the self term) and a bias: `rst = fc_self(h_dst) + fc_neigh(h_neigh) + bias`. One subtlety about where the linear map goes — apply it to features before message passing or to the aggregate after? Applying it before means transforming every source feature; applying it after means transforming once per destination. When the input and output widths are equal, as they are here, it's cheaper and equivalent in expressiveness to apply `fc_neigh` *after* the mean aggregation, so I'll do that. Initialize `fc_self` and `fc_neigh` with Xavier uniform (relu gain) so the variances are sensible at the start. I run this in both directions — ligand->pocket and pocket->ligand — with separate weights, because a contact summarized from the ligand atom's side and the same contact summarized from the pocket atom's side are two different views and I want both.

How do the relations combine into one layer? Each layer should update both node types from the *same* input features, running all four convolutions in parallel — covalent on the ligand, covalent on the pocket, non-covalent ligand->pocket, non-covalent pocket->ligand — and then, for each destination node type, combine the contributions that land on it. A pocket atom receives a covalent update from its own molecule and a non-covalent update from the ligand contacts pointing into it; a ligand atom receives its covalent update plus the non-covalent update from pocket contacts. I'll combine by summation: `lig_out = CIG_ligand(lig) + NIG_{pocket->ligand}(poc, lig)` and `poc_out = CIG_pocket(poc) + NIG_{ligand->pocket}(lig, poc)`. Sum is the natural choice — covalent and non-covalent contributions are additive influences on the atom, and using the same input features for both branches means the layer cleanly separates "what my own molecule tells me" from "what the binding partner tells me." Stack three of these. Why three? The contact graph is local — a 5-angstrom shell — and three rounds of message passing already let information propagate across a small neighborhood of the interface; go much deeper on a graph this size and message passing oversmooths, all atoms collapsing toward the same representation. Three matches the depth that works for the predecessor and keeps the receptive field sensible. Hidden width 256 throughout.

Now the part I actually care about — the readout that *is* my second assumption. After three layers I have a learned representation for every ligand atom and every pocket atom. My claim is: affinity = sum over interface contacts of a per-contact atom-atom affinity. So for each non-covalent edge — each contact between a ligand atom and a pocket atom — I want to produce a scalar, the affinity of *that* contact, and then sum the scalars over all contacts of the complex. The scalar has to depend on three things: the two atoms in contact, and the geometry of the contact. The cleanest way to let all three jointly determine a contact score, without a giant interaction tensor, is an elementwise triple product of projected versions of them: project the source node, the destination node, and the edge feature each to the hidden width, multiply them elementwise, `i_contact = e_proj ⊙ src_proj ⊙ dst_proj`, then a `Linear(., 1)` collapses that hidden vector to a single number. The elementwise product is a low-rank bilinear scorer — it fires on components where the source atom, the destination atom, *and* the contact geometry all agree, which is exactly "this kind of atom meeting that kind of atom at this geometry is a favorable contact." Sum these per-contact scalars over all the non-covalent edges of a complex and I get the predicted affinity as a literal sum of pairwise atom-atom affinities. That's the whole point: the output *form* encodes the physics, and each summand is the contribution of one named contact.

And I do this in both directions, ligand->pocket and pocket->ligand, with separate projection and scoring weights, giving two affinity estimates per complex, `atompairs_lp` and `atompairs_pl`. Two views of the same interface — one centered on aggregating contacts at pocket atoms, one at ligand atoms.

Let me stress-test the sum-of-contacts readout, because I think there's a problem with it as stated. A raw, unweighted sum over every contact within 5 angstroms treats all contacts as additive and on the same footing. But not every atom pair inside the cutoff is a real, favorable interaction — many are incidental, geometrically present but contributing little or even noise, and the cutoff itself is arbitrary. A bare sum over a variable number of contacts will carry a systematic, complex-dependent offset: complexes with more atoms simply have more contacts in the shell, so the sum drifts with size and with how many spurious pairs happen to fall inside the cutoff, independent of true binding strength. If I just regress that raw sum against affinity I'm asking it to absorb that offset implicitly, which it will do badly and unstably. I hit a wall: the additive form I want for interpretability also bakes in an additive nuisance.

So I need to subtract off that nuisance — a learned, complex-specific *bias correction*. What should it be? It should look at the same set of contacts and estimate the background offset to remove, and crucially it should know that not all contacts deserve equal weight in that estimate — the correction should be dominated by the contacts that matter, not diluted by the many marginal ones. That's an attention mechanism over the contact edges. Compute, for each contact, an attention logit from its source, destination, and edge features — sum the three projections and pass through a small head, `w = Linear(PReLU(prj_src(src) + prj_dst(dst) + prj_edge(e)))`, one scalar per contact. Normalize these logits with a softmax *over the contacts of each complex* — so the weights of a complex's contacts sum to one, making the correction scale-stable regardless of how many contacts there are, which is precisely the size-dependence I needed to kill. Then form a weighted aggregate: for each contact a feature `l = a ⊙ (w_edge(e) ⊙ w_src(src) ⊙ w_dst(dst))` — the attention weight times another elementwise triple product, with its own projection weights — sum these over the contacts, and pass the pooled vector through the same small fully-connected pattern the implementation uses for this correction (`Linear(H, 200) -> Dropout(0.1) -> LeakyReLU -> BatchNorm1d -> Linear(200, 200) -> Dropout(0.1) -> LeakyReLU -> BatchNorm1d -> Linear(200, 1)`) to a scalar bias. Subtract it: `pred_lp = atompairs_lp - bias_lp`, and similarly `pred_pl = atompairs_pl - bias_pl`. The attention-normalized aggregate gives a complex-specific baseline that the raw additive sum is then measured against; the softmax over contacts is what makes it a *correction* rather than another size-confounded sum. I do it per direction, separate weights for the ligand->pocket and pocket->ligand corrections.

Now I have two corrected estimates, `pred_lp` and `pred_pl`, from the two directional views. The final prediction is just their average, `(pred_lp + pred_pl) / 2` — both are estimates of the same physical affinity, and averaging two views reduces variance. But there's something better I can do during training than just average at the end. The two views are summarizing the *same* interface from opposite molecules, so they ought to agree — a disagreement between them is a signal that the model is being inconsistent. So beyond fitting each to the label, I can add a term that forces them toward each other. The loss becomes three mean-squared-error terms, averaged: fit the ligand->pocket estimate to the truth, fit the pocket->ligand estimate to the truth, and a consistency term that drives the two estimates together: `loss = (MSE(pred_lp, y) + MSE(pred_pl, y) + MSE(pred_lp, pred_pl)) / 3`. The third term is a multi-view agreement regularizer — it costs nothing extra at inference, and it pushes both directional heads to encode a consistent notion of the interface, which is a form of mutual distillation that should help generalization. The two views aren't redundant — they aggregate the same contacts at different atoms with different weights — but they must land on the same number, and saying so explicitly is free supervision.

Let me also double check the invariance survives all of this end to end. The node features are chemical, frame-independent; the edge features are angles, areas, distances — all invariant. Every message, every aggregate, every triple product, the attention, the bias, the readout sum — they're all functions of those invariant inputs only; coordinates never appear except through the precomputed invariants. So translating or rotating the complex leaves every intermediate unchanged, hence the prediction. Invariance holds by construction, no augmentation, exactly as I required at the start. And the second assumption is now structural: the prediction is `(sum_of_contact_affinities - attention_bias)` averaged over two directions, which is, up to the learned correction, a sum of pairwise atom-atom affinities. Each contact's contribution is a named, inspectable scalar — interpretability isn't bolted on afterward, it's the output's shape.

Let me write it as the model I'd actually ship, filling the two slots — the per-relation message passing and the contact-sum readout — in the heterogeneous PLA harness. I'll keep the message-passing layer as four convolutions combined by summation per node type, then the dual atom-atom-affinity readout with its bias correction, then the average and the three-term loss.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CIGConv(nn.Module):
    """Covalent Interaction Graph Convolution (intra-molecular).
    Edge-enhanced message ReLU(h_src + e), sum aggregation, residual, post-MLP."""
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
        msg = F.relu(x[src] + edge_attr)          # edge geometry/chemistry shapes the message
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, msg)               # sum over covalent neighbors
        rst = x + agg                             # residual: keep the atom's own identity
        return self.mlp(rst)                      # Dropout/LeakyReLU/BatchNorm stabilize depth


class NIGConv(nn.Module):
    """Non-covalent Interaction Graph Convolution (inter-molecular).
    Edge feature gates the source (h_src * e), mean aggregation, plus a self term.
    in==out, so fc_neigh is applied AFTER aggregation (cheaper, equivalent)."""
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
        msg = x_src[src] * edge_weight            # contact geometry gates the message
        agg = torch.zeros(num_dst, msg.size(-1), device=msg.device)
        count = torch.zeros(num_dst, 1, device=msg.device)
        agg.index_add_(0, dst, msg)
        count.index_add_(0, dst, torch.ones(src.size(0), 1, device=src.device))
        h_neigh = self.fc_neigh(agg / count.clamp(min=1))   # mean: contact degree varies a lot
        return self.fc_self(x_dst) + h_neigh + self.bias


class FC(nn.Module):
    """Small fully-connected head used by the bias-correction module."""
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

        # per-relation convolutions; all four run in parallel each layer
        self.cig_l = nn.ModuleList([CIGConv(H, H) for _ in range(num_layers)])
        self.cig_p = nn.ModuleList([CIGConv(H, H) for _ in range(num_layers)])
        self.nig_lp = nn.ModuleList([NIGConv(H, H, 0.1) for _ in range(num_layers)])
        self.nig_pl = nn.ModuleList([NIGConv(H, H, 0.1) for _ in range(num_layers)])

        # atom-atom affinity readout (both directions): scalar per contact = Linear(e*src*dst)
        self.prj_lp_src = nn.Linear(H, H); self.prj_lp_dst = nn.Linear(H, H)
        self.prj_lp_edge = nn.Linear(H, H); self.fc_lp = nn.Linear(H, 1)
        self.prj_pl_src = nn.Linear(H, H); self.prj_pl_dst = nn.Linear(H, H)
        self.prj_pl_edge = nn.Linear(H, H); self.fc_pl = nn.Linear(H, 1)

        # bias correction (attention-weighted, both directions)
        self.bc_lp_prj_src = nn.Linear(H, H); self.bc_lp_prj_dst = nn.Linear(H, H)
        self.bc_lp_prj_edge = nn.Linear(H, H); self.bc_lp_att = nn.Sequential(nn.PReLU(), nn.Linear(H, 1))
        self.bc_lp_w_src = nn.Linear(H, H); self.bc_lp_w_dst = nn.Linear(H, H)
        self.bc_lp_w_edge = nn.Linear(H, H); self.bc_lp_fc = FC(H, 200, 2, 0.1, 1)
        self.bc_pl_prj_src = nn.Linear(H, H); self.bc_pl_prj_dst = nn.Linear(H, H)
        self.bc_pl_prj_edge = nn.Linear(H, H); self.bc_pl_att = nn.Sequential(nn.PReLU(), nn.Linear(H, 1))
        self.bc_pl_w_src = nn.Linear(H, H); self.bc_pl_w_dst = nn.Linear(H, H)
        self.bc_pl_w_edge = nn.Linear(H, H); self.bc_pl_fc = FC(H, 200, 2, 0.1, 1)

    def _edge_softmax(self, scores, batch_idx, num_graphs):
        # softmax over each complex's contact edges -> attention sums to one per complex
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

        # message passing: all relations from the same inputs, summed per destination type
        for i in range(len(self.cig_l)):
            lig_in, poc_in = lig_h, poc_h
            lig_intra = self.cig_l[i](lig_in, batch.lig_edge_index, lig_e)
            poc_intra = self.cig_p[i](poc_in, batch.poc_edge_index, poc_e)
            lig_inter = torch.zeros_like(lig_in); poc_inter = torch.zeros_like(poc_in)
            if batch.l2p_edge_index.size(1) > 0:
                poc_inter = self.nig_lp[i](lig_in, poc_in, batch.l2p_edge_index, lp_e, poc_in.size(0))
            if batch.p2l_edge_index.size(1) > 0:
                lig_inter = self.nig_pl[i](poc_in, lig_in, batch.p2l_edge_index, pl_e, lig_in.size(0))
            lig_h = lig_intra + lig_inter        # covalent + non-covalent influence, summed
            poc_h = poc_intra + poc_inter

        # atom-atom affinity sum (L->P): score each contact, sum over the complex's contacts
        l2p_src, l2p_dst = batch.l2p_edge_index
        i_lp = self.prj_lp_edge(lp_e) * self.prj_lp_src(lig_h)[l2p_src] * self.prj_lp_dst(poc_h)[l2p_dst]
        logit_lp = self.fc_lp(i_lp)
        pred_lp = torch.zeros(B, 1, device=logit_lp.device)
        pred_lp.index_add_(0, batch.inter_batch, logit_lp)

        # atom-atom affinity sum (P->L); projection names stay tied to ligand/pocket roles
        p2l_src, p2l_dst = batch.p2l_edge_index
        p2l_batch = batch.lig_batch[p2l_dst]
        i_pl = self.prj_pl_edge(pl_e) * self.prj_pl_dst(poc_h)[p2l_src] * self.prj_pl_src(lig_h)[p2l_dst]
        logit_pl = self.fc_pl(i_pl)
        pred_pl = torch.zeros(B, 1, device=logit_pl.device)
        pred_pl.index_add_(0, p2l_batch, logit_pl)

        # bias correction (L->P): attention over contacts, weighted triple product, subtract
        w_lp = self.bc_lp_prj_src(lig_h)[l2p_src] + self.bc_lp_prj_dst(poc_h)[l2p_dst] + self.bc_lp_prj_edge(lp_e)
        a_lp = self._edge_softmax(self.bc_lp_att(w_lp), batch.inter_batch, B)
        s_lp = a_lp * self.bc_lp_w_edge(lp_e) * self.bc_lp_w_src(lig_h)[l2p_src] * self.bc_lp_w_dst(poc_h)[l2p_dst]
        bias_lp_agg = torch.zeros(B, s_lp.size(-1), device=s_lp.device)
        bias_lp_agg.index_add_(0, batch.inter_batch, s_lp)
        bias_lp = self.bc_lp_fc(bias_lp_agg)

        # bias correction (P->L)
        w_pl = self.bc_pl_prj_dst(poc_h)[p2l_src] + self.bc_pl_prj_src(lig_h)[p2l_dst] + self.bc_pl_prj_edge(pl_e)
        a_pl = self._edge_softmax(self.bc_pl_att(w_pl), p2l_batch, B)
        s_pl = a_pl * self.bc_pl_w_edge(pl_e) * self.bc_pl_w_dst(poc_h)[p2l_src] * self.bc_pl_w_src(lig_h)[p2l_dst]
        bias_pl_agg = torch.zeros(B, s_pl.size(-1), device=s_pl.device)
        bias_pl_agg.index_add_(0, p2l_batch, s_pl)
        bias_pl = self.bc_pl_fc(bias_pl_agg)

        return (pred_lp - bias_lp).squeeze(-1), (pred_pl - bias_pl).squeeze(-1)

    def forward(self, batch):
        pred_lp, pred_pl = self._forward_heads(batch)
        return (pred_lp + pred_pl) / 2           # average the two directional views

    def compute_loss(self, batch, labels):
        pred_lp, pred_pl = self._forward_heads(batch)
        # fit each direction to truth + force the two views to agree (consistency)
        return (F.mse_loss(pred_lp, labels) + F.mse_loss(pred_pl, labels)
                + F.mse_loss(pred_lp, pred_pl)) / 3
```

The causal chain, start to finish: binding affinity is a sum of non-covalent atom-atom contacts whose strength depends on 3D geometry, and the covalent bonds only set conformation — so I commit to two assumptions, a heterogeneous covalent/non-covalent graph and an output that is a sum of pairwise atom-atom affinities, and I make the model invariant for free by feeding it only rigid-motion invariants (angles, triangle areas, distances) instead of coordinates. Distance-only convolutions throw away most of that geometry and pool the output into a vector; the predecessor that splits the interaction types still uses only the scalar distance in its messages and still reads out by pooling node embeddings, so neither expresses the sum-over-contacts. I close that by giving each relation its own convolution — an edge-enhanced summing convolution for covalent edges, an edge-gated mean convolution for the variable-degree non-covalent contacts — combining them by summation per atom each layer, and then reading out the affinity as a literal sum of per-contact scalars, each scalar a low-rank triple product of the two atoms and the contact geometry. A bare sum carries a size- and cutoff-dependent offset, so I subtract an attention-normalized bias correction whose softmax over a complex's contacts makes it scale-stable. I compute the whole thing in both directions, average the two views at inference, and during training add a consistency term forcing the two views to agree, which regularizes by multi-view distillation — landing on a model that is invariant by construction, interpretable by the form of its output, and grounded in the physics of binding.
