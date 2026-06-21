HCC's numbers confirmed the ordering diagnosis and, in the same breath, exposed the limit the ranking term cannot reach. LIT-PCBA moved exactly where I predicted — AUROC 0.576 → 0.628, BEDROC 0.065 → 0.087, EF@0.5% nearly doubling 7.27 → 11.15 — while DUD-E firmed its AUROC to 0.920 and DEKOIS held at 0.922. So activity-aware ranking plus the sequence view did attack the graded-actives failure, and the relative gains *are* concentrated on the realistic benchmark, just as the falsifiable claim said. But LIT-PCBA's 0.628 AUROC is still soft and its 0.087 BEDROC is still an order of magnitude below DUD-E. The ranking term helped, and it stopped helping while the realistic benchmark is mostly unsolved — which is the alternative diagnosis I flagged at rung two: the residual is not ordering but the *geometry*. Euclidean space collapses *activity cliffs* — two ligands that are almost the same molecule, one substituent removed, that bind tens of fold differently. A structure encoder does its job and maps near-identical molecules to near-identical vectors; in Euclidean space distance is linear, so if $\|v_1 - v_2\|$ is small in the encoder's native space it stays small in the embedding, and the pocket-to-ligand score reads off essentially the same for both. The ranking term cannot fix this: it asks the score to order the pair, but the two embeddings are so close there is no *room* to place the order without forcing structurally similar molecules to large distance, which would fight the encoder's smoothness everywhere. The ranking loss ran out of room because the space has none.

I propose Full HypSeek: embed every tower on a **Lorentz hyperboloid** with learnable curvature, keep HCC's contrastive-plus-ranking core verbatim on top of it, and add an **affinity-tiered cone hierarchy** that uses radial depth and angular position together to pull activity cliffs apart. The reason negative curvature is the right space is that volume grows exponentially with radius instead of polynomially — there is exponentially more room near the boundary than a flat space provides — and graded binding affinity *is* a hierarchy: within an assay ligands fan from weak to strong, and across the library there is a coarse-to-fine "binds this family / this target / this pocket" structure. The bet is quantitative, not a slogan. On the Lorentz model — points on the upper sheet of a hyperboloid with the Lorentzian inner product, geodesic distance $d_L(x,y) = (1/\sqrt\kappa)\,\mathrm{arccosh}(-\kappa\langle x,y\rangle_L)$, encoder outputs lifted on with the exponential map at the origin — take two ligands as tangent vectors at the origin with nearly equal radial norm $r$ and a small angle $\theta \ll 1$ between them. The hyperbolic law of cosines on a unit-curvature sheet gives $\cosh d = \cosh^2 r - \sinh^2 r\cos\theta$; expand $\cos\theta \approx 1 - \theta^2/2$ with $\cosh^2 r - \sinh^2 r = 1$, so $\cosh d - 1 \approx (\sinh^2 r/2)\,\theta^2$, invert with $\mathrm{arccosh}(1+\epsilon) = \sqrt{2\epsilon}$ to get $d \approx \sinh r\cdot\theta$, and restoring curvature,
$$d_H \approx \frac{\sinh r}{\sqrt\kappa}\,\theta.$$
The separation is $\theta$ *amplified by $\sinh r$*, growing exponentially in $r$, where Euclidean gives only $r\cdot\theta$. So a cliff pair — tiny $\theta$, very different function — can be separated if the geometry gives the stronger tier access to larger radial scale and tighter angular control, amplifying the angular sliver exactly where it matters without distorting the metric uniformly. The design reduces to making the radial coordinate carry binding-strength tier and the angular position carry identity, then letting cliffs separate through the product of the two.

To *control* radial depth and angular spread per ligand as a function of affinity I need a structured prior on the manifold, and entailment cones supply it: attach to each point a cone opening away from the origin whose half-aperture *shrinks* as the point's norm grows — a point farther out projects a narrower cone, so a pocket pushed deep toward the boundary is "more specific" and admits only a tight set of ligand directions. The aperture transfers to the Lorentz model as $\omega(x) = \arcsin(2K/(\sqrt\kappa\|x_{\text{space}}\|))$. A single on/off cone is binary, but affinity is graded, so I turn it into a hierarchy of tiers indexed by activity along *both* axes. Per pocket-ligand pair I take two measurements, and the argument order is load-bearing: the radial one is the geodesic distance $d_{ij}$ from pocket $i$ to ligand $j$; the angular one is the first-argument angle $\phi_{ij} = \mathrm{oxy\_angle}(\text{ligand}_j, \text{pocket}_i)$ in the hyperbolic triangle O-ligand-pocket, with the aperture $\omega_i$ attached to the *pocket*, so the constraint is $\phi_{ij} \le \eta_{ij}\cdot\omega_i$ — pocket supplies the aperture, ligand-first angle supplies the measured lean. I bucket each ligand's pIC50 by the standard thresholds $\{5, 7, 9\}$ (5 is the $\sim 10$ µM "active" cutoff, each step a decade) into four buckets $b \in \{0,1,2,3\}$, and set $r_k = r_0 + b\,\Delta r$ and $\eta_k = \eta_0 - b\,\Delta\eta$ with $r_0, \eta_0$ the weakest-tier base and $\Delta r, \Delta\eta > 0$. The signs are the whole point: a *stronger* binder gets a *larger* radial cap — a one-sided hinge permitting it to occupy the larger radial scales where the $\sinh r$ amplification can act — and a *smaller* angular tolerance, because a strong, specific binding event should align more decisively with the pocket's admissible direction than a weak one. Strong = larger cap, tighter cone; weak = smaller cap, wider cone — the two knobs move in opposite directions with affinity, spreading tiers by both distance and angular selectivity, exactly the two-axis separation $\sinh r\cdot\theta$ requires.

The cone losses are one-sided hinges that penalize only violations and never pull a satisfied ligand: $L_{\text{rad}} = (1/\sqrt N)\sum\max(d_{ij} - r_{ij}, 0)$ and $L_{\text{ang}} = (1/\sqrt N)\sum\max(\phi_{ij} - \eta_{ij}\omega_i, 0)$, combined as $\lambda_{\text{rad}}L_{\text{rad}} + \lambda_{\text{ang}}L_{\text{ang}}$ with both weights $0.5$ — two halves of the same cone, no reason to prefer one. The $1/\sqrt N$ is the same assay-size discipline I have used since rung one. Two regularizers fall out of what can go wrong. The angular hinge is zero the instant $\phi \le \eta\omega$, so the optimizer has no pressure past touching the boundary and could collapse angles toward the axis trivially; a margin $m = 0.15$ rad — $R_{\text{ang}} = (1/\sqrt N)\sum\max(\phi - \eta\omega + m, 0)$ — keeps pushing until ligands sit decisively *inside* the cone. And because the metric is dominated by the very top of the list, a heterogeneity term weights threshold-selected entries by distance rank with $w_j = \exp(-\beta(\text{rank}_j - 1)/L_i)$ at $\beta = 80.5$ — the *same* focus parameter as BEDROC, shaping training to the evaluation's early-enrichment emphasis. Both regularizers are auxiliary at weight $0.10$.

The cone hierarchy rides on top of a retrieval objective that still has to find binders at all, so I keep HCC's contrastive-plus-ranking helper applied to the hyperbolic embeddings, run on both the pocket and the sequence query views, but apply the cone supervision *only* to the pocket branch — the pocket carries the structural, geometrically meaningful signal, and over-constraining the auxiliary sequence view geometrically would fight its role. Two scaffold details are load-bearing here. First, geodesic distance is the "correct" hyperbolic similarity, but at inference I need a plain dot product so retrieval stays a matmul over a cached matrix — non-negotiable, the whole reason screening is feasible — so I score with the inner product of the *spatial* components both inside the training softmax and at inference, and let the cone losses do the geometric shaping. The exp map returns only space components and the implementation treats the projector output as $[\text{lead}, \text{space}\ldots]$, dropping index 0 with `emb[:, 1:]` before the similarity so the manifold coordinates stay aligned; this is the reserved-coordinate convention that was inert in the flat rung and is now exactly what keeps the contrastive and cone math consistent. At *inference* the score uses the full embedding dot product (`pocket_reps @ mol_reps.T`, max over the target's pockets, plus the sequence contribution), because the geometry was already paid for at training time. Second, the exp map has initialization traps. CLIP-style init makes the Euclidean head output have norm $\approx\sqrt n$, so $\sinh(\sqrt\kappa\sqrt{128})$ is astronomical and training diverges immediately; the fix is a learnable per-tower scale $\alpha$ initialized to $1/\sqrt n = 128^{-1/2}$ so the scaled embedding has expected unit norm at init, learned in log space and clamped so $\exp(\alpha) \le 1$ — it can shrink but never blow the exp map back up. Each projection is $u = \text{head}(\text{feat})\cdot\exp(\alpha)$ then $h = \mathrm{exp\_map0}(u, \kappa)$, and the curvature $\kappa$ is itself learnable (init $\log 1$) clamped to $[\log 0.1, \log 10]$ so it can neither collapse to Euclidean nor detonate. The contrastive softmax keeps the detached $\log 13$ inverse temperature. The total is $\alpha_{\text{poc}}\text{loss}_{\text{poc}} + \alpha_{\text{prot}}\text{loss}_{\text{seq}} + \gamma_{\text{cone}}L_{\text{cone}} + \lambda_{\text{het}}R_{\text{het}} + \lambda_{\text{ang}}R_{\text{ang}}$ with $\gamma_{\text{cone}} = 0.1$, down-weighted because the cone is a structural prior shaping the space, not the primary retrieval signal — if it dominated, the model would satisfy the geometry at the expense of actually finding binders.

The falsifiable claims are specifically about the realistic benchmark, because that is where cliffs live. LIT-PCBA's AUROC and BEDROC should rise *again* past HCC's 0.628 / 0.087 — if the $\sinh r$ room is real, the model should now separate the near-identical-but-different-affinity pairs the Euclidean ranking term could not place — and DUD-E, already near its ceiling at 0.920, should hold or edge up on the early-enrichment metrics where the cone's head-of-list shaping helps. The honest risk is that the cone is a strong prior on top of already-strong features, so on a benchmark with easy decoys (DEKOIS) the extra geometric constraint is unnecessary and I would not be surprised to see it hold roughly flat. The claim that decides whether the geometry was worth it is DUD-E's BEDROC/EF and LIT-PCBA's AUROC moving up together — separation where it was already good, plus the cliff room the flat space never had.

```python
"""Full HypSeek scoring: Hyperbolic HCC + Cone Hierarchy."""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from unimol.losses import lorentz as L


class CustomScoring(nn.Module):
    """Full HypSeek: Lorentz hyperbolic embeddings + HCC + cone hierarchy.

    Maps projected features onto a Lorentz hyperboloid via exp_map0,
    trains with HCC contrastive-ranking loss plus cone hierarchy
    constraints (radial + angular).
    """

    def __init__(self, mol_dim=512, pocket_dim=512, protein_dim=480, embed_dim=128):
        super().__init__()
        # Projection heads (NonLinearHead equivalent: hidden=input_dim)
        self.mol_project = nn.Sequential(
            nn.Linear(mol_dim, mol_dim), nn.ReLU(), nn.Linear(mol_dim, embed_dim)
        )
        self.pocket_project = nn.Sequential(
            nn.Linear(pocket_dim, pocket_dim), nn.ReLU(), nn.Linear(pocket_dim, embed_dim)
        )
        self.protein_project = nn.Sequential(
            nn.Linear(protein_dim, protein_dim), nn.ReLU(), nn.Linear(protein_dim, embed_dim)
        )

        # Learnable scale parameters (log-space, clamped to exp(alpha) <= 1)
        self.mol_alpha = nn.Parameter(torch.tensor([embed_dim ** -0.5]).log())
        self.pocket_alpha = nn.Parameter(torch.tensor([embed_dim ** -0.5]).log())
        self.protein_alpha = nn.Parameter(torch.tensor([embed_dim ** -0.5]).log())

        # Learnable curvature (log-space)
        self.curv = nn.Parameter(torch.tensor([1.0]).log(), requires_grad=True)
        self._curv_minmax = {"max": math.log(10.0), "min": math.log(0.1)}

        # Temperature
        self.logit_scale = nn.Parameter(torch.ones([1]) * np.log(13))

        # Cone hierarchy hyperparameters
        self.bounds = torch.tensor([5.0, 7.0, 9.0], dtype=torch.float32)
        self.chl_r0 = 0.5
        self.chl_dr = 0.5
        self.chl_eta0 = 0.7
        self.chl_deta = 0.2
        self.lambda_rad = 0.5
        self.lambda_ang = 0.5
        self.gamma_chl = 0.1
        self.lambda_angu = 0.10
        self.lambda_het = 0.10

    def _clamp_params(self):
        """Clamp scale and curvature parameters."""
        self.mol_alpha.data = torch.clamp(self.mol_alpha.data, max=0.0)
        self.pocket_alpha.data = torch.clamp(self.pocket_alpha.data, max=0.0)
        self.protein_alpha.data = torch.clamp(self.protein_alpha.data, max=0.0)
        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)

    def _project_to_hyperboloid(self, feat, proj_head, alpha):
        """Project features to Lorentz hyperboloid."""
        u = proj_head(feat) * alpha.exp()
        with torch.autocast(u.device.type, dtype=torch.float32):
            h = L.exp_map0(u, self.curv.exp())
        return h

    def project_mol(self, mol_feat):
        self._clamp_params()
        return self._project_to_hyperboloid(mol_feat, self.mol_project, self.mol_alpha)

    def project_pocket(self, poc_feat):
        self._clamp_params()
        return self._project_to_hyperboloid(poc_feat, self.pocket_project, self.pocket_alpha)

    def project_protein(self, prot_feat):
        self._clamp_params()
        return self._project_to_hyperboloid(prot_feat, self.protein_project, self.protein_alpha)

    def _compute_hcc_pair(self, emb_poc, emb_mol, batch_list, act_list,
                          uniprot_poc, uniprot_mol, pocket_lig_smiles, lig_smiles,
                          logit_scale):
        """HCC loss for one pathway (space component dot product)."""
        B = emb_poc.size(0)
        emb_poc = emb_poc[:, 1:]
        emb_mol = emb_mol[:, 1:]
        logits = torch.matmul(emb_poc, emb_mol.T) * logit_scale

        N_mol = emb_mol.size(0)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        if uniprot_poc is not None and uniprot_mol is not None:
            for i in range(B):
                for j in range(N_mol):
                    if uniprot_poc[i] == uniprot_mol[j]:
                        mask[i, j] = True
        if pocket_lig_smiles is not None:
            for i in range(B):
                bad = pocket_lig_smiles[i]
                for j in range(N_mol):
                    if lig_smiles[j] in bad:
                        mask[i, j] = True

        minus_inf = torch.finfo(logits.dtype).min
        sim_masked = logits.masked_fill(mask, minus_inf)

        # Pocket retrieves ligands
        loss_mol_list, loss_rank_list = [], []
        for i in range(B):
            s, e = batch_list[i]
            acts = act_list[i]
            L_i = e - s
            out_i = sim_masked[i, s:e]
            for k in range(s, e):
                row_mask = torch.full_like(sim_masked[i], minus_inf)
                row_mask[k] = 0
                lprobs = F.log_softmax(row_mask + sim_masked[i], dim=-1)
                if L_i > 1 and acts[k - s] < 5:
                    continue
                loss_mol_list.append(-lprobs[k] / math.sqrt(L_i))
            if L_i > 2:
                for k_rel in range(L_i - 1):
                    m = torch.zeros_like(out_i)
                    for idx in range(L_i):
                        if idx == k_rel:
                            continue
                        if acts[k_rel] - math.log10(3) <= acts[idx]:
                            m[idx] = minus_inf
                    lprobs_rank = F.log_softmax(m + out_i, dim=-1)
                    loss_rank_list.append(-lprobs_rank[k_rel] / (math.log(k_rel + 2) * math.sqrt(L_i)))
        loss_mol = torch.stack(loss_mol_list).sum() if loss_mol_list else torch.tensor(0.0, device=logits.device)
        loss_rank = torch.stack(loss_rank_list).sum() if loss_rank_list else torch.tensor(0.0, device=logits.device)

        # Ligand-to-pocket
        idx2poc = []
        for i, (s, e) in enumerate(batch_list):
            idx2poc += [i] * (e - s)
        targets = torch.tensor(idx2poc, dtype=torch.long, device=logits.device)
        lprobs_pocket_all = F.log_softmax(sim_masked.T, dim=-1)
        loss_pocket_list = []
        for i, (s, e) in enumerate(batch_list):
            L_i = e - s
            if L_i == 0:
                continue
            rows = list(range(s, e))
            lprobs_sub = lprobs_pocket_all[rows]
            targ_sub = targets[rows]
            loss_tmp = F.nll_loss(lprobs_sub, targ_sub, reduction="none")
            loss_pocket_list.append(loss_tmp.sum() / math.sqrt(L_i))
        loss_pocket = torch.stack(loss_pocket_list).sum() if loss_pocket_list else torch.tensor(0.0, device=logits.device)

        total = loss_pocket + loss_mol + loss_rank
        return {"loss": total, "loss_pocket": loss_pocket, "loss_mol": loss_mol,
                "loss_rank": loss_rank, "sim_masked": sim_masked}

    def compute_loss(self, mol_emb, poc_emb, prot_emb,
                     batch_list, act_list,
                     uniprot_poc=None, uniprot_mol=None,
                     pocket_lig_smiles=None, lig_smiles=None):
        kappa = self.curv.exp().detach()
        logit_scale = self.logit_scale.exp().detach()
        B = poc_emb.size(0)

        # === Cone Hierarchy Loss ===
        poc_space = poc_emb[:, 1:]
        lig_space = mol_emb[:, 1:]
        poc_idx = []
        for i, (s, e) in enumerate(batch_list):
            poc_idx += [i] * (e - s)
        poc_idx = torch.tensor(poc_idx, device=poc_emb.device)

        poc_sel = poc_space[poc_idx]
        dist_mat = L.pairwise_dist(poc_sel, lig_space, curv=kappa)
        dist = dist_mat.diagonal()
        device = dist.device
        phi = L.oxy_angle(lig_space, poc_space[poc_idx], curv=kappa)
        omega = L.half_aperture(poc_space[poc_idx], curv=kappa)
        act_flat = torch.tensor(
            [x for sub in act_list for x in sub],
            device=poc_emb.device, dtype=torch.float32,
        )
        bounds = self.bounds.to(poc_emb.device)
        bucket = torch.bucketize(act_flat, bounds)
        r_k = self.chl_r0 + bucket.float() * self.chl_dr
        eta_k = self.chl_eta0 - bucket.float() * self.chl_deta
        Nl = dist.size(0)
        L_rad = F.relu(dist - r_k).sum() / math.sqrt(Nl)
        L_ang = F.relu(phi - eta_k * omega).sum() / math.sqrt(Nl)
        loss_cone = self.lambda_rad * L_rad + self.lambda_ang * L_ang

        # Angular regularization
        m_margin = 0.15
        R_ang = F.relu(phi - eta_k * omega + m_margin).sum() / math.sqrt(Nl)

        # Heterogeneous ranking regularization
        R_het = torch.zeros(1, device=device)
        cnt_het = 0
        beta = 80.5
        offset = 0
        for i_poc, (s, e) in enumerate(batch_list):
            L_i = e - s
            if L_i < 1:
                continue
            d_i = dist[offset : offset + L_i].detach()
            rank = (d_i.unsqueeze(0) < d_i.unsqueeze(1)).float().sum(1) + 1
            w = torch.exp(-beta * (rank - 1) / L_i)
            logits_row = torch.matmul(poc_space[i_poc : i_poc + 1], lig_space.T) * logit_scale
            row_probs = F.softmax(logits_row[0, s:e], dim=-1)
            pos_mask = act_flat[offset : offset + L_i] < 5
            if pos_mask.any():
                R_het += -(w[pos_mask] * row_probs[pos_mask].log()).sum() / (w[pos_mask].sum() + 1e-9)
                cnt_het += 1
            offset += L_i
        R_het = R_het / max(cnt_het, 1)
        loss_reg = self.lambda_het * R_het + self.lambda_angu * R_ang

        # === HCC for both pathways ===
        loss_dict_poc = self._compute_hcc_pair(
            poc_emb, mol_emb, batch_list, act_list,
            uniprot_poc, uniprot_mol, pocket_lig_smiles, lig_smiles, logit_scale,
        )
        loss_dict_prot = self._compute_hcc_pair(
            prot_emb, mol_emb, batch_list, act_list,
            uniprot_poc, uniprot_mol, pocket_lig_smiles, lig_smiles, logit_scale,
        )

        loss_hcc = loss_dict_poc["loss"] + loss_dict_prot["loss"]
        total_loss = loss_hcc + self.gamma_chl * loss_cone + loss_reg

        return total_loss, {
            "loss": total_loss.item(),
            "loss_hcc": loss_hcc.item(),
            "loss_cone": loss_cone.item(),
            "loss_reg": loss_reg.item(),
            "sim_masked": loss_dict_poc["sim_masked"],
        }

    def score(self, mol_reps, pocket_reps, prot_reps=None):
        """Score using full 128-d hyperbolic embedding (paper convention)."""
        poc_scores = (pocket_reps @ mol_reps.T).max(axis=0)
        if prot_reps is not None:
            prot_scores = (prot_reps @ mol_reps.T).max(axis=0)
            return poc_scores + prot_scores
        return poc_scores
```
