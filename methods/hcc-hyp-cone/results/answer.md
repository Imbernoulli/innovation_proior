# HypSeek, distilled

HypSeek is a retrieval-based protein–ligand virtual-screening model that embeds ligands,
protein pockets, and protein sequences into **Lorentz-model hyperbolic space** and trains them
with a hierarchical contrastive-ranking objective (HCC) plus an **entailment-cone hierarchy**.
The negative curvature gives the embedding room to separate **activity cliffs** — structurally
near-identical ligands with large affinity gaps — that a flat Euclidean space cannot, because a
small angular difference at large radial depth becomes an exponentially large geodesic
distance. Inference stays a cached-matrix dot product, so screening scales to huge libraries.

## Problem it solves

Contrastive dense-retrieval screening (DrugCLIP-style) in Euclidean space cannot separate
activity-cliff pairs (similar structure → near-identical embedding → near-identical score) and
has no mechanism for graded, within-assay affinity ranking. HypSeek replaces the geometry and
the objective so the embedding encodes binding strength radially and identity angularly.

## Key idea

Work on the Lorentz hyperboloid `L^n` (curvature `−κ`): `⟨x,y⟩_L = ⟨x_space,y_space⟩ −
x_time·y_time`, `x_time = sqrt(1/κ + ||x_space||²)`, geodesic distance `d_L(x,y) =
(1/√κ)·arccosh(−κ⟨x,y⟩_L)`. Lift each Euclidean encoder output onto the manifold with the
exponential map at the origin (space-only parameterization), then shape the space with:

- a **cone hierarchy** that ties each ligand's allowed radial depth and angular cone width to
  its affinity tier, exploiting the exponential angular amplification of curvature;
- DrugCLIP's symmetric in-batch contrastive loss + a Plackett–Luce listwise ranking term,
  applied on a pocket-query pathway and an auxiliary protein-sequence pathway.

### Why hyperbolic separates cliffs (the load-bearing derivation)

Two embeddings as tangent vectors at the origin, equal curvature-normalized radial coordinate `r`,
small included angle `θ`. Hyperbolic law of cosines (unit curvature):
`cosh d = cosh²r − sinh²r·cos θ`. Small-`θ`:
`cos θ ≈ 1 − θ²/2`, and `cosh²r − sinh²r = 1`, so `cosh d − 1 ≈ (sinh²r/2)θ²`. Invert with
`arccosh(1+ε) = √(2ε) + O(ε^{3/2})`, `ε = (sinh²r/2)θ²`, giving `√(2ε) = sinh r·θ`. Restoring κ:

```
d_H(exp_o(v_1), exp_o(v_2)) ≈ (sinh r / √κ)·θ + O(θ³).
```

Euclidean gives `≈ r·θ` (linear in r); hyperbolic gives `sinh r·θ` (exponential in r). So depth
amplifies a tiny angular wiggle into a large distance — separate cliffs without distorting local
similarity. (Activity-cliff separation proposition.)

## The geometry primitives (Lorentz model; from MERU)

For `x, y` given by their space components (time recovered from the constraint):

```
# geodesic distance
d_L(x,y) = (1/√κ)·arccosh( −κ·⟨x,y⟩_L ),   ⟨x,y⟩_L = ⟨x_space,y_space⟩ − x_time·y_time

# exponential map at origin (space-only)
exp_0^κ(v)_space = sinh(√κ·||v||)/(√κ·||v||) · v

# cone half-aperture (narrows as ||x|| grows); K = 0.1 (min_radius), via Ganea/MERU
ω(x) = arcsin( 2K / (√κ·||x_space||) )

# exterior angle (hyperbolic law of cosines): how far y leans off x's axis
φ(x,y) = arccos( (y_time + x_time·κ⟨x,y⟩_L) / (||x_space||·sqrt((κ⟨x,y⟩_L)² − 1)) )
```

## Training objective

Per ligand, bucket affinity `v` by thresholds `[5,7,9]` (pIC50 tiers) → `b ∈ {0,1,2,3}`; set

```
r_k = r_0 + b·Δr   (stronger binder → larger pocket-ligand radial cap),
η_k = η_0 − b·Δη   (stronger binder → tighter angular tolerance),
```

with `r_0=0.5`, `Δr=0.5`, `η_0=0.7`, `Δη=0.2`. Cone losses (one-sided hinges over a
pocket `i`, ligand `j`; `d_{ij}` = geodesic distance, `φ_{ij}=oxy_angle(ligand_j,pocket_i)` in
the implementation, `ω_i` = pocket half-aperture; `N` = #ligands):

```
L_rad  = (1/√N) Σ max(d_{ij} − r_{ij}, 0)
L_ang  = (1/√N) Σ max(φ_{ij} − η_{ij}·ω_i, 0)
L_cone = λ_rad·L_rad + λ_ang·L_ang             (λ_rad = λ_ang = 0.5)
```

Regularizers:

```
R_ang = (1/√N) Σ max(φ_{ij} − η_{ij}·ω_i + m, 0)     margin m = 0.15  (decisively inside cone)
R_het = (1/C) Σ_assay Σ_{j:v_{ij}<v_th} −w_j·log p_{ij},  w_j = exp(−β(rank_j−1)/L_i),  β = 80.5
```

`R_het` uses the implementation's literal threshold mask `v<5` and weights selected entries with
the same focus parameter as the BEDROC_80.5 metric; `p_{ij}` is the intra-assay softmax retrieval
probability; `C` = #assays with a nonempty threshold mask.

HCC contrastive + ranking core (per query pathway `u ∈ {pocket, seq}`, logits =
`u_space · mol_space · exp(logit_scale)`, with same-target and duplicate false-negative masks):

- **pocket→ligand**: one-positive InfoNCE per true binder (skip weak binders, pIC50<5, when
  the assay has >1 ligand), each `−log p` normalized by `1/√L_i`;
- **listwise ranking** (assays with >2 ligands): for ligand `k` in affinity order, mask ligands
  *not* at least 3-fold weaker (`act[k] − log10(3) ≤ act[idx]`), `−log p` with decay
  `1/(log(k+2)·√L_i)`;
- **ligand→pocket**: each ligand retrieves its own pocket (NLL over `sim.T`), `/√L_i`.

Total:

```
L_total = α_poc·(pocket pathway) + α_prot·(seq pathway)
          + γ_cone·L_cone + λ_het·R_het + λ_ang(reg)·R_ang
          α_poc = α_prot = 1 by default,  γ_cone = 0.1,  λ_het = λ_ang(reg) = 0.10
```

Hyperbolic cone supervision is applied to the **pocket** branch only; the sequence branch is an
auxiliary contrastive+ranking signal and is not geometrically constrained. Curvature `κ`,
per-tower scale `α` (clamped `exp(α) ≤ 1` to keep the exp map from
overflowing; init `1/√d`), and `logit_scale` (init `log 13`) are learnable.

## Inference

Embed the query pocket (and optionally the sequence) and each candidate ligand. The Lorentz helper
returns the 128-d space-component vector; evaluation scores by the full returned-vector **dot
product** — `s_j = h_poc · h_{mol,j}`, max over a target's candidate pockets, optionally plus the
sequence contribution — then ranks descending.
Training uses the on-manifold geometry; retrieval is a cheap matmul over cached embeddings.

## Working code

Fills the `CustomScoring` stubs from the scaffold: projection heads + geometry params in
`__init__`, the three `project_*` lifts, the HCC+cone objective in `compute_loss`, the matmul
`score`. Uses the Lorentz primitives `exp_map0`, `pairwise_dist`, `oxy_angle`, `half_aperture`.

```python
"""Full hyperbolic scoring: HCC contrastive-ranking + cone hierarchy on the Lorentz model."""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from unimol.losses import lorentz as L


class CustomScoring(nn.Module):
    def __init__(self, mol_dim=512, pocket_dim=512, protein_dim=480, embed_dim=128):
        super().__init__()
        self.mol_project = nn.Sequential(
            nn.Linear(mol_dim, mol_dim), nn.ReLU(), nn.Linear(mol_dim, embed_dim)
        )
        self.pocket_project = nn.Sequential(
            nn.Linear(pocket_dim, pocket_dim), nn.ReLU(), nn.Linear(pocket_dim, embed_dim)
        )
        self.protein_project = nn.Sequential(
            nn.Linear(protein_dim, protein_dim), nn.ReLU(), nn.Linear(protein_dim, embed_dim)
        )
        self.mol_alpha = nn.Parameter(torch.tensor([embed_dim ** -0.5]).log())
        self.pocket_alpha = nn.Parameter(torch.tensor([embed_dim ** -0.5]).log())
        self.protein_alpha = nn.Parameter(torch.tensor([embed_dim ** -0.5]).log())
        self.curv = nn.Parameter(torch.tensor([1.0]).log(), requires_grad=True)
        self._curv_minmax = {"max": math.log(10.0), "min": math.log(0.1)}
        self.logit_scale = nn.Parameter(torch.ones([1]) * np.log(13))
        self.alpha_poc, self.alpha_prot = 1.0, 1.0
        self.bounds = torch.tensor([5.0, 7.0, 9.0], dtype=torch.float32)
        self.chl_r0, self.chl_dr = 0.5, 0.5
        self.chl_eta0, self.chl_deta = 0.7, 0.2
        self.lambda_rad, self.lambda_ang = 0.5, 0.5
        self.gamma_chl = 0.1
        self.lambda_angu, self.lambda_het = 0.10, 0.10

    def _clamp_params(self):
        self.mol_alpha.data = torch.clamp(self.mol_alpha.data, max=0.0)
        self.pocket_alpha.data = torch.clamp(self.pocket_alpha.data, max=0.0)
        self.protein_alpha.data = torch.clamp(self.protein_alpha.data, max=0.0)
        self.curv.data = torch.clamp(self.curv.data, **self._curv_minmax)

    def _project_to_hyperboloid(self, feat, proj_head, alpha):
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
                    loss_rank_list.append(
                        -lprobs_rank[k_rel] / (math.log(k_rel + 2) * math.sqrt(L_i))
                    )
        loss_mol = torch.stack(loss_mol_list).sum() if loss_mol_list else torch.tensor(0.0, device=logits.device)
        loss_rank = torch.stack(loss_rank_list).sum() if loss_rank_list else torch.tensor(0.0, device=logits.device)

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
            loss_tmp = F.nll_loss(lprobs_pocket_all[rows], targets[rows], reduction="none")
            loss_pocket_list.append(loss_tmp.sum() / math.sqrt(L_i))
        loss_pocket = torch.stack(loss_pocket_list).sum() if loss_pocket_list else torch.tensor(0.0, device=logits.device)

        total = loss_pocket + loss_mol + loss_rank
        return {"loss": total, "loss_pocket": loss_pocket, "loss_mol": loss_mol,
                "loss_rank": loss_rank, "sim_masked": sim_masked}

    def compute_loss(self, mol_emb, poc_emb, prot_emb, batch_list, act_list,
                     uniprot_poc=None, uniprot_mol=None,
                     pocket_lig_smiles=None, lig_smiles=None):
        kappa = self.curv.exp().detach()
        logit_scale = self.logit_scale.exp().detach()

        poc_space = poc_emb[:, 1:]
        lig_space = mol_emb[:, 1:]
        poc_idx = []
        for i, (s, e) in enumerate(batch_list):
            poc_idx += [i] * (e - s)
        poc_idx = torch.tensor(poc_idx, device=poc_emb.device)

        poc_sel = poc_space[poc_idx]
        dist = L.pairwise_dist(poc_sel, lig_space, curv=kappa).diagonal()
        device = dist.device
        phi = L.oxy_angle(lig_space, poc_space[poc_idx], curv=kappa)
        omega = L.half_aperture(poc_space[poc_idx], curv=kappa)

        act_flat = torch.tensor([x for sub in act_list for x in sub],
                                device=poc_emb.device, dtype=torch.float32)
        bounds = self.bounds.to(poc_emb.device)
        bucket = torch.bucketize(act_flat, bounds)
        r_k = self.chl_r0 + bucket.float() * self.chl_dr
        eta_k = self.chl_eta0 - bucket.float() * self.chl_deta
        Nl = dist.size(0)
        L_rad = F.relu(dist - r_k).sum() / math.sqrt(Nl)
        L_ang = F.relu(phi - eta_k * omega).sum() / math.sqrt(Nl)
        loss_cone = self.lambda_rad * L_rad + self.lambda_ang * L_ang

        m_margin = 0.15
        R_ang = F.relu(phi - eta_k * omega + m_margin).sum() / math.sqrt(Nl)

        R_het = torch.zeros(1, device=device)
        cnt_het = 0
        beta = 80.5
        offset = 0
        for i_poc, (s, e) in enumerate(batch_list):
            L_i = e - s
            if L_i < 1:
                continue
            d_i = dist[offset:offset + L_i].detach()
            rank = (d_i.unsqueeze(0) < d_i.unsqueeze(1)).float().sum(1) + 1
            w = torch.exp(-beta * (rank - 1) / L_i)
            logits_row = torch.matmul(poc_space[i_poc:i_poc + 1], lig_space.T) * logit_scale
            row_probs = F.softmax(logits_row[0, s:e], dim=-1)
            pos_mask = act_flat[offset:offset + L_i] < 5
            if pos_mask.any():
                R_het += -(w[pos_mask] * row_probs[pos_mask].log()).sum() / (w[pos_mask].sum() + 1e-9)
                cnt_het += 1
            offset += L_i
        R_het = R_het / max(cnt_het, 1)
        loss_reg = self.lambda_het * R_het + self.lambda_angu * R_ang

        loss_dict_poc = self._compute_hcc_pair(
            poc_emb, mol_emb, batch_list, act_list,
            uniprot_poc, uniprot_mol, pocket_lig_smiles, lig_smiles, logit_scale,
        )
        loss_dict_prot = self._compute_hcc_pair(
            prot_emb, mol_emb, batch_list, act_list,
            uniprot_poc, uniprot_mol, pocket_lig_smiles, lig_smiles, logit_scale,
        )
        loss_hcc = self.alpha_poc * loss_dict_poc["loss"] + self.alpha_prot * loss_dict_prot["loss"]

        total_loss = loss_hcc + self.gamma_chl * loss_cone + loss_reg
        return total_loss, {
            "loss": total_loss.item(),
            "loss_hcc": loss_hcc.item(),
            "loss_cone": loss_cone.item(),
            "loss_reg": loss_reg.item(),
            "sim_masked": loss_dict_poc["sim_masked"],
        }

    def score(self, mol_reps, pocket_reps, prot_reps=None):
        poc_scores = (pocket_reps @ mol_reps.T).max(axis=0)
        if prot_reps is not None:
            prot_scores = (prot_reps @ mol_reps.T).max(axis=0)
            return poc_scores + prot_scores
        return poc_scores
```

## Relation to prior methods

- **DrugCLIP** = the Euclidean contrastive retrieval core (symmetric in-batch InfoNCE, dot-product
  cached retrieval). HypSeek keeps it but on the hyperboloid and adds graded ranking + cone tiers.
- **Listwise ranking (Plackett–Luce / LigUnity)** = the within-assay affinity-order term.
- **MERU** = the Lorentz lift (clamped learnable scale + exp map, learnable curvature) and the
  entailment cone (half-aperture, exterior angle). HypSeek turns MERU's single binary cone into an
  affinity-tiered radial+angular hierarchy and adds the screening-specific masks, pathways, and
  early-enrichment regularizer.
- **Ganea et al. entailment cones** = the origin of the half-aperture and the partial-order prior.
