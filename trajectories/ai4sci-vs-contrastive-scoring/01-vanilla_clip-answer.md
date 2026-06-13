**Problem.** Rank a $10^8$–$10^9$ molecule library against a target so true binders sit at the very
top (hit rate < 0.1%, metrics are top-heavy: BEDROC $\alpha$=80.5, EF@0.5/1/5%). Docking does not
amortize, regression needs scarce labels and per-pair passes, decoy classifiers overfit the
construction rule. The scoring module must be scalable, label-light, and good at the top of the list.

**Key idea (the floor).** Reframe screening as *retrieval*: a factorized score
$s(p,m)=\hat g(p)\cdot\hat f(m)$ over two independent towers, so the library embeds once and a new
target is a cached nearest-neighbor search. Train with a symmetric in-batch contrastive softmax — the
savage hit rate makes every other in-batch molecule a free, realistic negative; the softmax optimum is
a density ratio and its MI bound tightens with batch size. Embeddings come from one-hidden-layer
NonLinearHeads, L2-normalized (cosine), with a detached log-scale temperature so the loss cannot game
the scale.

**Adapted to this scaffold.** The data is grouped *by assay* (`batch_list` spans, `act_list`
activities), not clean one-positive pairs, so the diagonal target becomes a multi-positive **column**
term (each ligand column points at its owning pocket) plus a row-local selected-pair NLL; a
**false-negative mask** (same UniProt, or a known binder) drops true binders out of every softmax; each
per-pocket term is $\sqrt{L_i}$-normalized so big assays count more but not linearly. Evaluation scores
by max-over-pockets dot product.

**Deliberately left on the table (sets up rung 2).** The loss is **pocket↔molecule only** — the ESM-2
protein tower is projected but unused — and nothing is **activity-aware**: every active is an equal
positive, so strong binders are not pushed above weak ones. That is what makes this the weakest rung.

**Hyperparameters.** `embed_dim=128`; NonLinearHead `Linear(d,d)→ReLU→Linear(d,128)`; `logit_scale`
init `log(13)`, detached in the loss; per-pocket $1/\sqrt{L_i}$ normalization.

**What to watch.** Solid AUROC and DUD-E EF (property-matched decoys separate cleanly); weakest on
LIT-PCBA (graded, realistic actives), where the missing ordering signal leaves BEDROC/EF on the table.
That ordering gap, plus the unused activity numbers and sequence tower, forces rung 2.

```python
"""Custom scoring module for contrastive virtual screening — vanilla CLIP (default fill)."""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomScoring(nn.Module):
    """Scoring module for contrastive protein-ligand virtual screening."""

    def __init__(self, mol_dim=512, pocket_dim=512, protein_dim=480, embed_dim=128):
        super().__init__()
        # NonLinearHead pattern: Linear(in,in) -> ReLU -> Linear(in,embed_dim).
        self.mol_project = nn.Sequential(
            nn.Linear(mol_dim, mol_dim), nn.ReLU(), nn.Linear(mol_dim, embed_dim)
        )
        self.pocket_project = nn.Sequential(
            nn.Linear(pocket_dim, pocket_dim), nn.ReLU(), nn.Linear(pocket_dim, embed_dim)
        )
        self.protein_project = nn.Sequential(
            nn.Linear(protein_dim, protein_dim), nn.ReLU(), nn.Linear(protein_dim, embed_dim)
        )
        # Learnable temperature (log scale); init log(13) ~ CLIP-style.
        self.logit_scale = nn.Parameter(torch.ones([1]) * np.log(13))

    def project_mol(self, mol_feat):
        return F.normalize(self.mol_project(mol_feat), dim=-1)

    def project_pocket(self, poc_feat):
        return F.normalize(self.pocket_project(poc_feat), dim=-1)

    def project_protein(self, prot_feat):
        return F.normalize(self.protein_project(prot_feat), dim=-1)

    def compute_loss(self, mol_emb, poc_emb, prot_emb,
                     batch_list, act_list,
                     uniprot_poc=None, uniprot_mol=None,
                     pocket_lig_smiles=None, lig_smiles=None):
        logit_scale = self.logit_scale.exp().detach()
        B = poc_emb.size(0)

        # Similarity matrix: [N_poc, N_mol]
        logits = poc_emb @ mol_emb.T * logit_scale

        # False-negative mask: same target, or a known binder of this pocket.
        mask = torch.zeros_like(logits, dtype=torch.bool)
        if uniprot_poc is not None and uniprot_mol is not None:
            for i in range(B):
                for j in range(logits.size(1)):
                    if uniprot_poc[i] == uniprot_mol[j]:
                        mask[i, j] = True
        if pocket_lig_smiles is not None:
            for i in range(B):
                bad = pocket_lig_smiles[i]
                for j in range(logits.size(1)):
                    if lig_smiles[j] in bad:
                        mask[i, j] = True

        minus_inf = torch.finfo(logits.dtype).min
        sim_masked = logits.masked_fill(mask, minus_inf)

        # === Symmetric contrastive loss ===
        # Pocket-to-ligand (multi-positive column term): each pocket owns its ligand columns.
        idx2poc = []
        for i, (s, e) in enumerate(batch_list):
            idx2poc += [i] * (e - s)
        targets = torch.tensor(idx2poc, dtype=torch.long, device=logits.device)

        lprobs_pocket = F.log_softmax(sim_masked.T, dim=-1)
        loss_pocket_list = []
        for i, (s, e) in enumerate(batch_list):
            L_i = e - s
            if L_i == 0:
                continue
            rows = list(range(s, e))
            lprobs_sub = lprobs_pocket[rows]
            targ_sub = targets[rows]
            loss_tmp = F.nll_loss(lprobs_sub, targ_sub, reduction="none")
            loss_pocket_list.append(loss_tmp.sum() / math.sqrt(L_i))
        loss_pocket = torch.stack(loss_pocket_list).sum() if loss_pocket_list else torch.tensor(0.0, device=logits.device)

        # Ligand-to-pocket (row-local selected-pair NLL): each ligand retrieves its pocket.
        loss_mol_list = []
        for i in range(B):
            s, e = batch_list[i]
            for k in range(s, e):
                row_mask = torch.full_like(sim_masked[i], minus_inf)
                row_mask[k] = 0
                lprobs = F.log_softmax(row_mask + sim_masked[i], dim=-1)
                loss_mol_list.append(-lprobs[k] / math.sqrt(e - s))
        loss_mol = torch.stack(loss_mol_list).sum() if loss_mol_list else torch.tensor(0.0, device=logits.device)

        loss = loss_pocket + loss_mol

        return loss, {
            "loss": loss.item(),
            "loss_pocket": loss_pocket.item(),
            "loss_mol": loss_mol.item(),
            "sim_masked": sim_masked,
        }

    def score(self, mol_reps, pocket_reps, prot_reps=None):
        poc_scores = (pocket_reps @ mol_reps.T).max(axis=0)
        if prot_reps is not None:
            prot_scores = (prot_reps @ mol_reps.T).max(axis=0)
            return poc_scores + prot_scores
        return poc_scores
```
