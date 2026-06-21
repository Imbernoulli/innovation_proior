## Research question

Structure-based virtual screening ranks a compound library against one protein target so the few true binders sit at the very top. Libraries are $10^8$–$10^9$ molecules and the hit rate is far below 0.1%, so the whole game is *throughput at the top of the ranking*. The single thing being designed is the **scoring objective**: given fixed pretrained backbone encoders (Uni-Mol for the molecule and the pocket, ESM-2 for the protein sequence) that are fine-tuned jointly with the scoring module, how should their features be projected into a shared space, what training loss should align binders with their target, and how should a final per-molecule score be read off at evaluation. Everything else — the encoders, the data loaders, the training loop, the evaluation scripts — is fixed.

## Prior art / Background / Baselines

- **Molecular docking (Glide, AutoDock Vina, Gold).** Samples candidate ligand poses in the pocket and scores each pose with an empirical force field correlated with binding free energy. Gap: per-compound pose sampling takes roughly 10 s on a CPU core, so a $10^{10}$-compound library is thousands of years of compute — cost grows with the library, the wrong scaling for the "bigger is better" regime.
- **Supervised affinity regression (DeepDTA, GraphDTA, OnionNet).** Maps a protein-molecule representation to a numeric affinity and ranks by that value. Gap: depends on scarce affinity labels (~$10^4$ labeled complexes), sees almost no true negatives, and at inference screens $(\#\text{targets}) \times (\#\text{library})$ full network evaluations, so it does not amortize.
- **Supervised decoy classifiers (DrugVQA, AttentionSiteDTI).** Trains a binary classifier on actives versus rule-constructed decoys. Gap: the model latches onto the decoy-construction rule and fails to transfer to benchmarks built with different rules.
- **Single-tower 3D scorers.** Feeds the joint protein-ligand complex, including cross-distances, into one network. Gap: the score depends on protein-ligand cross-distances, which are only known once the ligand is posed in the pocket, so a docking step is required and docking's cost is inherited.
- **Dense retrieval and contrastive pretraining (DPR, InfoNCE, CLIP).** Embeds queries and documents independently and retrieves by dot-product similarity, training with a symmetric in-batch contrastive softmax. Gap: the softmax is dominated by easy negatives in large batches, and its global density-ratio optimum does not place the strongest binders at the very top of realistic, top-heavy benchmarks.

## Fixed substrate / Code framework

The backbone encoders and harness are frozen and must not be touched. Uni-Mol is an SE(3)-invariant 3D Transformer that returns a 512-d [CLS] vector for a molecule or pocket; ESM-2 returns a 480-d [CLS] vector for the protein sequence. These three encoders are loaded from pretrained weights and fine-tuned jointly with the scoring module. The fixed model wrapper extracts the three [CLS] features per assay, calls the module's three `project_*` methods, and hands `(prot_emb, poc_emb, mol_emb)` to the fixed loss wrapper, which calls `compute_loss`. Training data is organized **by assay**: one target paired with a contiguous block of tested ligands, with block spans, per-ligand pIC50 activities, and metadata for false-negative masking provided per batch. Evaluation calls `score` over a target's library and reports ranking metrics.

## Editable interface

Exactly one file is editable — `custom_scoring.py`, the `CustomScoring` module. Every method fills the same contract:

- `__init__(mol_dim=512, pocket_dim=512, protein_dim=480, embed_dim=128)` — projection heads and any objective parameters.
- `project_mol(mol_feat)` / `project_pocket(poc_feat)` / `project_protein(prot_feat)` — map a `[B, dim]` feature into the `[B, embed_dim]` comparison space.
- `compute_loss(mol_emb, poc_emb, prot_emb, batch_list, act_list, uniprot_poc, uniprot_mol, pocket_lig_smiles, lig_smiles)` — the training loss. `batch_list[i] = (start, end)` gives the contiguous column span of pocket `i`'s ligands in `mol_emb`; `act_list[i]` their pIC50 activities; the four metadata fields drive false-negative / duplicate masking. Returns `(loss, log_dict)`, and `log_dict["sim_masked"]` is read by the validation loop.
- `score(mol_reps, pocket_reps, prot_reps=None)` — numpy scoring for one target's library; returns one score per molecule.

The starting point is the scaffold default — a CLIP-style contrastive scorer over L2-normalized Euclidean embeddings, scored pocket↔molecule only, with the protein tower defined but unused in the loss.

```python
"""Custom scoring module for contrastive virtual screening — default fill (vanilla CLIP)."""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomScoring(nn.Module):
    """Scoring module for contrastive protein-ligand virtual screening.

    Projects frozen encoder features into a shared embedding space and computes
    the training loss for ranking actives above decoys.
    """

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

        # Symmetric contrastive loss.
        # Pocket-to-ligand: each pocket retrieves its tested ligands (multi-positive).
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

        # Ligand-to-pocket: each ligand retrieves its pocket (row-local).
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

## Evaluation settings

Zero-shot: no target-specific training. Three structure-based screening benchmarks, with metrics averaged across each benchmark's targets:

- **DUD-E** — 102 targets, actives versus property-matched ZINC decoys.
- **LIT-PCBA** — 15 targets, confirmed actives versus confirmed inactives from dose-response bioassays; built to remove DUD-E's artificial-decoy bias, so realistic and much harder.
- **DEKOIS 2.0** — 81 targets, a challenging decoy benchmark.

Metrics, all higher-is-better and all *early-recognition*: **AUROC**, **BEDROC** ($\alpha=80.5$), and **enrichment factor EF** at the top 0.5%, 1%, and 5% (also EF at 0.05%, 0.1%, 0.2%). The task score is the geometric mean across the three benchmarks of each benchmark's metric average. One seed (42).
