# HCC, distilled

HCC here is the **Euclidean** contrastive + ranking scoring objective for retrieval-based virtual
screening. On top of fixed, jointly fine-tuned backbones (Uni-Mol for molecules/pockets, ESM-2 for
protein sequences), it projects each modality to a shared 128-d unit-sphere embedding, scores by
Euclidean dot product on the learned coordinate slice, and trains each query view with the three
terms implemented by `compute_hcc_pair_official_style`: a multi-positive column term, the official
row-local selected-pair term, and an activity-aware ranking term that pushes stronger binders above
clearly-weaker ones within a pocket. The same helper runs on both the pocket view and the
protein-sequence view, sharing the ligand tower; at inference the weighted dot-product scores from
the available target views are summed.

## Problem it solves

Screening data is assay-organized: one target, a *set* of tested ligands with graded
within-assay affinities. The plain dense-retrieval contrastive objective (DrugCLIP) assumes one
positive ligand per pocket, so it (a) turns the other genuine actives of the same target into
false negatives, (b) learns binder-vs-other but never stronger-vs-weaker, and (c) misses the
top-heavy enrichment metrics (BEDROC α=80.5, EF at 0.5–1%) that reward putting the strongest
binders at the very top. HCC fixes all three while staying a fast normalize-and-dot retrieval
model.

## Key ideas

- **Multi-positive contrastive.** A pocket owns columns `[s,e)`; each of its ligand columns has
  a distribution over queries that should point at the owning pocket (column-softmax,
  cross-entropy), generalizing DrugCLIP's diagonal target to many positives per pocket.
- **False-negative masking.** Before any softmax, if optional metadata is present, mask logit
  `(i,j)` when `uniprot_poc[i] == uniprot_mol[j]` or `lig_smiles[j]` is in
  `pocket_lig_smiles[i]`. The helper applies this as a blunt `masked_fill(mask, finfo.min)` over
  the whole matrix; it does not exempt owner/target cells, so any supervised cell included in the
  mask is suppressed too. With the intended metadata contract, this generalizes DrugCLIP's
  exact-duplicate guard to target-level or known-binder collisions.
- **Activity-gated row-local term.** For each selected ligand `k`, the official helper builds a
  row mask that leaves only column `k` finite and reads `-log p_k`. For an unmasked selected
  cell this has no competitors and evaluates to zero; it is kept because it is part of the
  canonical helper, with the meaningful logic being the weak-ligand gate. In multi-ligand pockets,
  ligands with activity `< 5` are skipped; the column term is not gated.
- **Margin-gated, DCG-discounted ranking.** Ligands are sorted strongest-first. For each anchor
  `k_rel`, a softmax is taken over only the rivals that are *clearly weaker* — keep ligand `idx`
  iff `acts[idx] < acts[k_rel] - log10(3)` (strictly >3x weaker in IC50), mask the rest — and the anchor's
  negative-log-probability is weighted by `1/(log(k_rel+2)·√L_i)`. The 3× margin avoids
  over-fitting noisy within-assay ties; the log discount concentrates gradient on the top of the
  list. Applied only when `L_i > 2`.
- **√-assay-size normalization.** Every term divides by `√L_i` so large assays count more than
  small ones but do not dominate the batch gradient (sub-linear, unlike `1/L_i`).
- **Detached inverse-temperature.** Logits are dot products multiplied by `exp(logit_scale)`
  (init `log(13)`). `.detach()` inside the loss prevents this objective from trivially
  sharpening every softmax by inflating the scale.
- **Two query views.** The identical loss runs for the pocket embedding and the ESM-2 sequence
  embedding against the shared ligand embeddings; the total loss is their sum (weights 1).
- **Inference.** `score_j = alpha_poc * max_pocket(pocket·ligand_j) [+ alpha_prot *
  max_seq(protein·ligand_j)]`; rank by descending score. This is the dot-product the loss
  optimized.

## Loss (per query pathway u ∈ {pocket, sequence})

With L2-normalized embeddings carried in the helper convention, define `q = h_u[:, 1:]`,
`m = h_m[:, 1:]`, and `sim = (q m^T) * exp(logit_scale).detach()`. False negatives are masked to
the dtype minimum to give `sim_tilde`:

```
loss_pocket = sum_i (1/sqrt(L_i)) * sum_{c in [s_i,e_i)}
              -log softmax_over_queries(sim_tilde.T)[c, i]
loss_mol    = sum_i sum_{k in [s_i,e_i), (L_i == 1 or act_k >= 5)}
              -log softmax(row_mask_k + sim_tilde_i)[k] / sqrt(L_i)
              where row_mask_k[k] = 0 and row_mask_k[other] = -inf
loss_rank   = sum_{i: L_i > 2} sum_{k=0}^{L_i-2}
              -log softmax(mask_k + sim_tilde_{i,[s:e]})[k]
              / (log(k+2) * sqrt(L_i))
              where mask_k[idx] = -inf if idx != k and act_idx >= act_k - log10(3)
L_pathway   = loss_pocket + loss_mol + loss_rank
L_total     = alpha_poc * L_pocket_pathway + alpha_prot * L_sequence_pathway
```

`act_k` are within-assay affinities; the rank discount assumes the list is supplied strongest
first, so `k=0` is the strongest binder. The row mask in `loss_mol` leaves no ordinary negatives;
for a finite selected logit, that term contributes zero gradient.

## Working code

```python
"""HCC scoring module: Euclidean contrastive + ranking loss."""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomScoring(nn.Module):
    """HCC: contrastive + activity-aware ranking in Euclidean space.

    A column-softmax assigns tested ligands to their owning pocket, a gated
    row-local selected-pair term matches the official helper, and the ranking term pushes
    stronger binders above clearly-weaker ones."""

    def __init__(self, mol_dim=512, pocket_dim=512, protein_dim=480, embed_dim=128,
                 alpha_poc=1.0, alpha_prot=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.alpha_poc = alpha_poc
        self.alpha_prot = alpha_prot
        self.mol_project = nn.Sequential(
            nn.Linear(mol_dim, mol_dim), nn.ReLU(), nn.Linear(mol_dim, embed_dim)
        )
        self.pocket_project = nn.Sequential(
            nn.Linear(pocket_dim, pocket_dim), nn.ReLU(), nn.Linear(pocket_dim, embed_dim)
        )
        self.protein_project = nn.Sequential(
            nn.Linear(protein_dim, protein_dim), nn.ReLU(), nn.Linear(protein_dim, embed_dim)
        )
        self.logit_scale = nn.Parameter(torch.ones([1]) * np.log(13))

    def _with_reserved_coord(self, x):
        return torch.cat([torch.zeros_like(x[:, :1]), x], dim=-1)

    def project_mol(self, mol_feat):
        z = F.normalize(self.mol_project(mol_feat), dim=-1)
        return self._with_reserved_coord(z)

    def project_pocket(self, poc_feat):
        z = F.normalize(self.pocket_project(poc_feat), dim=-1)
        return self._with_reserved_coord(z)

    def project_protein(self, prot_feat):
        z = F.normalize(self.protein_project(prot_feat), dim=-1)
        return self._with_reserved_coord(z)

    def _compute_hcc_pair(self, emb_poc, emb_mol, batch_list, act_list,
                          uniprot_poc, uniprot_mol,
                          pocket_lig_smiles, lig_smiles, logit_scale):
        """One query->ligand pathway."""
        B = emb_poc.size(0)
        logits = torch.matmul(emb_poc[:, 1:], emb_mol[:, 1:].T) * logit_scale

        # False-negative/duplicate mask, applied globally exactly like the helper.
        mask = torch.zeros_like(logits, dtype=torch.bool)
        if uniprot_poc is not None and uniprot_mol is not None:
            for i in range(B):
                for j in range(B):
                    if uniprot_poc[i] == uniprot_mol[j]:
                        mask[i, j] = True
        if pocket_lig_smiles is not None:
            for i in range(B):
                bad = pocket_lig_smiles[i]
                for j in range(B):
                    if lig_smiles[j] in bad:
                        mask[i, j] = True
        minus_inf = torch.finfo(logits.dtype).min
        sim_masked = logits.masked_fill(mask, minus_inf)

        # Multi-positive column term over the column-softmax.
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
        loss_pocket = (torch.stack(loss_pocket_list).sum()
                       if loss_pocket_list else torch.tensor(0.0, device=logits.device))

        # Row-local selected-pair NLL; row_mask leaves only column k finite.
        loss_mol_list = []
        for i in range(B):
            s, e = batch_list[i]
            acts = act_list[i]
            L_i = e - s
            for k in range(s, e):
                row_mask = torch.full_like(sim_masked[i], minus_inf)
                row_mask[k] = 0
                lprobs = F.log_softmax(row_mask + sim_masked[i], dim=-1)
                if L_i > 1 and acts[k - s] < 5:
                    continue
                loss_mol_list.append(-lprobs[k] / math.sqrt(L_i))
        loss_mol = (torch.stack(loss_mol_list).sum()
                    if loss_mol_list else torch.tensor(0.0, device=logits.device))

        # Within-pocket ranking: anchor must outscore the clearly-weaker rivals (>3x),
        # DCG-discounted to the top; ligands are pre-sorted strongest-first.
        loss_rank_list = []
        for i in range(B):
            s, e = batch_list[i]
            acts = act_list[i]
            L_i = e - s
            if L_i <= 2:
                continue
            out_i = sim_masked[i, s:e]
            for k_rel in range(L_i - 1):
                m = torch.zeros_like(out_i)
                for idx in range(L_i):
                    if idx == k_rel:
                        continue
                    if acts[k_rel] - math.log10(3) <= acts[idx]:
                        m[idx] = minus_inf
                lprobs_rank = F.log_softmax(m + out_i, dim=-1)
                loss_rank_list.append(
                    -lprobs_rank[k_rel] / (math.log(k_rel + 2) * math.sqrt(L_i)))
        loss_rank = (torch.stack(loss_rank_list).sum()
                     if loss_rank_list else torch.tensor(0.0, device=logits.device))

        total = loss_pocket + loss_mol + loss_rank
        return {"loss": total, "loss_pocket": loss_pocket, "loss_mol": loss_mol,
                "loss_rank": loss_rank, "sim_masked": sim_masked}

    def compute_loss(self, mol_emb, poc_emb, prot_emb, batch_list, act_list,
                     uniprot_poc=None, uniprot_mol=None,
                     pocket_lig_smiles=None, lig_smiles=None):
        logit_scale = self.logit_scale.exp().detach()
        d_poc = self._compute_hcc_pair(
            poc_emb, mol_emb, batch_list, act_list, uniprot_poc, uniprot_mol,
            pocket_lig_smiles, lig_smiles, logit_scale)
        d_prot = self._compute_hcc_pair(
            prot_emb, mol_emb, batch_list, act_list, uniprot_poc, uniprot_mol,
            pocket_lig_smiles, lig_smiles, logit_scale)
        loss = self.alpha_poc * d_poc["loss"] + self.alpha_prot * d_prot["loss"]
        return loss, {"loss": loss.item(),
                      "loss_poc": d_poc["loss"].item(),
                      "loss_prot": d_prot["loss"].item(),
                      "sim_masked": d_poc["sim_masked"]}

    def score(self, mol_reps, pocket_reps, prot_reps=None):
        mol_space = mol_reps[:, 1:] if mol_reps.shape[1] == self.embed_dim + 1 else mol_reps
        poc_space = (pocket_reps[:, 1:]
                     if pocket_reps.shape[1] == self.embed_dim + 1 else pocket_reps)
        poc_scores = (poc_space @ mol_space.T).max(axis=0)
        if prot_reps is not None:
            prot_space = (prot_reps[:, 1:]
                          if prot_reps.shape[1] == self.embed_dim + 1 else prot_reps)
            prot_scores = (prot_space @ mol_space.T).max(axis=0)
            return self.alpha_poc * poc_scores + self.alpha_prot * prot_scores
        return self.alpha_poc * poc_scores
```

## Relation to prior methods

- **DrugCLIP** = the contrastive ancestor: same normalize-and-dot retrieval and symmetric
  in-batch InfoNCE, but with a diagonal one-positive-per-pocket target and only exact-duplicate
  masking. HCC keeps the dot-product scoring and two pathway views, replaces the diagonal target
  with a multi-positive column term, optionally masks target-level false-negative cells, keeps the
  official row-local selected-pair term with its activity gate, and adds the ranking term.
- **LigUnity** = the listwise ancestor: it adds a Plackett-Luce listwise ranking term over
  affinity-sorted ligands. HCC's ranking term differs by gating each comparison with a 3×-IC50
  margin (instead of enforcing a strict total order), which tolerates noisy within-assay ties.
- **CLIP / InfoNCE** supply the temperature-scaled softmax-over-similarities; DrugCLIP supplies
  the detached `logit_scale` style used here. **ListNet / Plackett-Luce** supply the listwise
  top-one-softmax form that the ranking term specializes.
