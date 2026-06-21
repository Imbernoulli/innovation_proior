Vanilla CLIP told me exactly where it breaks, and it told me in numbers. On DUD-E it is already strong — AUROC 0.895, BEDROC 0.703, EF@0.5% 51.3 — because property-matched ZINC decoys separate cleanly under a contrastive cosine boundary; DEKOIS holds too at AUROC 0.892. But LIT-PCBA is the tell: AUROC collapses to 0.576, barely above chance, and BEDROC craters to 0.065 with EF@0.5% at 7.27, a seventh of the DUD-E number. That split is structural, not noise. LIT-PCBA is the benchmark built to remove the artificial-decoy crutch — confirmed actives versus confirmed inactives, graded potencies crowded together — and a pure contrastive separation has no notion of "more active": two actives of the same target both get pulled toward it with equal force, nothing distinguishes a 10 nM binder from a 10 µM one. The metric is top-heavy and LIT-PCBA's top ranks are decided by *ordering* graded actives, which vanilla CLIP simply does not do. So the failure that bites is an *ordering* problem sitting on top of the *separation* I already have — and the two levers I left on the table at rung one, the per-ligand pIC50 in `act_list` and the ESM-2 sequence tower the loss never touched, are exactly what fix it.

I propose HCC — Hierarchical Contrastive Cosine — which keeps rung one's Euclidean contrastive core intact and adds an *activity-aware ranking* term plus the *sequence pathway*. The one constraint that shapes the ranking term is that affinities are only comparable *inside* one assay: across assays the pH, temperature, cofactors, and readout all differ, so whatever I learn must be relative-within-assay, never absolute-across-assays. The textbook move would be a listwise Plackett-Luce loss fitting a single *total order* over every ligand, but that commits to ordering pairs whose measured affinities differ by less than experimental error — I would be fitting the noise, and forcing spurious order among near-ties is precisely how you scramble the very top of the list, the opposite of what LIT-PCBA needs. The ordering I actually trust is coarse: in med-chem the rule of thumb for a "real" potency difference is roughly threefold in IC50, $\log_{10}(3)\approx 0.477$ in log-affinity units; anything inside that band is a wash. So I want order enforced *only between pairs clearly separated in affinity* and silence inside the $\sim 3\times$ band. I sort each pocket's ligands strongest-first, and for each ligand as an anchor at relative position `k_rel` I build a softmax over the in-pocket ligands but mask out every rival whose activity is *not* clearly below the anchor's — I keep `idx` in the denominator only if `acts[k_rel] - log10(3) > acts[idx]`, strictly more than threefold weaker, and always keep the anchor. The loss is $-\log$ of the anchor's softmax probability against that pruned set of clearly-weaker rivals: if it is genuinely strongest among them the loss is small; if a clearly-weaker ligand outscores it, it pays. Because I never compare against near-ties, I never punish ordering them. I run this only when $L_i > 2$, since one or two ligands carry no meaningful within-assay order, and I weight each anchor by the DCG discount $1/\log(\text{rank}+2)$ — anchor 0, the strongest, carries the largest weight $1/\log 2 \approx 1.44$, decaying for deeper anchors, the "+2" keeping the first log finite. This spends the ranking gradient on the head of the list, exactly where BEDROC and EF live.

Graded data also changes how I should treat the contrastive ligand-to-pocket direction. At rung one the row-local term matched every ligand to its pocket unconditionally, but a weak or inactive ligand is *noisy evidence* for a matched pair — pulling a 10 µM ligand toward its pocket as hard as a 10 nM one teaches the space to call marginal binders confident positives, exactly wrong when the metric lives at the top. So I gate it: for ligand $k$ in a multi-ligand pocket, if `acts[k - s] < 5` (below the conventional $\sim 10$ µM "active" cutoff in pIC50), I skip it as a contrastive positive; when $L_i = 1$ I keep the single ligand regardless, or that pocket contributes nothing. I do *not* gate the multi-positive column term the same way — the column term encodes which query owns the tested ligand column, and that ownership is part of the assay structure even when the ligand is weak. So each pathway is three pieces — the column term, the gated row-local selected-pair NLL, and the activity-gated ranking term — each $\sqrt{L_i}$-normalized and summed. The false-negative mask stays untouched from rung one, because on a sub-0.1% distribution a true binder treated as a negative is the one corruption I cannot afford.

The second lever is nearly free. The pocket is a single sampled conformation, but the ESM-2 protein sequence is a complementary, structure-free view of the same target that helps precisely when the pocket is noisy or ambiguous — the LIT-PCBA condition. The whole three-term computation is a function of one query embedding against the shared ligand embeddings, so I factor it into a helper `_compute_hcc_pair` and call it twice: once with the pocket embedding as query, once with the protein-sequence embedding as query, against the same ligand tower. The molecule embeddings are shared, so the ligand tower learns from both views at once; with both pathway weights at 1, the total loss is just pathway(pocket) + pathway(sequence). This is where the `prot_emb` I projected but ignored at rung one finally enters the loss. Everything else is held fixed as an honest minimal delta: this rung is still *Euclidean*, so I keep the three NonLinearHeads, L2-normalize, and score the *full* 128-d dot product `emb_poc @ emb_mol.T * logit_scale` with no reserved-coordinate slice — there is none to drop while the space is flat (that convention belongs to the manifold rung). The detached $\log 13$ inverse temperature stays, multiplied in inside the helper and `.detach()`-ed so neither the ranking nor the contrastive softmax can lower itself by sharpening every distribution via the scale alone rather than improving an embedding. At evaluation nothing changes in shape except that the sequence view now votes: max over the pocket conformers of `pocket · ligand`, plus the analogous max over the sequence view, ranked by the sum — the same Euclidean dot product the loss optimized, so train and test agree.

The falsifiable claim is sharp. DUD-E should hold or improve slightly — the ranking term cannot hurt a benchmark already separated well, and the sequence view adds a second target signal — while the real test is LIT-PCBA: its AUROC and BEDROC should *climb*, because the ranking term and the sequence view directly attack the graded-actives failure that left it at 0.576 / 0.065. If the ordering diagnosis is right, the largest relative gain on the ladder appears on LIT-PCBA, not DUD-E. And if instead LIT-PCBA barely moves, the problem was never ordering but the Euclidean geometry itself collapsing near-identical ligands — the diagnosis that would force the next rung.

```python
"""HCC scoring module: Euclidean contrastive + ranking loss."""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomScoring(nn.Module):
    """HCC: Hierarchical Contrastive Cosine in Euclidean space.

    Adds ranking loss that enforces more active ligands score higher
    within each pocket's ligand set, weighted by 1/log(rank+2) (DCG-style).
    """

    def __init__(self, mol_dim=512, pocket_dim=512, protein_dim=480, embed_dim=128):
        super().__init__()
        # NonLinearHead pattern used by the HypSeek implementation.
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

    def project_mol(self, mol_feat):
        return F.normalize(self.mol_project(mol_feat), dim=-1)

    def project_pocket(self, poc_feat):
        return F.normalize(self.pocket_project(poc_feat), dim=-1)

    def project_protein(self, prot_feat):
        return F.normalize(self.protein_project(prot_feat), dim=-1)

    def _compute_hcc_pair(self, emb_poc, emb_mol, batch_list, act_list,
                          uniprot_poc, uniprot_mol, pocket_lig_smiles, lig_smiles,
                          logit_scale):
        """Compute HCC loss for one pathway (pocket-mol or protein-mol)."""
        B = emb_poc.size(0)
        logits = emb_poc @ emb_mol.T * logit_scale

        # False-negative mask
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

        # Pocket retrieves ligands
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

        # Ligand retrieves pocket (skip low-activity ligands in multi-ligand pockets)
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
        loss_mol = torch.stack(loss_mol_list).sum() if loss_mol_list else torch.tensor(0.0, device=logits.device)

        # Ranking loss: within each pocket, rank by activity
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
                loss_rank_list.append(-lprobs_rank[k_rel] / (math.log(k_rel + 2) * math.sqrt(L_i)))
        loss_rank = torch.stack(loss_rank_list).sum() if loss_rank_list else torch.tensor(0.0, device=logits.device)

        total = loss_pocket + loss_mol + loss_rank
        return {
            "loss": total,
            "loss_pocket": loss_pocket,
            "loss_mol": loss_mol,
            "loss_rank": loss_rank,
            "sim_masked": sim_masked,
        }

    def compute_loss(self, mol_emb, poc_emb, prot_emb,
                     batch_list, act_list,
                     uniprot_poc=None, uniprot_mol=None,
                     pocket_lig_smiles=None, lig_smiles=None):
        logit_scale = self.logit_scale.exp().detach()

        # HCC for pocket-molecule pathway
        loss_dict_poc = self._compute_hcc_pair(
            poc_emb, mol_emb, batch_list, act_list,
            uniprot_poc, uniprot_mol, pocket_lig_smiles, lig_smiles,
            logit_scale,
        )
        # HCC for protein-molecule pathway
        loss_dict_prot = self._compute_hcc_pair(
            prot_emb, mol_emb, batch_list, act_list,
            uniprot_poc, uniprot_mol, pocket_lig_smiles, lig_smiles,
            logit_scale,
        )
        loss = loss_dict_poc["loss"] + loss_dict_prot["loss"]

        return loss, {
            "loss": loss.item(),
            "loss_poc": loss_dict_poc["loss"].item(),
            "loss_prot": loss_dict_prot["loss"].item(),
            "sim_masked": loss_dict_poc["sim_masked"],
        }

    def score(self, mol_reps, pocket_reps, prot_reps=None):
        poc_scores = (pocket_reps @ mol_reps.T).max(axis=0)
        if prot_reps is not None:
            prot_scores = (prot_reps @ mol_reps.T).max(axis=0)
            return poc_scores + prot_scores
        return poc_scores
```
