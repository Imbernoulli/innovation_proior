The floor I have to start from is the scoring objective alone, and the constraints that actually bite decide its shape. The library is $10^8$–$10^9$ molecules and the hit rate is savage — far below one in a thousand truly bind a given pocket — so almost any molecule I grab is a non-binder, and the metric I am judged on is violently top-heavy: BEDROC at $\alpha=80.5$ and EF at the top 0.5%/1%/5% give almost all the credit to the handful of ranks at the very top. So I need a score that puts true binders at the *top* specifically, can run over a billion molecules in feasible time, leans on neither scarce affinity labels nor hand-built decoy sets, and generalizes to unseen targets. Docking satisfies none of the throughput constraint — its per-compound pose sampling costs grow *with* the library, exactly backwards for a "bigger is better" regime. Affinity regression needs the ~$10^4$ reliable affinity labels that exist and re-runs a full forward pass per pocket-molecule pair, so it never amortizes. Decoy classifiers are worse in a subtler way: they manufacture negatives from a rule and learn the *decoy-construction rule* rather than binding, collapsing the moment they meet a benchmark built with a different rule.

I propose to reframe screening as *retrieval* and fill the scaffold with a CLIP-style contrastive scorer — vanilla CLIP, the plainest fill the contract admits. The single decision that makes the throughput problem disappear is the *factorized* score: a dot product $s(p,m)=\hat g(p)\cdot\hat f(m)$ of two independently computed embeddings, one tower per side, deliberately not a cross-encoder that jointly attends over the pair. Because the form is decomposable, I embed the whole library once, cache the vectors, and every new target becomes one pocket encoding plus a nearest-neighbor search against the cache — the expensive neural computation is paid once per molecule, ever, not once per (target, molecule) pair. That restriction is not a modeling concession; it is the entire reason the thing runs at scale. The signal it needs at training time is handed to me for free by the savage hit rate: any other molecule is essentially certainly a non-binder, so the cheapest negatives are the ones already in my batch. I form the full similarity matrix $S$ with $S_{ij}=s(p_i,m_j)$, treat the matched pairs as positives and every other cell as a negative-by-default, and train with a contrastive softmax — categorical cross-entropy over each pocket's row of logits. This is the right object, not a margin or a regression to $1/0$, for two reasons: its optimum drives the score toward a density ratio $p(\text{molecule}\mid\text{pocket})/p(\text{molecule})$, precisely the relevance signal for ranking and not an absolute affinity I would have to label; and minimizing it maximizes a lower bound on the pocket-molecule mutual information that *tightens as the batch grows*. I symmetrize over rows and columns so the space is good for retrieval in both directions and the molecule embeddings cannot collapse into a cluster that all looks alike to a pocket.

Two head choices carry the rest. I do not dot-product on the encoder's raw [CLS] feature: the contrastive objective is aggressive and will warp whatever space it acts on, so I interpose a nonlinear projection head — $\text{Linear}(d,d)\to\text{ReLU}\to\text{Linear}(d,128)$ — into a modest 128-d space, one head per modality since the three backbones have different statistics and dimensions. The head absorbs the contrastive distortion and protects the rich pretrained representation. After the head I L2-normalize each embedding onto the unit sphere, turning the dot product into bounded, scale-free cosine; without it the model could cheat the softmax by inflating vector norms instead of improving directions. But cosine lives in $[-1,1]$, far too compressed for a peaked softmax, so I scale the logits by an inverse temperature stored on a log axis and initialized so the multiplier is $\approx 13$, CLIP-ish. And I *detach* that scale inside the loss: otherwise the objective trivially lowers itself by cranking the scale up — sharpening every softmax — without improving a single embedding, and it would run away. Detaching forces the contrastive gradient into the embedding directions at a stable scale.

The scaffold's data is grouped *by assay*, not as clean one-positive-per-pocket pairs, and three adaptations follow from that. The harness hands me `batch_list[i] = (s, e)`, the contiguous column span of pocket $i$'s tested ligands, so a pocket typically owns *several* ligand columns; the literal one-positive diagonal of DrugCLIP would shove every other genuine active of the same target into pocket $i$'s negative set. The honest generalization is a multi-positive *column* term: log-softmax the *transpose* of the similarity matrix so each ligand column is a distribution over query pockets, and for every ligand in block $i$ the target class is "pocket $i$." The mirror direction is row-local by design: for ligand $k$ in pocket $i$, I build a row mask that is $0$ at column $k$ and $-\infty$ elsewhere, add it to the masked row, log-softmax, and take $-\log p_k$ — a selected-pair NLL, with the real competition carried by the column term, not by contrasting a pocket's own actives against each other. Second, assays vary wildly in size, so a straight sum lets one big assay dominate the gradient while dividing by $L_i$ over-corrects (a big, information-rich assay genuinely should count for more); the middle ground is $1/\sqrt{L_i}$, sub-linear, applied to each per-pocket term. Third, the in-batch construction has a quiet bug — an off-diagonal cell can be a *true* binder I am about to push down as a negative. So before any softmax I build a boolean mask, setting `mask[i, j] = True` whenever `uniprot_poc[i] == uniprot_mol[j]` (same target) or `lig_smiles[j]` is in pocket $i$'s known-binder set, and `masked_fill` those cells to the dtype minimum so they vanish from every softmax. On a sub-0.1% distribution a false negative is the one signal-corrupting mistake I cannot afford. At evaluation the score is the matching quantity: form `pocket_reps @ mol_reps.T` and take the max over the target's pocket conformers — a molecule that binds *any* conformation should rank high — then sort.

Two things this floor deliberately leaves on the table, and naming them is the point. The harness hands me a third tower — the ESM-2 protein-sequence embedding `prot_emb` — and the per-ligand pIC50 in `act_list`; vanilla CLIP ignores both. The loss is pocket↔molecule only, the protein tower is projected but never enters `compute_loss`, and nothing in the objective distinguishes a strong binder from a merely-okay one: two actives of the same target both get pulled toward it with equal force. That is exactly what makes this the weakest rung. The contrastive separation should already concentrate actives near the top far better than chance, so I expect solid AUROC across all three benchmarks and respectable EF on DUD-E, where property-matched ZINC decoys separate cleanly. But on LIT-PCBA — confirmed actives versus confirmed inactives, graded potencies crowded together, the benchmark built to remove the artificial-decoy crutch — I expect this fill to look weakest: decent AUROC but soft BEDROC and EF, because it ranks binders above decoys without ranking *strong* binders above *weak* ones. I have an *ordering* problem on top of a *separation* problem, and the unused activity numbers and the unused sequence tower are sitting right there to fix it at the next rung.

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
