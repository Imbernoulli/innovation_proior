Let me start from what the data actually looks like, because that's where the trouble is. I have screening data organized by assay: one protein target, and a set of ligands that were experimentally tested against it, each with an active/inactive label and, for the actives, an affinity number — pIC50, Kd, Ki, something. And the affinities are only comparable inside one assay; across assays the pH, the temperature, the cofactors, the readout all differ, so a pIC50 of 7 in one assay and a 7 in another don't mean the same thing. So whatever I learn has to be relative-within-assay, never absolute-across-assays. The thing I want to ship is a scoring objective: take the fixed backbone features — Uni-Mol gives me a 512-d vector for a pocket and for a molecule, ESM-2 gives me a 480-d vector for the target sequence — project them into a shared space, and train so that, for a query target, the active ligands float to the top of the ranked list and the strongest binders float to the very top.

The natural starting point is dense retrieval the way DrugCLIP set it up. Encode pocket and molecule each into a vector, L2-normalize so they sit on the unit sphere, score a pair by the dot product s(p,m) = ĝ(p)·f̂(m) which is just cosine similarity, and train contrastively so a binding pair has high cosine and everything else low. The loss is in-batch InfoNCE, symmetric: in a batch of N paired (pocket_k, ligand_k), for each pocket the positive is its own ligand and the negatives are the other N−1 ligands in the batch, and you do the same the other way (each ligand retrieves its pocket). Concretely the pocket-to-ligand batch loss is −(1/N) Σ_k log [ exp(s(p_k,m_k)/τ) / Σ_i exp(s(p_k,m_i)/τ) ], the ligand-to-pocket batch loss mirrors it with the sum over pockets, and the two batch losses are averaged. Why this and not a regression onto affinity? Because the InfoNCE softmax is exactly a density-ratio estimator — minimizing it lower-bounds the mutual information between the pocket view and the ligand view — and because in screening the positive rate is minuscule, far below 0.1%, so a random other-batch ligand is almost surely a true non-binder and makes a free, honest negative. In-batch negatives are cheap and well-justified here. I like this foundation. I'll keep the normalize-and-dot similarity and the two-view contrastive shape, but the assay grouping means I cannot keep the diagonal target literally.

But now I push the assay data through it and it breaks in a way I can't shrug off. DrugCLIP's target is the diagonal: pocket k's one and only positive is ligand k. My assays have *many* tested ligands per target — dozens, with graded activity. If I force one-positive-per-pocket, then for pocket k all the *other* genuine actives of the same target that happen to be in the batch get shoved into the negative set and the loss actively pushes them away from the pocket. That's not a small bias; the whole point of an assay is that it's a cluster of ligands around one target, so the false negatives aren't rare accidents, they're the dominant structure of the batch. Two distinct things have gone wrong at once and I should keep them separate. One: the contrastive target is wrong — a pocket should be allowed to call *all* of its tested binders positive, not just one. Two: even ligands of the same target that I'm *not* currently treating as this pocket's positive shouldn't be treated as its negative, because they really do bind it. Let me deal with the second one first since it's a masking question, then redesign the target.

DrugCLIP already saw a version of the false-negative problem and masked *exact* duplicates: if the same pocket string or the same molecule string appears twice in a batch, blank out those off-diagonal logits before the softmax so the model isn't told its own positive is a negative. That's necessary but it only catches literal repeats. My real false negatives are subtler: ligand j is a *different* molecule but it's a confirmed binder of pocket i's target, so calling it a negative for pocket i is a lie. When the batch gives me the metadata, I can catch this before the softmax: build a boolean mask over the logit matrix with mask[i,j] = true whenever uniprot_poc[i] == uniprot_mol[j] or whenever lig_smiles[j] is in pocket_lig_smiles[i]. Then sim_masked = logits.masked_fill(mask, a dtype-min value that behaves like −inf in the softmax). I have to be precise about this: the helper does not have a special exemption for the supervised owner cell. It is a blunt matrix mask. If the metadata marks a cell, that cell is suppressed everywhere it is later used, including in a target position, so the data contract has to make these optional metadata fields mean "remove this potential false negative or duplicate cell," not "relabel it as positive." With that understood, every downstream softmax sees sim_masked, never the raw logits.

Now I need the target itself. A pocket with L_i tested ligands occupying columns [s,e) should be allowed to own *all* of those ligand columns. The cleanest way to express that is a column cross-entropy. Build the per-ligand column distribution: take sim_masked, transpose it so each ligand column becomes a distribution over the query rows, and log-softmax along that query axis. Then for each pocket i, every ligand in its block [s,e) has the class label "query i," and I sum the negative-log-likelihood of those labels. So loss_pocket = Σ_i [ Σ_{ligand in block i} NLL(column-softmax, target=i) ], with the per-pocket sum divided by √L_i later. This is the honest generalization of DrugCLIP's diagonal target to the case where a pocket owns several ligand columns.

The mirror direction is where I need to slow down, because a normal row softmax over all molecules would quietly recreate the false-negative problem inside the assay block: the other ligands of the same pocket are not clean negatives. The official helper takes an extremely local version instead. For ligand k in pocket i, it builds row_mask = −inf everywhere except at column k, adds that to sim_masked[i], takes log_softmax, and reads −log p_k. If column k is finite, this softmax has no competitors and the loss is zero. So I should not pretend this is another full contrastive direction or the source of binder separation; it is a selected-pair NLL in the exact code shape, with the actual competition left to the column term and the ranking term. The nontrivial choice left in this row-local term is the activity gate. If a pocket has multiple ligands and ligand k is weak, acts[k−s] < 5, skip it. A weak or inactive ligand is noisy evidence for a matched pair, so the helper lets only confident actives through; when L_i == 1 it keeps the single ligand because otherwise that pocket contributes nothing to this term. I do not gate loss_pocket the same way: the column term says which query owns the tested ligand column, and that ownership is part of the assay structure even when the ligand is weak.

That gives me binder-versus-non-binder separation with the right multi-positive structure and the false negatives masked out. But it still doesn't do the thing the metric most rewards. BEDROC at α=80.5 and EF at 0.5–1% are violently top-heavy — almost all the credit is for the handful of ranks at the very top — and so far nothing in my loss distinguishes a strong binder from a merely-okay binder. Two actives of the same target both get pulled toward it with no preference for the stronger. I need a *ranking* signal inside the assay.

The textbook move is a listwise loss. LigUnity does exactly this: sort the assay's ligands by affinity and fit a Plackett-Luce model, where the probability of a ranking is the product, over positions, of softmaxes over the not-yet-ranked items — P(π|s) = Π_k exp(s_{π_k}) / Σ_{l≥k} exp(s_{π_l}) — and the listwise loss is the cross-entropy of the model's ranking against the affinity-sorted one, with a position decay so early ranks matter more. Let me try to just adopt that. I sort ligands by affinity, I ask the model to reproduce the *total order*, I weight by 1/(√B log(k+1))…

…and I hit the wall the data itself put there. The within-assay affinities are noisy and only loosely comparable. A full Plackett-Luce loss commits to a single total order over every ligand in the assay, which means it will penalize the model for failing to separate two ligands whose measured affinities differ by less than experimental error. I'd be fitting the noise. The ordering I actually trust is coarse: ligand A is *meaningfully* stronger than ligand B only when their affinities differ by more than the assay's noise floor. In med-chem the rule of thumb for a "real" potency difference is roughly threefold in IC50 — anything inside ~3× is a wash. So a strict total order is the wrong object; what I want is to enforce order *only between pairs that are clearly separated in affinity*, and stay silent inside a ~3× band. That tolerance is the fix for the noise the listwise loss would otherwise overfit.

So let me reshape the ranking loss around that. Threefold in IC50 is log10(3) ≈ 0.477 in log-affinity units. I'll sort each pocket's ligands by activity, strongest first, so position 0 is the strongest. Then for each ligand as an *anchor* — call it position k_rel — I want it to outscore the ligands that are clearly weaker than it, and I want to ignore the ligands that are within noise. So I build a softmax over the in-pocket ligands but I *mask out* every other ligand whose activity is not clearly below the anchor's: keep ligand idx in the denominator only if acts[idx] < acts[k_rel] − log10(3), i.e. it is strictly more than threefold weaker; mask (set −inf) everyone else, and always keep the anchor itself. Then the loss for the anchor is −log of its own softmax probability against that pruned set of clearly-weaker rivals. If the anchor really is the strongest among them it wins the softmax and the loss is small; if a clearly-weaker ligand is scoring above it, it pays. And because I only ever compare against ligands more than 3× weaker, I never punish the model for ordering two near-ties — exactly the noise tolerance I wanted. I only run this when L_i > 2, because with one or two ligands there's no meaningful within-assay order to enforce.

Now the weighting on the anchors. The metric is top-heavy, so getting the *strongest* ligand ranked above everything matters far more than getting the 20th-strongest exactly right. That's the DCG intuition: discount each position by 1/log(rank+1). Since I sorted strongest-first, anchor k_rel = 0 is the strongest ligand, and I want it to carry the most weight. So weight the anchor's loss by 1/log(k_rel + 2): at k_rel = 0 that's 1/log(2) ≈ 1.44, the largest, decaying for deeper, weaker anchors. The "+2" is just so the first anchor's log is log(2) rather than log(1) = 0 — a finite, large weight rather than a divide-by-zero. This makes the ranking loss spend its gradient on getting the top of the list right, which is precisely what BEDROC and EF reward.

There's a normalization issue threading through all three terms that I should settle now rather than let it bite me. Assays vary wildly in how many ligands they screened — some have a handful, some have dozens. If I sum each term straight, a single big assay with L_i = 50 ligands contributes ~50× the loss of an assay with one ligand and dominates the batch gradient, so the model effectively trains on a few big assays and ignores the rest. The over-correction would be dividing by L_i (mean per ligand), but that flattens it too far — a big, information-rich assay genuinely *should* count for more than a tiny one, just not linearly more. The middle ground is 1/√L_i: sub-linear, so big assays still pull harder than small ones but not overwhelmingly. So loss_pocket sums the per-pocket NLL and divides by √L_i; loss_mol divides each ligand's term by √L_i; and the ranking loss divides each anchor's term by √L_i on top of the 1/log(k_rel+2) discount. Three terms, each √-normalized by assay size, summed.

So the per-pathway loss is loss_pocket + loss_mol + loss_rank, where loss_mol is included exactly as the helper computes it, even though its row mask leaves no ordinary negatives. And I have two query views of the target, not one — the pocket structure and the protein sequence. The pocket is the primary signal, but the sequence is a complementary, structure-free view that helps when the pocket is noisy or ambiguous, and it costs me almost nothing: I run the identical three-term helper a second time with the protein-sequence embedding as the query instead of the pocket embedding, against the same ligand embeddings, and add it in with its own weight. With both weights at 1 the total training loss is just pathway(pocket) + pathway(sequence). The molecule embeddings are shared between the two pathways, so the ligand tower learns from both views at once.

One more thing the contrastive loss can sabotage if I'm not careful: the scale. After L2-normalization the dot product lives in [−1,1], which is far too compressed for a peaked softmax — the differences between a good and a bad match are tenths, and exp of tenths is nearly flat. So I multiply the logits by a scale before the softmax, the inverse temperature. I want the usual CLIP-shaped parameterization, so I store logit_scale and use exp(logit_scale) as the multiplier. Initialize it at log(13), so the starting multiplier is ~13 (an inverse temperature around 0.08), the same ballpark DrugCLIP used with log(14). But inside this helper I must consume exp(logit_scale).detach(). If I don't, the contrastive objective can trivially lower itself by cranking the scale up without improving a single embedding — it would just sharpen every softmax — so the scale would run away. Detaching means this loss treats the scale as fixed while still making the initialization explicit in the module.

The backbone hands me 512-d (mol, pocket) and 480-d (sequence); I need each in a shared 128-d space. A single linear map is probably too rigid to disentangle binding-relevant directions from the backbone's general-purpose features, and a deep MLP is overkill and would overfit on top of already-strong pretrained features. The standard projection-head shape is one hidden layer with a nonlinearity: Linear(d → d) → ReLU → Linear(d → 128). Keep the hidden width at the input dimension so the head has enough capacity to rotate and recombine features before compressing to 128, and use ReLU as the cheap default nonlinearity. One head per modality, since the three backbones have different statistics and dimensions. After the head I L2-normalize, so every learned embedding is on the unit sphere and the dot product is cosine — which is what the contrastive/ranking machinery above assumed. To mirror the helper's tensor convention, I can carry a leading reserved coordinate and score only the learned coordinates with `[:, 1:]`; the objective itself is still Euclidean dot-product scoring on the normalized learned slice.

At test time I embed the query target and every candidate ligand, and I score by similarity. But a target in these benchmarks can come with *several* candidate pocket structures (different conformations or crystal forms of the same binding site), and a ligand binds if it matches *any* of them — so I shouldn't average over pockets, I should take the best-matching one: score_j = max over the pocket embeddings of (pocket · ligand_j). That's (pocket_reps @ mol_reps.T).max(axis=0) on the learned coordinate slice. And since I trained a sequence pathway too, I add its analogous score with its own weight: if I have protein-sequence embeddings, score_j += alpha_prot times max over those of (protein · ligand_j), while the pocket branch carries alpha_poc. The two views vote through the same dot product. Then rank all ligands by descending score. Note this scoring is the plain Euclidean dot product on the normalized embeddings — the exact quantity the training loss optimized — so train and test agree.

Let me make sure I can write all of this as one coherent module that fills the slots. The pocket and sequence pathways are the same function with a different query embedding, so I factor the three-term computation into one helper and call it twice. Let me write it.

```python
"""Scoring module: Euclidean contrastive + activity-aware ranking loss."""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomScoring(nn.Module):
    """Contrastive + ranking scoring in Euclidean space.

    A column-softmax assigns tested ligands to their owning pocket, a gated
    row-local selected-pair term matches the official helper, and the
    ranking term pushes stronger binders above clearly-weaker ones."""

    def __init__(self, mol_dim=512, pocket_dim=512, protein_dim=480, embed_dim=128,
                 alpha_poc=1.0, alpha_prot=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.alpha_poc = alpha_poc
        self.alpha_prot = alpha_prot
        # one projection head per modality: Linear -> ReLU -> Linear -> 128-d
        self.mol_project = nn.Sequential(
            nn.Linear(mol_dim, mol_dim), nn.ReLU(), nn.Linear(mol_dim, embed_dim)
        )
        self.pocket_project = nn.Sequential(
            nn.Linear(pocket_dim, pocket_dim), nn.ReLU(), nn.Linear(pocket_dim, embed_dim)
        )
        self.protein_project = nn.Sequential(
            nn.Linear(protein_dim, protein_dim), nn.ReLU(), nn.Linear(protein_dim, embed_dim)
        )
        # parameterized inverse-temperature; detached inside the loss helper
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
                          uniprot_poc, uniprot_mol, pocket_lig_smiles, lig_smiles,
                          logit_scale):
        """One query->ligand pathway (query = pocket or protein sequence)."""
        B = emb_poc.size(0)
        logits = torch.matmul(emb_poc[:, 1:], emb_mol[:, 1:].T) * logit_scale

        # mask false negatives: same target, or a known binder of this pocket
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
        sim_masked = logits.masked_fill(mask, minus_inf)    # drop them from every softmax

        # multi-positive column term: each ligand column's distribution over
        # queries should point at the pocket that owns it
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
            loss_pocket_list.append(loss_tmp.sum() / math.sqrt(L_i))   # sqrt assay-size norm
        loss_pocket = (torch.stack(loss_pocket_list).sum()
                       if loss_pocket_list else torch.tensor(0.0, device=logits.device))

        # row-local selected-pair NLL; row_mask leaves only column k finite
        loss_mol_list = []
        for i in range(B):
            s, e = batch_list[i]
            acts = act_list[i]
            L_i = e - s
            for k in range(s, e):
                row_mask = torch.full_like(sim_masked[i], minus_inf)
                row_mask[k] = 0                                # positive is column k only
                lprobs = F.log_softmax(row_mask + sim_masked[i], dim=-1)
                if L_i > 1 and acts[k - s] < 5:
                    continue
                loss_mol_list.append(-lprobs[k] / math.sqrt(L_i))
        loss_mol = (torch.stack(loss_mol_list).sum()
                    if loss_mol_list else torch.tensor(0.0, device=logits.device))

        # within-pocket ranking: each anchor must outscore the ligands that are
        # clearly weaker (>3x in IC50); ties within ~3x are not penalized; the
        # top (strongest) anchors are weighted most (DCG-style discount)
        loss_rank_list = []
        for i in range(B):
            s, e = batch_list[i]
            acts = act_list[i]                                # sorted strongest-first
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
                        m[idx] = minus_inf                    # keep only strictly weaker rivals
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
        logit_scale = self.logit_scale.exp().detach()         # detach: loss can't game scale
        d_poc = self._compute_hcc_pair(poc_emb, mol_emb, batch_list, act_list,
                                       uniprot_poc, uniprot_mol,
                                       pocket_lig_smiles, lig_smiles, logit_scale)
        d_prot = self._compute_hcc_pair(prot_emb, mol_emb, batch_list, act_list,
                                        uniprot_poc, uniprot_mol,
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

Let me trace the causal chain back. I started from DrugCLIP's dense-retrieval contrastive scoring — normalize, dot, symmetric in-batch InfoNCE — because it's a principled, mutual-information-grounded way to align binders with targets and in-batch negatives are nearly free when the positive rate is tiny. But assay data has many graded-activity ligands per target, so the one-positive diagonal target is wrong and turns genuine actives into false negatives. I apply the optional metadata mask exactly as a blunt matrix mask before the softmaxes, then replace the diagonal target with a multi-positive column term that lets a pocket own all its tested ligand columns. I keep the official row-local selected-pair term, but I do not rely on it for competition because its row mask leaves only the selected column finite; its activity gate keeps weak ligands out in multi-ligand pockets. The column term gives the assay ownership signal, but it has no preference among binders, while the metrics are top-heavy — so I add a within-pocket ranking term. A full Plackett-Luce listwise loss would overfit the noisy within-assay affinities by demanding a total order, so instead I gate each comparison by a threefold-IC50 margin, only keeping ligands strictly more than log10(3) below the anchor, and discount anchors by 1/log(rank+2) so the gradient concentrates on getting the strongest binders to the top. Every term is √(assay-size)-normalized so big assays count more but do not dominate; the softmaxes use exp(logit_scale).detach() so this objective cannot trivially sharpen itself; the embeddings come from one-hidden-layer ReLU projection heads, L2-normalized and scored by their learned Euclidean coordinates; the same helper runs on the pocket view and the ESM-2 sequence view sharing the ligand tower; and at inference I score by the max over candidate pocket conformers plus the weighted sequence view, ranking by that dot product — the same quantity the loss optimized.
