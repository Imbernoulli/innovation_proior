## Research question

Virtual screening ranks a large compound library against a protein target to surface likely
binders. The retrieval-based framing treats this as dense retrieval: encode each protein pocket
and each candidate molecule into a shared vector space, then rank molecules by their similarity
to the pocket. With strong pretrained backbone encoders already in hand — an SE(3) 3D graph
transformer (Uni-Mol) that turns a pocket or a molecule into a 512-dimensional vector, and a
protein language model (ESM-2) that turns an amino-acid sequence into a 480-dimensional vector,
all fine-tuned jointly with the scoring module — the open question is the *scoring objective*:
how to project those backbone features into a shared embedding space, how to score pairs, and
what training loss best discriminates active binders from decoys and orders the actives by
strength.

The pain that makes this non-trivial is the structure of the training data. Screening data comes
organized by *assay*: one protein target, screened against a *set* of ligands, each with a binary
active/inactive label and, for the actives, a graded affinity value (pIC50, Kd, Ki). Crucially,
affinity values are only comparable *within* the same assay — pH, temperature, cofactors, and
measurement type differ across assays — so the supervision is "this ligand binds this target, and
it binds more/less strongly than that other ligand of the *same* assay," never an absolute affinity
comparable across targets. A good objective has to (1) align each target with *all* of its tested
binders, not just one; (2) push the *stronger* binders above the weaker ones within an assay, so
the very top of the ranked list is enriched (the metric that matters, EF/BEDROC, is dominated by
the first ~1-2% of ranks); (3) survive measurement noise and the fact that within-assay affinities
are only loosely ordered; and (4) not punish a molecule for being similar to other genuine actives
that happen to share the batch. Designing that objective — heads, embedding space, loss — is the
problem.

## Background

**Dense retrieval and contrastive alignment.** The retrieval framing rests on contrastive
representation learning. Given two views of a thing (here, a binding pocket and a ligand that
binds it), encode both, normalize to the unit sphere, and train so that a matched pair has high
dot product (cosine similarity) and mismatched pairs have low. The canonical loss is InfoNCE
(van den Oord et al. 2018; popularized for cross-modal alignment by CLIP, Radford et al. 2021):
for an anchor `x` with one positive `x⁺` and a pool of negatives `{x⁻_i}`,

```
L_InfoNCE = -log [ exp(sim(x, x⁺)/τ) / ( exp(sim(x, x⁺)/τ) + Σ_i exp(sim(x, x⁻_i)/τ) ) ].
```

This is a softmax-cross-entropy that asks the positive to win a multiway classification against
the negatives. Two facts about it are load-bearing. First, it estimates a density ratio
`p(x⁺|x)/p(x⁺)` and so its minimization lower-bounds the mutual information between the two views —
it is a principled alignment objective, not just a heuristic. Second, the temperature `τ` controls
how peaked the distribution is: similarities live in `[-1,1]` after normalization, which is too
narrow a range for a useful softmax, so one divides by a small `τ` (equivalently multiplies by a
large `1/τ`); small `τ` makes the loss focus on the hardest negatives. CLIP makes `1/τ` a learned
scalar (`logit_scale`), clamped to keep it from running away.

**In-batch negatives and their failure mode.** Explicit negatives are scarce in screening — assays
record what was tested, and confirmed non-binders are few — so the standard trick (CLIP, dense
passage retrieval) is *in-batch* negatives: in a batch of paired `(pocket_k, ligand_k)`, treat
`ligand_k` as the positive for `pocket_k` and every *other* batch ligand as a negative. This is
justified when the positive rate is tiny (a random library ligand binds a given target with
probability far below 0.1%), so a random in-batch ligand is almost surely a true negative. The
exception, which is exactly the screening regime, is when the batch contains *another genuine
active of the same target*, or the same ligand twice: those become **false negatives** the loss
will wrongly push apart.

**Learning to rank, listwise.** Ordering a list by a score is the learning-to-rank problem. The
listwise approach (ListNet, Cao et al. 2007) models a ranking with the Plackett-Luce distribution:
the probability of drawing a permutation is a product of softmaxes, each over the items not yet
chosen,

```
P(π | s) = Π_{k=1}^{C}  exp(s_{π_k}) / Σ_{l=k}^{C} exp(s_{π_l}),
```

so the top-one probability is just `softmax(s)` and the listwise loss is the cross-entropy between
the target ranking distribution and the model's. The metrics that screening cares about are
top-heavy: BEDROC with α=80.5 puts ~80% of its weight on roughly the top 2% of ranks, and EF_x%
counts actives in the top x%. The discounted-cumulative-gain family handles top-heaviness with a
position discount `1/log(rank+1)`, so an error at rank 1 costs far more than an error at rank 20.

**Backbone encoders (given, fixed).** Uni-Mol (Zhou et al., ICLR 2023) is an SE(3)-equivariant 3D
graph transformer producing a 512-d representation for a molecule or a pocket from atom types and
coordinates. ESM-2 (Lin et al., Science 2023) is a protein language model; its per-target
sequence embedding (here 480-d, from the 35M-parameter variant) is a complementary, structure-free
view of the target. These are pretrained and fine-tuned jointly; the scoring module sits on top of
their outputs.

## Baselines

**DrugCLIP (Gao et al., NeurIPS 2023).** The method that reframed virtual screening as dense
retrieval. Two encoders (pocket `g_φ`, molecule `f_θ`) produce L2-normalized embeddings; the
similarity of a pair is `s(p_i, m_j) = g_φ(p_i)ᵀ f_θ(m_j)`. Training is a symmetric in-batch
InfoNCE over a batch of `N` paired examples, with a scale `exp(logit_scale)` (init `log(14)`,
read with `.detach()` in the loss code) standing in for `1/τ`:

```
L^p = -(1/N) Σ_k log [ exp(s(p_k, m_k)/τ) / Σ_i exp(s(p_k, m_i)/τ) ]      # pocket retrieves mol
L^m = -(1/N) Σ_k log [ exp(s(p_k, m_k)/τ) / Σ_i exp(s(p_i, m_k)/τ) ]      # mol retrieves pocket
L   = (L^p + L^m) / 2.
```

To blunt the false-negative problem it masks logits of *exact* duplicates — a pocket or a molecule
string that appears more than once in the batch has those entries set to a large negative before
the softmax. At inference, molecule embeddings are precomputed and cached, and a query pocket is
scored against all of them by dot product, enabling billion-scale retrieval. **Gaps:** the training
target is the *diagonal* — exactly one positive ligand per pocket. An assay that screened dozens of
ligands of graded activity is forced into one-positive-per-pocket, so all the other genuine actives
of that target are treated as negatives; the loss separates binder-from-other but has *no notion of
stronger-versus-weaker*, so it does not specifically enrich the very top of the list with the
strongest binders; and its false-negative guard catches only exact string duplicates, not a
*different* active of the same target sharing the batch.

**LigUnity (Feng et al., 2025).** Extends the retrieval framework toward affinity ranking by adding
a listwise term on top of contrastive screening: within each assay, ligands are ordered by measured
affinity and a Plackett-Luce listwise loss (with a rank-position decay `μ_k = 1/(√B · log(k+1))`)
encourages the model to reproduce that order, while the contrastive term keeps the global
binder/non-binder separation. It thus learns both global interaction patterns and intra-assay
preference in one embedding space. **Gap:** a full listwise (Plackett-Luce) loss commits to a single
*total order* over all the assay's ligands. Within-assay affinities are noisy and only loosely
comparable, so a strict total order over-fits measurement noise and ties — it penalizes the model
for not separating two ligands whose affinities differ within experimental error.

**Plain Euclidean contrastive scoring (vanilla CLIP-style).** The minimal retrieval baseline:
L2-normalized Euclidean embeddings with the symmetric in-batch InfoNCE above, dot-product scoring.
**Gap:** it inherits DrugCLIP's one-positive assumption and has no ranking signal at all.

## Evaluation settings

The natural yardsticks, all zero-shot (no target-specific training; the screening test targets are
excluded from training by UniProt id):

- **DUD-E** (Mysinger et al. 2012): 102 protein targets, each with experimentally verified actives
  and ~50 property-matched decoys per active — tests enrichment against decoys chosen to be
  physicochemically similar but presumed non-binding.
- **LIT-PCBA** (Tran-Nguyen et al. 2020): 15 targets with >400K experimentally confirmed inactives
  — a realistic, harder setting free of synthetic-decoy bias.
- **DEKOIS 2.0**: 81 targets, a challenging decoy benchmark.
- **Metrics**, averaged across targets, all top-heavy except AUROC: **AUROC** (probability a random
  active outranks a random inactive); **BEDROC** with α=80.5 (Boltzmann-weighted early enrichment,
  ~80% of weight on the top ~2%); **EF** at 0.5% / 1% / 5% (fold-enrichment of actives in the top
  cut versus random). Higher is better for all. The strong top-heaviness of BEDROC/EF is the
  feature any objective must be tuned to.
- Protocol: backbone encoders, data loaders, training loop, and evaluation scripts are fixed;
  backbone parameters are loaded from pretrained weights and fine-tuned jointly with the scoring
  module. Training data is assay-organized (target + a set of screened ligands with activities),
  drawn from ChEMBL / BindingDB / PDBBind, with each batch grouping ligands by their pocket.

## Code framework

The scoring module plugs into a fixed harness. The backbones already produce features; the data
loader already groups a batch into per-pocket ligand spans (`batch_list`: for pocket `i`, its
ligands occupy columns `[s, e)`), carries each ligand's activity (`act_list`), and may carry
target and ligand identity metadata (`uniprot_*`, `*lig_smiles`). What is not settled is how to
project the backbone features, how to build the score matrix, which logits to suppress before a
softmax, and what training loss to put on top. The substrate is just the generic retrieval
machinery: projection heads, a pairwise similarity matrix, optional metadata masks, and a training
loop that hands the loss the batch's embeddings plus the grouping/activity metadata.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomScoring(nn.Module):
    """Scoring module on top of fixed, jointly fine-tuned backbones.
    Projects backbone features into a shared embedding space, defines the
    training loss over assay-grouped batches, and scores at inference."""

    def __init__(self, mol_dim=512, pocket_dim=512, protein_dim=480, embed_dim=128):
        super().__init__()
        # TODO: the projection heads and any embedding / loss parameters we will design.
        pass

    def project_mol(self, mol_feat):       # [B, 512] -> [B, embed_dim]
        # TODO
        pass

    def project_pocket(self, poc_feat):    # [B, 512] -> [B, embed_dim]
        # TODO
        pass

    def project_protein(self, prot_feat):  # [B, 480] -> [B, embed_dim]
        # TODO
        pass

    def compute_loss(self, mol_emb, poc_emb, prot_emb,
                     batch_list, act_list,
                     uniprot_poc=None, uniprot_mol=None,
                     pocket_lig_smiles=None, lig_smiles=None):
        # batch_list[i] = (s, e): pocket i's ligands are columns [s, e)
        # act_list[i]   = activities of pocket i's ligands
        # uniprot_*, *lig_smiles: optional target and ligand identity metadata
        # TODO: the training objective we will design.
        pass

    def score(self, mol_reps, pocket_reps, prot_reps=None):
        # evaluation scoring over numpy arrays; returns one score per molecule
        # TODO
        pass


# existing fixed training loop the scoring module plugs into
def train(model, scoring, data_loader, optimizer):
    for batch in data_loader:                       # assay-grouped batch
        optimizer.zero_grad()
        mol_feat, poc_feat, prot_feat = model(batch)        # fixed backbones (fine-tuned)
        mol_emb  = scoring.project_mol(mol_feat)
        poc_emb  = scoring.project_pocket(poc_feat)
        prot_emb = scoring.project_protein(prot_feat)
        loss, _ = scoring.compute_loss(
            mol_emb, poc_emb, prot_emb,
            batch.batch_list, batch.act_list,
            batch.uniprot_poc, batch.uniprot_mol,
            batch.pocket_lig_smiles, batch.lig_smiles,
        )
        loss.backward()
        optimizer.step()
```

The empty slots are the projection heads, the similarity/embedding choice, the training loss, and
the inference score — the scoring objective itself.
