# Context: large-scale structure-based virtual screening (circa 2022-2023)

## Research question

Given one protein target with a known binding pocket and a compound library of $10^8$–$10^9$
small molecules, rank the library so that the few true binders sit at the very top, where a
handful can be pulled for wet-lab validation. The accepted wisdom is "bigger is better": the
larger the library that can actually be screened, the higher the chance of finding a viable
drug candidate — statistics show the number of true ligands in the top-1000 jumps sharply as
the library grows from $10^5$ to $10^8$. The binding constraint is not accuracy on any one
pair; it is throughput at fixed (and very small) hit rate. The goal is a scoring function
$s(\text{pocket}, \text{molecule})$ that ranks true binders above non-binders specifically in
the top fraction (0.5%, 1%) that will ever be looked at, evaluated over a billion molecules in
feasible time, without depending on scarce binding-affinity labels or hand-built decoy sets, and
generalizing to protein targets never seen in training — the realistic "zero-shot" screening
setting.

## Background

Virtual screening is a core computer-aided drug discovery step. The field state rests on a few
load-bearing facts and observed phenomena, all knowable before any new method:

- **The hit rate is brutally low.** Across realistic libraries the proportion of true binders
  to a given pocket is far below 0.1%. This is the central statistical fact: a molecule drawn
  at random from the library is, with overwhelming probability, a non-binder. It means the
  problem is one of finding needles, and it means an essentially unlimited supply of *almost
  certainly correct* negative examples is lying around for free, while confirmed positives are
  rare and precious.

- **Binding is governed by 3D structure, and structure is increasingly available.** Whether a
  molecule binds depends on the 3D complementarity between the ligand's conformation and the
  pocket's shape and chemistry. Advances in cryo-EM and high-accuracy structure prediction have
  made many more protein pockets available as 3D coordinates, which makes structure-based
  approaches broadly applicable.

- **Representation learning for 3D molecules and pockets exists and is strong.** A pretrained
  SE(3)-invariant 3D Transformer (UniMol; Zhou et al. 2023) encodes a molecule or a pocket
  given atom types and 3D coordinates. It maintains an atom representation and a pair
  representation: the pair representation is an invariant spatial positional encoding built from
  pairwise Euclidean distances through a Gaussian kernel, and it enters self-attention as a bias
  term, $\text{Attention}(Q,K,V)=\mathrm{softmax}(QK^\top/\sqrt{d}+q_{ij})V$, with the pair
  representation itself updated by $q_{ij}^{l+1}=q_{ij}^{l}+Q_iK_j^\top/\sqrt{d}$. It is
  pretrained, BERT-style, by masked atom-type prediction together with a 3D position-recovery
  task — corrupt 15% of atom coordinates with uniform noise in $[-1,1]\,\text{Å}$ and recover
  the original pair distances and coordinates. A special [CLS] atom placed at the centroid
  yields a single whole-molecule or whole-pocket vector. Molecule and pocket encoders are
  pretrained separately on their own large corpora.

- **Contrastive representation learning is mature.** The InfoNCE objective (Oord et al. 2018;
  N-pair loss, Sohn 2016) trains a scoring function $f(x,c)$ by the categorical cross-entropy
  of picking the one positive out of a set of one positive and $N-1$ negatives,
  $\mathcal{L}_N=-\mathbb{E}\big[\log \frac{f(x,c)}{\sum_{x_j} f(x_j,c)}\big]$. Two facts about
  it are load-bearing: its optimum has $f(x,c)\propto p(x|c)/p(x)$, a density ratio that is
  exactly a relevance/binding signal rather than an absolute value; and minimizing it maximizes
  a lower bound on mutual information, $I(x;c)\ge \log N-\mathcal{L}_N$, which *tightens as the
  number of negatives $N$ grows*. In vision-language pretraining (CLIP; Radford et al. 2021) the
  same objective is applied symmetrically across two modalities, with cosine similarity, a
  learnable log-parameterized temperature (initialized so the similarity is scaled by $\sim$14
  and clipped so the scale never exceeds 100, which they found necessary to avoid training
  instability), and projection heads onto a shared space. SimCLR's analysis adds that a
  projection head onto the contrastive space — and L2-normalizing before computing similarity —
  matters: the representation *before* the projection head generalizes better than the one
  after, so the contrastive objective should act through a small head rather than on the
  backbone feature directly.

- **Decomposable similarity enables offline indexing.** Dense retrieval in NLP (DPR; Karpukhin
  et al. 2020) established that if the scoring function factorizes as $s(q,p)=E_Q(q)^\top E_P(p)$
  — a dot product of independently computed embeddings — then all document embeddings can be
  computed once, stored, and searched with libraries built for billion-scale nearest-neighbor
  search (FAISS). The contrast is with a cross-encoder that jointly attends over the pair: more
  expressive, but every query-document pair needs a fresh forward pass, which does not amortize.

- **Rule-based decoys leak a shortcut.** It is documented (Wang et al. 2022) that classifiers
  trained to separate actives from rule-constructed decoys (e.g. DUD-E's property-matched ZINC
  decoys) learn the *decoy-construction rule* rather than binding, and fail to transfer to other
  benchmarks. This is a diagnostic observation about existing systems: the negatives you train
  on shape what the model learns, and artificial decoys teach an artificial boundary.

## Baselines

These are the prior families a new screening scorer is measured against and reacts to.

**Molecular docking (Glide, AutoDock Vina, Gold).** Model the protein-molecule interaction
physically: sample candidate ligand poses (genetic algorithms, Monte Carlo) by exploring the
conformational space of ligand and receptor, then evaluate each pose with a scoring function
(empirical force fields) correlated with binding free energy, iterating to convergence. This is
the dominant method and the only family that reliably beats chance on hard benchmarks.

**Supervised regression scorers (DeepDTA, OnionNet, GraphDTA, SG-CNN).** Learn a map from a
protein-molecule representation to a numeric binding-affinity value, train by regression on
labeled affinities, then rank molecules by predicted affinity.

**Supervised classification scorers (DrugVQA, AttentionSiteDTI).** Obtain negatives from
predefined rules (e.g. DUD-E) and train a binary classifier to separate active from inactive
protein-molecule pairs.

**Single-tower 3D scorers (e.g. OnionNet, SG-CNN style).** Feed the joint protein-ligand
complex — including the relative protein-ligand geometry — into one network that outputs a
score, $k_\gamma(h_p, h_m, \text{distances})$. This captures interaction geometry directly.

## Evaluation settings

The natural yardsticks already in use, all pre-existing:

- **DUD-E** — 102 protein targets, 22,886 bioactive molecules, each target padded with 50
  topologically dissimilar but physicochemically property-matched decoys retrieved from ZINC.
  The standard structure-based screening benchmark.
- **LIT-PCBA** — 15 targets, 7,844 experimentally confirmed actives and 407,381 confirmed
  inactives derived from dose-response PubChem bioassays; built specifically to remove the
  artificial-decoy bias of DUD-E, so it matches realistic hit rates and is much harder.
- **DEKOIS 2.0** — 81 targets, a challenging decoy benchmark.
- **Zero-shot protocol** — to mimic real screening against new targets, train on an external
  dataset (e.g. the labeled-complex set) with all benchmark targets *excluded*, then test
  directly with no target-specific tuning. The pretrained molecule and pocket encoders, the
  labeled-complex training data, and the RDKit conformation generator are the available
  ingredients.
- **Metrics — early-recognition, not global.** AUROC is the familiar classification metric but
  averages the false-positive rate over $[0,1]$, which is the wrong emphasis when only the top
  fraction is ever inspected. The screening-appropriate metrics are: BEDROC with $\alpha=80.5$
  (exponentially up-weights early ranks),
  $\text{BEDROC}_\alpha=\frac{\sum_{i=1}^{\text{NTB}_t}e^{-\alpha r_i/N}}{R_\alpha\frac{1-e^{-\alpha}}{e^{\alpha/N}-1}}\cdot\frac{R_\alpha\sinh(\alpha/2)}{\cosh(\alpha/2)-\cosh(\alpha/2-\alpha R_\alpha)}+\frac{1}{1-e^{\alpha(1-R_\alpha)}}$;
  and the enrichment factor $\text{EF}_\alpha=\frac{\text{NTB}_\alpha}{\text{NTB}_t\cdot\alpha}$
  at $\alpha\in\{0.5\%,1\%,5\%\}$ — how many times more actives appear in the top $\alpha$
  fraction than random would give. Higher is better for all.

## Code framework

A scoring module plugs into a fixed harness: pretrained molecule and pocket encoders produce
per-instance feature vectors (a molecule CLS feature `mol_feat` of dim 512, a pocket CLS
feature `poc_feat` of dim 512), a data pipeline yields batches of paired pocket-molecule
instances along with bookkeeping (which molecules pair with which pocket, and identifiers to
spot repeats in a batch), an optimizer and training loop call the module's loss, and an
evaluation loop calls the module's scorer over a target's molecules and reports the ranking
metrics. What is *not* settled — and is exactly what is to be designed — is everything inside
the scoring module: how to map each encoder feature into whatever space the comparison happens
in, what the training objective is, and how a final score is computed at evaluation. So the
substrate is only the generic machinery that already exists, with empty slots for the module.

```python
import torch
import torch.nn as nn


class CustomScoring(nn.Module):
    """Scoring module over pretrained pocket/molecule encoder features.

    The encoders are fixed (fine-tuned jointly upstream); this module owns the
    mapping of their features into a comparison space, the training objective,
    and the evaluation score. All of that is open."""

    def __init__(self, mol_dim=512, pocket_dim=512, embed_dim=128):
        super().__init__()
        # TODO: the parameters this module needs (the feature->space mapping,
        #       and any objective parameters we will design).
        pass

    def project_mol(self, mol_feat):
        # TODO: map a molecule feature [B, mol_dim] into the comparison space.
        pass

    def project_pocket(self, poc_feat):
        # TODO: map a pocket feature [B, pocket_dim] into the comparison space.
        pass

    def compute_loss(self, mol_emb, poc_emb, batch_list, **bookkeeping):
        # mol_emb / poc_emb: this module's projections of a batch.
        # batch_list: which molecules go with which pocket (the positive pairs).
        # bookkeeping: identifiers for spotting repeats within the batch.
        # TODO: the training objective we will design, plus optional stats.
        pass

    def score(self, mol_reps, pocket_reps):
        # mol_reps / pocket_reps: numpy projections for one target's library.
        # TODO: the evaluation score per molecule we will design.
        pass


# fixed training loop the module plugs into
def train(model, scoring, data_loader, optimizer):
    for batch in data_loader:                       # paired pocket/molecule instances
        optimizer.zero_grad()
        mol_feat, poc_feat = model.encode(batch)    # fixed pretrained encoders
        mol_emb = scoring.project_mol(mol_feat)
        poc_emb = scoring.project_pocket(poc_feat)
        loss, stats = scoring.compute_loss(mol_emb, poc_emb, batch.pairs, **batch.bookkeeping)
        loss.backward()                             # encoders fine-tuned jointly
        optimizer.step()
```
