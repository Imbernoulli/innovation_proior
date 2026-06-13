# DrugCLIP, distilled

DrugCLIP reformulates structure-based virtual screening as a dense-retrieval problem and
trains it with a CLIP-style symmetric in-batch contrastive objective. Two independent
SE(3)-invariant 3D encoders — one for protein pockets, one for molecules — map each input to a
vector; a small projection head plus L2-normalization put pockets and molecules in a shared
unit-sphere embedding space; the binding score is the dot product (cosine similarity) of the
two embeddings scaled by a detached log-scale initialized to $\exp(\log 14)$. Because the score
factorizes over the two
towers, every molecule in a library is encoded once, offline, and screening a new pocket is a
single matrix of dot products against the cached index, with no dependence on scarce affinity
labels or hand-built decoys.

## Problem it solves

Rank a $10^8$–$10^9$-compound library against a protein pocket so true binders land in the top
fraction (0.5%, 1%) that gets validated, fast enough for the whole library and without
target-specific labels — the zero-shot screening setting. Docking's per-compound pose sampling
(~10 s/compound) and supervised scorers' per-pair forward passes both scale the wrong way;
rule-based-decoy classifiers learn the decoy rule and don't transfer.

## Key idea

Screening needs *which* molecules bind, not the affinity value or the pose — a matching/retrieval
problem. Make the score decomposable, $s(p,m)=g_\phi(p)^\top f_\theta(m)$, so molecule embeddings
precompute and index. Train it contrastively with in-batch negatives: in a batch of $N$ binding
pairs, the $N\times N$ similarity matrix has the true pairs on the diagonal and $N(N-1)$ free,
realistic negatives off it (the sub-0.1% hit rate makes any other molecule almost surely a
non-binder). Optimize the symmetric softmax that picks the right molecule for each pocket and the
right pocket for each molecule.

## Why the design choices

- **Two towers + dot product (not a cross-encoder):** decomposability lets molecule embeddings be
  computed once and searched with billion-scale nearest-neighbor libraries (FAISS); a
  cross-encoder needs a fresh pass per pair.
- **Dual tower also removes docking:** with an SE(3)-invariant encoder built on relative-distance
  maps $D$, a first-order expansion gives the molecule-tower score deviation under a conformation
  perturbation $\delta$ as $\propto \partial f_\theta/\partial D(c_m,c_m)\cdot[D(c_m+\delta,c_m+\delta)-D(c_m,c_m)]$,
  which is $0$ when the perturbation is a rigid re-placement (intra-molecular distances unchanged),
  so $\lim_{\delta\to0} \{s(\tilde x_m,x_p)-s(x_m,x_p)\}=0$. A single tower keeps a
  molecule-to-pocket cross term $\partial k_\gamma/\partial D(c_m,c_p)\cdot[D(c_m+\delta,c_p)-D(c_m,c_p)]$
  that does **not** vanish, so it needs a docked pose. Hence DrugCLIP trains and tests on cheap
  RDKit conformations and never docks.
- **In-batch contrastive (not regression / rule-based decoys):** no affinity labels needed; the
  contrastive optimum drives $\exp(s)\propto p(m\mid p)/p(m)$ (a relevance density ratio, exactly
  the ranking signal); minimizing it maximizes $I(p;m)\ge \log N-\mathcal{L}_N$, tightening with
  batch size — so larger batches are provably better. In-batch negatives are real molecules, not
  artifacts of a construction rule, avoiding the shortcut that breaks rule-based classifiers.
- **Symmetric loss:** balances the space for both pocket→molecule screening and molecule→pocket
  target fishing, and prevents one-sided embedding collapse.
- **Duplicate masking:** off-diagonal cells sharing a pocket id or molecule id are true pairs;
  mask them to $-10^6$ before softmax (subtract $2I$ so the diagonal positives stay) to avoid
  false negatives.
- **Detached log-scale on normalized embeddings:** cosine sims live in $[-1,1]$, too flat for the
  softmax; the scorer stores $\rho=\log(14)$ and applies $\exp(\rho).\text{detach}()$ to the
  logits. L2-normalization makes the geometry scale-free, and the detached scale keeps the
  contrastive gradients focused on embedding directions rather than on changing a scalar
  temperature.
- **Nonlinear projection head:** the contrastive objective distorts whatever space it acts on; a
  $\text{Linear}\to\text{ReLU}\to\text{Linear}$ head absorbs that distortion and keeps it off the
  pretrained backbone (the pre-head representation generalizes better).
- **max-over-pockets at inference:** a target may have several pocket conformations; a molecule
  binding *any* of them should rank high, so reduce the score matrix by the max over pockets.

## Final objective

Per pocket $k$ (row) and molecule $k$ (column) of the temperature-scaled similarity matrix:
$$
\mathcal{L}^p_k = -\log\frac{\exp(s(x^p_k,x^m_k)/\tau)}{\sum_i \exp(s(x^p_k,x^m_i)/\tau)},\qquad
\mathcal{L}^m_k = -\log\frac{\exp(s(x^p_k,x^m_k)/\tau)}{\sum_i \exp(s(x^p_i,x^m_k)/\tau)},
$$
$$
\mathcal{L}_{\text{avg}} = \tfrac{1}{2N}\sum_{k=1}^N\big(\mathcal{L}^p_k + \mathcal{L}^m_k\big),
\qquad s(x^p_i,x^m_j)=\widehat{g_\phi(x^p_i)}^\top\,\widehat{f_\theta(x^m_j)},
$$
with $\widehat{\cdot}$ denoting L2-normalized projection-head outputs and
$1/\tau=\exp(\rho)$ applied with stop-gradient. In the code form, each directional cross-entropy
is accumulated with the identity as target over the rows of $S$ and $S^\top$, then combined as
$0.5\,\text{loss\_pocket}+0.5\,\text{loss\_mol}$; the training wrapper carries $N$ as the sample
size for normalization.

## Training and inference

Encoders: separately pretrained SE(3)-invariant 3D Transformers (UniMol), [CLS] vector per
molecule/pocket, fine-tuned jointly with the heads. Optimizer Adam, learning rate $10^{-3}$, batch
size 192, up to 200 epochs, validation on CASF-2016 by $\text{BEDROC}_{85}$. Molecule input uses
RDKit conformations (consistency with the pose-free test setting); biological data augmentation
(HomoAug) pairs PDBBind ligands with homologous pockets. Inference: precompute and cache all
molecule embeddings; encode a query pocket, dot-product against the cache, max over the target's
pockets, sort. Benchmarks DUD-E / LIT-PCBA / DEKOIS 2.0, zero-shot (benchmark targets excluded
from training), metrics AUROC, $\text{BEDROC}_{\alpha=80.5}$, $\text{EF}@\{0.5,1,5\}\%$ where
$\text{EF}_\alpha=\text{NTB}_\alpha/(\text{NTB}_t\cdot\alpha)$.

## Working code

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NonLinearHead(nn.Module):
    """Projection head: Linear(d, d) -> ReLU -> Linear(d, embed_dim)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, in_dim)
        self.linear2 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


class DrugCLIP(nn.Module):
    """Two-tower contrastive scorer over pretrained pocket/molecule encoders.

    score(p, m) = <normalize(head_p(g(p))), normalize(head_m(f(m)))> * exp(rho)
    Trained by symmetric in-batch softmax; evaluated by max-over-pockets dot.
    """

    def __init__(self, mol_dim=512, pocket_dim=512, embed_dim=128):
        super().__init__()
        self.mol_project = NonLinearHead(mol_dim, embed_dim)
        self.pocket_project = NonLinearHead(pocket_dim, embed_dim)
        # log-scale initialized so cosine sims are multiplied by ~14
        self.logit_scale = nn.Parameter(torch.ones([1]) * np.log(14))

    def project_mol(self, mol_feat):
        return F.normalize(self.mol_project(mol_feat), dim=-1)

    def project_pocket(self, poc_feat):
        return F.normalize(self.pocket_project(poc_feat), dim=-1)

    def compute_loss(self, mol_emb, poc_emb, pocket_ids, mol_ids):
        # N x N similarity, scaled by detached log-scale.
        logits = poc_emb @ mol_emb.T * self.logit_scale.exp().detach()
        B = logits.size(0)

        # mask accidental in-batch duplicates (same pocket / same molecule)
        # so true pairs aren't treated as negatives; keep the diagonal.
        pid = np.array(pocket_ids, dtype=str)[:, None]
        mid = np.array(mol_ids, dtype=str)[:, None]
        poc_dup = torch.tensor(pid == pid.T, dtype=logits.dtype, device=logits.device)
        mol_dup = torch.tensor(mid == mid.T, dtype=logits.dtype, device=logits.device)
        indicator = poc_dup + mol_dup - 2 * torch.eye(B, dtype=logits.dtype, device=logits.device)
        logits = logits + indicator * -1e6

        # symmetric in-batch cross-entropy with the identity as target
        targets = torch.arange(B, dtype=torch.long, device=logits.device)
        loss_pocket = F.nll_loss(F.log_softmax(logits, dim=-1), targets, reduction="sum")
        loss_mol = F.nll_loss(F.log_softmax(logits.T, dim=-1), targets, reduction="sum")
        loss = 0.5 * loss_pocket + 0.5 * loss_mol
        return loss, {"loss": loss.item(),
                      "sample_size": B,
                      "loss_pocket": loss_pocket.item(),
                      "loss_mol": loss_mol.item()}

    @torch.no_grad()
    def score(self, mol_reps, pocket_reps):
        # one target's library: per-molecule score = best over the target's pockets
        res = pocket_reps @ mol_reps.T          # [n_pockets, n_mols]
        return res.max(axis=0)                  # [n_mols]
```
