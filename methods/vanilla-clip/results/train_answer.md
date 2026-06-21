The task I actually face in structure-based virtual screening is narrower and harsher than the one the field usually answers. I have a single protein target with a known pocket and a library of $10^8$–$10^9$ small molecules, and I need to rank that library so the few true binders sit at the very top — the top $0.5\%$ or $1\%$ that anyone will ever pull for wet-lab validation. The hit rate is brutal: across realistic libraries far fewer than one molecule in a thousand truly binds a given pocket. That single number reframes everything. It tells me I am hunting needles, and it tells me that almost any molecule grabbed at random is, with overwhelming probability, a non-binder — so realistic negatives are free and essentially unlimited, while confirmed positives are scarce and precious. The binding constraint is therefore not accuracy on any one pair; it is throughput at a fixed and tiny hit rate, on a target never seen in training.

Against this, every existing family answers a harder question than screening poses and pays for it at scale. Docking samples ligand poses in the pocket and scores each with a force-field function correlated with binding free energy; it is the one family that reliably beats chance on hard benchmarks, but its per-compound pose sampling runs on the order of ten seconds on a CPU core, so $10^{10}$ compounds is roughly three thousand years of compute — the cost grows *with* the library, exactly backwards for a regime whose premise is "bigger is better." Supervised regression scorers predict a numeric affinity and rank by it, but they need reliable affinity labels (only $\sim 10^4$ labeled complexes exist), they see almost no true negatives so false positives explode, and at inference every pocket-molecule pair needs a full forward pass, so a library against many targets costs (#targets) $\times$ (#library) network evaluations — no amortization. Classification scorers manufacture negatives from rules (property-matched ZINC decoys), and it is documented that they learn the *decoy-construction rule* rather than binding and collapse on benchmarks built with different rules. Single-tower 3D scorers eat the joint complex including the protein-ligand cross-distances, which are only known once the ligand is posed — so they implicitly require docking and inherit its cost and its pose-sensitivity.

I propose DrugCLIP. The move that unlocks everything is to stop predicting a pose or an affinity value and recognize that screening is a *matching* problem: given the pocket as a query, find the molecules in the library that go with it. That is retrieval, and the template I trust from dense passage retrieval is a deliberate architectural restriction — define the relevance score as a plain dot product of two independently computed embeddings rather than a cross-encoder that jointly attends over the pair. Committing to the factorized form
$$ s(p, m) = g_\phi(p)^\top f_\theta(m), $$
with one encoder $g_\phi$ for pockets and a separate encoder $f_\theta$ for molecules, is not a modeling concession; it is the whole reason the thing runs at scale. Because the score decomposes over the two towers, I encode the entire molecule library once, offline, cache the vectors, and screening a new pocket becomes a single matrix of dot products against the cached index searched with billion-scale nearest-neighbor libraries. The expensive neural cost is paid once per molecule, ever, not once per (target, molecule) pair — which directly answers the throughput constraint that sank docking and regression.

There is a second, less obvious reason for two towers, and it removes docking entirely. With an SE(3)-invariant encoder everything is built from a relative-distance map $D(c_x, c_y)$. Perturbing the molecule's input conformation by $\delta$ and expanding to first order, the molecule tower's score deviation is
$$ s(\tilde{x}_m, x_p) - s(x_m, x_p) \approx \frac{\partial f_\theta}{\partial D(c_m, c_m)} \cdot \big[ D(c_m+\delta, c_m+\delta) - D(c_m, c_m) \big], $$
because the tower only ever sees the molecule's *intra*-molecular distances. If $c_m+\delta$ is merely a rigid re-placement of the same shape — a rotation and translation that map it back to $c_m$ — those pairwise distances are unchanged, so the bracket is exactly zero and the deviation vanishes. A single tower instead keeps a molecule-to-pocket cross term $\frac{\partial k_\gamma}{\partial D(c_m, c_p)}\big[ D(c_m+\delta, c_p) - D(c_m, c_p) \big]$ that does *not* vanish, since no rotation simultaneously fixes the molecule's internal distances and its distances to a fixed pocket unless it is already correctly posed. So the dual tower is blind to rigid re-posing while the single tower genuinely depends on the pose. The practical payoff is that I can feed the molecule tower a cheap, unbound RDKit conformation both at training and at test — the score is, to first order, insensitive to the error as long as the shape is roughly right — and I never invoke docking. For the encoders themselves I take pretrained SE(3)-invariant 3D Transformers (UniMol), separately pretrained for molecules and pockets by masked atom-type prediction plus 3D position recovery, with the pair (distance) representation entering self-attention as a bias; I read off the [CLS] vector from each as my raw feature. The encoders are not the contribution; the scorer wrapped around them is.

For training I have positive pairs from solved complexes and almost no labeled true negatives, but the savage hit rate hands me negatives for free: any other molecule in the batch is, almost surely, a non-binder. This is the in-batch-negatives idea. For a batch $\{(x^p_k, x^m_k)\}_{k=1}^N$, form the full $N \times N$ similarity matrix $S$ with $S_{ij} = s(x^p_i, x^m_j)$; the diagonal holds the true binding pairs and every off-diagonal cell is a realistic negative — $N$ positives and $N(N-1)$ negatives from one matrix multiply, with no decoy mining and no construction rule to overfit. I turn the matrix into an InfoNCE / contrastive loss: for pocket $k$, treat its row as logits over candidate molecules and identify the true one,
$$ \ell^p_k = -\log \frac{\exp(S_{kk}/\tau)}{\sum_{i=1}^N \exp(S_{ki}/\tau)}. $$
This is the right objective for two derivable reasons. Its optimum drives the score toward a density ratio, $\exp(s) \propto p(m \mid p)/p(m)$ — exactly the relevance signal I want for ranking, and notably not an absolute affinity I would have to label. And minimizing it maximizes a mutual-information lower bound, $I(p; m) \ge \log N - \mathcal{L}_N$, which *tightens as $N$ grows* — so a larger batch is provably better, a concrete reason to push the batch as large as memory allows rather than an empirical knob. Training only pocket-retrieves-molecule would shape the space asymmetrically and risk molecule-embedding collapse, so I symmetrize, reading the matrix column-wise as well,
$$ \ell^m_k = -\log \frac{\exp(S_{kk}/\tau)}{\sum_{i=1}^N \exp(S_{ik}/\tau)}, \qquad \mathcal{L} = \frac{1}{2N}\sum_{k=1}^N \big(\ell^p_k + \ell^m_k\big). $$
In code this is two cross-entropies with the identity as target — one over the rows of $S$, one over the rows of $S^\top$ — combined as $0.5\,\text{loss\_pocket}+0.5\,\text{loss\_mol}$.

Three details make it well-posed. First, the in-batch construction hides a bug: if two examples share a pocket or share a molecule, an off-diagonal cell can be a *true* binding pair that I would wrongly push down — a false negative, which is exactly the signal-corrupting mistake I cannot afford on a sharp, low-hit-rate distribution. So before the softmax I build an indicator that is $1$ where row $i$ and column $j$ share a pocket id plus $1$ where they share a molecule id, subtract $2I$ so the genuine diagonal positives are *not* masked, and add $-10^6$ to the flagged cells so they vanish from the denominator. Second, the temperature: cosine similarities live in $[-1,1]$, a range over which the softmax is nearly flat, so I scale them up. I store the scale on a log axis, $\rho = \log(1/\tau)$, because it is multiplicative and must stay positive, and initialize it so similarities are multiplied by roughly fourteen, $\rho = \log 14$. Crucially I *detach* the scale when applying it, $S \leftarrow \exp(\rho)_{\text{stop-grad}}\cdot S$, so the contrastive gradients sharpen embedding directions at a stable scale rather than letting the softmax cheat by moving one scalar. Third, two head choices: I do not compute the dot product on the raw [CLS] feature, because the aggressive contrastive objective would warp the backbone and degrade the general pretrained representation; instead a small nonlinear projection head, $\text{Linear}(d,d) \to \text{ReLU} \to \text{Linear}(d,128)$, absorbs the contrastive-specific distortion and keeps it off the backbone — the pre-head representation generalizes better, and a bare linear head would be strictly weaker for no gain. After projecting I L2-normalize, putting every vector on the unit sphere so the dot product is cosine similarity — bounded and scale-free, exactly the geometry the detached log-scale rescales; without normalization the model could cheat by inflating norms instead of improving directions. So each tower is encoder $\to$ nonlinear head $\to$ L2-normalize, and the similarity is the detached-scaled dot product of the two normalized embeddings, masked and fed into the symmetric in-batch cross-entropy.

At inference a target may be described by several pocket conformations, so I encode and project every molecule once into the cache and the target's pockets, form $\text{res} = \text{pocket\_reps}\cdot\text{mol\_reps}^\top$, and reduce per molecule by the *max* over pockets — a molecule that binds strongly to *any* of the target's pocket conformations should rank high. Sort descending and read out by early-recognition metrics, $\text{BEDROC}$ at $\alpha = 80.5$ and enrichment factor at the top $\{0.5, 1, 5\}\%$, because what matters is binders concentrated at the very top, not average behavior over the whole curve. Two towers, one dot product, one symmetric in-batch softmax — and a billion-molecule library becomes a single matrix of dot products against a cached index.

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
