Let me start from what actually breaks when I try to screen a real library. I have one protein target with a known pocket, and a library of molecules — and the libraries that matter are huge, $10^8$ to $10^9$ compounds, with the field's own statistics saying the bigger the library I can actually screen the more true ligands I find in the top thousand. So throughput is not a nicety; it is the whole game, because the hit rate is savage. Across realistic libraries the fraction of molecules that truly bind a given pocket is well under one in a thousand. That single number reframes everything. It tells me I am looking for needles, and it tells me that almost any molecule I grab at random is, with overwhelming probability, a non-binder — there is a free and essentially unlimited supply of correct negatives lying around, while the confirmed positives are the scarce thing.

What do people do today? Docking. Take the molecule, sample its poses in the pocket — genetic algorithms, Monte Carlo over the conformational space of ligand and receptor — and score each pose with a force-field function correlated with binding free energy, iterating to convergence. It works; on the hard benchmarks it is basically the only family that reliably beats chance. But the pose sampling is the bottleneck: order ten seconds per compound on a CPU core, which means ten billion compounds is something like three thousand years of compute. The cost scales *with the library*, which is exactly backwards for a regime whose entire premise is "use a bigger library." Docking cannot be the answer at this scale.

So people reach for learned scoring functions to skip the sampling. Regression models — take a protein-molecule representation, predict the numeric binding affinity, rank by predicted affinity. Three things go wrong and they all trace to the same root. First, regression needs reliable affinity labels, and there are only on the order of $10^4$ labeled complexes; that is a tiny, precious dataset. Second, those datasets are almost all positives, so the model never really learns what a non-binder looks like and false positives explode. Third — and this is the one that quietly kills you — at inference every single pocket-molecule pair needs a full forward pass through the scoring network, so screening a fixed library against many targets is (number of targets) times (size of library) network evaluations. That does not amortize at all. The classification variants are worse in a subtler way: they manufacture negatives from rules — take a pocket's actives, pad with property-matched decoys from ZINC — and train a binary classifier. It has been observed cleanly that such a model learns the *decoy-construction rule* and not binding: train on one benchmark's decoys, test on another built with different rules, and it collapses. The negatives you feed a contrastive boundary *are* what the model learns. Artificial decoys teach an artificial boundary.

Let me stare at what I am actually being asked for, because I think the whole field has been answering a harder question than the one screening poses. Docking predicts a *pose*. Regression predicts an *affinity value*. But for screening I do not need either of those. I need to know *which* molecules are likely to bind, ranked, in the top fraction I will ever look at. That is not regression and it is not pose prediction. That is a matching problem — given the pocket as a query, find the molecules in the library that go with it. It is retrieval.

The moment I say "retrieval," a template I trust snaps into focus. Dense passage retrieval for question answering had exactly this shape: a question, a corpus of millions of passages, find the relevant ones. And the thing that made it work at scale was a deliberate architectural restriction. They defined the relevance score as a plain dot product of two independently computed embeddings, $s(q,p) = E_Q(q)^\top E_P(p)$, one encoder for the query and a separate encoder for the passage. Not a cross-encoder that jointly attends over the (question, passage) pair — that is more expressive, but it forces a fresh forward pass for every pair, which is the exact non-amortizing cost that just sank the regression scorers. The dot-product form is *decomposable*: I can encode all the passages once, store the vectors, and then every query is just a nearest-neighbor search against a fixed index — and there are libraries built to do that search over billions of vectors. So the restriction to a factorized score is not a modeling concession, it is the whole reason the thing runs at scale.

Map that onto screening. The pocket is the query, the molecule library is the corpus. If I commit to a score of the form
$$ s(p, m) = g_\phi(p)^\top f_\theta(m), $$
one encoder $g_\phi$ for pockets, a separate encoder $f_\theta$ for molecules, then I encode the entire molecule library *offline*, cache the vectors, and online screening of a new pocket is: encode the pocket once, take dot products against the cache, sort. The expensive neural computation is paid once per molecule, ever, not once per (target, molecule) pair. That directly answers the throughput constraint that defeated docking and regression. Two towers, dot product, precompute — that is the skeleton.

Now there is a second, less obvious reason to want two towers, and I only see it when I think about *which* conformation of the molecule I feed in. At training time, in a solved complex, I have the molecule in its bound (holo) pose. But at screening time I have no bound pose — the whole point is I have not docked. A single-tower scorer eats the joint complex including the protein-ligand *cross*-distances, $k_\gamma(h_p, h_m, \text{distances between molecule and pocket atoms})$, and those cross-distances are precisely what you only know after you have placed the ligand in the pocket. So a single tower implicitly needs a pose, which means it needs docking, which is the cost I am trying to escape. Let me check whether the two-tower form actually escapes it or just hides it. Suppose I perturb the molecule's input conformation by $\delta$ and ask how much the score moves. Using an SE(3)-invariant encoder, everything is built from a relative-distance map $D(c_x, c_y)$ with entries $d_{ij} = \|x_i - x_j'\|$. To first order,
$$ s(\tilde{x}_m, x_p) - s(x_m, x_p) \approx \frac{\partial f_\theta}{\partial D(c_m, c_m)} \cdot \big[ D(c_m+\delta, c_m+\delta) - D(c_m, c_m) \big]. $$
The molecule tower only ever sees the molecule's *intra*-molecular distance map $D(c_m, c_m)$. If the perturbed conformation $c_m + \delta$ is just a rigid re-placement of the same shape — there exists a rotation $R$ and translation $t$ with $(c_m+\delta)R^\top + t = c_m$ — then the *pairwise* distances are completely unchanged, so $D(c_m+\delta, c_m+\delta) - D(c_m, c_m) = 0$ exactly, and the whole first-order deviation is zero. The molecule tower is blind to rigid re-posing; it only cares about the molecule's internal geometry. Contrast the single tower:
$$ k_\gamma(\ldots) \text{ deviation} \approx \frac{\partial k_\gamma}{\partial D(c_m,c_m)}[\,\cdot\,] + \frac{\partial k_\gamma}{\partial D(c_m, c_p)}\big[ D(c_m+\delta, c_p) - D(c_m, c_p) \big]. $$
The first term can vanish for the same reason, but the second — the molecule-to-pocket cross term — does *not*, because moving the molecule changes its distances to the pocket atoms, and there is no rotation that fixes both the molecule's internal distances and its distances to a fixed pocket simultaneously unless it is already correctly posed. So the single tower's score genuinely depends on the pose; the dual tower's does not. That means I can train the molecule tower on a *cheap, approximate* conformation — say one generated by a fast cheminformatics routine rather than docked — and the score is, to first order, insensitive to the error, as long as it is roughly the right shape. The dual-tower restriction I adopted for throughput turns out to also buy me freedom from docking at both train and test time. Good — two independent reasons now point at two towers.

For the encoders themselves I do not have to invent anything: there are pretrained SE(3)-invariant 3D Transformers that take atom types and coordinates and give an invariant representation, with a [CLS]-style atom at the centroid yielding one vector for the whole molecule or pocket. They were pretrained by masked atom-type prediction plus a 3D position-recovery task, and crucially the pair representation — the invariant spatial encoding from pairwise distances — enters self-attention as a bias, $\mathrm{softmax}(QK^\top/\sqrt{d} + q_{ij})V$, so 3D geometry is baked in. I will take one such encoder for molecules and a separately pretrained one for pockets, read off the [CLS] vector from each, and treat those as my raw features $f_\theta(x^m)$ and $g_\phi(x^p)$. The encoders are not the contribution; the scorer wrapped around them is.

So I have two towers and a dot-product score. Now: how do I train it? I have positive pairs — solved complexes, a pocket with a molecule that genuinely binds it. I have almost no labeled true negatives. What I *do* have, from the savage hit rate, is that any random other molecule is essentially certainly a non-binder. So the negatives should be *other molecules*, and the cheapest other molecules around are the ones already in my training batch. This is the in-batch-negatives idea from dense retrieval: take a batch of $N$ paired examples, form every cross pairing. Let me write the batch as $\{(x^p_k, x^m_k)\}_{k=1}^N$ and look at the full $N \times N$ similarity matrix $S$ with $S_{ij} = s(x^p_i, x^m_j)$. The diagonal entries $S_{kk}$ are the true binding pairs; every off-diagonal $S_{ij}$, $i \neq j$, is a pocket paired with some *other* example's molecule, which I treat as a negative. With one matrix multiply I get $N$ positives and $N(N-1)$ negatives for free — no external decoy mining, no rules to overfit. And because these negatives are just real drug-like molecules that happen to bind *some* pocket, they are realistic hard negatives, not artifacts of a construction rule. That sidesteps exactly the shortcut that made the rule-based classifiers fail to transfer.

How do I turn that matrix into a loss? The natural object is the contrastive / InfoNCE loss: among a positive and a pile of negatives, maximize the score of the positive relative to the rest. For a fixed pocket $k$, treat its row of $S$ as logits over candidate molecules and ask the model to pick out its true molecule:
$$ \ell^p_k = -\log \frac{\exp(S_{kk}/\tau)}{\sum_{i=1}^N \exp(S_{ki}/\tau)}. $$
This is just the categorical cross-entropy of identifying the right molecule among the $N$ in the batch. Why is this the right objective and not, say, a margin or a regression toward 1/0? Two reasons I can actually derive. First, the optimum of this loss is not an arbitrary calibration — it drives the learned score toward a density ratio: the contrastive optimum has $\exp(s) \propto p(\text{molecule} \mid \text{pocket}) / p(\text{molecule})$, i.e. exactly "how much more likely is this molecule given this pocket than at random," which is precisely the relevance signal I want for ranking, and notably *not* an absolute affinity value I would have to have labeled. Second, minimizing this loss maximizes a lower bound on the mutual information between pocket and molecule, $I(\text{pocket}; \text{molecule}) \ge \log N - \mathcal{L}_N$, and that bound *tightens as $N$ grows*. So more in-batch negatives is not just convenient, it is provably better — a larger batch makes the bound I am pushing on tighter. That is a concrete argument for wanting the batch as large as memory allows, and it falls straight out of the objective rather than being an empirical knob.

But wait — if I only ever train pocket-retrieves-molecule, I am shaping the space asymmetrically. The space should be just as good when a *molecule* is the query and I want its pocket (target fishing is a real task, and even setting that aside, a one-sided objective can let the molecule embeddings collapse into clusters that all look similar to a given pocket). The fix is symmetry: also read the matrix column-wise. For a fixed molecule $k$, treat the column as logits over pockets and identify its true pocket,
$$ \ell^m_k = -\log \frac{\exp(S_{kk}/\tau)}{\sum_{i=1}^N \exp(S_{ik}/\tau)}, $$
and combine the two directions,
$$ \mathcal{L} = \frac{1}{2N}\sum_{k=1}^N \big(\ell^p_k + \ell^m_k\big). $$
In code this is two cross-entropies with the identity as the target: one $\log\text{-softmax}$ over the rows of $S$ with targets $\{0,1,\dots,N-1\}$ (the diagonal), one over the rows of $S^\top$ with the same targets, combined as $0.5\,\text{loss\_pocket}+0.5\,\text{loss\_mol}$. If the loss function returns summed cross-entropies, the training wrapper carries the batch size as the normalizer; mathematically it is the same batch average. Symmetric, balanced, and it makes the embedding good for retrieval in both directions.

Now there is a quiet bug waiting in the in-batch construction, and I want to catch it before it poisons training. I assumed every off-diagonal cell is a negative. But what if two examples in the same batch share a pocket — the same pocket appears twice, paired with two different known binders — or two examples share the *same molecule*? Then an off-diagonal cell $S_{ij}$ might actually be a *true* binding pair that I am about to push *down* as a negative. That is a false negative, and on a sharp, low-hit-rate distribution false negatives are exactly the signal-corrupting mistake I cannot afford. So before forming the loss I should mask such cells out of the denominator. Concretely: build an indicator that is 1 where row $i$ and column $j$ share a pocket identity, plus 1 where they share a molecule identity, and subtract twice the identity so the genuine diagonal positives are *not* masked. Set those flagged off-diagonal cells to a large negative number, $-10^6$, before the softmax so they contribute nothing. The diagonal stays a clean positive; the accidental duplicates stop acting as fake negatives.

Let me pin down the temperature $\tau$, because it is doing more than it looks. The logits going into the softmax are similarities $S_{ij}$, and if I use cosine similarity those live in $[-1, 1]$ — a tiny range, over which the softmax is nearly flat and the loss can barely separate the positive from the negatives. I need to scale them up. I will store the scale on a log axis, $\rho=\log(1/\tau)$, because the scale is multiplicative and must stay positive, and I will initialize it so the similarities are multiplied by roughly fourteen. But I do not want the softmax to solve the task by moving only this scalar. When I apply the scale to the similarity matrix, I detach it: $S \leftarrow \exp(\rho)_{\text{stop-grad}}\cdot S$. The contrastive gradients then sharpen the embedding directions at a stable scale instead of flowing through the temperature.

Two more design choices in the scorer head, and I want their reasons, not just the shapes. First, I should not compute the dot product directly on the encoder's [CLS] feature. The contrastive objective is aggressive — it will warp whatever space it acts on to make the softmax happy, and if that space *is* the backbone feature, I degrade the rich, general representation the pretraining gave me. The clean separation is a projection head: a small network mapping the encoder feature into a dedicated comparison space, with the contrastive loss acting only there. The analysis from self-supervised vision is explicit that the representation *before* such a head generalizes better than the one after — so the head absorbs the contrastive-specific distortion and protects the backbone. I will make the head nonlinear, $\text{Linear}(d, d) \to \text{ReLU} \to \text{Linear}(d, 128)$, projecting to a modest 128-dimensional space (small enough to index cheaply, large enough to separate). A bare linear map would be a strictly weaker head with no reason to prefer it here. Second, after projecting I L2-normalize each embedding. Normalizing puts every vector on the unit sphere, which turns the dot product into cosine similarity — bounded, scale-free, and exactly the geometry the detached log-scale is meant to rescale. Without normalization the model could cheat the contrastive loss by inflating vector norms instead of improving directions, and the temperature scaling would be fighting an uncontrolled magnitude. Normalize, then apply the detached scale: that is a clean, well-posed pipeline.

So the molecule tower is: encoder $\to$ nonlinear projection head $\to$ L2-normalize, and identically for the pocket tower with its own head. The similarity is the dot product of the two normalized embeddings, scaled by the detached log-scale parameter, with duplicate cells masked, fed into the symmetric in-batch cross-entropy.

Let me deal with the train/test conformation gap concretely, since I argued the dual tower tolerates it. At training I have holo (bound) molecule conformations from solved complexes, but at test I will have no bound pose. To make training look like testing, I feed the molecule tower a *noisy, unbound* conformation even during training — generate one with a fast cheminformatics conformer routine and use those coordinates, $\tilde{x}_m = \{c_m + \delta, h_m\}$. By the first-order argument, since the tower only sees intra-molecular distances and those are nearly preserved under a rigid re-placement, the score barely moves; so training on the cheap conformation is consistent with testing on one, and I never invoke docking. This is the practical payoff of the robustness property, not a separate trick.

Now the evaluation score. At test I have one target, possibly described by several pocket instances (a target can present more than one pocket conformation), and a library of molecules to rank. I encode and project every molecule once into $\text{mol\_reps}$ and the target's pockets into $\text{pocket\_reps}$, then form the score matrix $\text{res} = \text{pocket\_reps} \cdot \text{mol\_reps}^\top$. Per molecule I need one number. Taking the *max* over the target's pockets is the right reduction: a molecule that binds strongly to *any* of the target's pocket conformations should rank high, and a max is exactly "best match among the available pockets." So $\text{score}(m) = \max_{\text{pocket}} \text{res}[\text{pocket}, m]$, then sort descending. That is the ranking I report, scored by the early-recognition metrics — BEDROC at $\alpha = 80.5$ and enrichment factor at the top 0.5%, 1%, 5% — because what matters is binders concentrated at the very top, not average behavior over the whole curve.

Let me put the whole thing into the scorer module, filling the empty slots — the two projection heads, the detached log-scale, the symmetric masked in-batch loss, and the max-over-pockets evaluation score.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NonLinearHead(nn.Module):
    """Projection head: Linear(d,d) -> ReLU -> Linear(d, embed_dim).
    Isolates the contrastive subspace so the backbone feature is protected."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, in_dim)
        self.linear2 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


class CustomScoring(nn.Module):
    """Two-tower contrastive scorer over pretrained pocket/molecule features.

    score(p, m) = <normalize(head_p(g(p))), normalize(head_m(f(m)))> * exp(rho).
    Trained by symmetric in-batch softmax; evaluated by max-over-pockets dot."""

    def __init__(self, mol_dim=512, pocket_dim=512, embed_dim=128):
        super().__init__()
        # separate projection heads, one per tower (towers stay independent
        # so embeddings precompute and index for billion-scale retrieval)
        self.mol_project = NonLinearHead(mol_dim, embed_dim)
        self.pocket_project = NonLinearHead(pocket_dim, embed_dim)
        # log-scale initialized so cosine sims are multiplied by ~14;
        # detached when applied to the logits
        self.logit_scale = nn.Parameter(torch.ones([1]) * np.log(14))

    def project_mol(self, mol_feat):
        # project then L2-normalize -> unit sphere -> dot product is cosine
        return F.normalize(self.mol_project(mol_feat), dim=-1)

    def project_pocket(self, poc_feat):
        return F.normalize(self.pocket_project(poc_feat), dim=-1)

    def compute_loss(self, mol_emb, poc_emb, pocket_ids, mol_ids):
        # full N x N similarity matrix, scaled by the detached log-scale.
        logits = poc_emb @ mol_emb.T * self.logit_scale.exp().detach()
        B = logits.size(0)

        # mask accidental in-batch duplicates so true pairs aren't pushed down
        # as false negatives (same pocket or same molecule appearing twice);
        # subtract 2*I so the genuine diagonal positives are NOT masked.
        pid = np.array(pocket_ids, dtype=str)[:, None]
        mid = np.array(mol_ids, dtype=str)[:, None]
        poc_dup = torch.tensor(pid == pid.T, dtype=logits.dtype, device=logits.device)
        mol_dup = torch.tensor(mid == mid.T, dtype=logits.dtype, device=logits.device)
        indicator = poc_dup + mol_dup - 2 * torch.eye(B, dtype=logits.dtype, device=logits.device)
        logits = logits + indicator * -1e6

        # symmetric in-batch cross-entropy; the positive for row/col k is k
        targets = torch.arange(B, dtype=torch.long, device=logits.device)
        loss_pocket = F.nll_loss(F.log_softmax(logits, dim=-1), targets, reduction="sum")
        loss_mol = F.nll_loss(F.log_softmax(logits.T, dim=-1), targets, reduction="sum")
        loss = 0.5 * loss_pocket + 0.5 * loss_mol
        return loss, {"loss": loss.item(),
                      "sample_size": B,
                      "loss_pocket": loss_pocket.item(),
                      "loss_mol": loss_mol.item()}

    def score(self, mol_reps, pocket_reps):
        # one target's library: score per molecule = best match over the
        # target's pocket conformations (a molecule binding ANY pocket ranks high)
        res = pocket_reps @ mol_reps.T          # [n_pockets, n_mols]
        return res.max(axis=0)                  # [n_mols]
```

The hit rate is below one in a thousand and the libraries are billions, so throughput at the top of the ranking is everything, and docking's per-compound pose sampling and regression's per-pair forward passes both scale the wrong way. Reframing screening as retrieval rather than affinity prediction or pose prediction lets me commit to a factorized dot-product score across two independent towers, which is what makes offline precomputation and billion-scale nearest-neighbor search possible — paying the encoder cost once per molecule, ever. A first-order argument on the SE(3)-invariant distance maps shows the dual tower is, unlike a single tower, insensitive to the molecule's exact pose, so I can train and test on cheap unbound conformations and never dock. The savage hit rate also hands me free, realistic negatives — every other molecule in the batch — so an in-batch contrastive softmax over the $N\times N$ similarity matrix trains the ranking directly, with no scarce affinity labels and no rule-based decoys to overfit; its optimum is a relevance density ratio and it maximizes a mutual-information bound that tightens with batch size, which is why I want the batch large. Symmetrizing over rows and columns balances the space for retrieval in both directions; masking accidental in-batch duplicates removes false negatives; a detached log-scale on L2-normalized embeddings gives a well-posed, scale-free softmax; and a nonlinear projection head keeps the contrastive distortion off the pretrained backbone. At inference, max-over-pockets reduces the score matrix to one number per molecule, ranked and read out by early-recognition metrics. Two towers, one dot product, one symmetric in-batch softmax — and a billion-molecule library becomes a single matrix of dot products against a cached index.
