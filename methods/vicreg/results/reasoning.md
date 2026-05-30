OK, let me think this through from scratch. I want representations of images learned without labels, and the lever everyone uses is augmentation invariance: take an image, make two distorted views, push both through a network, and ask for the two embeddings to land in the same place. Whatever survives random crops, color jitter, blur, grayscale — that's the semantic content. So the core of the objective wants to be "make the embedding of view A equal to the embedding of view B."

But the instant I write that down I see the rot in it. A network that emits a constant vector regardless of its input satisfies "A equals B" perfectly — every pair agrees, the loss is zero, and the representation tells me nothing. Collapse. So "maximize similarity between the two views" is degenerate on its own, and the entire field has been about bolting some extra contraption onto that objective to make the constant solution unappealing. Let me enumerate the contraptions, because I want to understand exactly what each one buys, what it costs, and whether I can throw the whole genre away.

The contrastive route — SimCLR, MoCo, InfoNCE. I don't just pull the two views of one image together; I also push each image away from every *other* image in the batch. On l2-normalized embeddings it's a softmax over cosine similarities: bring the positive pair close, drive all the in-batch negatives apart. Now the constant solution is catastrophic — if everything maps to one point, every negative inner product is maximal and the denominator explodes — so collapse is dead. But what did it cost? That repulsion term is, in effect, a non-parametric estimate of the spread of the embedding distribution computed from all the pairwise distances in the batch. Spread estimates from pairwise distances in high dimensions are brutally sample-hungry, which is exactly why SimCLR wants enormous batches and MoCo has to hoard a queue of tens of thousands of past embeddings. I'm chained to a sample-space spread estimate, and the chain is the batch size.

The clustering route — DeepCluster, SwAV. Assign views to clusters, force the two views of an image into the same cluster, and keep the assignment balanced so clusters don't go empty. SwAV does this online with a Sinkhorn–Knopp balancing step. It's really contrastive at the level of clusters — it still leans on a pile of prototype vectors and a balancing constraint to avoid its own flavor of collapse. Same genre of fragility, just moved up a level.

The asymmetric route — BYOL, SimSiam. Throw the negatives away entirely. The loss is just a similarity between the two views, which on its own *should* collapse — and yet it doesn't, because they break the symmetry of the branches. One branch gets a predictor MLP stacked on top; the other is treated as a detached target (SimSiam stops the gradient through it; BYOL additionally keeps the target as a slow exponential moving average of the online weights). SimSiam's ablation is the sharp diagnostic: knock out the stop-gradient and it collapses to a constant instantly — the stop-gradient is load-bearing. It's batch-size robust, since there's no cross-sample term at all. But stare at what it *is*: there's no single scalar being minimized. Genuine trivial solutions to that similarity loss exist; the method just never reaches them because of an asymmetry and the resulting dynamics, and nobody can cleanly say why. And the asymmetry forces a *dependency* between the branches — shared weights, or one being an EMA of the other. That bothers me on two counts. I don't want "it empirically doesn't collapse if you arrange the gradients just so," and I don't want the two branches welded together.

Information-maximization — Barlow Twins, W-MSE. These feel closest to what I want, because they replace the trick with an actual statement about what a good representation *is*: it should be informative, and a clean operational proxy for "informative" is "the dimensions don't duplicate each other" — decorrelate them. Barlow Twins builds the normalized cross-correlation matrix between the two branches' embeddings and drives it to the identity: diagonal to 1 is invariance, off-diagonal to 0 is decorrelation. W-MSE whitens the embeddings (eigen-decompose the covariance, apply the inverse square root) and spreads them on the sphere. I like the decorrelation idea a lot — it kills the "copy one good direction into all dimensions" cheat, which is informational collapse — but both have warts. W-MSE has to invert a covariance matrix, which is costly and unstable, and it builds the whitening operator over several batches, biasing the loss unless you slice batches. And Barlow's cross-correlation is computed *between* the two branches — it couples them. Each matrix entry mixes a dimension of branch A with a dimension of branch B, so the method implicitly wants the two branches to have similar output statistics; that's exactly the coupling I'm trying to escape if I ever want the branches to be different architectures or different modalities. Barlow also has to *standardize* the embeddings (divide each dimension by its batch standard deviation) for the cross-correlation entries to be correlations in [−1,1] — that's a batch-normalization-like operation, and without it the target value of 1 on the diagonal is arbitrary and the pre-standardization embeddings can shrink to numerical zero. So even the method I like best smuggles in normalization and inter-branch coupling.

Here's the shape of my dissatisfaction, now sharp. Everyone treats "agree" as the objective and "don't collapse" as a side condition enforced by machinery — negatives, asymmetry, cluster balancing, whitening, cross-branch coupling. And every mechanism that prevents collapse also ties the two branches together. What I want is to prevent collapse with a constraint applied to **each branch by itself**, so explicit that I can point at the term and say "this is the line that forbids the constant," and so local to one branch that the two branches need share nothing — not weights, not architecture, not even input modality.

So let me ask the most literal possible question. The constant solution is bad because each embedding dimension carries no variation across the batch — every sample gets the same value. What is the *most direct* way to forbid that? Don't go through negatives, don't go through an asymmetry, don't go through a cross-branch matrix. Just look at one branch's embeddings, compute, for each dimension, its variance across the batch, and demand that it not be zero. Force every dimension to vary across samples. That is collapse-prevention stated as bluntly as the failure itself: collapse = no variance per dimension, so the cure = require variance per dimension.

Let me set up the objects. A batch of n images, two augmentation pipelines giving views, each pushed through an encoder and then a head into embeddings Z = [z₁,…,z_n] and Z′ = [z′₁,…,z′_n], each of dimension d. Write z^j for the vector of the j-th coordinate across all n samples in the batch — the j-th column. I want each column z^j to have variance bounded away from zero.

First instinct: add a reward for variance, something like −(1/d) Σ_j Var(z^j), minimized. But that's wrong — it would push the variance to grow without bound, which fights the invariance term (which wants the two views' embeddings on top of each other) and just makes the embeddings explode. I don't want to *maximize* variance; I want a *floor*. Each dimension should have *at least* some standard deviation γ, and once it's above γ there's no further pressure. That's a hinge. Define

  v(Z) = (1/d) Σ_j max(0, γ − S(z^j)),

where S(z^j) is the standard deviation of column j and γ is the target floor. As long as a dimension's std is ≥ γ, it contributes nothing; the moment it dips below γ, the hinge turns on and pushes it back up. Set γ = 1 — pick a scale and stick to it; the absolute value is arbitrary because the network can rescale, what matters is that it's a fixed positive floor. This term I'll apply to each branch separately: v(Z) and v(Z′), each computed from that branch's own batch statistics. No coupling.

Now a subtlety I almost walked past — should the hinge act on the variance or the standard deviation? They carry the same information, so it feels like it shouldn't matter. Let me actually differentiate. If I put the variance straight in the hinge, S(z^j) = Var(z^j) = (1/n) Σ_b (z_{b,j} − z̄_j)², then the gradient of Var with respect to a single entry z_{b,j} is proportional to (z_{b,j} − z̄_j). When the embeddings are near-collapsed, every z_{b,j} sits right at the mean z̄_j, so (z_{b,j} − z̄_j) ≈ 0, so the gradient of Var ≈ 0, so the gradient of the hinge ≈ 0 — precisely in the regime I most need a push *out of*, the term goes silent and the embeddings stay collapsed. The variance-in-the-hinge cure does nothing exactly when the disease is worst. Now use the standard deviation instead, S = √(Var). Its derivative with respect to Var is 1/(2√Var), which *blows up* as Var → 0. So d S/d z_{b,j} = (1/(2√Var))·(2/n)(z_{b,j} − z̄_j) = (z_{b,j} − z̄_j)/(n√Var); the (z − z̄) factor still shrinks near collapse, but it's now divided by √Var which itself shrinks to zero, and the two effects cancel — the deviations (z − z̄) are themselves of order √Var, so the ratio stays order 1 instead of vanishing. The variance-in-the-hinge gradient dies as Var → 0; the standard-deviation gradient does not. The standard deviation gives me a restoring force that survives at collapse, exactly where I need it. That's the version I need. (I'll add a tiny ε inside the square root, S = √(Var + ε), so the gradient is large-but-finite rather than literally infinite, and the whole thing is numerically stable.) So:

  v(Z) = (1/d) Σ_{j=1}^d max(0, γ − √(Var(z^j) + ε)),   γ = 1, ε small.

Let me check that this single term actually kills the constant. If the network emits a constant, every column z^j has zero variance, √(0 + ε) ≈ 0 < γ, the hinge is fully active on every dimension, and its gradient pushes each dimension to spread out. Collapse is no longer a free lunch — it's the *most* penalized configuration. Good. And crucially I got here without a single negative sample, without an asymmetry, without touching the other branch. This term plays the role that the repulsive negative term played in contrastive learning — it's what keeps the embeddings from imploding — but it does it per-dimension, per-branch, locally, instead of by comparing samples to each other across the batch. That's the whole reason it doesn't need big batches: I'm not estimating a sample-space spread, I'm just measuring d cheap per-dimension variances.

But now I have to be honest about whether the variance floor is *enough*. Suppose I train with only invariance + this variance term. Does anything go wrong? Imagine the network keeps every dimension's std at exactly γ = 1 — variance floor satisfied everywhere — but makes all d dimensions carry the *same* signal, perfect copies of one informative direction (or a few). Each dimension individually has variance 1, so the variance term is perfectly happy. The invariance term is happy too if that shared direction is invariant. And yet the representation is garbage in the sense that matters: it has d coordinates but only one (or a few) effective degrees of freedom — all the variance is crammed into a tiny subspace, the dimensions are maximally redundant. That's informational collapse: not a shrink to a point, but a shrink to a low-dimensional subspace. The variance floor is blind to it, because variance is a per-dimension statistic and says nothing about whether two dimensions are the *same*.

So I need a second term that forbids redundancy *between* dimensions — exactly the decorrelation idea I admired in Barlow Twins, but I want it on a single branch, not as a cross-correlation between branches. The natural object is the covariance matrix of one branch's embeddings across the batch:

  C(Z) = (1/(n−1)) Σ_{i=1}^n (z_i − z̄)(z_i − z̄)ᵀ,   z̄ = (1/n) Σ z_i.

This is d×d. Its diagonal entries are the per-dimension variances (already handled by the variance term). Its off-diagonal entry C_{ij} (i ≠ j) is the covariance between dimension i and dimension j across the batch — how much they co-vary. Redundancy is exactly co-variation: if two dimensions carry the same information, they're highly correlated, so C_{ij} is large. To forbid redundancy, drive every off-diagonal entry to zero:

  c(Z) = (1/d) Σ_{i≠j} [C(Z)]_{ij}².

Sum of squared off-diagonals, scaled by 1/d so the criterion grows sensibly with the dimension (there are ~d² off-diagonal terms; the 1/d keeps it on a comparable footing as I widen the embedding). Like the variance term, this is computed on each branch separately — c(Z) and c(Z′) — so again, no coupling between branches. And note what I did *not* do: I didn't normalize the embeddings into correlations. Barlow had to standardize because its cross-correlation entries needed to be in [−1,1] for the diagonal-to-1 target to mean anything. I have no diagonal target — the diagonal of my covariance is governed by the *variance* term, which already pins each dimension's scale to γ. So I can leave the covariance unnormalized; the variance term holds the scale and the covariance term decorrelates. That's why I can throw out the standardization that everyone else needs.

Let me re-examine the division of labor, because it's the crux. The variance term forbids *trivial* collapse: each dimension must keep std ≥ γ, so nothing can shrink to a point. The covariance term forbids *informational* collapse: the dimensions must be decorrelated, so the variance the first term guarantees gets *spread across all d dimensions* instead of being duplicated into a low-dimensional subspace. Neither alone is sufficient. Variance alone allows the "copy one direction" redundancy I just described. And covariance alone is even worse — think about it: the off-diagonal-to-zero objective has *no* repulsive effect against shrinking. The cheapest way to make all off-diagonal covariances zero is to send everything to a constant, where the entire covariance matrix is zero, off-diagonals included. So covariance without variance collapses outright. The two terms are genuinely complementary: variance gives the covariance term something to spread (it can't zero the off-diagonals by collapsing, because collapsing is blocked), and covariance makes the variance meaningful (the guaranteed per-dimension variance becomes d independent directions of information rather than d copies of one).

And invariance is the third leg — it's what makes the representation *about the image* rather than just "any high-variance decorrelated noise." Without it, the variance and covariance terms are perfectly satisfied by, say, mapping each dimension to independent Gaussian noise: high variance, decorrelated, and utterly disconnected from the input. The invariance term ties the two views of the same image together, forcing the variance-and-decorrelation budget to be spent on features that are stable under augmentation. So I need all three, and I can already predict the ablation in my head: invariance only → collapse; invariance + covariance (no variance) → collapse (nothing stops the shrink); invariance + variance (no covariance) → works but leaves performance on the table (informational collapse caps the quality); all three → the real thing.

Now the invariance term itself. Everyone l2-normalizes the embeddings before comparing them, because they're measuring cosine similarity or feeding a cross-correlation. Do I need to? Projecting onto the unit sphere implicitly forces each dimension's batch std toward 1/√d — it *imposes a variance structure* as a side effect. But I'm already controlling the variance explicitly, per dimension, with my own term and my own target γ. If I also l2-normalize, I'm fighting my own variance term: the normalization pins the std to 1/√d while my hinge wants it at γ = 1. So I should *not* normalize. Just use the plain mean-squared Euclidean distance between the paired embeddings:

  s(Z, Z′) = (1/n) Σ_i ‖z_i − z′_i‖₂².

No normalization at all. This is the cleanest possible "pull the two views together," and it's compatible with the variance term because I let the variance term — not a normalization — own the scale. (I can already guess that if I forced l2-normalization back in and set γ = 1/√d to match the sphere, I'd be over-constraining the network and lose a few points; and that standardizing the embeddings would turn my covariance into a correlation in a narrow [−1,1] range, which empirically helps the network *less* than letting the covariance roam free. The whole design philosophy is: own the variance explicitly so I never need a normalization.)

Put the three terms together into one scalar, with a weight on each, and apply the variance and covariance terms to both branches:

  ℓ(Z, Z′) = λ·s(Z, Z′) + μ·[v(Z) + v(Z′)] + ν·[c(Z) + c(Z′)].

Three knobs: λ on invariance, μ on variance, ν on covariance. How should I set them? Start with ν = 1 as the reference and reason about the rest. The variance and invariance terms are the two that fight each other most directly — invariance wants the two views collapsed onto each other (which, taken alone, would also shrink variance), variance wants each dimension to keep spreading. If I let λ and μ drift far apart, one wins and the other can't do its job — too much invariance and the variance floor can't hold (slow collapse), too much variance and the views never actually agree. So I want them balanced, λ = μ, and both larger than the covariance weight, because the covariance term has ~d² terms and even with the 1/d scaling it's a strong gradient that, if over-weighted relative to the others, destabilizes training (decorrelating aggressively before the representation is informative is counterproductive). So the recipe is λ = μ > ν. Reasoning about magnitudes alone won't pin the exact number, but I'd expect a broad basin: pick ν = 1, λ = μ somewhere comfortably above it, and the exact value of λ = μ should matter little. Something like λ = μ = 25, ν = 1 is the kind of setting I'd land on, and I'd expect λ = μ with ν > μ to be unstable and λ = μ with ν < μ to be stable, which is consistent with the balance argument.

The overall objective is this loss summed over all images in the unlabeled dataset and over the sampled augmentation pairs, minimized over the encoder and head parameters by ordinary gradient descent. Nothing else — no momentum encoder, no target network, no predictor, no stop-gradient, no memory bank, no Sinkhorn, no whitening, no standardization. Every collapse-prevention mechanism the other methods needed has been replaced by two explicit per-branch regularizers on the embedding statistics, and because they're per-branch, the two branches share nothing by necessity. I could give them different architectures, different weights, even different input modalities (an image on one side, its caption or its audio on the other), regularize each branch with its *own* μ and ν if their statistics differ, and the method still makes sense — there's no cross-branch matrix that would break. That's a capability none of the coupled methods have.

A couple of architecture choices fall out of the reasoning. The head: after the encoder produces a representation y, I map it through an MLP into the embedding space where the loss lives, and I make that MLP *expand* the dimension (wider output than the representation) rather than project down. Two reasons. First, the loss should be free to throw away exactly the information by which the two views differ (the augmentation nuisances) without having to destroy representation capacity — give it a separate space to do that in. Second, and this is why *expand*: my covariance term decorrelates the embedding dimensions, i.e. it removes pairwise linear correlations. Linear decorrelation in a non-linearly-expanded space corresponds to removing higher-order *dependencies* back in the representation space, not just correlations — the non-linear expansion lets "decorrelate the embedding" do more than "decorrelate the representation" would. So a wide expander (the same lever Barlow found helps) should help here too, and I'd expect performance to climb with embedding width and saturate only at large widths — consistent with information-maximization wanting many decorrelated dimensions to spread variance across. I'll keep batch normalization in the *hidden* layers of the expander for optimization stability, but the final embedding is left un-normalized, because the variance term is what governs its scale. After pretraining the expander is discarded and the encoder's representation is what I keep for downstream tasks.

Let me now write it as code, mirroring how I'd actually implement it. The forward pass takes the two views, runs each through backbone-then-expander to get the two embedding batches, computes the three terms, and returns the weighted sum.

```python
import torch
import torch.nn.functional as F
from torch import nn


def off_diagonal(x):
    # return a flat view of all off-diagonal entries of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class VarianceInvarianceCovariance(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = int(args.mlp.split("-")[-1])   # embedding dim d
        self.backbone, repr_dim = build_resnet(args.arch)  # encoder f_theta, repr y
        self.projector = Expander(args, repr_dim)          # expander h_phi -> embeddings z

    def forward(self, x, y):
        # two views -> two embeddings, each (n, d)
        z = self.projector(self.backbone(x))
        z_prime = self.projector(self.backbone(y))

        # invariance: plain MSE between paired views, NO normalization
        # (the variance term owns the scale, so we never l2-normalize)
        sim_loss = F.mse_loss(z, z_prime)

        # center each dimension across the batch (needed for variance & covariance;
        # centering does not change the variance, so the std below is unaffected)
        z = z - z.mean(dim=0)
        z_prime = z_prime - z_prime.mean(dim=0)

        # variance: hinge on the STANDARD DEVIATION (not the variance) of each dim,
        # so the restoring gradient blows up at collapse instead of vanishing.
        # eps keeps it finite; gamma = 1 is the target floor, hence (1 - std).
        std_z = torch.sqrt(z.var(dim=0) + 1e-4)
        std_z_prime = torch.sqrt(z_prime.var(dim=0) + 1e-4)
        std_loss = torch.mean(F.relu(1 - std_z)) / 2 + torch.mean(F.relu(1 - std_z_prime)) / 2

        # covariance: push every off-diagonal of each branch's covariance to zero,
        # decorrelating the dimensions; 1/d scaling. Computed PER BRANCH (no cross term).
        n = z.shape[0]
        cov_z = (z.T @ z) / (n - 1)
        cov_z_prime = (z_prime.T @ z_prime) / (n - 1)
        cov_loss = (
            off_diagonal(cov_z).pow_(2).sum().div(self.num_features)
            + off_diagonal(cov_z_prime).pow_(2).sum().div(self.num_features)
        )

        # weighted sum: lambda * invariance + mu * variance + nu * covariance
        loss = (
            self.args.sim_coeff * sim_loss     # lambda = 25
            + self.args.std_coeff * std_loss   # mu     = 25
            + self.args.cov_coeff * cov_loss   # nu     = 1
        )
        return loss


def Expander(args, repr_dim):
    # MLP that EXPANDS the representation into a wider embedding space.
    # Linear + BN + ReLU on the hidden layers (BN for optimization stability),
    # final layer is a plain Linear with no normalization on the output.
    dims = [repr_dim] + list(map(int, args.mlp.split("-")))   # e.g. 2048-8192-8192-8192
    layers = []
    for i in range(len(dims) - 2):
        layers += [nn.Linear(dims[i], dims[i + 1]), nn.BatchNorm1d(dims[i + 1]), nn.ReLU(True)]
    layers += [nn.Linear(dims[-2], dims[-1], bias=False)]
    return nn.Sequential(*layers)
```

The training loop around this is ordinary: sample a batch, make the two views, forward to get the loss, backprop, step a LARS optimizer with a warmup-then-cosine learning-rate schedule. No second network to maintain, no queue, no stop-gradient placement to get right.

So the causal chain, start to finish: "make two views agree" collapses to a constant; instead of preventing that with negatives or an asymmetry or a cross-branch coupling, I attacked the failure directly — collapse is the absence of per-dimension variance, so I added a hinge that forces each embedding dimension's standard deviation above a floor (standard deviation, not variance, so the gradient grows rather than vanishes at collapse). That floor alone permits the dimensions to redundantly copy one direction — informational collapse — so I added a covariance term that drives every off-diagonal of each branch's embedding covariance to zero, spreading the guaranteed variance across all dimensions. Both regularizers act on each branch by itself, which removes any need to normalize the embeddings (the variance term owns the scale) and removes any coupling between the branches (so they can differ in architecture, weights, or modality). The plain-MSE invariance term ties the two views together so the variance-and-decorrelation budget is spent on augmentation-stable features. Three explicit terms — invariance, variance, covariance — and the constant solution is impossible at the optimum by construction, not by a training-dynamics accident.
