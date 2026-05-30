OK, let me think this through from scratch. I want representations of images learned without labels, and the lever everyone uses is augmentation invariance: take an image, make two distorted views of it, and ask the network to map them to the same place. Whatever survives random crops, color jitter, blur, grayscale — that's the semantic content. So the objective wants to be something like "make the embedding of view A equal to the embedding of view B."

But the instant I write that down I see the rot in it. A network that emits a constant vector regardless of its input satisfies "view A equals view B" perfectly — every pair agrees, the loss is zero, and the representation tells me nothing. Collapse. So "maximize similarity between the two views" is degenerate on its own, and the whole game in this field has been bolting some extra contraption onto that objective to make the constant solution unappealing.

Let me actually enumerate the contraptions, because I want to understand what each one is buying and what it costs, and whether I can avoid the whole genre.

The contrastive route — SimCLR, MoCo, the InfoNCE loss. Here I don't just pull the two views of one image together; I also push each image away from the *other* images in the batch. Write it with feature-normalized embeddings so it's a cosine similarity: for a positive pair (z^A_b, z^B_b), the denominator compares that positive against the other examples in the batch,

  L = - sum_b <z^A_b, z^B_b> / (tau ||z^A_b|| ||z^B_b||)  +  sum_b log sum_{b'} exp( <z^A_b, z^B_b'> / (tau ||z^A_b|| ||z^B_b'||) ).

The first term is "agree"; the second term is "be more similar to the positive than to everyone else." Now the constant solution is bad for the classification problem inside the loss: if everything maps to one point, the positive logit and all negative logits are equal, so the loss is stuck at the uniform-classification value instead of going to zero. So collapse is avoided. Good. But what did it cost? That second term is, in effect, a non-parametric estimate of the spread (the entropy, the "uniformity") of the embedding distribution, computed by looking at all the pairwise distances in the batch. Non-parametric estimates of spread in high dimensions are brutal: they need a lot of samples to be any good. That's exactly why SimCLR wants enormous batches and why MoCo has to keep a queue of tens of thousands of past embeddings around — they're feeding the entropy estimate. And there's a second symptom I keep seeing reported: if I crank up the embedding dimensionality, contrastive methods don't improve, they saturate. Curse of dimensionality again — the pairwise-distance estimate gets *worse* in higher dimensions, so there's no reason to go wide. So contrastive buys non-collapse but chains me to big batches and forbids high-dimensional embeddings.

The asymmetric route — BYOL, SimSiam. These throw the negatives away entirely. The loss is just a cosine similarity between the two views,

  L = - sum_b <z^A_b, z^B_b> / (||z^A_b|| ||z^B_b||),

which, written alone, *should* collapse — and yet it doesn't, because they break the symmetry of the two branches. One branch gets an extra "predictor" MLP stacked on top; the other branch is treated as a fixed target, detached from the gradient (BYOL additionally keeps that target as a slow exponential moving average of the online weights). The SimSiam ablation is the sharp diagnostic here: knock out the stop-gradient and the thing collapses to a constant instantly; it's the stop-gradient that's load-bearing. Fine — it's batch-size robust, since there's no cross-sample term at all. But stare at what it *is*: there's no single function being minimized. There genuinely exist trivial (collapsed) solutions to that cosine loss; the method simply never reaches them because of an asymmetry trick and the resulting learning dynamics. Nobody can cleanly say why. That bothers me. I don't want "it empirically doesn't collapse if you arrange the gradients just so." I want collapse to be impossible at the optimum, by construction.

Clustering — DeepCluster, SwAV, SeLa — assigns views to clusters and enforces that two views of an image get the same code, with a clustering step in the loop. But clustering collapses too (empty clusters), avoided by balance constraints and careful plumbing, plus you store features when there are more clusters than batch. Same flavor of fragility.

So here's the shape of the dissatisfaction. All of these treat "agree" as the real objective and "don't collapse" as a side condition enforced by machinery — negatives, or asymmetry, or cluster balancing. What if non-collapse isn't a side condition I have to *enforce*, but a property that falls out of a richer statement of what a good representation *is*?

Let me go back to first principles about what I actually want from the representation, beyond "the two views agree." I want the representation to be *informative* about the image. A constant is uninformative — that's precisely the failure. So somewhere in the objective there should be a term that rewards informativeness, and if it's there, the constant solution should be ruled out not by a trick but because it scores terribly on informativeness.

There's an old idea from neuroscience that says exactly this in usable terms. Barlow's redundancy-reduction principle: the goal of sensory coding is to recode redundant input into a *factorial code* — components that are statistically independent. The intuition is that if two units always carry the same information, one of them is wasted; a good code spreads distinct, non-overlapping information across its units. So "informative" gets a concrete operational meaning: don't let the components duplicate each other. Decorrelate them.

Now I have two desiderata for the representation of two views, and they pull in opposite directions in a way I like:

  (1) Invariance: the two views of one image should produce the same feature values.
  (2) Non-redundancy: distinct feature components should not duplicate each other — they should be decorrelated.

Invariance alone collapses (everything to a point). But decorrelation actively forbids that: a constant has no variance, and a set of identical/duplicated features is maximally correlated, not decorrelated. The two desiderata fight, and the place where they balance can't be the constant. That's the thing I was looking for — non-collapse as a *consequence* of asking for non-redundancy, not as an enforced side condition.

So how do I turn "the two views agree, and the components are decorrelated" into one differentiable scalar? Let me set up the objects. A batch of N images. Two augmentation pipelines give views A and B. I push both through the *same* network f_theta — same weights, fully symmetric, no predictor, no target branch — getting embeddings Z^A and Z^B, each N×D (D the embedding dimension, b indexes the batch sample, i indexes the feature).

What statistic captures both desiderata at once? I want to compare feature i of view A against feature j of view B, *across the batch*. That's a correlation between two batch-vectors. Let me mean-center each feature across the batch first (subtract its batch mean), then form, for every pair (i, j), the correlation of feature i in A with feature j in B:

  C_ij = ( sum_b z^A_{b,i} z^B_{b,j} ) / ( sqrt( sum_b (z^A_{b,i})^2 ) · sqrt( sum_b (z^B_{b,j})^2 ) ).

This is a D×D matrix — the cross-correlation matrix between the two views' embeddings, computed along the batch dimension. Each entry sits in [-1, 1]: +1 perfect correlation, -1 perfect anti-correlation, 0 decorrelated. The normalization by the per-feature batch standard deviation is what makes each entry a *correlation* rather than a raw covariance, so the scale of any individual feature drops out.

Now read my two desiderata directly off this matrix. The diagonal entry C_ii is the correlation between feature i of view A and feature i of view B. If feature i is invariant to the distortion, then across the batch its A-value and its B-value move together perfectly: C_ii = 1. So desideratum (1), invariance, is "make every diagonal entry 1." The off-diagonal entry C_ij (i ≠ j) is the correlation between feature i and a *different* feature j across the views. If the components carry non-redundant information, feature i and feature j shouldn't co-vary: C_ij = 0. So desideratum (2), non-redundancy, is "make every off-diagonal entry 0."

Put together: I want the cross-correlation matrix to be the identity. C = I. And the loss is just the squared distance of C from the identity, with a knob to trade the two parts:

  L = sum_i (1 - C_ii)^2  +  lambda · sum_i sum_{j≠i} C_ij^2.

The first sum — call it the invariance term — drives the diagonal to 1. The second sum — the redundancy-reduction term — drives the off-diagonal to 0. That's the whole objective. It's symmetric in the two views, it's one scalar, there are no negatives, no predictor, no stop-gradient, no moving average, no clustering.

Now the load-bearing question: does this actually exclude collapse, by construction? Let me try to collapse it and watch it fail. Suppose the network ignores its input and emits a constant: z_{b,i} = c_i for all b. After I mean-center feature i across the batch, every value becomes c_i − c_i = 0. So the numerator of C_ii is sum_b 0·0 = 0, and the denominator is sqrt(0)·sqrt(0) = 0 — the diagonal is 0/0, certainly not 1. More carefully than the 0/0: a *correlation* requires non-zero variance in both arguments to even be 1; a feature with zero batch-variance can never achieve C_ii = 1. The invariance term penalizes it with the full (1 − 0)^2 = 1 per dimension. So the diagonal target by itself already forbids the constant solution — because I normalized by the per-feature batch standard deviation, "be perfectly correlated with myself across the batch" is impossible for a feature that doesn't vary across the batch. The act of normalizing forces every feature to carry batch variance, i.e. to actually respond to the input.

So the constant is dead. But there's a subtler cheap escape I should check: what if the network finds *one* informative direction that's invariant, and then just copies it into all D features? Then every diagonal C_ii = 1 (each copy is invariant), and the invariance term is happy. But the copies are all the same signal, or a sign-flipped version of it, so feature i and feature j have |C_ij| = 1 off-diagonal, and the redundancy-reduction term slams the pair with lambda. So duplicating features is exactly what the off-diagonal term forbids. The only zero-loss correlation target is C = I: D features that are each invariant to the distortion *and* mutually decorrelated. The decorrelation term is doing the job that negatives or stop-gradients did in the other methods — it's what makes the trivial solution impossible — but it does it as part of the objective, not as a bolted-on mechanism. That's the payoff.

Let me sanity-check the no-collapse claim by deleting pieces. If I delete the off-diagonal term and keep only invariance, the cheap "copy one direction" escape comes back — the representation can satisfy invariance without spreading information. If I delete the invariance term and keep only decorrelation, nothing ties the two views together at all; the network can decorrelate features without learning anything view-consistent. Both terms have to be there.

Why normalize along the *batch* dimension and not the feature dimension? This is the precise fork from contrastive. Contrastive normalizes each *sample's* embedding to unit length (feature-dim normalization) and then measures cosine similarity *between samples* — its statistic lives in sample-vs-sample space. I'm doing the opposite: I normalize each *feature* across the batch and measure correlation *between features*. My matrix is D×D (feature × feature), not N×N (sample × sample). That's the structural reason I don't need negatives: I never compare sample to sample, so I never need a pile of negative samples to estimate a sample-space spread. The "spread" I care about is feature decorrelation, and that's a D×D object estimated by averaging over the batch, which is statistically cheap. (If I instead normalize along the feature dimension and use the unnormalized covariance, I'd expect a small regression — I'd be drifting back toward the cosine-similarity framing. Worth a check, but it shouldn't help.)

And why the normalized cross-correlation and not the raw cross-covariance? Because the normalization pins the diagonal scale: each feature is rescaled to unit batch-variance, so C_ii's target of 1 is meaningful and scale-free, and the off-diagonal targets of 0 aren't competing with arbitrary feature magnitudes. If I drop that normalization and run on the bare covariance, the diagonal isn't pinned, the loss has to also fight feature scales, and I'd expect a substantial regression. (Indeed, removing the feature normalization and using covariance should hurt a lot more than the feature-dim-normalization variant — the normalization is doing real work.)

Now the trade-off knob lambda. Why not 1? Count the terms. There are D diagonal entries but D(D−1) off-diagonal entries — order D^2 of them. With D in the thousands, the off-diagonal sum has vastly more terms than the diagonal sum, so if I weight them equally the redundancy term swamps invariance and the gradient barely cares about agreement. So lambda should be small — small enough to bring the two terms' *total* magnitudes into the same ballpark. Something like a few×10^−3 puts D versus lambda·D^2 in balance for D ~ a few thousand, which is the right order. And because both terms are squared correlations bounded in [0,1], the objective isn't knife-edge sensitive to lambda — I'd expect it to work across a range of small values. So lambda ≈ 5×10^−3, not finely tuned.

This flips the usual intuition: I should *want* very high-dimensional embeddings. The contrastive methods saturate or get worse as D grows because their non-parametric entropy estimate degrades in high dimensions. But my "spread" is captured by the off-diagonal structure of a correlation matrix — a quantity I can estimate from few samples even when D is large, because it's effectively a parametric (second-moment) object rather than a non-parametric density estimate. So increasing D gives me more decorrelated directions to fill with non-redundant information, and the objective stays estimable. I'd expect performance to keep climbing as I widen the projector well past the backbone's own width — even though the ResNet output is fixed at 2048 and acts as an intrinsic bottleneck, projecting up to 8192 and beyond should still help. That's a falsifiable prediction and it's the cleanest fingerprint distinguishing this from contrastive learning.

Let me make the "second-moment object" intuition rigorous, because it also tells me where lambda comes from at a deeper level. What does "informative representation, invariant to nuisance" mean information-theoretically? The Information Bottleneck. Let X be the underlying image, Y a distorted view, Z = f_theta(Y) the representation. I want Z to keep information about the sample while discarding information about which specific distortion was applied. Write the IB objective as

  IB = I(Z, Y) − beta · I(Z, X).

Use the identity I(Z, U) = H(Z) − H(Z|U):

  IB = [ H(Z) − H(Z|Y) ] − beta · [ H(Z) − H(Z|X) ].

H(Z|Y) = 0, because f_theta is deterministic: once I fix the distorted input Y, the representation Z is fully determined, no entropy left. So that term cancels:

  IB = H(Z) − beta·H(Z) + beta·H(Z|X) = (1 − beta) H(Z) + beta · H(Z|X).

Overall scale of a loss doesn't matter, so divide by beta:

  IB ∝ H(Z|X) + ((1 − beta)/beta) · H(Z).

Two terms. H(Z|X) is the entropy of the representation given the original image — i.e. how much Z still varies once X is fixed, which is variability caused purely by the distortion. Minimizing it makes Z invariant to the distortion: that's my invariance term. The second term is H(Z), the overall entropy/variability of the representation, which I want to *maximize* to keep Z informative. Check the sign of its coefficient: if beta ≤ 1 then (1−beta)/beta ≥ 0 and I'd be *minimizing* H(Z) too — the optimum is then a constant Z, uninteresting, so that regime is useless and I discard it. For beta > 1 the coefficient (1−beta)/beta is negative, so I can write it as −gamma with gamma = (beta−1)/beta > 0: now the loss minimizes H(Z|X) while maximizing gamma·H(Z). The final squared-correlation loss will have its own practical weight lambda, but the reason a positive trade-off weight belongs there is this beta-controlled entropy term.

Now how do I actually compute H(Z)? Entropy of a high-dimensional signal needs vastly more data than a batch. So I make the one simplifying assumption that turns this tractable: model Z as Gaussian. The entropy of a Gaussian is (up to a constant) (1/2) log|Cov(Z)| — the log-determinant of its covariance. So the IB objective becomes, schematically,

  IB = E_X log|Cov(Z|X)| + ((1−beta)/beta) · log|Cov(Z)|,

invariance via the first log-det, informativeness via maximizing the second. Two practical moves get me from here to my loss. First, directly optimizing log-determinants is numerically finicky; but maximizing log|Cov(Z)| under the constraint that each feature is rescaled to unit batch-variance is *the same global optimum* as making the off-diagonals vanish — by Hadamard's inequality, a positive semidefinite covariance with unit diagonal has determinant at most 1, with equality exactly at the identity. So I can replace "maximize log-det" by the much friendlier "drive the off-diagonal correlations to zero," i.e. minimize the Frobenius norm of the off-diagonal block. Same optimum, smoother objective. Second, to be perfectly faithful to the derivation the informativeness term should use the auto-correlation of a single view's embedding, but the same objective is already measuring cross-view alignment; if I use the *cross*-correlation off-diagonals, I tie redundancy and invariance into one matrix C and keep the loss symmetric. So my redundancy-reduction term is a Gaussian-parametrized proxy for entropy maximization — which is exactly why it survives high dimensions and small batches where the non-parametric contrastive estimate dies. The whole loss is one instantiation of IB applied to SSL.

Let me also place this against the relatives so I'm sure I'm not reinventing one of them. The contrastive InfoNCE, recast: its similarity term is my invariance, and its contrastive log-sum-exp term is a non-parametric estimate of embedding entropy — same *role* as my off-diagonal term (both maximize variability using batch statistics), but it estimates entropy by pairwise sample distances (needs many negatives, dies in high D), while I estimate it by feature decorrelation under a Gaussian proxy (few samples, loves high D). Same skeleton, opposite statistics; InfoNCE also carries a temperature that I simply don't have, and I carry a lambda tied to the IB trade-off that it doesn't.

The whitening method is the nearest cousin: it *hard*-whitens each batch (Cholesky) so the covariance is exactly identity, then takes cosine similarity. My redundancy term asks for the same end state — decorrelated, unit-variance features — but as a *soft* penalty folded into the loss rather than a hard transform applied to the activations. Soft-whitening, not hard-whitening. That should be the more scalable choice.

And the oldest cousin, IMAX from the early SSL days: log|Cov(Z^A − Z^B)| − log|Cov(Z^A + Z^B)|, which maximizes information between the twin representations under an additive-Gaussian-noise-on-a-shared-signal model. It has the same two-flavored feel — a term wanting the two views similar, a term wanting units decorrelated — but it's *directly* an information quantity with no trade-off knob, and it never scaled. My version isn't literally an information quantity; it's a proxy with the lambda knob, which is exactly the slack that lets it scale.

I'm convinced. Let me write the network and the loss as real code.

The encoder is a ResNet-50 with its classification head removed, output width 2048 (call these the "representations" — what I'll actually use downstream). On top of it sits a projector: three linear layers, all 8192 wide, with batch-norm + ReLU after the first two and a bare linear at the end (call the projector output the "embeddings" — only these go into the loss). The projector being very wide is the high-D bet I argued for; three layers give the head enough nonlinear capacity without putting any supervised labels into it. For the per-feature batch normalization that defines C, I run each view's embeddings through a non-affine BatchNorm1d — that *is* the mean-center-and-divide-by-batch-std of my definition — and then the cross-correlation matrix is just the matrix product of the two normalized embedding batches, divided by N.

```python
import torch
import torch.nn as nn
import torchvision

def off_diagonal(x):
    # flattened view of the off-diagonal elements of a square (n x n) matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowTwins(nn.Module):
    def __init__(self, projector="8192-8192-8192", lambd=0.0051, batch_size=2048):
        super().__init__()
        self.lambd = lambd
        self.batch_size = batch_size

        # encoder: ResNet-50 backbone, classifier head removed -> 2048-d "representations"
        self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        self.backbone.fc = nn.Identity()

        # projector: 3 wide linear layers -> high-dim "embeddings" that enter the loss.
        # wide on purpose: decorrelation is a second-moment object, so high D keeps helping.
        sizes = [2048] + list(map(int, projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # non-affine BN = mean-center + divide by batch std per feature: this is the
        # normalization in the definition of the cross-correlation matrix C.
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):
        # same network on both views -- fully symmetric, no predictor / target / stop-grad.
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))

        # cross-correlation matrix C (D x D), normalized along the batch dimension.
        c = self.bn(z1).T @ self.bn(z2)
        c.div_(self.batch_size)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(c)

        # invariance term: drive the diagonal of C to 1  ->  features invariant to the distortion.
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        # redundancy-reduction term: drive the off-diagonal of C to 0  ->  decorrelate components.
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss
```

The training loop is the ordinary one — two augmented views per image (BYOL's augmentation pipeline, with the blur/solarize probabilities differing between the two views so the views aren't statistically interchangeable), LARS optimizer with the learning rate scaled by batch/256, a short warmup and cosine decay, biases and BN parameters excluded from LARS adaptation and weight decay. In distributed training I convert the model to synchronized batch normalization before wrapping it, because the normalization in C is a batch statistic. Batch size can be large (2048) but, unlike contrastive learning, it doesn't *have* to be — there's no sample-space estimate that needs feeding, so it should work down to a few hundred.

```python
for x in loader:           # x already yields (y1, y2): two augmented views per image
    y1, y2 = x
    loss = model(y1, y2)
    loss.backward()
    optimizer.step()       # LARS; lr scaled by batch/256, warmup + cosine decay
    optimizer.zero_grad()
```

The causal chain, start to finish: bare view agreement collapses to a constant, and every existing fix — negatives, predictors, stop-gradients, clustering — props up "agree" with external machinery. Redundancy reduction reframes the goal as two desiderata that fight: make the two views agree, and decorrelate the components. Encode both in one object, the cross-correlation matrix between the two views computed across the batch, and ask for it to be the identity — diagonal→1 for invariance, off-diagonal→0 for non-redundancy. Normalizing each feature by its batch std makes "correlate perfectly with myself" impossible for a constant feature, so collapse is excluded by construction, not by a trick; the off-diagonal term kills the copy-one-direction escape; the zero-loss target C=I is a decorrelated, invariant, informative code. Because decorrelation is a Gaussian-proxy entropy estimate tied to the IB trade-off, the objective is cheap to estimate from small batches and high-dimensional embeddings — the exact reverse of contrastive learning. One symmetric loss, a wide projector, a plain ResNet, and no asymmetry anywhere.
