# Barlow Twins — synthesis

## Pain point / research question
SSL-by-joint-embedding learns representations invariant to augmentations by maximizing similarity
of two views. The trivial solution is a constant output (collapse). All existing methods bolt on a
mechanism to dodge collapse:
- Contrastive (SimCLR / MoCo / InfoNCE): push negatives apart → needs many negatives → large batches
  (SimCLR) or a momentum-encoder + ~65k-entry queue (MoCo). Non-parametric entropy estimate → curse
  of dimensionality, no benefit from high-dim embeddings.
- Asymmetric no-negative (BYOL / SimSiam): predictor network + stop-gradient (+ EMA target in BYOL).
  Works but not an optimization of a single objective; collapse avoided by dynamics/implementation,
  not by construction. SimSiam ablation: stop-grad is the critical piece.
- Clustering (DeepCluster / SwAV / SeLa): non-differentiable assignment, balanced-cluster constraints
  to avoid empty clusters.
- Whitening (W-MSE): hard Cholesky whitening per batch then cosine — not shown to scale.

Goal: a single symmetric objective whose optimum is non-trivial by construction — no negatives, no
predictor, no stop-grad, no EMA, no clustering.

## The idea — redundancy reduction (Barlow 1961)
Barlow's principle: sensory processing should recode redundant inputs into a factorial code
(statistically independent components). Two desiderata for a representation of two views:
1. INVARIANCE: the two views should map to the same representation.
2. NON-REDUNDANCY: distinct components should be decorrelated → each carries non-redundant info.
These two pull apart: invariance alone collapses; decorrelation forbids the constant solution.

## The cross-correlation matrix and loss (DERIVE)
Two views A,B of a batch X through the SAME net f_theta → embeddings Z^A, Z^B (NxD). Mean-center each
feature over the batch. Define cross-correlation (d x d, d=embedding dim):
  C_ij = sum_b z^A_{b,i} z^B_{b,j} / [ sqrt(sum_b (z^A_{b,i})^2) sqrt(sum_b (z^B_{b,j})^2) ]
C_ij in [-1,1]. Want C = I:
  L_BT = sum_i (1 - C_ii)^2  +  lambda * sum_i sum_{j!=i} C_ij^2
- invariance term sum_i (1-C_ii)^2: C_ii=1 means feature i of view A and feature i of view B are
  perfectly correlated across the batch → feature i is invariant to the distortion.
- redundancy-reduction term lambda sum_{i!=j} C_ij^2: C_ij=0 means feature i (view A) and feature j
  (view B) are decorrelated → distinct components carry non-redundant information.

## No-collapse argument (DERIVE, central)
Collapse = constant output: z_{b,i} = const_i for all b. Then after mean-centering over batch,
z^A_{b,i}=0 for all b → numerator and denominator of C are 0/0, C undefined; more sharply, any
direction with zero batch-variance cannot satisfy C_ii=1 (correlation requires non-zero variance).
Because C is normalized by per-feature batch std, the diagonal target C_ii=1 *forces* every feature to
have non-zero variance across the batch — a constant feature can never score 1. So the invariance term
alone already rules out the collapsed constant. Meanwhile the off-diagonal term forbids the cheap
escape of making all features identical/collinear (which would let one informative direction satisfy
invariance while the rest copy it): copies are perfectly correlated → C_ij=1 off-diagonal → penalized.
The optimum is C=I: D features each invariant to the distortion AND mutually decorrelated. This is a
non-trivial optimum reachable by a symmetric network with no asymmetry tricks. Decorrelation is what
prevents the trivial solution — that is why no negatives/stop-grad/EMA are needed.

Note on batchnorm: in the impl the per-feature normalization is done by a BatchNorm1d(affine=False)
on each view, then C = bn(z1).T @ bn(z2) / N. BN subtracts batch mean and divides by batch std, which
is exactly the mean-centering + std-normalization of eq. 2. So C is literally the empirical
cross-correlation. (BN here is the normalization in the loss, not a collapse-avoidance trick.)

## Why high-dim embeddings help (vs contrastive)
Off-diagonal decorrelation is a proxy entropy estimate under a Gaussian parametrization of Z
(Appendix A: IB). Gaussian entropy = log|Cov|; under unit-variance rescaling, maximizing it =
driving off-diagonals to 0. Gaussian-parametrized "entropy" estimate is stable from few samples and
in high dimension, unlike the non-parametric InfoNCE entropy estimate (curse of dimensionality). Hence
BT keeps improving as projector output dim grows (up to 8192+), while SimCLR/BYOL saturate. ResNet
output stays 2048 (a bottleneck), yet the >2048-dim projector still helps.

## IB derivation (Appendix A, DERIVE)
IB objective: IB_theta = I(Z,Y) - beta I(Z,X), Y = distorted sample, X = original.
Want Z informative about X but invariant (uninformative) about the specific distortion Y.
Wait — convention in appendix: maximize info about sample, minimize info about distortion. Using
I(Z,Y) - beta I(Z,X) with the identity I(Z,U)=H(Z)-H(Z|U):
  IB = [H(Z) - H(Z|Y)] - beta[H(Z) - H(Z|X)]
H(Z|Y)=0 (f deterministic: Z fully determined by the distorted input Y). So
  IB = H(Z) - beta H(Z) + beta H(Z|X) = (1-beta)H(Z) + beta H(Z|X).
Divide by beta (scale irrelevant): IB ∝ H(Z|X) + (1-beta)/beta H(Z).
Gaussian: H(Z) = (1/2)log|Cov_Z| + const. So
  IB = E_X log|C_{Z|X}| + ((1-beta)/beta) log|C_Z|.
beta<=1: optimum is constant Z → uninteresting, ignore. beta>1: (1-beta)/beta<0; write it as -lambda,
lambda>0. First term log|C_{Z|X}| (entropy given the sample, i.e. variability due to distortion) →
minimized → invariance (same optimum as making C diagonal=1). Second term -lambda log|C_Z| →
maximize log|C_Z| = maximize variability/entropy of Z. Directly optimizing log-det doesn't reach SoTA;
replace by the proxy of minimizing Frobenius norm of the (cross-)correlation off-diagonals after
unit-variance rescaling — same global optimum (decorrelate all units). For consistency the second term
should use the auto-correlation of one net; using cross-correlation works equally well in practice.
The trade-off scalar lambda IS the IB beta repackaged.

## InfoNCE comparison (Discussion)
InfoNCE: -sum_b <z^A_b,z^B_b>/(tau ||.||||.||) + sum_b log sum_{b'!=b} exp(<z^A_b,z^B_b'>/(tau||.||||.||)).
- similarity term ≈ invariance; contrastive term ≈ non-parametric entropy of embedding distribution
  (Wang & Isola 2020 uniformity).
- BT normalizes along the BATCH dimension (decorrelate features); InfoNCE normalizes along the FEATURE
  dimension (cosine sim between samples). BT replaces the pairwise-distance entropy estimate with a
  Gaussian-proxy decorrelation → small batches OK, high dim helps.
- lambda <-> IB beta; InfoNCE temperature tau has no analogue in BT.

## IMAX (Becker & Hinton 1992; Zemel & Hinton 1990) — ancestor
L_IMAX = log|C_{Z^A-Z^B}| - log|C_{Z^A+Z^B}|: maximizes info between twin reps under
additive-independent-Gaussian-noise-on-shared-signal assumption. Has a "similar" + "decorrelate" feel.
Difference: IMAX is directly an info quantity, no lambda; didn't scale (their ImageNet attempts failed).

## Design decisions -> why
- Same network for both views (symmetric, shared weights): redundancy reduction needs no asymmetry; the
  ablation shows adding predictor/stop-grad slightly HURTS (71.4 -> 70.5/70.2, both -> 61.3).
- Normalize along batch dim (not feature dim): the object we decorrelate is feature-feature correlation
  across the batch; cosine/feature-norm would give InfoNCE-style sample similarity. Feature-dim
  normalization ablation: 71.4 -> 69.8 (slightly worse).
- Cross-correlation (normalized) not cross-covariance: removing batch normalization of features
  (using covariance) drops to 53.4. Normalization fixes the diagonal scale so the loss is scale-free.
- Projector: 3 linear layers, 8192 wide, BN+ReLU on first two, last linear no BN/ReLU. High dim helps
  (unique to BT); depth saturates at 3.
- lambda = 5e-3: balances invariance vs redundancy; D off-diagonals (~D^2) vastly outnumber D diagonals,
  so lambda<1 keeps the two terms comparable in magnitude. BT robust to lambda over orders of magnitude.
- LARS optimizer, lr scaled by batch/256, 10-epoch warmup, cosine decay to lr/1000, weight decay
  1.5e-6, bias/BN excluded from LARS+WD. Batch size 2048 (works down to 256). Follows BYOL protocol.
- BYOL-style augmentations; asymmetric blur/solarize probabilities for the two views.
- zero_init_residual on ResNet-50.

## Canonical code = facebookresearch/barlowtwins/main.py
Loss: c = bn(z1).T @ bn(z2) / N ; on_diag = (diag(c)-1)^2 sum ; off_diag = off_diagonal(c)^2 sum ;
loss = on_diag + lambd*off_diag. off_diagonal trick: flatten()[:-1].view(n-1,n+1)[:,1:].flatten().
