# VICReg synthesis

## Pain point / research question
Joint-embedding SSL: two augmented views of an image, push both through encoder(s), make the
embeddings agree. Bare "make views agree" (invariance) is solved by a constant → **collapse**. Every
working method bolts on machinery to avoid the constant:
- **Contrastive (SimCLR, MoCo)**: InfoNCE, push negatives apart → needs large batch / memory bank.
- **Clustering (SwAV, DeepCluster)**: cluster codes + balanced assignment (Sinkhorn) → still negative-like,
  needs many prototypes.
- **Distillation / asymmetric (BYOL, SimSiam)**: predictor on one branch + stop-grad (SimSiam) / EMA
  momentum encoder (BYOL). Empirically no collapse, but *why* is not understood; dynamics-dependent,
  requires dependency between branches (shared weights / EMA), and BN is implicated.
- **Information-max (Barlow Twins, W-MSE)**: decorrelate embedding dimensions to maximize info content.
  Barlow drives the **cross-correlation** matrix between the two branches to identity (requires
  standardization/batch-norm of embeddings; the cross-correlation couples the two branches). W-MSE
  whitens (matrix inverse → costly/unstable, needs batch slicing).

Question: can we prevent collapse with an **explicit, interpretable** constraint applied to **each
branch separately**, needing none of weight sharing / EMA / stop-grad / predictor / negatives /
batch-or-feature normalization / quantization? That would also free the two branches to differ
(different architectures, weights, modalities → multi-modal).

## The three terms (DERIVE)
Setup: batch of n vectors, embeddings Z=[z_1..z_n], Z'=[..], each dim d. z^j = column j across batch.

1. **Invariance** s(Z,Z') = (1/n) Σ_i ||z_i − z'_i||². Plain MSE between paired views, NO normalization
   (this is the contrast with everyone who l2-normalizes). Pulls the two views together.

2. **Variance** v(Z) = (1/d) Σ_j max(0, γ − S(z^j, ε)), where S(x,ε)=√(Var(x)+ε), γ=1.
   Hinge keeping each embedding dimension's std (across the batch) ABOVE γ. A constant has zero
   variance per dim → hinge maximally active → this is the term that DIRECTLY forbids collapse. Applied
   to each branch separately.
   - **WHY std not variance**: if we used Var directly inside the hinge, d/dx of Var → 0 as x→ x̄ (mean),
     so the gradient of v vanishes exactly when the embedding is near-collapsed — the term can't push
     out of collapse. √Var has a gradient that blows up as Var→0 (1/(2√Var)), so it pushes hardest
     precisely when near collapse. ε also guards numerics. This is the central design subtlety.
   - **WHY hinge / threshold γ=1, not maximize variance**: we want std ≥ γ, not unbounded growth.
     Once each dim has enough variance there's no incentive to keep inflating (which would fight
     invariance / explode). Floor, not a target to maximize.

3. **Covariance** c(Z) = (1/d) Σ_{i≠j} [C(Z)]_{i,j}², with C(Z) = (1/(n−1)) Σ_i (z_i−z̄)(z_i−z̄)ᵀ.
   Sum of squared off-diagonal covariances → 0. Decorrelates dimensions, prevents **informational
   collapse** (dims carrying redundant info / all variance crammed into a subspace). Borrowed
   decorrelation idea from Barlow but on the COVARIANCE of EACH branch (not cross-correlation between
   branches) → branches independent, and no standardization needed (variance term already pins scale).
   1/d scaling so the criterion scales with dimension.

Total: ℓ(Z,Z') = λ s(Z,Z') + μ[v(Z)+v(Z')] + ν[c(Z)+c(Z')].
λ=μ=25, ν=1, ε=1e-4, γ=1 on ImageNet.

## WHY variance + covariance together prevent BOTH collapses
- Variance alone (Inv+Var, λ=μ=1, ν=0) → 57.5% (works, no collapse, but all info can pile into few
  correlated dims → informational collapse limits quality).
- Covariance alone (no variance) → collapse: covariance term has NO repulsive effect; off-diagonals
  can be zeroed by shrinking everything to a constant (0 covariance trivially). So Cov needs Var.
- Inv alone → collapse. Inv+Cov (no Var) → collapse (ablation table: λ=25,μ=0,ν=1 collapse; λ=0,μ=25,ν=1 collapse).
- So: **Variance forbids trivial (norm) collapse — each dim must keep variance ≥ γ. Covariance forbids
  informational collapse — dims must be decorrelated so variance is spread across all d dims, not
  duplicated.** Together they force d genuinely-different, individually-varying dims = informative
  representation. Invariance ties the two views.

## Coefficient choices (appendix)
- ν=1, grid search λ=μ with base condition λ=μ>1. λ=μ=25 best (small margin). Very different λ,μ or
  λ=μ with ν>μ → unstable. λ=μ with ν<μ → stable.
- Without variance reg → immediate collapse, covariance has no impact.
- Std (variance) gradient argument is the key derivation subtlety (see above).

## Architecture / impl
- encoder f_θ = ResNet-50, 2048-d representation (kept for downstream; expander discarded after).
- expander h_φ: 3 FC layers size 8192, BN+ReLU on first two, last linear (bias=False).
  WHY expander (not projector that reduces): (1) remove info by which the two reps differ, (2) expand
  dim non-linearly so decorrelating EMBEDDING dims reduces *dependencies* (not just correlations) at
  the representation level. Larger expander than representation dim → better (like Barlow). Perf rises
  with dim 256→16384 (55.9→68.8), saturates ~8192.
- Optimizer LARS, 1000 epochs, wd 1e-6, lr = base_lr × batch/256, base_lr 0.2, batch 2048, cosine
  decay with 10 warmup epochs, final lr 0.002.
- Augmentations: SimCLR/BYOL pipeline symmetrized: RandomResizedCrop 224, hflip 0.5, colorjitter p0.8,
  grayscale p0.2, gaussian blur p0.5, solarization p0.1, ImageNet normalize.
- NO normalization of embeddings (first SSL joint-embed method w/o it). l2-norm hurts 3.5%,
  standardizing embeddings hurts 0.2% (covariance becomes correlation, range [-1,1] — narrower range
  hurts). BN in expander HIDDEN layers helps stability (+1.2%).

## Canonical code notes (facebookresearch/vicreg main_vicreg.py)
- VICReg.forward(x,y): project(backbone(x)), project(backbone(y)). repr_loss = mse(x,y).
- IMPORTANT impl detail: gathers across GPUs (FullGatherLayer), then **centers x,y (x-=mean)** before
  BOTH std and cov (paper pseudocode only centers for cov; centering doesn't change variance so std is
  same). std_loss = mean(relu(1-std_x))/2 + mean(relu(1-std_y))/2  (the /2 because each v averaged,
  matches μ[v(Z)+v(Z')]). γ=1 hardcoded as `1 - std`. ε=0.0001.
- cov = (x.T@x)/(batch_size-1); cov_loss = off_diagonal(cov).pow(2).sum()/num_features per branch.
  off_diagonal trick: flatten()[:-1].view(n-1,n+1)[:,1:].flatten().
- Projector: dims embedding-8192-8192-8192, Linear+BN+ReLU for first two, last Linear bias=False.
- coeffs: sim 25, std 25, cov 1.
- LARS with exclude_bias_and_norm filter (ndim==1 → no wd, no lars adaptation).

## Equations to flag for Codex gate
- v(Z) = (1/d) Σ_j max(0, γ − √(Var(z^j)+ε))   (hinge on STD, not var — gradient argument)
- C(Z) = (1/(n−1)) Σ (z_i−z̄)(z_i−z̄)ᵀ ;  c(Z) = (1/d) Σ_{i≠j} [C(Z)]²_{i,j}
- s(Z,Z') = (1/n) Σ ||z_i−z'_i||²
- ℓ = λs + μ(v(Z)+v(Z')) + ν(c(Z)+c(Z'))
- std-gradient: d√Var/dVar = 1/(2√Var) → ∞ as Var→0 (pushes out of collapse); dVar/dx→0 near mean.
