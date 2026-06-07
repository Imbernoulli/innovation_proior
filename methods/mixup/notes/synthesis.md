# mixup — synthesis notes

## Pain point
ERM minimizes the empirical risk R_δ(f) = (1/n)Σ ℓ(f(x_i), y_i), i.e. expected loss under the empirical distribution P_δ = (1/n)Σ δ(x_i, y_i) — Dirac masses on the training points. Two consequences for large nets (params ~ n):
- Memorization: a trivial minimizer of R_δ memorizes the data even under regularization or with random labels (Zhang et al. 2016).
- Off-manifold misbehavior: ERM only constrains f at n points; between/outside them f is unconstrained → erratic predictions, adversarial fragility (Szegedy et al. 2013): tiny gradient-ascent perturbations flip predictions.
Classical VC theory says ERM converges iff machine size doesn't grow with n — but modern nets violate this. So the empirical distribution is the wrong object to minimize against.

## VRM (Chapelle et al. 2000, "Vicinal Risk Minimization")
Replace each Dirac with a vicinity distribution ν(x̃,ỹ | x_i,y_i):
P_ν(x̃,ỹ) = (1/n)Σ ν(x̃,ỹ | x_i,y_i).
Then sample a virtual dataset D_ν = {(x̃_i,ỹ_i)} and minimize empirical vicinal risk R_ν(f)=(1/m)Σ ℓ(f(x̃_i),ỹ_i).
Chapelle's instance: Gaussian vicinity ν = N(x̃−x_i, σ²)·δ(ỹ=y_i) = additive Gaussian noise on inputs, label unchanged. This is exactly the formalization of data augmentation (Simard 1998: rotations/translations/crops/flips). Drawbacks of standard augmentation: dataset-dependent, needs expert/domain knowledge, and it assumes the vicinity shares the SAME class — it never models the relation ACROSS classes.

## The mixup vicinal distribution (the contribution)
Want a vicinity that is (a) data-agnostic (no domain knowledge), (b) crosses class boundaries. Generic vicinal distribution:
μ(x̃,ỹ | x_i,y_i) = (1/n) Σ_j E_λ[ δ(x̃ = λx_i+(1−λ)x_j, ỹ = λy_i+(1−λ)y_j) ], λ ~ Beta(α,α).
Sampling produces:
  x̃ = λ x_i + (1−λ) x_j
  ỹ = λ y_i + (1−λ) y_j   (one-hot labels → soft label)
The encoded prior: linear interpolations of inputs → linear interpolations of targets. α controls interpolation strength; α→0 recovers ERM (Beta(0,0) puts mass at {0,1} → λ∈{0,1} → no mixing). α=1 → Beta(1,1)=Uniform[0,1].

## Why linear behavior / why it regularizes (the "what is mixup doing")
Training on convex combos and asking the prediction to be the convex combo of the endpoint labels pushes f toward acting linearly between training points. Linear interpolation is the simplest off-data behavior (Occam) → reduces oscillation/overconfidence in-between points, smoother uncertainty estimates, smaller input-gradient norms between examples → robustness to adversarial perturbation (adversarial noise = gradient ascent on input; smaller gradient norms along plausible directions = harder attack). Harder to memorize random labels: interpolations between real examples are easy to fit, interpolations involving random labels are not → resists memorization (large α → virtual points farther from real → memorization harder).

## Lipschitz/complexity argument (appendix proof, lived inline)
Define the expected mixup model f̃(x) = E_{x',λ} f̂(λx+(1−λ)x'). Assume f̂ achieves zero training error so on interpolated points f̂(λx+(1−λ)x'')=λf(x)+(1−λ)f(x''). Lipschitz over data Lip̂(g)=sup_{x,x'∈D} ‖g(x')−g(x)‖/‖x'−x‖.
  Lip̂(f̃) = sup ‖f̃(x')−f̃(x)‖/‖x'−x‖
          = sup ‖E_{x'',λ}[ f̂(λx'+(1−λ)x'') − f̂(λx+(1−λ)x'') ]‖/‖x'−x‖
          = sup ‖E_{x'',λ}[ λf(x')+(1−λ)f(x'') − λf(x) − (1−λ)f(x'') ]‖/‖x'−x‖   (zero-error → linear on interps)
          = sup ‖E_λ[ λ(f(x')−f(x)) ]‖/‖x'−x‖
          ≤ E[λ] · sup ‖f(x')−f(x)‖/‖x'−x‖
          = E[λ] · Lip̂(f).
So E[λ] upper-bounds the complexity (Lipschitz) of the mixup model relative to the target. The (1−λ)f(x'') terms cancel — the third equality requires knowing the target value AT the interpolated location, which is exactly what mixup supplies. Gaussian-noise augmentation or shrinking-to-constant cannot make that equality. For Beta(α,α), E[λ]=1/2.

## Design choices (from ablation §4.9) — why mixup and not alternatives
- Interpolate raw INPUTS, not latent layers (Layer 1-5): mixing higher-layer features gives weaker regularization (large weight decay helps more → less regularizing). Input mixing strongest.
- Mix ALL classes (AC), not same-class (SC): same-class mixing (and SMOTE-style nearest-neighbor same-class interpolation, Chawla 2002) gives no notable gain — the cross-class signal is what matters.
- Mix random pairs (RP), not kNN (k=200): nearest-neighbor restriction unneeded; random pairs strongest regularization.
- Mix LABELS too, not single hard label: using a convex combo of one-hot labels beats picking the closer example's label. (Label smoothing / confidence penalty also use soft labels but independent of the inputs — mixup ties label softness to the input interpolation.)
- 3+ examples via Dirichlet: no extra gain, more compute. Use pairs.
- One data loader + shuffle (randperm) the same minibatch instead of two loaders: equally good, less I/O. This is the canonical implementation trick.

## Loss-linearity implementation identity (canonical code)
Because the label is mixed and CE is linear in the (one-hot/soft) target:
ℓ(f(x̃), λy_a+(1−λ)y_b) = λ ℓ(f(x̃), y_a) + (1−λ) ℓ(f(x̃), y_b) for cross-entropy.
So instead of building a soft-label tensor, code computes:
  mixed_x = lam*x + (1-lam)*x[perm]
  loss = lam*CE(out, y) + (1-lam)*CE(out, y[perm])
This is the few-line implementation. Beta sampled once per minibatch (scalar λ).

## Canonical code structure (facebookresearch/mixup-cifar10)
- mixup_data(x,y,alpha): lam=np.random.beta(a,a) if a>0 else 1; index=randperm(B); mixed_x=lam*x+(1-lam)*x[index]; return mixed_x, y_a=y, y_b=y[index], lam.
- mixup_criterion(crit,pred,y_a,y_b,lam)=lam*crit(pred,y_a)+(1-lam)*crit(pred,y_b).
- train loop: standard SGD (lr0.1, momentum0.9, wd1e-4), PreAct ResNet-18, CIFAR-10 with standard crop+flip+normalize augmentation underneath, step LR /10 at epoch 100,150, 200 epochs, batch 128, alpha default 1.0.

## In-frame ancestors to cite (prior art, OK to name)
Vapnik (ERM, VC theory); Chapelle et al. 2000 (VRM); Simard et al. 1998 (data augmentation / tangent prop lineage); Zhang et al. 2016 (memorization / random labels); Szegedy et al. 2013 + Goodfellow et al. 2014 (adversarial examples, FGSM); Chawla et al. 2002 (SMOTE); DeVries & Taylor 2017 (feature-space interpolation/extrapolation); Szegedy et al. 2016 (label smoothing); Pereyra et al. 2017 (confidence penalty). Do NOT name the target paper/authors.
