# RealNVP — Synthesis notes (Phase 1.5)

## The pain point / research question
Unsupervised learning of a probabilistic model over high-dimensional, highly-structured
continuous data (natural images). We want a *single* model that is good at all four of:
1. **exact log-likelihood evaluation** (a principled training objective, not a bound / surrogate),
2. **exact, efficient sampling** (ideally parallel, not sequential),
3. **exact, efficient inference** of the latent representation z given x,
4. an **interpretable / usable latent space**.

No prior family hits all four. That is the gap.

## The landscape at the time (load-bearing ancestors and where each falls short)

- **Undirected graphical models** — RBM (Smolensky 1986), DBM (Salakhutdinov & Hinton 2009).
  Trained via the conditional-independence of bipartite structure. But the partition function /
  marginal over latents is intractable → training, evaluation, sampling all need approximations
  (mean-field, MCMC). MCMC mixing time is undetermined; samples highly correlated. Evaluation
  needs AIS. → No exact likelihood, no exact sampling.

- **Directed latent-variable models / VAE** (Kingma & Welling 2013; Rezende et al. 2014).
  Ancestral sampling is simple. But exact posterior inference is intractable, so they
  maximize the ELBO — a *lower bound* on log-likelihood — using an amortized approximate
  inference network and the reparameterization trick. The bound's looseness and the
  approximate posterior limit it. Crucially the reconstruction term is a *fixed-form* cost
  (Gaussian decoder → L2), which over-weights low-frequency content → blurry samples.
  → Only a bound, only approximate inference.

- **Autoregressive models** — fully-visible Bayes nets / NADE / MADE / PixelRNN / PixelCNN
  (Frey 1998; Bengio & Bengio 1999; Larochelle & Murray 2011; Germain et al. 2015; van den Oord
  et al. 2016). Decompose p(x) = ∏_i p(x_i | x_{<i}) by the chain rule under a fixed ordering.
  Exact tractable likelihood, very flexible. BUT: sampling is inherently **sequential** (D steps,
  non-parallelizable) — cumbersome for high-dim images / audio / real-time. Ordering is arbitrary
  yet matters. No natural latent representation. These are the "tractable maximum-likelihood
  nonlinear ICA" benchmark to beat on the *sampling-speed + latent-space* axes.

- **GANs** (Goodfellow et al. 2014; DCGAN, Radford et al. 2015; LAPGAN, Denton et al. 2015).
  Train any differentiable generator g: z→x via a discriminator instead of likelihood. Sharp
  samples, no fixed reconstruction cost. BUT: no likelihood (can't evaluate density / diversity
  intractable), training unstable, no inference network. → No evaluation, no inference, unstable.

- **The change-of-variables / bijective generator idea itself.** If g: z→x is a *bijection*,
  no discriminator and no approximate inference are needed — maximum likelihood works directly via
  p_X(x) = p_Z(z) |det ∂g/∂z|^{-1}. Discussed for ICA max-likelihood (Bell & Sejnowski 1995;
  Hyvärinen 2004), Gaussianization (Chen & Gopinath 2000), and deep density models
  (Rippel & Adams 2013; Bengio 1991). The catch: **naive** change-of-variables needs the Jacobian
  determinant of an arbitrary D×D map → O(D^3) and badly conditioned. So large-scale models of
  this type "have not entered general use." THIS is the precise difficulty to crack.

- **NICE (Dinh et al. 2014)** — the direct parent.
  - Objective: log p_X(x) = log p_H(f(x)) + log|det ∂f/∂x|, f bijective.
  - **Additive coupling layer**: split x into x_{I1}, x_{I2}; set
      y_{I1} = x_{I1},   y_{I2} = x_{I2} + m(x_{I1}),  m an arbitrary neural net.
    Inverse: x_{I1}=y_{I1}, x_{I2}=y_{I2}−m(y_{I1}). No inverse of m required.
    Jacobian is lower-triangular with **unit diagonal** → det = 1. So you get a tractable
    determinant AND m can be arbitrarily complex (its own Jacobian never enters). Stack with
    alternating partitions so every unit gets transformed.
  - **Limitation #1 (the one RealNVP fixes):** unit-diagonal Jacobian ⇒ the map is
    **volume-preserving** (det=1 everywhere). The model cannot locally expand/contract volume,
    so it can't allocate density mass freely; a separate final **diagonal scaling layer** S
    (det = ∏|S_ii|, contributing Σ log|S_ii|) is bolted on just to get *some* non-unit volume,
    but the coupling layers themselves stay rigid.
  - **Limitation #2:** NICE was fully-connected / not built for the 2-D local structure of images,
    no multi-scale hierarchy, no batch norm in the flow.
  - Prior p_H factorial (logistic or Gaussian).

## The core idea RealNVP lands on
Keep NICE's coupling trick (arbitrary conditioner, triangular Jacobian, cheap inverse) but make
the coupling **affine** instead of additive, so it is **non-volume-preserving** while *still*
keeping the Jacobian triangular and its log-det cheap. Then make it work on images: masked
convolutions for the partition, a squeeze op + multi-scale factoring-out, residual nets with
batch/weight norm inside s,t, and batch norm folded into the flow as just another bijector.

### Affine coupling (the central object)
Given D-dim x, choose d<D:
  y_{1:d}   = x_{1:d}
  y_{d+1:D} = x_{d+1:D} ⊙ exp(s(x_{1:d})) + t(x_{1:d})
s, t: R^d → R^{D−d} (scale, translation), arbitrary nets.
Jacobian:
  ∂y/∂xᵀ = [[ I_d , 0 ],
            [ ∂y_{d+1:D}/∂x_{1:d}ᵀ , diag(exp(s(x_{1:d}))) ]]
Lower-triangular ⇒ det = ∏_j exp(s(x_{1:d})_j) = exp(Σ_j s(x_{1:d})_j).
**log-det = Σ_j s(x_{1:d})_j** — no Jacobian of s or t needed; s,t arbitrarily complex.
Inverse (no inverse of s,t needed):
  x_{1:d}   = y_{1:d}
  x_{d+1:D} = (y_{d+1:D} − t(y_{1:d})) ⊙ exp(−s(y_{1:d})).
Forward and inverse cost identical ⇒ sampling as cheap as inference. det≠1 ⇒ non-volume-preserving,
so the final diagonal-scaling crutch of NICE is gone; the layers themselves can reshape volume.

### Composition
For a composition f_b ∘ f_a: chain rule on Jacobians, det(AB)=det(A)det(B) ⇒ log-dets add;
(f_b∘f_a)^{-1} = f_a^{-1}∘f_b^{-1}. So stacking preserves both tractable det and tractable inverse.
Alternate which partition is frozen so all units get updated.

### Masking (how to partition on images)
y = b⊙x + (1−b)⊙( x⊙exp(s(b⊙x)) + t(b⊙x) ), b a binary mask.
Two masks exploiting image locality:
- **spatial checkerboard**: b=1 where (i+j) odd.
- **channel-wise**: b=1 for first half of channels, 0 for second half.
s,t are rectified (ReLU) conv nets so they see 2-D structure.

### Squeeze + multi-scale
- **Squeeze**: reshape s×s×c → (s/2)×(s/2)×4c (group each 2×2×c block into 1×1×4c). Trades space
  for channels so channel-wise masking becomes meaningful and the receptive field grows.
- Per scale: 3 coupling layers with alternating checkerboard masks → squeeze → 3 coupling layers
  with alternating channel-wise masks. Final scale: 4 coupling layers, checkerboard only.
- **Factor out** half the dims at each scale (recursive):
    h^{(0)}=x;  (z^{(i+1)}, h^{(i+1)}) = f^{(i+1)}(h^{(i)});  z^{(L)}=f^{(L)}(h^{(L−1)});
    z=(z^{(1)},…,z^{(L)}).
  Reasons: (a) cuts compute/memory/parameters (don't push all D dims through all layers);
  (b) distributes the loss through the net (deep-supervision philosophy, Lee et al. 2014 /
  VGG factoring, Simonyan & Zisserman 2014); (c) gives a hierarchy — finer-scale (earlier) units
  must be Gaussianized before coarser-scale (later) ones → coarse-to-fine latent levels.

### Batch norm inside the flow (as a bijector)
BN acts as a per-dimension linear rescale x ↦ (x−μ̃)/√(σ̃²+ε). Its Jacobian det is
(∏_i (σ̃_i²+ε))^{−1/2}; this log-det term is folded straight into the total log-det. Novel variant:
use a running average over recent minibatches for μ̃, σ̃² (only backprop through current-batch
stats μ̂,σ̂²), so it is robust with very small minibatches:
  μ̃_{t+1}=ρμ̃_t+(1−ρ)μ̂_t,  σ̃²_{t+1}=ρσ̃²_t+(1−ρ)σ̂²_t.
Why it matters: training the change-of-variables objective with a scale parameter (exp(s)) is
numerically unstable (linear+exponential interplay); BN reshapes the cost and lets a much deeper
stack of coupling layers train. (Analogous to reward normalization in RL.)

### Data preprocessing (the logit trick, needed for continuous density on bounded pixels)
Pixels are in [0,256] after jittering/dequantization. Model the density of
  logit(α + (1−α)·x/256),  α=.05,
to remove boundary effects; the log-det of this bijection is added to the likelihood. (k=256
discrete levels → subtract D·log k to report bits/dim; dequantize x←(x·255+u)/256, u~U[0,1].)

### Prior
p_Z = isotropic unit Gaussian (factorial). Could be anything (autoregressive, VAE) but Gaussian
is the simplest factorial latent.

## Design-decision → why table
- **Affine (×exp(s)+t) vs NICE additive (+m)?** Additive ⇒ det=1 ⇒ volume-preserving ⇒ can't
  reallocate density mass; needs a final diagonal scaling crutch. Affine makes det=exp(Σs)≠1
  (non-volume-preserving) while keeping the Jacobian triangular ⇒ strictly more expressive at no
  extra cost. THIS is the whole "real NVP" = real-valued **non-volume-preserving**.
- **exp(s) (multiplicative in log space) vs raw scale?** Need scale > 0 to stay invertible and to
  make log-det additive (= Σ s, the pre-exp value) — no log/abs needed, and exp keeps the inverse
  a clean ⊙exp(−s).
- **s = tanh(·)·(learned per-channel scale), t = affine?** Unconstrained s exponentiated explodes
  exp(s); tanh bounds the raw scale to (−1,1) for stability, the learned per-channel factor
  restores range. t needs no bound (additive). Wrap the rescale in weight-norm. (Numerical
  stability of exp is the reason; without it training diverges / NaNs.)
- **Triangular Jacobian (coupling) vs general bijector?** General det is O(D^3) and ill-conditioned
  — the exact reason naive change-of-variables never scaled. Triangular ⇒ det = product of diagonal
  ⇒ O(D). The single key trick.
- **Conditioner depends only on the frozen half?** So the transformed half's Jacobian wrt itself
  is diagonal (the off-diagonal block ∂y_{d+1:D}/∂x_{1:d} sits *below* the diagonal and never
  affects the determinant). Lets s,t be arbitrarily deep/complex without entering the det or the
  inverse.
- **Alternating masks / composing couplings?** One coupling leaves half the units untouched;
  alternate so everything is transformed; composition keeps det tractable (dets multiply) and
  inverse tractable.
- **Checkerboard then channel-wise (around a squeeze)?** Checkerboard exploits spatial local
  correlation at full resolution; squeeze trades spatial for channel dims; channel-wise mask then
  exploits the now-meaningful channel grouping. Chosen so the channel partition isn't redundant
  with the prior checkerboard.
- **Squeeze 2×2×c→1×1×4c?** Makes channel-wise coupling possible and enlarges effective receptive
  field; it's a pure deterministic reshape (a permutation), Jacobian det = 1.
- **Factor out half at each scale?** Compute/memory/param savings + distributed loss (deep
  supervision) + coarse-to-fine hierarchy of latents. Borrowed factoring idea from VGG.
- **Double hidden features after each squeeze?** Spatial resolution halves, channels quadruple;
  doubling s,t width keeps capacity matched to the data at each scale.
- **Residual nets + ReLU in s,t?** Need deep, expressive conditioners; ResNet + skip connections
  (PixelCNN-style) make them trainable; s,t complexity is free (not in det/inverse).
- **Batch norm folded into the flow?** It's a bijection with a trivial diagonal Jacobian; its
  log-det just adds in. Stabilizes the exp(s) objective and enables much deeper stacks. Running-avg
  variant for small-minibatch robustness.
- **logit(α+(1−α)x/256) preprocessing?** Map bounded pixels to unbounded reals (the model assumes
  unbounded continuous support) and kill boundary effects; α=.05 keeps logit finite at 0/255.
- **Isotropic Gaussian prior?** Simplest factorial latent; log-prior = −½(z²+log2π) per dim,
  trivially differentiable; any factorial prior would do.

## Canonical implementations (for grounding the code)
- Official TensorFlow: tensorflow/models research/real_nvp (real_nvp_multiscale_dataset.py) — saved
  to code/real_nvp_multiscale_dataset.py.
- Clean PyTorch reimplementation (Chris Chute), widely used — code/chrischute-realnvp/:
  models/real_nvp/coupling_layer.py (affine coupling, tanh+weight-norm rescale, checkerboard &
  channel-wise, forward/reverse with sldj), models/real_nvp/real_nvp.py (recursive multi-scale:
  3 checkerboard couplings → squeeze → 3 channel couplings → squeeze/split → recurse; logit
  preprocessing with its log-det), models/real_nvp/real_nvp_loss.py (Gaussian prior ll + sldj −
  D·log k, bits/dim), util/array_util.py (squeeze_2x2, checkerboard_mask), models/resnet (s,t net).
  The final code in answer.md/reasoning.md mirrors this structure.

## Notes on in-frame discipline
- Do NOT name "RealNVP" or cite the paper in context.md/reasoning.md as a published artifact;
  may name the method in answer.md as the thing being built.
- NICE, VAE, GAN, PixelRNN, ICA, Gaussianization, ResNet, BN, VGG, deep-supervision are prior art
  → cite by author/year freely.
- Skip the proposed-method *evaluation results* (bits/dim table, sample quality claims). The
  motivating facts about prior systems (blurry VAE samples, sequential AR sampling, naive CoV is
  O(D^3) ill-conditioned, NICE volume-preserving rigidity) ARE context.
