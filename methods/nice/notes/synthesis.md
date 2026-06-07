# NICE synthesis (Phase 1.5)

Verified: arXiv 1410.8516, "NICE: Non-linear Independent Components Estimation",
Dinh, Krueger, Bengio (ICLR 2015 workshop). Title confirmed from src/nice.tex line 123.

## Pain point / research question
Unsupervised density modeling of complex high-dim continuous data. Want exact
tractable log-likelihood (not a bound, not adversarial), easy unbiased ancestral
sampling, and a representation in which the data distribution is easy to model.
View: "a good representation is one in which the data distribution is easy to
model" → ask for a transformation h=f(x) such that p_H(h)=∏_d p_{H_d}(h_d) factorizes
(independent latent components).

## Central object: change of variables
p_X(x) = p_H(f(x)) |det ∂f/∂x|, with f invertible, dim h = dim x.
log p_X(x) = log p_H(f(x)) + log|det ∂f/∂x|.
Factorial prior → NICE criterion:
log p_X(x) = Σ_d log p_{H_d}(f_d(x)) + log|det ∂f/∂x|.
The det term penalizes contraction / rewards expansion at data points (counteracts
the trivial likelihood-inflation by contraction). Interpretation: f = encoder,
f^{-1} = decoder, deterministic, perfect autoencoder pair → reconstruction term is
constant; what remains of the variational criterion is the prior (KL) term +
volume/entropy (log-det) term.

## Central difficulty
For arbitrary f on R^D, det of D×D Jacobian is O(D^3) and ill-conditioned, and need
f invertible too. So: design f expressive yet with (a) trivial Jacobian determinant
and (b) trivial inverse.

## Triangular structure (Sec 3.1)
det of triangular matrix = product of diagonal entries → O(D). If Jacobian is
triangular by construction, det collapses. (Same reason autoregressive/NADE are
tractable: their adjacency is strictly triangular.) Composition: f = f_L∘...∘f_1;
forward/backward compose; det = product of layer dets. So design elementary layers.

## Coupling layer (Sec 3.2)
General: partition [1..D] into I_1,I_2, d=|I_1|, m fn on R^d:
  y_{I_1}=x_{I_1};  y_{I_2}=g(x_{I_2}; m(x_{I_1}))
g = coupling law, invertible in first arg given second. Jacobian:
  [[I_d, 0],[∂y_{I_2}/∂x_{I_1}, ∂y_{I_2}/∂x_{I_2}]]
→ det = det(∂y_{I_2}/∂x_{I_2}). Off-diagonal (containing m's derivatives, the messy
part) never enters det → m can be ARBITRARY (a deep ReLU MLP). Inverse:
  x_{I_1}=y_{I_1}; x_{I_2}=g^{-1}(y_{I_2}; m(y_{I_1})). m never inverted.

Additive coupling law: g(a;b)=a+b →
  y_{I_2}=x_{I_2}+m(x_{I_1}); inverse x_{I_2}=y_{I_2}-m(y_{I_1}).
∂y_{I_2}/∂x_{I_2}=I → unit Jacobian determinant (det=1). Inverse as cheap as forward.
Chosen over multiplicative g(a;b)=a⊙b (b≠0) and affine g(a;b)=a⊙b_1+b_2 (b_1≠0) for
NUMERICAL STABILITY: with rectified (ReLU) m the additive transform is piecewise
linear. (Affine/multiplicative would be RealNVP's later move; here rejected.)

## Combining (Sec 3.2)
Coupling leaves a block unchanged → must alternate which block is updated. Examining
the Jacobian: at least THREE coupling layers needed for all dims to influence one
another; "we generally use four." (Experiments: 4 coupling layers.)

## Allowing rescaling (Sec 3.3) — THE WALL & PATCH
Composition of unit-det additive couplings is necessarily VOLUME-PRESERVING
(det=1 everywhere). But density estimation needs to contract volume where data piles
up. Patch: a diagonal scaling matrix S as TOP layer, x_i → S_ii x_i. In practice
parametrized exponentially: h = exp(s) ⊙ h^{(4)}, scale param s, so S_ii = exp(s_i).
Then log-det contribution = Σ_i log|S_ii| = Σ_i s_i. Criterion:
log p_X(x) = Σ_i [ log p_{H_i}(f_i(x)) + log|S_ii| ] = Σ_i log p_{H_i}(f_i(x)) + Σ_i s_i.
Interpretation: σ_d = S_dd^{-1} ~ scale of each independent component, analogue of PCA
eigenspectrum; large S_ii → unimportant dim. As S_ii→+∞ effective dim drops by 1
(manifold). Prior term pushes S_ii small; the log S_ii det term prevents S_ii→0.

## Prior (Sec 3.4)
Factorial. Gaussian: log p_{H_d} = -1/2 (h_d^2 + log 2π).
Logistic: log p_{H_d} = -log(1+exp(h_d)) - log(1+exp(-h_d)) = -softplus(h)-softplus(-h).
Use logistic generally — "better behaved gradient" (bounded score, ∈(-1,1)); gradient
of log-logistic is tanh-like / bounded vs gaussian's unbounded -h.

## Experiments setup (context only)
MNIST(784), TFD(2304), SVHN(3072), CIFAR-10(3072). Dequantization (Uria/RNADE 2013):
add uniform noise 1/256, rescale to [0,1]^D; CIFAR-10 noise 1/128, rescale to [-1,1]^D.
Preprocessing: none (MNIST), approx whitening (TFD), exact ZCA (SVHN, CIFAR-10).
Partition = odd (I_1) / even (I_2) components. 4 coupling layers each m a deep ReLU
net w/ linear outputs: MNIST 5 hidden layers ×1000; TFD 4×5000; SVHN/CIFAR 4×2000.
Prior: logistic (MNIST,SVHN,CIFAR), gaussian (TFD).
Optimizer Adam, lr=1e-3, momentum(β1)=0.9, β2=0.01 (!), λ=1, ε=1e-4. 1500 epochs,
best by val log-lik. Bits/dim reporting adds log(256)*D. (Results numbers are
proposed-method outcomes → excluded from context/reasoning.)

## Appendix derivations (must be lived in reasoning)
- Approx whitening (App B): z = Lx + b, L lower triangular, b bias — done inside the
  NICE framework with an affine map + standard gaussian prior = learning a gaussian.
  Lower-triangular L: its log-det = Σ log|L_ii|, still tractable. Equivalent to learning
  a Gaussian (Cholesky).
- VAE-as-NICE (App C / appendix:vae): SGVB maximizes log-likelihood on the pair
  (x, ε) in a NICE model with two affine coupling layers + gaussian prior. Recognition
  net z = g_φ(ε|x), ε~N(0,I). Define ξ = (x - f_θ(z))/σ. Standard gaussian prior on
  h=(z,ξ). Then
  log p_{(x,ε)}(x,ε) = log p_H(h) - D_X log σ + log|det ∂g_φ/∂ε(ε;x)|.
  Subtract log p_ε(ε):
  ... = log p_H(h) - D_X log σ - log q_{Z|X}(z)   [since the change-of-var of g_φ relates
        p to q]
      = log p_ξ(ξ) + log p_Z(z) - D_X log σ - log q_{Z|X}(z)
      = log p_{X|Z}(x|z) + log p_Z(z) - log q_{Z|X}(z)
  which is the MC estimate of the SGVB/ELBO cost. So VAE = NICE on (x,ε) with two
  affine couplings + gaussian prior. (z→x is one affine coupling via f_θ giving ξ; the
  recognition net g_φ is the other coupling ε→z given x.) The -D_X log σ is the
  log-det of the x = f_θ(z) + σξ scaling.

## Code (fmu2/NICE PyTorch, canonical clean reimpl) — grounding
- Coupling: reshape (B,W)->(B,W//2,2), split on/off by parity (mask_config), shift =
  ReLU MLP(off) [in_block Linear+ReLU, hidden-1 mid blocks Linear+ReLU, out_block
  Linear], on = on ± shift, restack. No log-det (additive → 0).
- Scaling: self.scale param init zeros (so exp(scale)=1). forward: log_det_J =
  sum(scale); x = x*exp(scale) (fwd) or x*exp(-scale) (reverse). NOTE: in this impl
  scaling sits at end of f (encoder x->z) which matches "top layer".
- NICE: coupling list with alternating mask_config=(mask_config+i)%2, scaling at top.
  f: couplings then scaling -> (z, log_det_J). g: scaling reverse then couplings
  reversed. log_prob: z,logdet=f(x); sum(prior.log_prob(z),dim=1)+log_det_J. sample:
  z=prior.sample; g(z).
- StandardLogistic prior log_prob = -(softplus(z)+softplus(-z)). Adam betas=(0.9,0.01),
  eps=1e-4, lr=1e-3. bits/dim = (mean NLL + log(256)*D)/(D*log2)... per impl
  (mean_loss + log(256)*full_dim).

## Design decisions → why
- Change-of-variables (vs VAE bound / GAN / Boltzmann): only route giving EXACT
  likelihood + exact deterministic inference + exact ancestral sampling at once.
- Triangular Jacobian: to kill the O(D^3) det. Why coupling not triangular-weight nets:
  triangular-weight nets over-constrain architecture (only depth & nonlinearity left).
- Additive (vs affine/multiplicative): numerical stability / piecewise-linear with ReLU.
  Cost: unit det → volume preserving → needs the diagonal scaling patch.
- ≥3 (use 4) coupling layers, alternating: one layer freezes a block; need enough to
  mix all dims both ways.
- Diagonal scaling top layer, exp-parametrized: restore ability to change volume; exp
  keeps S_ii>0 and gives the clean Σ s_i log-det; relates to PCA spectrum / manifold.
- Logistic prior: better-behaved (bounded) gradient than gaussian.
- Dequantization: discrete pixels would let a continuous density spike to ∞ likelihood;
  uniform noise gives a proper upper bound (Theis/Uria argument).

## In-frame notes
NICE is the method being BUILT. Ancestors to cite by author/year: ICA/maximum-likelihood
ICA (Hyvärinen), Gaussianization (Chen & Gopinath 2000), Bengio 1991 (learning the
transform), Rippel & Adams 2013 (regularized AE proxy), nonlinear ICA via ensemble
learning (Hyvärinen 1999; Roberts; Lappalainen), VAE/SGVB (Kingma & Welling 2014;
Rezende et al 2014), Helmholtz machine (Dayan 1995), DBM (Salakhutdinov & Hinton 09),
AIS (Salakhutdinov & Murray 08), GAN (Goodfellow et al 2014), NADE (Larochelle & Murray
2011), Bengio & Bengio 1999, RNADE/dequantization (Uria et al 2013), Adam (Kingma & Ba),
Bengio et al ICML2013 (volume-expansion-of-interesting-regions view).
