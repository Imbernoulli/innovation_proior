# IAF synthesis (Phase 1.5)

Verified: arXiv 1606.04934, PDF title "Improved Variational Inference with Inverse
Autoregressive Flow", Kingma, Salimans, Jozefowicz, Chen, Sutskever, Welling (NIPS 2016).
(The task hint "Improving Variational Inference..." is the running-head/v1 form; published
title per PDF metadata = "Improved Variational Inference with Inverse Autoregressive Flow".)
Canonical code: openai/iaf (TensorFlow, official, ResNet-VAE with AR convolutions); clean
MADE-based PyTorch IAF step (Ritchie Vink / community) for the readable code grounding.

## Pain point / research question
Stochastic variational inference / VAEs: maximize the variational lower bound (ELBO). The
inference model q(z|x) must be (1) cheap to evaluate its density and differentiate, (2) cheap
to SAMPLE (both needed per datapoint per step), and — for high-dim z and GPUs — (3)
PARALLELIZABLE across dimensions. These force people to diagonal Gaussian posteriors
q(z|x)=N(μ(x),σ²(x)). But the bound L = log p(x) - KL(q(z|x)‖p(z|x)) is tight only when q
matches the true posterior p(z|x), which is generally NOT factorized. So we need a q that is
far more flexible than diagonal Gaussian, yet still cheap-density + cheap-sample + parallel +
SCALES to high-dim z (where existing flows fail). That is the gap.

## Background
- VI/ELBO: log p(x) ≥ E_{q(z|x)}[log p(x,z) - log q(z|x)] = L(x;θ).
  L = log p(x) - KL(q(z|x)‖p(z|x)). Maximizing L jointly raises log p(x) and lowers
  KL(q‖posterior). Reparameterization (Kingma & Welling 2013; Rezende et al 2014) makes it
  differentiable. Increasing flexibility of q OR p helps the objective.
- Normalizing flow (Rezende & Mohamed 2015): start z_0~q(z_0|x), apply chain z_t=f_t(z_{t-1},x),
  t=1..T; if each Jacobian det computable, log q(z_T|x)=log q(z_0|x) - Σ_t log|det dz_t/dz_{t-1}|.
  Their planar flow f_t(z)=z + u h(wᵀz+b): all info through a SINGLE unit bottleneck → needs a
  LONG chain in high-dim; planar/radial only work for low-dim z (a few hundred). Gap: doesn't
  scale.
- Gaussian autoregressive functions (MADE Germain 2015, PixelCNN/PixelRNN van den Oord 2016,
  WaveNet, RNNs): for y with order, [μ(y),σ(y)] with ∂[μ_i,σ_i]/∂y_j = [0,0] for j≥i
  (autoregressive) → Jacobian lower-triangular w/ zeros on diagonal. Sampling sequential:
  y_0=μ_0+σ_0ε_0, y_i=μ_i(y_{1:i-1})+σ_i(y_{1:i-1})ε_i. Cost ∝ D, sequential → uninteresting
  for direct posterior sampling. BUT the INVERSE is the gold: given y, ε_i=(y_i-μ_i)/σ_i.

## Two key observations (the crux)
1. The INVERSE autoregressive transformation ε=(y-μ(y))/σ(y) can be PARALLELIZED: each ε_i
   depends only on y (all available), the ε_i don't depend on each other. One pass.
2. It has a SIMPLE Jacobian determinant: dε/dy lower-triangular (∂ε_i/∂y_j=0 for j>i), diagonal
   ∂ε_i/∂y_i = 1/σ_i, so log|det dε/dy| = Σ_i -log σ_i(y).
Flexible + parallel + simple log-det → ideal flow for HIGH-DIM latent space. This is the
inverse of the AR sampling step, hence "inverse autoregressive flow."

## IAF (Sec 4)
Encoder outputs μ_0, σ_0, AND an extra context h (extra input to every flow step). Init chain:
  z_0 = μ_0 + σ_0 ⊙ ε,   ε~N(0,I).
Each of T steps uses a DIFFERENT autoregressive net with inputs z_{t-1} and h, outputs μ_t,σ_t:
  z_t = μ_t + σ_t ⊙ z_{t-1}.      (eq 10)
AR w.r.t. z_{t-1} → dz_t/dz_{t-1} triangular with σ_t on diagonal, det = ∏_i σ_{t,i}. (Jacobian
w.r.t. h has no constraint — h is extra context, fine.) Density of final iterate (eq 11):
  log q(z_T|x) = -Σ_i ( ½ ε_i² + ½ log 2π + Σ_{t=0}^{T} log σ_{t,i} ).
Derivation: z_0=μ_0+σ_0ε so log q(z_0|x)= log N(ε;0,I) - Σ_i log σ_{0,i}
= -Σ_i(½ε_i²+½log2π+log σ_{0,i}); each later step subtracts Σ_i log σ_{t,i} (eq 5). Sum gives 11.

## Numerically stable LSTM-style step (eqs 12-14, Algorithm 1) — the actual implementation
AR net outputs two unconstrained vectors [m_t, s_t]:
  σ_t = sigmoid(s_t)                       (13)
  z_t = σ_t ⊙ z_{t-1} + (1 - σ_t) ⊙ m_t    (14)
This is a particular case of eq 10 (with μ_t=(1-σ_t)⊙m_t), so eq 11 still applies. Why this form:
the sigmoid σ_t∈(0,1) keeps the scale bounded (stable), and it's an LSTM-like convex blend of
"keep z" and "write m". KEY init trick: initialize s_t so σ_t starts near 1 (close to +1 or +2
before sigmoid; the +1.5 forget-gate-bias) → step ≈ identity at start (z stays z), a known
"forget gate bias" (Jozefowicz et al 2015), so the flow begins as ~identity and learns deviations.
Algorithm 1 (full): [μ,σ,h]←EncoderNN(x); ε~N(0,I); z←σ⊙ε+μ; l←-sum(log σ + ½ε² + ½log2π);
for t in 1..T: [m,s]←AutoregressiveNN[t](z,h); σ←sigmoid(s); z←σ⊙z+(1-σ)⊙m; l←l-sum(log σ).
(So l = log q(z|x), accumulating -Σ log σ each step — matches eq 11.)

## Special cases / relations (Sec 5)
- Linear IAF (App A): simplest IAF = single linear step. Any full-cov Gaussian N(m,C) is an AR
  model y_i=μ_i+σ_i ε_i with μ_i = m_i + C[i,1:i-1]C[1:i-1,1:i-1]^{-1}(y_{1:i-1}-m_{1:i-1}),
  σ_i² = C[i,i]-C[i,1:i-1]C[...]^{-1}C[1:i-1,i]. Inverting: ε=(y-μ)/σ = L(y-m), L = inverse
  Cholesky of C (lower-triangular). So make L(x),m part of encoder; start from factorized
  Gaussian y=μ(x)+σ(x)⊙ε, then ONE linear IAF step z=L(x)·y turns factorized Gaussian into a
  full-covariance Gaussian posterior (set m=0, ones on L's diagonal since overparameterized).
- vs planar/radial flows (Rezende & Mohamed 2015): those scale poorly to high-dim (bottleneck);
  IAF scales (the whole point). Volume-conserving AR nets first in Deco & Brauer 1995 (nonlinear ICA).
- vs NICE (Dinh 2014) / RealNVP (Dinh 2016): NICE/RealNVP update only HALF the variables as a
  function of the other half (coupling) — large blocks → cheap inverse but generally LESS
  powerful per Rezende & Mohamed; IAF transforms EVERY variable as function of all previous.
- vs Hamiltonian VI (Salimans et al 2014): powerful but very compute-heavy, needs auxiliary bound.
- Reversing order of variables after each step: a volume-preserving permutation, so eq 11
  unchanged; helps (each step attacks dependencies the previous order can't).

## Use in VAE / ResNet VAE (Sec 6, App B/C) — context
Improve VAEs: each layer's posterior parameterized by its own IAF. ResNet VAE: L stacked
residual blocks, each a bottom-up residual unit (inference) + top-down residual unit (inference &
generation). Generative model has AUTOREGRESSIVE PRIOR over latents:
  p(x,z_{1:L}) = p(x|z_{1:L}) p(z_{1:L}),  p(z_{1:L}) = p(z_L)∏_{l<L} p(z_l|z_{l+1:L}).
Autoregressive prior increases flexibility of true posterior → tighter bound without sacrificing
generative flexibility. Bottom-up vs bidirectional inference (App C): bidirectional does a
deterministic bottom-up pass then samples top-down in the generative ordering; posterior
conditioned on bottom-up h^(q) and top-down h^(p) inputs. Approx posterior q(z_l|.) either
diagonal Gaussian or IAF (context c = h^(q) and/or h^(p)). AR nets = masked PixelCNN (van den
Oord 2016) with 0/1/2 hidden layers, ELU nonlinearities; IAF with 0 hidden layers = linear =
Gaussian with off-diagonal covariance. Full IAF with learned σ_t vs fixed σ_t=1: difference
nearly negligible in their CIFAR bpp. (All result NUMBERS excluded from context/reasoning.)

## Evaluation settings (context only)
MNIST (dynamically/statically binarized; Burda et al 2015 setup), convolutional VAE w/ ResNet
(He et al 2015) blocks, single layer of 32 Gaussian stochastic units, 2-layer MADE for IAF,
ELU activations, weight normalization + data-dependent init (Salimans & Kingma 2016). CIFAR-10
natural images, ResNet VAE many stochastic layers. Metrics: variational lower bound (VLB) and
importance-sampled estimate of marginal log-likelihood (128 samples); bits/dim for CIFAR. Adam.
Compare expressiveness via depth/width of IAF.

## Design decisions → why
- Use the INVERSE AR transform (not the forward AR): forward AR sampling is sequential (∝D);
  the inverse is parallel in one pass + simple log-det -Σ log σ → only the inverse is usable as
  a flow for sampling-based VI in high dim.
- z_t = σ_t z_{t-1} + (1-σ_t) m_t with σ=sigmoid (LSTM blend) instead of raw μ_t+σ_t z_{t-1}:
  numerical stability (bounded σ∈(0,1)); plus the forget-gate-bias init makes the chain start at
  identity so optimization starts from a clean factorized Gaussian and learns deviations.
- Extra context h from encoder fed to every step: lets each AR step condition on x (the datapoint)
  without re-running the encoder; h's Jacobian is unconstrained so it doesn't affect the log-det.
- Reverse variable order between steps: volume-preserving, leaves eq 11 unchanged, lets the flow
  mix dependencies both ways across the chain.
- Autoregressive prior in the ResNet VAE generative model: a flexible prior makes the true
  posterior (which is a function of the prior) more flexible, so the IAF posterior can match it →
  tighter bound, without making the generative model itself less flexible.
- Bidirectional inference: lets the posterior of each layer see both bottom-up evidence and
  top-down prior context, matching the generative ordering.

## In-frame
IAF is the method being BUILT. Ancestors to cite: VI/SVI (Blei et al 2012; Hoffman et al 2013),
VAE/reparam (Kingma & Welling 2013; Rezende et al 2014), normalizing flows + planar/radial
(Rezende & Mohamed 2015), MADE (Germain et al 2015), PixelCNN/PixelRNN/WaveNet (van den Oord et al
2016), NICE (Dinh et al 2014), RealNVP (Dinh et al 2016), Hamiltonian VI / auxiliary variables
(Salimans et al 2014; Ranganath et al 2015; Tran et al 2015), forget-gate bias (Jozefowicz et al
2015), ResNet (He et al 2015, 2016), weight norm (Salimans & Kingma 2016), IWAE (Burda et al
2015), Deco & Brauer 1995 (AR nonlinear ICA). NEVER reference the IAF paper itself.
