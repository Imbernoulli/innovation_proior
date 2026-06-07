# MAF synthesis (Phase 1.5)

Verified: arXiv 1705.07057, "Masked Autoregressive Flow for Density Estimation",
Papamakarios, Pavlakou, Murray (NIPS 2017). Title confirmed src/tex/preamble.tex line 54.
Canonical code: gpapamak/maf (Theano, official); clean PyTorch reimpl: e-hulten/maf
(made.py, maf_layer.py, maf.py, batch_norm_layer.py) — code grounded here.

## Pain point / research question
Neural density estimation: model p(x) flexibly AND tractably (exact density + tractable
learning). Two flexible+tractable families: autoregressive models and normalizing flows.
Want: state-of-the-art general-purpose density estimation, fast to TRAIN/EVALUATE on GPU
(so density of an arbitrary external datapoint in one pass), more flexible than a single
autoregressive model.

## Background pieces
- Autoregressive: p(x)=∏_i p(x_i|x_{1:i-1}). RNADE (Uria 2013): mixtures of Gaussian/Laplace
  conditionals + linear hidden-state rule. Recurrent versions sequential: D sequential
  computations to compute p(x) — bad for GPU.
- MADE (Germain 2015): a fully-connected autoencoder with binary MASKS multiplying weight
  matrices, dropping connections so output i depends only on inputs 1..i-1. Gives the
  autoregressive property AND a single-forward-pass evaluation on GPU. Mask design: assign
  each input/hidden unit a "degree" m∈{1..D} (input degree = its order index; outputs degrees
  0..D-1); a unit may connect only to units of lower-or-equal degree; every hidden layer must
  contain every degree (necessary+sufficient so output i connects to all inputs <i, no extra
  conditional independences). Connectivity mask M[j,k] = 1{ m_next[j] >= m_prev[k] }.
- Normalizing flows (Rezende & Mohamed 2015): p(x)=π_u(f^{-1}(x))|det df^{-1}/dx|, f invertible
  with tractable Jacobian; composable. Prior flows: Gaussianization (Chen & Gopinath 2000);
  nonsingular-weight invertible nets (Rippel & Adams 2013; Ballé 2016) — det is O(D^3) general;
  planar/radial flows and IAF (Kingma 2016) tractable-by-design but for VARIATIONAL INFERENCE,
  can only efficiently get density of THEIR OWN samples, not externally-provided datapoints;
  NICE (Dinh 2014) & RealNVP (Dinh 2017) tractable Jacobian, suitable for density estimation.

## The key realization (Kingma 2016, used as starting point)
An autoregressive model, when GENERATING, is a differentiable transform of an external source
of randomness u (the randn() calls). That transform has a triangular Jacobian by design, and
for Gaussian conditionals it is invertible → it IS a normalizing flow. So can increase
flexibility by STACKING: each model models the random numbers of the next.

## MAF core (Sec 3)
Autoregressive model with single-Gaussian conditionals:
  p(x_i|x_{1:i-1}) = N(x_i; μ_i, (exp α_i)^2),  μ_i = f_{μ_i}(x_{1:i-1}), α_i = f_{α_i}(x_{1:i-1}).
Generation recursion (u→x):  x_i = u_i exp(α_i) + μ_i,  u_i~N(0,1).
This is x=f(u), u~N(0,I). f is easily invertible:
  u_i = (x_i - μ_i) exp(-α_i),  μ_i,α_i functions of x_{1:i-1}.  [x→u in ONE pass via masking]
Jacobian of f^{-1} triangular (∂u_i/∂x_j=0 for j>i; diagonal ∂u_i/∂x_i = exp(-α_i)):
  |det df^{-1}/dx| = exp(-Σ_i α_i),  α_i=f_{α_i}(x_{1:i-1}).
Substitute into change-of-vars → exact density.
DIAGNOSTIC: transform train data {x_n}→{u_n} via x→u; if u_i not iid standard normal, the model
is a bad fit (the figure: order (x1,x2) MADE fails, scatter non-Gaussian).
STACK: M_1,...,M_K; M_2 models u_1, ..., M_K models u_{K-1}, final u_K ~ N(0,I). Adds flexibility
(5-layer MAF learns multimodal conditionals though each MADE has unimodal). Implement each
{f_{μ_i}, f_{α_i}} with a MADE with Gaussian conditionals (outputs μ and α for all i in one pass).
ORDER: first layer uses dataset's default order; reverse order each successive layer (same as IAF).

## Relation to IAF (Sec 3.2)
IAF layer: x_i = u_i exp(α_i) + μ_i, but μ_i = f_{μ_i}(u_{1:i-1}), α_i = f_{α_i}(u_{1:i-1}) —
conditioners read the RANDOM NUMBERS u, not the data x.
Consequence (the whole trade-off):
- MAF: density p(x) of ANY x in ONE pass (x→u parallel via masking); SAMPLING needs D sequential
  passes (must compute x_i before x_{i+1}'s conditioner).
- IAF: SAMPLE + get its density in ONE pass (u→x parallel); density of EXTERNAL x needs D
  sequential passes (must invert to find u).
So: connect conditioner to x_{1:i-1} → MAF (density estimation); to u_{1:i-1} → IAF (variational
inference / recognition net, only needs density of own samples).
THEORETICAL EQUIVALENCE (App A): training MAF by max-likelihood = fitting an implicit IAF to the
base density via stochastic variational inference. p_x(x)=π_u(f^{-1}(x))|det df^{-1}/dx|;
implicit p_u(u)=π_x(f(u))|det df/du|. Then
  KL(π_x ‖ p_x) = E_{π_x}[log π_x - log π_u(f^{-1}) - log|det df^{-1}/dx|]
  change of vars x↦u: = E_{p_u}[log π_x(f(u)) - log π_u(u) + log|det df/du|]
                      = E_{p_u}[log p_u(u) - log π_u(u)] = KL(p_u ‖ π_u).
So minimizing KL(π_x‖p_x) (=max-likelihood of MAF) ≡ minimizing KL(p_u‖π_u) (the VI objective for
an IAF with base π_x, transform f^{-1}, reparam trick). MAF training = variationally training an
implicit IAF with MAF's base π_u as its target posterior.

## Relation to RealNVP (Sec 3.3)
RealNVP coupling layer: x_{1:d}=u_{1:d}; x_{d+1:D}=u_{d+1:D}⊙exp(α)+μ; μ=f_μ(u_{1:d}),
α=f_α(u_{1:d}). NICE = special case α=0. Coupling = special case of BOTH MAF and IAF:
recover from MAF by μ_i=α_i=0 for i≤d, and μ_i,α_i functions of only x_{1:d} for i>d (for IAF use
u_{1:d}). So MAF and IAF are more flexible (but different) generalizations of RealNVP, where
EVERY element is scaled+shifted as function of ALL previous elements (not just a fixed split).
RealNVP's advantage: BOTH sample and density in ONE pass; MAF needs D passes to sample, IAF needs D
to estimate density.

## Conditional MAF (Sec 3.4)
p(x|y)=∏_i p(x_i|x_{1:i-1}, y). Augment MADE inputs with y; don't drop any connection from y;
y becomes an extra input to every layer. RealNVP made conditional same way.

## Batch normalization in flows (App B) — MUST live in reasoning
BN as an invertible flow layer between every two autoregressive layers (and before base). x near
data, u near base:
  x = (u - β) ⊙ exp(-γ) ⊙ (v+ε)^{1/2} + m   (forward u→x)
  u = (x - m) ⊙ (v+ε)^{-1/2} ⊙ exp(γ) + β    (inverse x→u)
  |det df^{-1}/dx| = exp(Σ_i (γ_i - 1/2 log(v_i+ε))).
γ exponentiated (unlike standard BN) to ensure positivity and simplify log-det. m,v = minibatch
mean/var at train (batch 100); whole-train mean/var at test. ε=1e-5. Reduces training time,
increases stability, improves performance (also seen by RealNVP).

## Experiments setup (context only — no outcomes)
Datasets: UCI POWER(D6) GAS(8) HEPMASS(21) MINIBOONE(43), BSDS300 8x8 patches (D63, bottom-right
pixel discarded, dequantize+rescale[0,1], subtract patch mean), MNIST(784) CIFAR-10(3072)
conditional (one-hot y, p(x)=Σ_y p(x|y)p(y), p(y)=1/10). Image preprocessing (RealNVP-style):
dequantize uniform noise, rescale [0,1], logit transform x↦logit(λ+(1-2λ)x), λ=1e-6 MNIST,
λ=0.05 CIFAR; CIFAR augment horizontal flips. Models compared: MADE (Gaussian), MADE MoG (C=10),
RealNVP(5/10), MAF(5/10), MAF MoG(5). Hidden units: ReLU (tanh for GAS). RealNVP general-purpose
impl: two nets f_α (tanh hidden) and f_μ (ReLU hidden), linear output; odd/even alternation.
Optimizer Adam, minibatch 100, step 1e-3 (MADE/MADE MoG) or 1e-4 (RealNVP/MAF), L2 coef 1e-6,
early stopping (30 epochs no val improvement). Metric: avg test log-likelihood (nats); bits/pixel
for images. (All result NUMBERS excluded.)
Bits/pixel conversion (App C): from logit-space density, b(x) = -log p(x)/(D log2) - log2(1-2λ)
  + 8 + (1/D)Σ_i (log2 σ(x_i) + log2(1-σ(x_i))).
Param counts (App): MADE 3/2 DH + 1/2(L-1)H^2; MADE MoG (C+1/2)DH+1/2(L-1)H^2; RealNVP
2KDH+2K(L-1)H^2; MAF 3/2 KDH+1/2 K(L-1)H^2. RealNVP ~1.3-2x params of comparable MAF.

## Design decisions → why
- Why view AR model as a flow: a single AR model is limited; seeing it as f(u) lets you STACK to
  add flexibility while keeping tractability (composition of flows is a flow).
- Single-Gaussian conditionals (not MoG): so the u→x map is a clean invertible affine recursion
  with a triangular Jacobian; flexibility recovered by stacking instead of by mixture components.
  (MAF MoG = put a MADE MoG as base to also get universal-approximation.)
- MADE (masking) not recurrence: parallel one-pass x→u on GPU, removes the D-step sequential loop
  in density evaluation — the whole speed point.
- Conditioner reads x (MAF) not u (IAF): MAF is for DENSITY ESTIMATION (one-pass density of
  external data), the target task; IAF's u-conditioning is for VI (one-pass own-sample density).
- Reverse order each layer: a single fixed order is a modelling limitation (AR models are
  order-sensitive); reversing lets later layers attack the dependencies the earlier order can't.
- BN between layers, γ exponentiated: stabilizes/normalizes activations between flow layers
  (deep flow training), invertible + tractable log-det so it's a legal flow layer; exp(γ) keeps
  scale positive and simplifies log-det.
- α = log-σ parameterization (exp α): keeps σ>0 unconstrained and gives the clean log-det -Σα.
- step size 1e-4 for MAF/RealNVP vs 1e-3 for MADE: deeper composed flows need a smaller step.

## In-frame
MAF is the method being BUILT. Ancestors to cite: MADE (Germain et al 2015), IAF (Kingma et al
2016) — the "AR-as-flow" realization is theirs and is the starting point, cite it; RealNVP (Dinh
et al 2017) and NICE (Dinh et al 2014); RNADE (Uria et al 2013); normalizing flows (Rezende &
Mohamed 2015); Gaussianization (Chen & Gopinath 2000); Rippel & Adams 2013; batchnorm (Ioffe &
Szegedy 2015); Adam (Kingma & Ba 2015); chain rule / deep belief net stacking (Hinton 2006),
deep MFA stacking (Tang 2012).
