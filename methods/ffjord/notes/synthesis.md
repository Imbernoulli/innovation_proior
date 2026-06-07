# FFJORD — synthesis (arXiv 1810.01367, verified — Grathwohl, Chen, Bettencourt, Sutskever, Duvenaud, ICLR 2019)

## Pain point
Reversible/flow generative models: warp a simple base p_z through invertible f: R^D→R^D. Change of variables: log p_x(x) = log p_z(z) − log|det ∂f/∂z|, z=f^{-1}(x). The log-det is O(D^3) in general. To dodge it, prior flows RESTRICT f's architecture so the Jacobian is structured:
- Normalizing flows (planar/Sylvester, Rezende 2015; Berg 2018): special functional forms (e.g. rank-1 / det identities). Planar = 1-layer NN with a SINGLE hidden unit per layer. Can't both train-on-data and sample (no tractable analytic inverse) — used for variational posteriors.
- Autoregressive (IAF Kingma 2016; MAF Papamakarios 2017; TAN Oliva 2018): impose an ordering → triangular Jacobian → det = product of diagonal. Great density estimation but O(D) SEQUENTIAL passes to invert/sample.
- Partitioned/coupling (NICE Dinh 2014, RealNVP Dinh 2016, Glow Kingma 2018): split dims, affine-transform one half conditioned on the other → cheap triangular det, inverse same cost as forward, convolution-friendly. But still architecturally constrained; need MANY stacked coupling layers.
All three pay for tractable det with constrained, hand-engineered, low-capacity-per-layer architectures.

## Continuous normalizing flow (CNF), Chen et al. 2018 (Neural ODE) — the ancestor
Replace the discrete stack with continuous dynamics. Sample z_0 ~ p_{z0}; define ODE ∂z(t)/∂t = f(z(t), t; θ); solve IVP from t_0 to t_1 to get x = z(t_1). "Instantaneous change of variables":
   ∂ log p(z(t)) / ∂t = − Tr( ∂f/∂z(t) ).
Integrate: log p(z(t_1)) = log p(z(t_0)) − ∫_{t0}^{t1} Tr(∂f/∂z(t)) dt.
The DETERMINANT becomes a TRACE — and trace is linear, doesn't need structured Jacobian. Reduces O(D^3) → O(D^2) (trace of Jacobian still costs ~D evals of f, one per diagonal entry). Allows a freer architecture (planar CNF = 1-layer NN with MANY hidden units) but still O(D^2), still effectively restricted.
To get z_0 and log p(x) from data x: solve combined reverse IVP (z and log-density) from t_1 to t_0 with initial [z(t1), Δlogp] = [x, 0]:
   [z_0; log p(x) − log p(z(t1))] = ∫_{t1}^{t0} [f; −Tr(∂f/∂z)] dt.
Then log p(x) = log p_{z0}(z_0) − (accumulated −Tr) ... i.e. log p̂(x) = log p_{z0}(z_0) − Δlogp.
Existence/uniqueness needs f and its first derivatives Lipschitz → use smooth Lipschitz activations (tanh, softplus).

## Adjoint method (Pontryagin; Chen 2018) for backprop through ODE solve
For scalar loss L(z(t_1)) = L(∫ f dt), adjoint a(t) = −∂L/∂z(t), and
   dL/dθ = − ∫_{t1}^{t0} (∂L/∂z(t))^T ∂f/∂θ dt.
Solve this second IVP backward with init ∂L/∂z(t_1). Continuous-time analog of backprop; O(1) memory (don't store activations), enables huge batch sizes.

## Core contribution: O(D) unbiased log-density via Hutchinson trace estimator
Exact Tr(∂f/∂z) is O(D^2) (D separate vJPs to get all diagonal entries). Two facts:
1) vector-Jacobian product v^T ∂f/∂z costs ~1 eval of f via reverse-mode autodiff.
2) Hutchinson: for any DxD matrix A and any p(ε) with E[ε]=0, Cov(ε)=I:
   Tr(A) = E_{p(ε)}[ ε^T A ε ].
   (Proof: E[ε^T A ε] = E[Σ_ij ε_i A_ij ε_j] = Σ_ij A_ij E[ε_i ε_j] = Σ_ij A_ij δ_ij = Σ_i A_ii = Tr A.)
So ε^T (∂f/∂z) ε: ONE vJP (ε^T ∂f/∂z) then a dot with ε → O(D), one eval of f. Unbiased estimate of the trace.
Plug into log-density and pull the expectation OUTSIDE the time integral (fix ε for the whole solve so dynamics stay deterministic per solve — Fubini, no bias):
   log p(z(t1)) = log p(z(t0)) − E_{p(ε)}[ ∫_{t0}^{t1} ε^T (∂f/∂z(t)) ε dt ].
ε ~ standard Gaussian or Rademacher (±1 entries; Cov=I). Cost O((DH + D) L̂) vs CNF's O((DH + D^2) L̂) vs discrete flow's O((DH + D^3) L).
Name: Free-form Jacobian of Reversible Dynamics. f is now ANY (Lipschitz) NN — unrestricted.

## Bottleneck trick (variance reduction)
Variance of Hutchinson for Tr(A) grows ~ ||A||_F^2. If f = g∘h with hidden width H < D, by cyclic property of trace:
   Tr(∂f/∂z) = Tr( (∂g/∂h)(∂h/∂z) ) = Tr( (∂h/∂z)(∂g/∂h) )  [HxH instead of DxD]
   = E_ε[ ε^T (∂h/∂z)(∂g/∂h) ε ] with ε ∈ R^H.
Smaller matrix → smaller Frobenius norm → lower variance. Choose H = smallest hidden dim. Empirically: faster convergence with Gaussian ε, not with Rademacher ε.

## Time conditioning
f takes (z(t), t). Tried hypernetworks; settled on simply CONCATENATING t onto z (or the layer input) at every layer (ConcatLinear/ConcatConv). Simple, worked best.

## VAE / amortized variant
Encoder outputs flow params as function of x led to too-hard-to-integrate ODEs. Instead: low-rank update to a GLOBAL weight + input-dependent bias:
   layer(h; x) = σ( (W + Û(x) V̂(x)^T) h + b + b̂(x) ),  Û: D_out×k, V̂: D_in×k (rank k), b̂: D_out.

## Practical details
- ODE solver: Runge-Kutta 4(5) (Dormand-Prince dopri5), adaptive step. atol=rtol=1e-5 typical; tabular atol=1e-8 rtol=1e-6. NFE grows during training, converges to a value independent of D (depends on distribution complexity, not dimension). Thought experiment: Gaussian data → Gaussian base → optimal ODE is zero → 0 evaluations.
- Train with Hutchinson estimator; report test with EXACT trace (except MNIST/CIFAR where exact infeasible — trace-estimator variance of val log-lik < 1e-4).
- Adam, lr 1e-3 decayed to 1e-4. Batch up to 10,000 (tabular), 900 (images) thanks to O(1)-memory adjoint.
- Smooth activations: tanh, softplus, swish.
- Limitations: NFE can grow prohibitively (weight decay/spectral norm reduce it, slight perf cost); only non-stiff ODEs solvable by general solvers (small weight decay keeps non-stiff); continuous data only (dequantize discrete).

## Code grounding (rtqichen/ffjord)
- CNF (cnf.py): wraps odefunc; forward solves odeint_adjoint(odefunc, (z, logpz), [t0,t1]); reverse flips times; train_T optional. Returns z_t, logpz_t. logp accumulated as the SECOND state, init 0.
- ODEfunc.forward(t, states) (odefunc.py): y=states[0]; sample & FIX self._e (Rademacher or Gaussian) in before_odeint; with grad enabled: dy = diffeq(t,y); divergence = divergence_approx(dy, y, e) = sum( (∂(dy)·e via vjp) * e ); returns [dy, −divergence, ...]. So the log-density dynamics is −Tr̃. divergence_bf = brute-force exact (loops D dims) used for test/2D.
- divergence_approx(f, y, e): e_dzdx = autograd.grad(f, y, e) = e^T ∂f/∂y (vJP); approx_tr = sum(e_dzdx * e). EXACT Hutchinson ε^T ∂f/∂z ε.
- divergence_bf(dx, y): loops i over dims, grad(dx[:,i].sum(), y)[:,i], sums → exact diagonal sum = trace.
- sample_rademacher_like / sample_gaussian_like for ε.
- ODEnet: stack of ConcatLinear/ConcatConv (t concatenated), smooth nonlinearity (softplus default). AutoencoderODEfunc implements the bottleneck trick: two chained vJPs e^T (∂h/∂z)(∂g/∂h) over hidden h.
- residual flag: dy -= y and divergence -= D (an identity-residual reparam).

## Design decisions → why
- Continuous dynamics (ODE) over discrete stack: det → trace (linear, no structured-Jacobian requirement), O(D^3)→O(D^2).
- Hutchinson trace estimator: turns the O(D^2) exact trace into an O(D) UNBIASED stochastic estimate (one vJP + dot), removing the LAST architectural constraint → f free-form.
- Fix ε across the solve: keeps the per-solve ODE deterministic (needed for the adaptive solver to be consistent); pulling E outside integral via Fubini keeps it unbiased.
- Gaussian vs Rademacher ε: both have Cov=I (unbiased); Rademacher has lower variance for a single matrix (entries ±1), but bottleneck trick helps Gaussian more empirically.
- Bottleneck trick (cyclic trace): variance ~ ||A||_F^2, shrink matrix to HxH → lower variance, faster training.
- Adjoint method: O(1) memory backprop through the solver → very large batches.
- Concatenate t (not hypernet): simpler, integrates more stably.
- Low-rank weight update for VAE: full data-dependent weights → too-stiff ODEs; low-rank keeps it integrable.
- Smooth activations (softplus/tanh): ODE existence/uniqueness needs Lipschitz f and derivatives; smoothness keeps solver happy (non-stiff).
- Exact trace at test: report unbiased true log-lik; estimator only at train for speed.
