# IWAE synthesis notes

## Pain point
VAE maximizes ELBO L = E_{q(z|x)}[log p(x,z)/q(z|x)] = log p(x) - KL(q||p(z|x)). The bound is loose
exactly when the recognition network q can't match the true posterior (factorial Gaussian assumption,
amortized nonlinear regression). The ELBO penalizes q-samples that fail to explain x harshly: even if
q puts only e.g. 20% of mass in high-posterior region, ELBO punishes the rest. This forces the model
to learn posteriors that ARE factorial/predictable, limiting expressiveness -> overly simple reps,
latent dimensions "die out".

## Central question
Importance sampling estimates p(x) = E_q[w], w = p(x,z)/q(z|x), with multiple samples. Can a k-sample
estimator be turned into a tighter TRAINING bound that is still a valid lower bound on log p(x)?

## The bound (Eqn 8)
L_k = E_{z_1..z_k ~ q}[ log (1/k) Σ_{i=1}^k w_i ],  w_i = p(x,z_i)/q(z_i|x).
k=1 recovers the VAE ELBO.

## Three properties (Appendix A) — proofs to live out inline

1. L_k <= log p(x). Jensen on concave log:
   E[log (1/k Σ w_i)] <= log E[1/k Σ w_i] = log p(x), since E_q[w_i] = ∫ q (p(x,z)/q) dz = p(x).

2. L_{k} >= L_{m} for k >= m (monotone non-decreasing). Key combinatorial averaging argument:
   Let I = {i_1..i_m} uniform random m-subset of {1..k}. Observation:
   E_I[ (a_{i_1}+..+a_{i_m})/m ] = (a_1+..+a_k)/k for any numbers a_i (each index appears with prob m/k).
   So (1/k) Σ_{i=1}^k w_i = E_I[ (1/m) Σ_{j=1}^m w_{i_j} ].
   Then
   L_k = E_{z_1..z_k}[ log E_I[ (1/m)Σ w_{i_j} ] ]
       >= E_{z_1..z_k}[ E_I[ log (1/m)Σ w_{i_j} ] ]    (Jensen, log concave, over I)
       = E_{z_1..z_m}[ log (1/m) Σ_{j=1}^m w_j ] = L_m.   (each m-subset i.i.d. q, exchangeable)
   Setting m=k, k+1=k gives L_{k+1} >= L_k.

3. L_k -> log p(x) as k->inf, if w = p(x,z)/q(z|x) bounded.
   M_k = (1/k)Σ w_i. By SLLN M_k -> E_q[w] = p(x) a.s. log continuous => log M_k -> log p(x) a.s.
   Boundedness of w => bounded convergence/dominated convergence => E[log M_k] -> log p(x).
   So L_k -> log p(x).

## Variance argument (Appendix B) — log of average, not the average
Importance sampling estimator of p(x) can have huge/infinite variance. But we use log of the average.
For positive unbiased Ẑ of Z, log Ẑ underestimates in expectation (Jensen), bias δ = log Z - E[log Ẑ].
Markov on Ẑ/Z: Pr(Ẑ > Z e^b) = Pr(Ẑ/Z > e^b) <= E[Ẑ/Z]/e^b = e^{-b}, so Pr(log Ẑ > log Z + b) <= e^{-b}.
MAD = E|log Ẑ - E log Ẑ| = 2 E[(log Ẑ - E log Ẑ)_+]
    <= 2 E[(log Ẑ - log Z)_+] + 2(log Z - E log Ẑ)
    = 2 ∫_0^inf Pr(log Ẑ - log Z > t) dt + 2δ
    <= 2 ∫_0^inf e^{-t} dt + 2δ = 2 + 2δ.
So the log-domain estimator has MAD bounded by 2 + 2δ regardless. The log tames the heavy tail.

## Gradient via reparameterization + self-normalized weights (Eqn 13)
Reparameterize z_i = h(eps_i, x, theta), eps_i ~ N(0,I). Push grad inside expectation:
∇_θ L_k = E_{eps}[ ∇_θ log (1/k) Σ w_i ].
∇_θ log (1/k Σ w_i) = (Σ ∇_θ w_i)/(Σ w_i) = Σ_i (w_i/Σ_j w_j) ∇_θ log w_i = Σ_i w̃_i ∇_θ log w_i,
where w̃_i = w_i / Σ_j w_j are SELF-NORMALIZED weights, and we used ∇w_i = w_i ∇ log w_i.
So  ∇_θ L_k = E_eps[ Σ_i w̃_i ∇_θ log w_i ].
k=1: w̃_1 = 1 => recovers VAE update.
∇ log w_i = ∇ log p(x, z_i) - ∇ log q(z_i|x): first term = autoencoder reconstruction signal pulling
generative + recognition toward better explanation; second = entropy term spreading q out.
Weighted by importance weights => "importance weighted autoencoder".
Note: no separate-KL trick (VAE's lower-variance trick) available for k>1.

## Cost trick
k forward+backward passes (linear in k). Can reduce: do k forward passes (cheap), sample ONE index
proportional to w̃_i, do one backward. ~3x fewer add-multiplies for large k, higher variance.

## Code grounding
Original: yburda/iwae (Theano). Key: log_mean_exp for the bound estimate; gradient via
theano.clone replacing a dummy vector with ws_normalized (detached) — implements Σ w̃_i ∇ log w_i.
Clean PyTorch: xqding/Importance_Weighted_Autoencoders. train_loss:
  log_weight = log p(x,z) - log q(z|x)  [per sample]
  log_weight -= max; weight = exp; weight /= sum; weight = detach(weight)
  loss = -mean( sum_k weight * log_weight )   # surrogate: d/dθ gives Σ w̃ ∇ log w
test_loss: -mean(log(mean(exp(log_weight)))) = -L_k estimate (logmeanexp).
Encoder: BasicBlock (Linear200-tanh-Linear200-tanh -> mu, logsigma; sigma=exp). Decoder Bernoulli sigmoid.
Reparam: h = mu + sigma*eps. Samples handled by expanding input to (k, batch, dim).

## Design choices -> why
- log (1/k Σ w_i) not (1/k Σ log w_i): the latter is just k independent ELBOs, averages to L_1 (no gain).
  The log-of-mean is what makes it tighter and a valid bound that improves with k.
- self-normalized weights detached in gradient: they are the coefficients; their own θ-dependence is
  already accounted (the surrogate Σ w̃ log w differentiated with w̃ fixed reproduces the true grad).
- exp nonlinearity on predicted variance: enforce positivity of sigma.
- tanh deterministic layers; Gaussian diagonal stochastic layers; Bernoulli visible for binary images.
- max-subtraction before exp: numerical stability (logsumexp trick).
