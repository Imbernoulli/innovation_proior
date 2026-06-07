# Research notes — VAE ancestors, field state, explainers, code

## Load-bearing ancestors (from the paper's own bibliography + background)

### EM algorithm (Dempster 1977; via Roweis 1998 for PCA/SPCA, cited [Row98])
- Decomposition: log p(x) = ELBO(q) + KL(q(z) || p(z|x)). Since KL >= 0, ELBO is a lower bound.
- E-step: set q(z) = p(z|x; theta_old) -> KL = 0, bound becomes tight.
- M-step: theta_new = argmax_theta E_{q}[log p(x,z|theta)] (complete-data log-lik, log inside expectation moved outside).
- Breaks down exactly when the posterior p(z|x) is intractable: cannot do the E-step in closed form. For a neural-network likelihood p(x|z), the posterior has no closed form -> EM dead.
- Roweis 1998 [Row98]: PCA = ML solution of a special linear-Gaussian model p(z)=N(0,I), p(x|z)=N(Wz, eps I) as eps->0. Shows the lineage from linear autoencoders to generative latent-variable models — but only linear/Gaussian, tractable posterior.

### Classical (mean-field) variational inference / Stochastic VI (Hoffman 2013 [HBWP13])
- Replace intractable posterior with q from a tractable family; maximize ELBO instead of doing exact E-step.
- Mean-field: q(z) = prod_j q_j(z_j). Coordinate-ascent (CAVI) updates: log q_j* = E_{-j}[log p(x,z)] + const.
- LIMITATION 1: CAVI requires analytic expectations E_q[log p(x,z)] — only tractable for conditionally-conjugate exponential-family models. A nonlinear NN likelihood blows this up.
- LIMITATION 2: factorial assumption can't capture posterior correlations.
- LIMITATION 3 (scale): per-datapoint local variational params must be optimized to convergence before each global update -> costly for big data. SVI (Hoffman) fixes the scale issue with stochastic natural-gradient on global params, but still needs conjugacy / analytic local updates.

### Score-function / REINFORCE gradient estimator + its variance (Blei 2012 [BJP12], Ranganath BBVI 2014 [RGB13])
- naive MC gradient: grad_phi E_{q_phi}[f(z)] = E_{q_phi}[ f(z) grad_phi log q_phi(z) ].
  (log-derivative / score-function identity, = REINFORCE.)
- Unbiased and "black-box" (only needs to evaluate f, not differentiate it), works for discrete z.
- LIMITATION: very high variance — f(z) multiplies the score, which is noisy; variance does not shrink with smooth f. Needs control variates / Rao-Blackwellization / baselines (Blei 2012, Ranganath 2014) just to be usable. Paper explicitly says "impractical for our purposes."

### Wake-sleep / Helmholtz machine (Hinton, Dayan, Frey, Neal 1995 [HDFN95])
- The only prior online method for the same general class of continuous-latent directed models.
- Uses a separate recognition network q(z|x) to approximate posterior (the "encoder" idea predates VAE here).
- Wake phase: sample z ~ q(z|x), update generative weights to raise log p(x,z).
- Sleep phase: "dream" x,z ~ p, update recognition weights to predict z from x (fits q to p's posterior by reversed-KL on fantasy data).
- LIMITATION: optimizes TWO different objectives that do NOT jointly correspond to a single bound on log p(x); sleep phase minimizes KL(p||q) on dreamed data, not KL(q||p) on real data -> no guarantee of improving marginal likelihood. VAE's pitch: one objective (the ELBO) for both networks.

### Salimans & Knowles 2013 [SK13]
- Independently used a reparameterization-like idea inside a stochastic-linear-regression VI for exponential-family approximations. Closest prior use of the trick, but tied to exp-family natural params, not a general NN recognition model.

### Rezende, Mohamed, Wierstra 2014 [RMW14] (concurrent, independent)
- Same connection (stochastic backprop / DLGM) developed independently — additional perspective, not a predecessor.

### Autoencoder lineage (context, not derivation core)
- Denoising/contractive/sparse AEs (Vincent 2010 [VLL+10], Bengio 2013 [BCV13]): reconstruction + ad-hoc regularizer; infomax view -> reconstruction lower-bounds mutual info I(X;Z). VAE's KL term is a *principled* regularizer with no nuisance hyperparameter.
- PSD (Kavukcuoglu 2008 [KRL08]): predictive encoder for sparse coding — inspiration for an amortized encoder.

## State of the field (~2013, pre-method)
- Deep directed generative models with continuous latents + NN likelihoods were attractive but "untrainable": exact posterior intractable (no EM), mean-field needs conjugacy (no NN), score-function gradients too noisy, MCMC-EM too slow per-datapoint for big data, wake-sleep has no single coherent objective. SGD + backprop + GPUs were mature; Adagrad (Duchi 2010) available. The missing piece: a low-variance, differentiable, minibatch-friendly estimator of the ELBO that works through a neural-net q.

## Explainers consulted
- Gregory Gundersen, "The Reparameterization Trick" — clean statement that grad of expectation (params in the measure) is not an expectation; reparameterize z=g(eps,phi) -> grad moves inside -> expectation of a grad, low variance, differentiable.
- Krasser, "Latent variable models part 1" — EM via ELBO/KL decomposition; why intractable posterior kills the E-step.
- Jaan Altosaar tutorial, Doersch tutorial (arXiv 1606.05908), Kingma & Welling F&T monograph (1906.02691) — standard derivations of ELBO + reparam + Gaussian KL.
- Blei "VI: A Review for Statisticians" (1601.00670) — mean-field/CAVI and the analytic-expectation requirement.

## Canonical code
- pytorch/examples vae/main.py (fetched into vae/code/): MLP encoder 784->400->(mu20, logvar20), reparameterize std=exp(0.5*logvar); eps~N(0,1); z=mu+eps*std; decoder 20->400->784 sigmoid; loss = BCE(recon,x) (sum) + KLD where KLD = -0.5*sum(1+logvar-mu^2-exp(logvar)). Adam lr=1e-3, M=128. README notes ReLU+Adam vs paper's sigmoid+Adagrad.
