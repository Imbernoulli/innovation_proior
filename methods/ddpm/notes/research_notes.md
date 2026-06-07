# DDPM research notes (Phase 1)

## Primary paper (read in full incl. appendices)
Ho, Jain, Abbeel 2020, arXiv 2006.11239. Source in src/main.tex (812 lines), bbl/bib read.
Key equations confirmed against src/main.tex:
- Forward: q(x_t|x_{t-1}) = N(sqrt(1-beta_t) x_{t-1}, beta_t I)  [eq:forwardprocess]
- Closed form: q(x_t|x_0) = N(sqrt(abar_t) x_0, (1-abar_t) I), abar_t = prod alpha_s  [eq:q_marginal]
- ELBO: L = E_q[-log p(x_T) - sum log p_theta/q]  [eq:vb_original]
- Reduced: L_T = KL(q(x_T|x_0)||p(x_T)); L_{t-1}=KL(q(x_{t-1}|x_t,x_0)||p_theta); L_0=-log p(x_0|x_1)  [eq:vb]
- Posterior: q(x_{t-1}|x_t,x_0)=N(mu_tilde, betatilde I);
  mu_tilde = (sqrt(abar_{t-1}) beta_t/(1-abar_t)) x_0 + (sqrt(alpha_t)(1-abar_{t-1})/(1-abar_t)) x_t
  betatilde_t = (1-abar_{t-1})/(1-abar_t) beta_t
- L_{t-1} = E[ 1/(2 sigma_t^2) ||mu_tilde - mu_theta||^2 ] + C  [eq:vb_term_orig]
- eps reparam: x_t = sqrt(abar_t)x_0 + sqrt(1-abar_t) eps
- mu_theta = 1/sqrt(alpha_t) (x_t - beta_t/sqrt(1-abar_t) eps_theta)  [eq:mu_func_approx]
- eps loss term: E[ beta_t^2/(2 sigma_t^2 alpha_t (1-abar_t)) ||eps - eps_theta||^2 ]  [eq:vb_term_langevin_eps]
- L_simple = E_{t,x0,eps} ||eps - eps_theta(sqrt(abar_t)x0+sqrt(1-abar_t)eps, t)||^2
- Full appendix-A derivation of eq:vb present (telescoping via Bayes q(x_t|x_{t-1})=q(x_{t-1}|x_t,x_0)q(x_t|x_0)/q(x_{t-1}|x_0)).
- Alternate L form: KL(q(x_T)||p(x_T)) + sum KL(q(x_{t-1}|x_t)||p_theta) + H(x_0).
- sigma_t^2 = beta_t (fixedlarge) or betatilde_t (fixedsmall), both similar.
- T=1000, linear beta 1e-4 -> 0.02. U-Net (PixelCNN++/Wide ResNet backbone), sinusoidal time embed, self-attn at 16x16, group norm. CIFAR10 FID 3.17, IS 9.46.
- Sampling: x_{t-1} = 1/sqrt(alpha_t)(x_t - (1-alpha_t)/sqrt(1-abar_t) eps_theta) + sigma_t z.
- Connection: score s = -eps_theta/sqrt(1-abar_t); sampling resembles annealed Langevin.

## Load-bearing ancestors
1. Sohl-Dickstein 2015 (nonequilibrium thermodynamics diffusion). Forward destroys structure, learn reverse; same functional form when steps small; trained on variational bound. Gap: never shown to produce high-quality images (CIFAR NLL <= 5.40 only). DDPM inherits its math exactly (appendix says "this material is from Sohl-Dickstein").
2. Kingma & Welling 2013 (VAE). ELBO = recon - KL; reparameterization trick z=mu+sigma*eps for low-variance pathwise gradients. DDPM = a many-layer VAE with fixed, parameter-free encoder q and Gaussian decoders; reparam used to get the closed-form x_t(x_0,eps).
3. Song & Ermon 2019 (NCSN). Estimate score grad log p at multiple Gaussian noise scales via denoising score matching; sample by annealed Langevin. Loss sum_sigma lambda(sigma) ||s_theta - target||^2. Gaps DDPM fixes: (a) sampler step sizes/noise set by hand post-hoc; (b) no sqrt(1-beta) rescaling so variance grows; (c) prior not matched (forward doesn't destroy signal); (d) not trained as a latent-variable model by variational inference. DDPM's eps-loss = NCSN-style weighted DSM, but derived from an ELBO.
4. Vincent 2011 (denoising score matching). J_DSM = E ||s_theta(x_tilde) - grad log q_sigma(x_tilde|x)||^2; for q=N(x,sigma^2 I), target = (x - x_tilde)/sigma^2; equals score matching up to const. This is why predicting noise eps == predicting the score (up to scale): grad log q(x_t|x_0) = -(x_t - sqrt(abar)x_0)/(1-abar) = -eps/sqrt(1-abar).
5. Langevin dynamics: x <- x + (delta/2) grad log p(x) + sqrt(delta) z. NCSN's sampler; DDPM's reverse step has the same shape with eps_theta as learned gradient.

## Explainer
Lilian Weng blog (lilianweng.github.io/posts/2021-07-11-diffusion-models) confirms forward closed form, ELBO split, posterior mean/var, eps reparam, L_simple, and score connection s = -eps_theta/sqrt(1-abar_t).

## Canonical code (code/)
- hojonathanho_official/diffusion_utils_2.py (official TF GaussianDiffusion2): q_sample, q_posterior_mean_variance, _predict_xstart_from_eps, p_mean_variance (eps/xstart/xprev), p_sample, training_losses (kl or mse), _vb_terms_bpd. Buffers: alphas_cumprod, posterior_variance/mean_coef1/coef2, sqrt_recip(m1)_alphas_cumprod. fixedlarge=betas, fixedsmall=posterior_variance.
- lucidrains_pytorch/denoising_diffusion_pytorch.py (PyTorch GaussianDiffusion): same buffers/structure, predict_start_from_noise, q_posterior, p_mean_variance, p_sample, q_sample, p_losses (pred_noise default), loss_weight (min-SNR). Cleaner for the final code snippet.

## Injection check
No prompt-injection encountered. One ref host (umontreal) was behind Anubis; refetched Vincent 2011 from gwern mirror, confirmed J_DSM and target = (x - x_tilde)/sigma^2.
