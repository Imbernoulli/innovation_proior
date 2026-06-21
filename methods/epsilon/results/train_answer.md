The problem is to make a fixed-forward Gaussian diffusion model generate sharp, high-fidelity natural images. The forward process is fixed: it gradually scales the signal down by sqrt(1 - beta_t) and injects Gaussian variance beta_t, so that after T small steps any data distribution becomes nearly N(0, I). What remains to design is the reverse process: what the network should output at each noisy sample, and how the per-timestep loss should be weighted. The obvious choices have already been tried and failed. Predicting the reverse mean directly forces the network to reproduce the input x_t plus a tiny t-dependent correction, wasting capacity on the trivial part of the target. Predicting the clean image x_0 directly is even worse under a flat MSE: at small t the target is almost the identity, while at large t the network must hallucinate a full image from near-pure noise, so the loss is dominated by the high-noise regime where x_0 is least recoverable. What is needed is a target whose difficulty and scale are well-behaved at every noise level.

The right target is the added noise itself. From the reparameterized forward step x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) epsilon, the only unknown given x_t is epsilon. If we parameterize the reverse mean as mu_theta(x_t, t) = (1/sqrt(alpha_t)) (x_t - (beta_t / sqrt(1 - alpha_bar_t)) epsilon_theta(x_t, t)), then substituting into the variational bound collapses the per-step KL to a scaled squared error between the true noise and the predicted noise. Predicting epsilon gives a regression target of constant N(0, I) scale across all timesteps, which is far better conditioned than predicting mu_tilde or x_0. It also reveals that the variational diffusion and denoising score matching are the same construction: the score of the noised data density is -epsilon / sqrt(1 - alpha_bar_t), so epsilon_theta is a learned score, and the reverse step is one annealed Langevin move.

The method is called epsilon prediction, or the DDPM noise-prediction parameterization. The network is trained to predict the noise epsilon that was added to produce the current noisy sample x_t. Training uses the simplified objective L_simple, which drops the per-timestep weight from the true variational bound and uses an unweighted MSE across timesteps. That weighting is important: the bound's true weight peaks at small t, where the network is asked to denoise almost-clean images, and is smallest at large t, where the image is mostly noise and the network must recover global structure. By using a uniform weight, L_simple redirects gradient toward the hard high-noise terms that actually shape image content, matching the equal-magnitude-per-level weighting that score matching had already found to work well empirically. At sampling time, the predicted noise is inverted to a clean-image estimate via x0_hat = (x_t - sqrt(1 - alpha_bar_t) epsilon_theta) / sqrt(alpha_bar_t), and the reverse step draws from a Gaussian whose mean is the corresponding posterior mean.

The standard settings close the chain. Use T = 1000 diffusion steps with beta_t linear from 1e-4 to 0.02, small enough relative to data scaled to [-1, 1] that each reverse step remains approximately Gaussian and x_T is essentially standard normal. The reverse variance is fixed to a principled constant, either beta_t or beta_tilde_t, rather than learned. The data is linearly scaled to [-1, 1] so the network inputs share the same scale as the prior. The whole method reduces to a matched training-target and inversion pair, plus the usual noise-prediction training loop.

```python
import torch


def get_schedule(betas):
    """Fixed forward-diffusion tensors (no learnable parameters)."""
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones_like(alphas_cumprod[:1]), alphas_cumprod[:-1]])
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    return {
        "betas": betas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alpha": alphas_cumprod.sqrt(),
        "sqrt_one_minus_alpha": (1.0 - alphas_cumprod).sqrt(),
        "posterior_variance": posterior_variance,
        "posterior_mean_coef1": betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod),
        "posterior_mean_coef2": (1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod),
    }


def q_sample(x_0, noise, t, schedule):
    """Forward process: x_t = sqrt(abar_t) x_0 + sqrt(1-abar_t) eps."""
    sa = schedule["sqrt_alpha"][t].view(-1, 1, 1, 1)
    soma = schedule["sqrt_one_minus_alpha"][t].view(-1, 1, 1, 1)
    return sa * x_0 + soma * noise


def compute_training_target(x_0, noise, timesteps, schedule):
    # Epsilon prediction: model learns to predict the added noise.
    # Target is N(0, I) at every timestep -> well-conditioned, unweighted MSE.
    return noise


def predict_x0(model_output, x_t, timesteps, schedule):
    # Recover x_0 from the predicted noise:
    # x_t = sqrt(abar) x_0 + sqrt(1-abar) eps  =>  x_0 = (x_t - sqrt(1-abar) eps) / sqrt(abar)
    sqrt_alpha = schedule["sqrt_alpha"][timesteps].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha = schedule["sqrt_one_minus_alpha"][timesteps].view(-1, 1, 1, 1)
    return (x_t - sqrt_one_minus_alpha * model_output) / sqrt_alpha.clamp(min=1e-8)


def q_posterior_mean(x_start, x_t, t, schedule):
    """Mean of q(x_{t-1} | x_t, x_0)."""
    coef1 = schedule["posterior_mean_coef1"][t].view(-1, 1, 1, 1)
    coef2 = schedule["posterior_mean_coef2"][t].view(-1, 1, 1, 1)
    return coef1 * x_start + coef2 * x_t


def train_step(model, x_0, schedule, T):
    """Training: L_simple = E || eps - eps_theta(x_t, t) ||^2."""
    B = x_0.shape[0]
    t = torch.randint(0, T, (B,), device=x_0.device)
    noise = torch.randn_like(x_0)
    x_t = q_sample(x_0, noise, t, schedule)
    target = compute_training_target(x_0, noise, t, schedule)
    pred = model(x_t, t)
    return ((pred - target) ** 2).mean()


@torch.no_grad()
def p_sample_step(model, x_t, t, schedule, sigma_t, add_noise=True):
    """One reverse step x_t -> x_{t-1}."""
    eps_theta = model(x_t, t)
    x0_hat = predict_x0(eps_theta, x_t, t, schedule).clamp(-1.0, 1.0)
    mean = q_posterior_mean(x0_hat, x_t, t, schedule)
    z = torch.randn_like(x_t) if add_noise else torch.zeros_like(x_t)
    nonzero = (t != 0).float().view(-1, *([1] * (x_t.ndim - 1)))
    return mean + nonzero * sigma_t * z
```
