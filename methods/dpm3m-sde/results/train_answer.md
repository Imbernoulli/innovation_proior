Fast guided sampling of a pre-trained diffusion model is the goal: produce high-quality, prompt-aligned images with classifier-free guidance at the scales practitioners actually use, around s = 7.5, in only fifteen to twenty network evaluations. The simplest reliable baseline is DDIM, a first-order deterministic ODE step. It is stable under large guidance because it never estimates high derivatives, but it needs roughly a hundred to two hundred steps to converge, so it is far too slow for interactive use. Existing high-order ODE solvers such as DPM-Solver and DEIS cut unconditional sampling down to ten to twenty steps, but when they are applied to guided sampling they become visibly worse than DDIM at large guidance scales. The reason is that guidance multiplies both the network output and its derivatives along the trajectory; the higher the derivative order, the more it is amplified, which shrinks the convergence radius that the high-order Taylor extrapolation needs. With the step budget fixed and tight, the solver cannot take small enough steps to stay inside that radius, so the extra order backfires rather than helping. A second, independent problem is that large guidance pushes the converged clean image outside the valid data range, and the usual thresholding fix needs a clean-image estimate to clip, which a noise-prediction solver never forms.

The method I propose is DPM-Solver++(3M) SDE. It keeps the exponential-integrator framework but rebuilds it on the data-prediction parameterization, reading the trained network as a clean-image predictor x_theta(x_t, t) = (x_t - sigma_t eps_theta) / alpha_t instead of a noise predictor eps_theta. In this face the probability-flow ODE has the exact solution x_t = (sigma_t / sigma_s) x_s + sigma_t integral_{lambda_s}^{lambda_t} e^lambda x_theta(x_lambda, lambda) d lambda, where lambda is half the log-SNR and equals -log sigma in the alpha = 1 convention. The solver only ever approximates an integral of x_theta, a bounded clean-image estimate, so clipping or dynamic thresholding can be applied directly to the quantity that needs to stay inside the data range. Because the data-prediction formulation contracts the first-derivative error by an extra exponential factor compared with the noise face, the same large guidance scale is far less destabilizing.

DPM-Solver++(3M) SDE is a third-order multistep stochastic solver. It uses one network call per step and reuses the last two clean predictions to estimate the first and second lambda-derivatives of x_theta through Newton divided differences. Reusing past values is crucial at a fixed budget: a singlestep method such as Heun needs two calls per step and therefore can only afford half as many coarse steps, while the multistep method keeps all twenty fine steps. Smaller lambda-steps shrink the local truncation error and keep the extrapolation inside the guidance-narrowed convergence radius. Stochasticity is introduced by solving the reverse SDE rather than the probability-flow ODE. A single parameter eta interpolates the renoising rate via h_eta = h (eta + 1), so eta = 0 recovers the deterministic ODE, eta = 1 recovers the full reverse SDE, and eta slightly above one adds extra Langevin self-correction. The first-order step is a convex combination exp(-h_eta) x + (1 - exp(-h_eta)) denoised plus Gaussian noise with standard deviation sigma_next sqrt(1 - exp(-2 h eta)). The second- and third-order terms add phi2 d1 and -phi3 d2 curvature corrections, with phi2 = (exp(-h_eta) - 1) / h_eta + 1 and phi3 = phi2 / h_eta - 0.5. The Karras power schedule with rho = 7 concentrates steps at low noise where the per-step truncation error is largest, and expm1 keeps the small-argument exponentials numerically stable.

Deterministic high-order extrapolation can drift off the data manifold because discretization error accumulates with no mechanism to remove it. Re-injecting a controlled amount of Gaussian noise at each step and letting the next denoising step remove it acts as a Langevin correction that pulls the trajectory back toward the data manifold and cancels accumulated drift. The order ramps automatically with available history: the first step is first order, the second step uses one past value for a second-order correction, and subsequent steps use two past values for the third-order curvature correction. The final step lands at sigma = 0 and simply returns the clean prediction, because there is no noise left to re-inject. The implementation below follows the standard k-diffusion style. The model is assumed to return the clean-image estimate x_theta for a given noisy input and noise level; a simple reproducible noise sampler is provided so the snippet is self-contained.

```python
import torch
from tqdm.auto import trange


class SimpleNoiseSampler:
    """Reproducible standard-normal noise increments for the SDE path."""
    def __init__(self, x, sigma_min, sigma_max, seed=None):
        self.shape = x.shape
        self.device = x.device
        self.dtype = x.dtype
        self.generator = torch.Generator(device=x.device)
        if seed is not None:
            self.generator.manual_seed(seed)

    def __call__(self, sigma_from, sigma_to):
        return torch.randn(self.shape, generator=self.generator,
                           device=self.device, dtype=self.dtype)


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
    """Karras et al. power schedule: more steps where sigma is small."""
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat([sigmas, sigmas.new_zeros([1])])


@torch.no_grad()
def sample_dpmpp_3m_sde(model, x, sigmas, extra_args=None, callback=None,
                        disable=None, eta=1., s_noise=1., noise_sampler=None):
    """DPM-Solver++(3M) SDE: third-order multistep stochastic data-prediction solver."""
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    if noise_sampler is None:
        noise_sampler = SimpleNoiseSampler(x, sigma_min, sigma_max)
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    denoised_1, denoised_2 = None, None  # last two data predictions
    h_1, h_2 = None, None                # last two lambda-step sizes

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i],
                      'sigma_hat': sigmas[i], 'denoised': denoised})

        if sigmas[i + 1] == 0:
            x = denoised  # final step: return the clean estimate
        else:
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            h_eta = h * (eta + 1)  # eta = 0 ODE, eta = 1 full SDE

            # Constant data-prediction step: convex combination
            x = torch.exp(-h_eta) * x + (-h_eta).expm1().neg() * denoised

            if h_2 is not None:  # third order
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                x = x + phi_2 * d1 - phi_3 * d2
            elif h_1 is not None:  # second order
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                x = x + phi_2 * d

            if eta:  # Langevin renoise
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] \
                    * (-2 * h * eta).expm1().neg().sqrt() * s_noise

        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1

    return x
```
