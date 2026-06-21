The problem is to generate clean data x_0 from an informative endpoint x_T using a pretrained diffusion bridge. The bridge is pinned at both ends, with a forward kernel q(x_t | x_0, x_T) = N(a_t x_T + b_t x_0, c_t^2 I) where the coefficients come from the underlying noise schedule. Existing samplers simulate the reverse SDE or probability-flow ODE with generic high-order discretizers, which is slow: each step needs a full U-Net forward pass, and many small steps are required to keep discretization error under control. Ordinary diffusion models have dedicated fast samplers like DDIM, but those are built for a one-sided data-to-noise process and cannot directly handle a bridge with a second informative endpoint.

The key observation is that the bridge training loss only depends on the per-time marginals q(x_t | x_0, x_T), not on the full joint distribution over trajectories. So we are free to replace the inference joint with any process that preserves those marginals, reusing the same pretrained network. This is the same lever that makes DDIM work for ordinary diffusion, but it must be rebuilt for the bridge kernel because the mean now contains an x_T term and the noise scale is c_t rather than sigma_t.

I propose DBIM, Diffusion Bridge Implicit Models. DBIM defines a family of non-Markovian inference processes over the discretized sampling times, indexed by a per-step injected standard deviation rho_n. The reverse conditional is chosen so that the marginal at each time matches the bridge kernel exactly: q^rho(x_{t_n} | x_0, x_{t_{n+1}}, x_T) = N(a_{t_n} x_T + b_{t_n} x_0 + sqrt(c_{t_n}^2 - rho_n^2) * (x_{t_{n+1}} - a_{t_{n+1}} x_T - b_{t_{n+1}} x_0)/c_{t_{n+1}}, rho_n^2 I). The recycled term is the normalized noise of the later state, and rho_n is fresh injected noise. A backward induction proof confirms that every marginal is preserved: the recycled-direction term averages to zero in expectation, and the variance bookkeeping gives rho_n^2 + (c_{t_n}^2 - rho_n^2) = c_{t_n}^2. The variational objective reduces to a weighted sum of data-prediction errors, which converts to the same denoising bridge score-matching loss the network was trained on, so the pretrained network remains optimal.

In practice rho_n is parameterized by an eta dial in [0, 1]: rho_n = eta * sigma_{t_n} * sqrt(1 - SNR_{t_{n+1}} / SNR_{t_n}). Setting eta = 0 gives a deterministic implicit model with clean large jumps, eta = 1 gives a Markovian DDPM-like stochastic sampler, and intermediate values trade off sharpness and diversity. The first step is special: because the bridge is genuinely stochastic under a fixed x_T, a fully deterministic first step would divide by c_T = 0, which is singular. DBIM therefore uses the Markovian boundary rho_{N-1} = c_{t_{N-1}} at step one, injecting a single shot of booting noise that serves as the latent variable for the whole run. Fresh noise is dropped on the final step to keep the endpoint sharp.

The deterministic eta = 0 update can be rewritten by dividing by c_t, revealing a clean ODE d(x_t/c_t) = x_T d(a_t/c_t) + x_theta(x_t, t, x_T) d(b_t/c_t). This is equivalent to the bridge probability-flow ODE but written in coordinates that integrate easily. Using variation-of-constants with integrating factor c_t/c_s and changing variable to lambda_t = log(b_t/c_t), the linear part is integrated exactly and all approximation error lives in an exponentially weighted integral of the smooth network output. Taylor-expanding the network output in lambda and estimating derivatives by finite differences of past predictions yields second- or third-order solvers with no extra denoiser calls. The first step and optionally the last step drop to first order.

```python
import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from .nn import append_dims
from .random_util import BatchedSeedGenerator


@torch.no_grad()
def sample_dbim(denoiser, diffusion, x, ts, eta=1.0, mask=None, seed=None, **kwargs):
    x_T = x
    path, pred_x0 = [], []
    ones = x.new_ones([x.shape[0]])
    indices = tqdm(range(len(ts) - 1), disable=(dist.get_rank() != 0))

    nfe = 0
    x0_hat = denoiser(x, diffusion.t_max * ones)
    generator = BatchedSeedGenerator(seed)
    noise = generator.randn_like(x0_hat)
    first_noise = noise
    if mask is not None:
        x0_hat = x0_hat * mask + x_T * (1 - mask)
    x = diffusion.bridge_sample(x0_hat, x_T, ts[0] * ones, noise)
    path.append(x.detach().cpu()); pred_x0.append(x0_hat.detach().cpu()); nfe += 1

    for _, i in enumerate(indices):
        s = ts[i]
        t = ts[i + 1]

        x0_hat = denoiser(x, s * ones)
        if mask is not None:
            x0_hat = x0_hat * mask + x_T * (1 - mask)

        a_s, b_s, c_s = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(s * ones)]
        a_t, b_t, c_t = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(t * ones)]
        _, _, rho_s, _ = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_alpha_rho(s * ones)]
        alpha_t, _, rho_t, _ = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_alpha_rho(t * ones)]

        omega_st = eta * (alpha_t * rho_t) * (1 - rho_t**2 / rho_s**2).sqrt()
        tmp_var = (c_t**2 - omega_st**2).sqrt() / c_s
        coeff_xs = tmp_var
        coeff_x0_hat = b_t - tmp_var * b_s
        coeff_xT = a_t - tmp_var * a_s

        noise = generator.randn_like(x0_hat)
        x = (coeff_x0_hat * x0_hat + coeff_xT * x_T + coeff_xs * x
             + (1 if i != len(ts) - 2 else 0) * omega_st * noise)

        path.append(x.detach().cpu()); pred_x0.append(x0_hat.detach().cpu()); nfe += 1

    return x, path, nfe, pred_x0, ts, first_noise
```

The deterministic high-order variant reuses past predictions to estimate lambda-derivatives. It keeps the same booting-noise setup and drops to first order on the first and last steps. With order 2 or 3, each step still costs one denoiser call, but the local truncation error is much smaller, so the same NFE budget reaches lower FID on hard translation tasks. Both variants plug into the existing harness without retraining: the only change is the per-step transition, and the network is used exactly as trained.
