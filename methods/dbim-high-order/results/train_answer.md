The problem is to sample a pretrained diffusion bridge for image-to-image translation or restoration under a very tight budget of denoiser calls. A diffusion bridge pins the trajectory at both ends, so ordinary diffusion fast samplers do not apply: they assume a Gaussian prior and have no fixed endpoint. The bridge ships with a generic hybrid Heun solver that treats the reverse dynamics as a black box, which costs well over a hundred network evaluations because it pays discretization error on parts of the dynamics that are analytically integrable. Even the first-order bridge DDIM analog helps, but it still approximates the predictor as constant over each large step, which limits accuracy on hard translation tasks.

The right move is to exploit the bridge's semi-linear structure. The deterministic limit of the bridge non-Markovian family is an Euler discretization of a clean ordinary differential equation for the normalized state x_t / c_t, and that ODE is exactly the bridge probability-flow ODE. The linear part can be cancelled analytically by variation of constants, leaving only an exponentially-weighted integral of the data predictor. Taylor-expanding the predictor in the bridge log-SNR variable and estimating derivatives from past predictor outputs gives a multistep high-order solver that still uses only one new denoiser call per step.

The method is the DBIM high-order solver, short for Diffusion Bridge Implicit Models high-order solver. It builds on the same marginal-preserving non-Markovian bridge family as DBIM, so the trained predictor is reused unchanged. The first step is necessarily stochastic because the bridge is pinned at the endpoint with c_T = 0, which would make a deterministic first step singular. After that booting sample the solver follows the deterministic rho = 0 ODE. Second- and third-order updates are obtained by integrating the Taylor expansion of the predictor against e^lambda, where lambda_t = log(b_t / c_t) is the bridge log-SNR. The derivatives with respect to lambda are estimated by Adams-Bashforth finite differences of previously computed predictor outputs, so the budget of N calls buys N steps rather than N/k steps as in a single-step high-order method. The first loop transition and the final step fall back to first order because there is no reliable history at the start and because endpoint sharpness is better protected by a clean low-order finish.

Here is a concise implementation that fills the bridge sampler slot. It assumes the existing diffusion object exposes get_abc for the bridge coefficients a_t, b_t, c_t and bridge_sample for the booting step, and that denoiser is the metered data predictor x_theta(x_t, t, x_T). The schedule ts decreases from t_max to t_min.

```python
import torch
import torch.distributed as dist
from tqdm.auto import tqdm
from ddbm.random_util import BatchedSeedGenerator


def append_dims(x, target_dims):
    return x[(...,) + (None,) * (target_dims - x.ndim)]


@torch.no_grad()
def sample_dbim_high_order(
    denoiser,
    diffusion,
    x,
    ts,
    mask=None,
    order=2,
    lower_order_final=True,
    seed=None,
    **kwargs,
):
    if order not in [2, 3]:
        raise NotImplementedError("Only order 2 or 3 supported")

    x_T = x
    path, pred_x0, nfe = [], [], 0
    ones = x.new_ones([x.shape[0]])
    indices = tqdm(range(len(ts) - 1), disable=(dist.get_rank() != 0))

    # Booting step: the bridge is pinned at T, so c_T = 0 and a deterministic
    # first step is singular. Use the Markovian boundary rho = c at the first
    # non-terminal node; the injected Gaussian is the diversity latent.
    x0_hat = denoiser(x, diffusion.t_max * ones)
    generator = BatchedSeedGenerator(seed)
    noise = generator.randn_like(x0_hat)
    first_noise = noise
    if mask is not None:
        x0_hat = x0_hat * mask + x_T * (1 - mask)
    x = diffusion.bridge_sample(x0_hat, x_T, ts[0] * ones, noise)
    path.append(x.detach().cpu())
    pred_x0.append(x0_hat.detach().cpu())
    nfe += 1

    # Buffers for multistep derivative estimates.
    u = diffusion.t_max
    if u == 1.0:
        u -= 5e-5
    u = [u for _ in range(order - 1)]
    xu_hat = [x0_hat.detach().clone() for _ in range(order - 1)]

    for _, i in enumerate(indices):
        s = ts[i]        # current node, larger time
        t = ts[i + 1]    # target node, smaller time

        # First-order: first loop transition after boot, or final step.
        if (lower_order_final and i + 1 == len(ts) - 1) or (i == 0):
            a_s, b_s, c_s = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(s * ones)]
            a_t, b_t, c_t = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(t * ones)]
            tmp = c_t / c_s
            x0_hat = denoiser(x, s * ones)
            if mask is not None:
                x0_hat = x0_hat * mask + x_T * (1 - mask)
            nfe += 1
            x = tmp * x + (b_t - tmp * b_s) * x0_hat + (a_t - tmp * a_s) * x_T

        # Second-order multistep.
        elif order == 2 or i == 1:
            a_u, b_u, c_u = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(u[-1] * ones)]
            a_s, b_s, c_s = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(s * ones)]
            a_t, b_t, c_t = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(t * ones)]
            lambda_u = torch.log(b_u / c_u)
            lambda_s = torch.log(b_s / c_s)
            lambda_t = torch.log(b_t / c_t)
            x0_hat = denoiser(x, s * ones)
            if mask is not None:
                x0_hat = x0_hat * mask + x_T * (1 - mask)
            nfe += 1
            h = lambda_t - lambda_s
            h2 = lambda_s - lambda_u
            integral = torch.exp(lambda_t) * (
                (1 - torch.exp(-h)) * x0_hat
                + (torch.exp(-h) + h - 1) * (x0_hat - xu_hat[-1]) / h2
            )
            x = x * (c_t / c_s) + x_T * (a_t - a_s * (c_t / c_s)) + c_t * integral

        # Third-order multistep.
        elif order == 3:
            a_u1, b_u1, c_u1 = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(u[-1] * ones)]
            a_u2, b_u2, c_u2 = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(u[-2] * ones)]
            a_s, b_s, c_s = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(s * ones)]
            a_t, b_t, c_t = [append_dims(v, x0_hat.ndim) for v in diffusion.noise_schedule.get_abc(t * ones)]
            lambda_u2, lambda_u1, lambda_s, lambda_t = (
                torch.log(b_u2 / c_u2),
                torch.log(b_u1 / c_u1),
                torch.log(b_s / c_s),
                torch.log(b_t / c_t),
            )
            x0_hat = denoiser(x, s * ones)
            if mask is not None:
                x0_hat = x0_hat * mask + x_T * (1 - mask)
            nfe += 1
            h = lambda_t - lambda_s
            h1 = lambda_s - lambda_u1
            h2 = lambda_u1 - lambda_u2
            D1 = (x0_hat - xu_hat[-1]) / h1
            D2 = (xu_hat[-1] - xu_hat[-2]) / h2
            dx0_hat = (D1 * (2 * h1 + h2) - D2 * h1) / (h1 + h2)
            d2x0_hat = 2 * (D1 - D2) / (h1 + h2)
            integral = torch.exp(lambda_t) * (
                (1 - torch.exp(-h)) * x0_hat
                + (torch.exp(-h) + h - 1) * dx0_hat
                + (h ** 2 / 2 - h + 1 - torch.exp(-h)) * d2x0_hat
            )
            x = x * (c_t / c_s) + x_T * (a_t - a_s * (c_t / c_s)) + c_t * integral

        u.append(s)
        u.pop(0)
        xu_hat.append(x0_hat)
        xu_hat.pop(0)
        path.append(x.detach().cpu())
        pred_x0.append(x0_hat.detach().cpu())

    return x, path, nfe, pred_x0, ts, first_noise
```
