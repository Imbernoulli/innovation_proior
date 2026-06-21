Diffusion models are built around a single transport problem: move data to Gaussian noise and back again. That is perfect for unconditional generation, but most practical image-to-image tasks are not noise-to-data. They are data-to-data: a sketch must become a photo, a blurred image must become a sharp one, a daytime scene must become nighttime. In each case the input is itself a structured sample from a source distribution, and the goal is to sample from the conditional target distribution given that source. Existing approaches all bend the standard diffusion recipe in awkward ways. Conditioning the denoiser on the source is an ad-hoc plug-in. SDEdit injects an arbitrary amount of noise into the source and hopes an unconditional model will denoise toward the right target, which forces a painful trade-off between preserving structure and generating diversity. DDIB trains two separate unconditional models and stitches them through a shared latent, but it is one-directional and loses cycle consistency. Flow matching and rectified flow do connect two arbitrary distributions, yet their deterministic maps tend to collapse to the blurry conditional mean, and they stand outside the diffusion SDE formalism, so they cannot reuse the pretrained preconditioning, noise schedules, and fast samplers that made diffusion work. What is needed is a process whose endpoints are the actual paired data distributions, with a closed-form training objective and a sampler that can stay sharp and diverse under a tight compute budget.

The right tool is Denoising Diffusion Bridge Models, or DDBM. The starting point is Doob's h-transform: given a base diffusion with a Gaussian transition kernel, one can add a drift proportional to the gradient of the backward transition probability of hitting a fixed endpoint. That pins the process to a target point. Pinning both endpoints gives a diffusion bridge. DDBM turns this bridge into a learnable generative model between paired data distributions. For a base diffusion with kernel p(x_t | x_0) = N(α_t x_0, σ_t² I), the doubly-pinned bridge q(x_t | x_0, x_T) is Gaussian by Bayes' rule. Its mean linearly interpolates the scaled endpoints and its variance is zero at both ends, so it genuinely ties the trajectory to the source and target. Training is denoising bridge score matching: sample a pair (x_0, x_T), sample x_t from the closed-form bridge, and regress a network onto the closed-form conditional score. The L₂ minimizer of this regression is exactly the marginal bridge score, so no path simulation is needed. For sampling, the bridge's reverse dynamics split cleanly into a reverse SDE and a probability-flow ODE. The reverse SDE drift is f − g²(s − h), and the ODE drift is f − g²(½ s − h), where s is the learned bridge score and h is the analytic h-transform; only the learned score is halved in the ODE, because the half comes from the continuity-equation conversion on q, not from the bridge's endpoint-pinning drift. To avoid the blurry conditional mean that a purely deterministic ODE from a fixed source would produce, the sampler injects noise via a short stochastic Euler "churn" step before each accurate deterministic Heun step, on EDM's ρ = 7 time grid. The result is a principled translation model that contains ordinary diffusion and flow matching as special cases and reuses the full diffusion toolbox.

```python
import torch
from tqdm.auto import tqdm
import torch.distributed as dist

from .nn import append_dims

def get_d(denoiser, noise_schedule, x, x_T, t, stochastic):
    ones = x.new_ones([x.shape[0]])
    f_t, g2_t = [append_dims(item, x.ndim) for item in noise_schedule.get_f_g2(t * ones)]
    alpha_t, alpha_bar_t, _, rho_bar_t = [
        append_dims(item, x.ndim) for item in noise_schedule.get_alpha_rho(t * ones)
    ]
    a_t, b_t, c_t = [append_dims(item, x.ndim) for item in noise_schedule.get_abc(t * ones)]
    denoised = denoiser(x, t * ones)
    grad_logq = -(x - (a_t * x_T + b_t * denoised)) / c_t**2
    grad_logpxTlxt = -(x - alpha_bar_t * x_T) / (alpha_t**2 * rho_bar_t**2)
    d = f_t * x - g2_t * ((0.5 if not stochastic else 1) * grad_logq - grad_logpxTlxt)
    return d, g2_t, denoised


def ddbm_simulate(denoiser, noise_schedule, x, x_T, t_cur, t_next, stochastic, second_order=False):
    dt = t_next - t_cur
    if isinstance(noise_schedule, I2SBNoiseSchedule):
        dt = dt * (noise_schedule.n_timestep - 1)
    d, g2_t, pred_x0 = get_d(denoiser, noise_schedule, x, x_T, t_cur, stochastic)
    x_new = x + d * dt + (0 if not stochastic else 1) * torch.randn_like(x) * (dt.abs() ** 0.5) * g2_t.sqrt()
    if second_order:
        d_2, _, pred_x0 = get_d(denoiser, noise_schedule, x_new, x_T, t_next, stochastic)
        d_prime = (d + d_2) / 2
        x_new = x + d_prime * dt + (0 if not stochastic else 1) * torch.randn_like(x) * (dt.abs() ** 0.5) * g2_t.sqrt()
    return x_new, pred_x0


@torch.no_grad()
def sample_heun(denoiser, diffusion, x, ts, churn_step_ratio=0.0, **kwargs):
    x_T = x
    path, pred_x0, nfe = [], [], 0
    indices = tqdm(range(len(ts) - 1), disable=(dist.get_rank() != 0))
    assert churn_step_ratio < 1

    for _, i in enumerate(indices):
        if churn_step_ratio > 0:
            t_hat = (ts[i + 1] - ts[i]) * churn_step_ratio + ts[i]
            x, _p = ddbm_simulate(denoiser, diffusion.noise_schedule, x, x_T, ts[i], t_hat, stochastic=True)
            nfe += 1
            path.append(x.detach().cpu()); pred_x0.append(_p.detach().cpu())
        else:
            t_hat = ts[i]

        if ts[i + 1] == 0:
            x, _p = ddbm_simulate(denoiser, diffusion.noise_schedule, x, x_T, t_hat, ts[i + 1], stochastic=False)
            nfe += 1
        else:
            x, _p = ddbm_simulate(denoiser, diffusion.noise_schedule, x, x_T, t_hat, ts[i + 1],
                                  stochastic=False, second_order=True)
            nfe += 2
        path.append(x.detach().cpu()); pred_x0.append(_p.detach().cpu())
    return x, path, nfe, pred_x0, ts, None
```
