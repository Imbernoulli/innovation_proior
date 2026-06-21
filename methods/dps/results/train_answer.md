We need to solve a noisy inverse problem using only a pre-trained unconditional diffusion model as a prior. The problem is to recover a clean signal x_0 from a measurement y = A(x_0) + n, where A is a known forward operator and n is measurement noise. Because A is many-to-one, the right goal is to sample from the posterior p(x_0 | y), not to produce a single deterministic inverse. Existing approaches each fail on at least one of the requirements. Projection-based solvers such as score-SDE/ILVR enforce A x = y exactly, which overfits measurement noise and amplifies it through A^T for ill-conditioned operators; they also require a linear, easily projectable A. Spectral methods such as DDRM/SNIPS handle noise, but only when an explicit, cheap SVD of A is available, which rules out general nonlinear operators. Methods like robust-CSGM use the clean-model likelihood score at t = 0 and patch it with a hand-tuned annealing schedule, so they use the wrong time-level term and remain limited to linear A. None of them gives a single sampler that is training-free, noise-robust, and applicable to arbitrary differentiable forward operators.

The method I propose is Diffusion Posterior Sampling, abbreviated DPS. It builds on the Bayesian decomposition of the conditional score. At every noise level t, Bayes' rule gives grad_{x_t} log p_t(x_t | y) = grad_{x_t} log p_t(x_t) + grad_{x_t} log p_t(y | x_t). The first term is the pre-trained unconditional score. The second term, the likelihood score at the noised iterate x_t, is intractable because y depends on x_t only through the unknown clean x_0. DPS makes it tractable by first marginalizing x_0 exactly: p_t(y | x_t) = E_{x_0 ~ p(x_0 | x_t)} [ p(y | x_0) ]. The remaining difficulty is the expectation over the denoising posterior p(x_0 | x_t). Although that full posterior is intractable, its mean is available in closed form through Tweedie's formula: x_0_hat := E[x_0 | x_t] = (1 / sqrt(alpha_bar(t))) ( x_t + (1 - alpha_bar(t)) s_theta(x_t, t) ), where s_theta is the pre-trained score network. DPS therefore approximates the expectation by evaluating the clean likelihood at the posterior mean, p_t(y | x_t) ≈ p(y | x_0_hat). This is a Jensen-gap approximation, and the gap can be bounded by a Lipschitz constant of the measurement density times the operator Jacobian norm times the first central moment of the denoising posterior. For Gaussian measurement noise the bound tightens as the noise level grows, which is the opposite of projection methods and explains why DPS behaves well in the noisy regime.

Differentiating the surrogate likelihood with respect to x_t gives the guidance term. For Gaussian noise it is the gradient of the squared residual || y - A(x_0_hat) ||^2, backpropagated through x_0_hat(x_t) and through A. Because the gradient is obtained by ordinary automatic differentiation, A can be nonlinear or even a neural forward model; no SVD, pseudo-inverse, or projection subspace is required. For Poisson or shot noise, the same structure holds with a weighted least-squares surrogate where the per-bin variance is frozen to the measured value. A literal fixed noise-variance coefficient would not work well across the whole reverse trajectory, because the squared residual changes by orders of magnitude as the sample becomes clean. DPS therefore residual-normalizes the guidance step: the squared-loss gradient is scaled by 0.5 / || y - A(x_0_hat) ||, which is equivalent to taking a constant-size step on the unsquared residual norm. This yields a single stable guidance scale. The sampler is an unconditional reverse-diffusion step followed by a soft likelihood-guidance correction; it never enforces exact measurement consistency, so it avoids noise amplification while remaining compatible with nonlinear operators.

```python
import torch
from tqdm import tqdm
from .base import Algo
from utils.scheduler import Scheduler
import numpy as np


class DPS(Algo):
    """Diffusion Posterior Sampling.
    Unconditional reverse diffusion plus soft likelihood guidance computed by
    backpropagating a data-fidelity loss through the in-loop Tweedie estimate.
    Works for any differentiable forward operator A."""

    def __init__(self, net, forward_op, diffusion_scheduler_config, guidance_scale, sde=True):
        super().__init__(net, forward_op)
        self.scale = guidance_scale
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.scheduler = Scheduler(**diffusion_scheduler_config)
        self.sde = sde

    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        if num_samples > 1:
            observation = observation.repeat(num_samples, 1, 1, 1)

        x_initial = torch.randn(
            num_samples, self.net.img_channels,
            self.net.img_resolution, self.net.img_resolution,
            device=device
        ) * self.scheduler.sigma_max
        x_next = x_initial
        x_next.requires_grad = True

        pbar = tqdm(range(self.scheduler.num_steps))
        for i in pbar:
            x_cur = x_next.detach().requires_grad_(True)

            sigma = self.scheduler.sigma_steps[i]
            factor = self.scheduler.factor_steps[i]
            scaling_factor = self.scheduler.scaling_factor[i]

            # Tweedie posterior mean x_0_hat = E[x_0 | x_t]
            denoised = self.net(
                x_cur / self.scheduler.scaling_steps[i],
                torch.as_tensor(sigma).to(x_cur.device)
            )

            # grad_{x_0_hat} ||A(x_0_hat) - y||^2 and loss = ||A(x_0_hat) - y||^2
            gradient, loss_scale = self.forward_op.gradient(
                denoised, observation, return_loss=True
            )

            # VJP: backprop through x_0_hat(x_cur) to get grad_{x_cur} ||y - A(x_0_hat)||^2
            ll_grad = torch.autograd.grad(denoised, x_cur, gradient)[0]
            # Residual normalization: 0.5/sqrt(loss) turns grad ||.||^2 into grad ||.||
            ll_grad = ll_grad * 0.5 / torch.sqrt(loss_scale).clamp(min=1e-6)

            # Unconditional reverse step (EDM-scaled score; SDE or PF-ODE)
            score = (
                (denoised - x_cur / self.scheduler.scaling_steps[i])
                / sigma ** 2 / self.scheduler.scaling_steps[i]
            )
            pbar.set_description(
                f'Iteration {i + 1}/{self.scheduler.num_steps}. '
                f'Data fitting loss: {torch.sqrt(loss_scale)}'
            )

            if self.sde:
                epsilon = torch.randn_like(x_cur)
                x_next = (
                    x_cur * scaling_factor + factor * score
                    + np.sqrt(factor) * epsilon
                )
            else:
                x_next = x_cur * scaling_factor + factor * score * 0.5

            # Likelihood guidance: subtract zeta' * (residual-normalized fidelity gradient)
            x_next = x_next - ll_grad * self.scale

        return x_next
```
