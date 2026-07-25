The goal is to draw posterior samples x_0 ~ p(x_0 | y) for a noisy inverse problem y = A(x_0) + n using only a pretrained diffusion prior, without retraining, and for general forward operators A including nonlinear ones. The difficulty is the time-level conditional score: Bayes splits it into the free prior score plus the likelihood score, but the latter requires marginalizing over the unknown clean x_0 and has no closed form because the denoising posterior p(x_0 | x_t) is multimodal. Existing training-free solvers handle this by approximating a guidance term and injecting it into a single reverse step. DPS collapses the denoising posterior to its Tweedie mean, which is least reliable at high noise and costs a denoiser Jacobian per step. LGD replaces the point mass with a Gaussian surrogate and Monte-Carlos the gradient, but it still backpropagates through the denoiser and still couples the prior and likelihood on the noisy latent. RED-diff steps off the trajectory entirely and optimizes a single Gaussian on the clean variable, removing the network Jacobian but giving up stochastic exploration and often landing on a blurred mode. The recurring failure is coupling: the prior lives on the noisy latent x_t while the likelihood lives on the clean signal x_0, and forcing both into one update makes nonlinear or ill-scaled problems fragile.

The right move is to keep the variables separate. At each annealing level we should extract a clean estimate from the prior, correct it against the likelihood on the clean variable where the likelihood actually lives, and then re-noise the corrected clean sample to the next noise level. This decouples consecutive iterates so a bad likelihood correction does not poison the latent trajectory, removes the denoiser Jacobian from the data term, and preserves stochastic exploration so the method remains a true sampler rather than a mode-seeking optimizer.

The method is DAPS, Decoupled Annealing Posterior Sampling. It is a three-stage cycle repeated over an annealing schedule of noise levels. First, the prior update runs the unconditional probability-flow ODE starting from the current noisy x_t down to a clean estimate x_0hat. Using a short ODE sub-sampler instead of a single Tweedie call gives a sharper sample from p(x_0 | x_t) and is affordable because the outer steps are decoupled. Second, the likelihood correction samples the clean conditional p(x_0 | x_t, y) by Langevin dynamics on x_0. Approximating p(x_0 | x_t) as a Gaussian centered at x_0hat with variance tied to the current noise level, the conditional score is the sum of a prior anchor -(x_0 - x_0hat)/sigma^2 and the data-fidelity gradient on the clean variable. Importantly, A is evaluated and differentiated directly at x_0, never through the denoiser, so nonlinear operators are no harder than linear ones and the expensive network is outside the inner loop. Third, re-noising draws the next iterate x_{t-1} from N(x_0^{(N)}, sigma_next^2 I), freshly building the next latent from a corrected clean sample rather than inheriting a small step. As sigma shrinks, the anchor tightens and the conditional concentrates on the true posterior p(x_0 | y), while the Langevin noise prevents mode collapse.

The inner Langevin step size must decay with the noise level so that the 1/sigma^2 anchor term remains well-conditioned across the schedule. A cosine decay works well. The number of annealing levels, the number of ODE steps per level, and the number of Langevin steps trade cost for accuracy; with the denoiser outside the inner loop, running on the order of a hundred Langevin steps is cheap. The known measurement noise sigma_y sets the strength of the data term relative to the prior anchor, so a new forward operator needs only this one scalar, not a delicate guidance schedule on a coupled trajectory.

```python
import torch
import numpy as np
from .base import Algo
from utils.scheduler import Scheduler
from utils.diffusion import DiffusionSampler


class LangevinDynamics:
    """Sample p(x0 | x_t, y) on the CLEAN variable. The denoiser is NOT here —
    only the cheap forward operator, so many steps are affordable."""

    def __init__(self, num_steps, lr, sigma_y, lr_min_ratio=0.01):
        self.num_steps = num_steps
        self.lr = lr
        self.sigma_y = sigma_y
        self.lr_min_ratio = lr_min_ratio

    def sample(self, x0hat, forward_op, observation, sigma):
        x = x0hat.clone().detach()
        for j in range(self.num_steps):
            ratio = j / self.num_steps
            lr = self.lr * (self.lr_min_ratio + (1 - self.lr_min_ratio)
                            * 0.5 * (1 + np.cos(np.pi * ratio)))
            x = x.detach().requires_grad_(True)
            data_grad, _ = forward_op.gradient(x, observation, return_loss=True)
            grad = (x - x0hat) / (sigma ** 2) + data_grad / (2 * self.sigma_y ** 2)
            x = x.detach() - lr * grad + np.sqrt(2 * lr) * torch.randn_like(x)
        return x.detach()


class DAPS(Algo):
    def __init__(self, net, forward_op, annealing_scheduler_config,
                 diffusion_scheduler_config, lgvd_lr, lgvd_steps=100, sigma_y=0.05):
        super().__init__(net, forward_op)
        self.annealing_scheduler = Scheduler(**annealing_scheduler_config)
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.lgvd = LangevinDynamics(lgvd_steps, lgvd_lr, sigma_y)

    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        if num_samples > 1:
            observation = observation.repeat(num_samples, 1, 1, 1)
        sigmas = self.annealing_scheduler.sigma_steps
        x_t = torch.randn(num_samples, self.net.img_channels, self.net.img_resolution,
                          self.net.img_resolution, device=device) * sigmas[0]

        for i in range(self.annealing_scheduler.num_steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1] if i + 1 < len(sigmas) else 0.0

            # (1) unconditional PF-ODE prior update: x_t -> x0hat
            diff_sched = Scheduler(sigma_max=sigma, **self.diffusion_scheduler_config)
            x0hat = DiffusionSampler(diff_sched).sample(self.net, x_t, SDE=False)

            # (2) clean-variable Langevin correction: sample p(x0 | x_t, y)
            x0y = self.lgvd.sample(x0hat, self.forward_op, observation, sigma)

            # (3) re-noise to the next lower level
            x_t = x0y + torch.randn_like(x0y) * sigma_next

        return x_t
```
