RED-diff is the strongest frame yet, and its numbers tell me exactly what is still missing. The decoupling-into-an-optimization paid off where I bet it would: $\texttt{inv-scatter}$ leapt to PSNR $38.25$ / SSIM $0.98$, far past LGD's $30.20$, by fitting the clean EM operator hard with a tiny prior weight on a stable objective; the $\texttt{blackhole}$ regression LGD caused was repaired, $\texttt{cp\_chi2}$ from LGD's $17.58$ back to $3.01$ and $\texttt{camp\_chi2}$ from $11.46$ to $3.22$, both *below* DPS's $5.89/4.11$; and $\texttt{inpainting}$ improved on both axes, PSNR $21.54$ past LGD's $20.50$ and LPIPS down to $0.1677$ from DPS's $0.2202$. But one number did not move: the blackhole PSNR is stuck at $21.86$, essentially LGD's $21.50$ and below DPS's $25.60$, even with excellent chi-squared. That combination — consistency statistics excellent, PSNR mediocre — is the signature of the failure I flagged when I built RED-diff: by committing to a single Gaussian mode with $\sigma\to 0$, the variational fit is a point estimate of $\mu$, and mode-seeking KL on a multimodal posterior lands a *consistent-but-blurred* image. The sparse-interferometer posterior is genuinely multimodal, so the closure statistics are satisfied by a smooth image that lacks the sharp structure PSNR rewards. RED-diff stopped being a *sampler* and became an optimizer that found a blurred mode. The fix I named at the close of that reasoning is to restore stochastic exploration on the clean variable rather than optimize a single point estimate of $\mu$ — keep the decoupling that fixed the chi-squared, but make the clean-variable step a genuine *sample* again. Looking at what every method so far has in common that I have not yet attacked: RED-diff decoupled but collapsed to one mode; DPS and LGD stayed samplers but coupled the likelihood into a single reverse step on the noisy latent. Neither has both properties — decoupled *and* a true sampler — and that gap is where the remaining gain is.

I propose DAPS — Decoupled Annealing Posterior Sampling — which does the likelihood correction on the *clean* variable, as a genuine sample, and re-injects the prior separately, so the two operations live on the two variables they each belong to. I do not carry a noisy latent forward by small coupled steps at all. At each annealing noise level $\sigma$ I run three separate operations. First, the *prior update*: take the current noisy iterate $x_t$ and turn it into a sharp clean estimate $\hat{x}_0$ by solving the *unconditional* reverse process (the PF-ODE) from $x_t$ down to clean. This is the diffusion model doing its job, knowing nothing about $y$ — "what clean signal does the prior think this iterate came from" — and because I can spend a few ODE steps here it is sharper than DPS's single Tweedie call, while the decoupling means it need not be exact. Second, the *likelihood correction*: now that I have a clean estimate, I sample the clean conditional $p(x_0\mid x_t, y) \propto p(x_0\mid x_t)\,p(y\mid x_0)$. Approximating $p(x_0\mid x_t) \approx \mathcal{N}(\hat{x}_0, \sigma^2 I)$ — the conjugate width tracking the level's noise — its score is
$$\nabla_{x_0}\log p(x_0\mid x_t, y) = -\frac{x_0 - \hat{x}_0}{\sigma^2} - \texttt{data\_scale}\cdot\nabla_{x_0}\|A(x_0) - y\|^2,$$
a clean-variable score with $A$ evaluated *directly at $x_0$* and no denoiser anywhere in it. I sample it with Langevin dynamics, $x_0 \leftarrow x_0 - \eta\big[(x_0 - \hat{x}_0)/\sigma^2 + \texttt{data\_scale}\cdot\nabla\|A(x_0)-y\|^2\big] + \sqrt{2\eta}\,z$, for $N$ steps from $x_0^{(0)} = \hat{x}_0$. The injected Langevin noise $z$ is exactly what RED-diff threw away — it is what keeps this a *sample* of the conditional rather than a mode-collapsing optimization, and it is what should restore the exploration the stuck blackhole PSNR is asking for. Third, the *re-noise*: take the corrected clean sample and add fresh noise at the next, lower level, $x_{t-1} = x_0^{(N)} + \sigma_\text{next}\cdot\epsilon$.

The re-noising is the structural heart. In DPS and LGD each $x_t$ is a small reverse step from the last, so the steps are *coupled* — a bad guidance correction at level $t$ corrupts the latent feeding level $t-1$, and the error propagates down the whole trajectory; that coupling is precisely why LGD's sampled surrogate, fine on inv-scatter, *propagated* into a blackhole collapse as one operator's instability rode the shared latent all the way down. Here each $x_t$ is built *fresh* from a corrected clean estimate plus new noise, so consecutive iterates may differ a lot and a bad correction at one level does not poison the next — the noise wipes the slate to the next level's scale. The trajectory is decoupled, which is the property that should let me fix the blackhole PSNR without re-breaking the chi-squared. And it anneals to the right thing: at the largest $\sigma$, $x_t$ is near pure noise, the prior update samples broadly, and the wide anchor ($1/\sigma^2$ small) lets the data term and Langevin noise explore; as $\sigma$ decreases the anchor tightens, the prior's clean estimate sharpens, and $p(x_0\mid x_t, y)$ concentrates; at $\sigma\to 0$ the re-noising adds vanishing noise and the conditional *is* the posterior $p(x_0\mid y)$. The sequence of clean conditionals is an annealing path from a data-dominated exploratory distribution to the sharp posterior — annealed Langevin with the prior re-injected at every level through the unconditional clean estimate. That is the both-properties object: decoupled (re-noised, errors do not propagate) *and* a true sampler (Langevin noise, no mode collapse).

In this task's edit surface the harness gives me everything the construction needs. $\texttt{DiffusionSampler(scheduler).sample(net, x\_start, SDE=False)}$ is the unconditional PF-ODE prior update, and I build the inner scheduler fresh with $\texttt{Scheduler(sigma\_max=}\sigma\texttt{, \dots)}$ per level so the sub-sampler starts at the current annealing noise. The Langevin inner loop calls $\texttt{forward\_op.gradient(x, observation, return\_loss=True)}$ for $\nabla\|A(x)-y\|^2$ at the *clean* $x$ — the denoiser is never inside this loop, so the $N=100$ inner steps are cheap, and a nonlinear $A$ (inv-scatter) is just a different gradient with no network Jacobian, the robustness property DPS/LGD lacked. The data term carries a per-operator $\texttt{data\_scale}$ playing the role of $1/(2\sigma_y^2)$, set per $\texttt{ENV}$ exactly as DPS/LGD/RED-diff all needed because the three operators produce residuals on wildly different scales — $\texttt{inv-scatter}$ $\texttt{data\_scale}=1.0$ (clean large EM residual), $\texttt{blackhole}$ $10^{-5}$ (sparse residual orders of magnitude smaller), $\texttt{inpainting}$ $1/(2\cdot 0.05^2)$ (small pixel-domain residual at $\sigma_y = 0.05$) — alongside the Langevin step size; the algorithm itself is unchanged across problems. The one numerical detail the method requires is that the Langevin step size is cosine-decayed across the inner loop from the configured $\texttt{lr}$ down to a small floor ($\texttt{lr\_min\_ratio}=0.01$), so the $1/\sigma^2$ anchor stays $O(1)$ and the inner sampler does not explode at small $\sigma$. The three pieces line up with the canonical method: the unconditional PF-ODE for $\hat{x}_0$ (not a single Tweedie call), the two-term clean-variable Langevin score with $\sqrt{2\,\texttt{lr}}$ injected noise, and the $\mathcal{N}(x_0^{(N)}, \sigma_\text{next}^2)$ re-noise.

Against RED-diff's measured numbers, the bar is precise. The blackhole PSNR stuck at $21.86$ with excellent chi-squared is the target: if restoring stochastic exploration on the clean variable is the right diagnosis, DAPS should raise it *above* $21.86$ — ideally back toward DPS's $25.60$ — while *holding* $\texttt{cp\_chi2}$/$\texttt{camp\_chi2}$ near RED-diff's low level, the decoupling and re-noising being what let it improve PSNR without re-breaking consistency; a blackhole PSNR that rises while the chi-squared stays low is the falsifiable win. On $\texttt{inv-scatter}$, where RED-diff reached $38.25/0.98$ by aggressive clean-variable fitting, the clean-variable Langevin should match or exceed that — landing below $38$ would mean the Langevin exploration is costing fidelity on the easy operator and the step size needs tightening. On $\texttt{inpainting}$, the true-sampler exploration should push LPIPS below $0.1677$ while holding PSNR at or above $21.54$ — and if LPIPS does not improve, that says the inpainting posterior was already near-unimodal and exploration buys nothing there, a clean negative rather than a regression. The single way DAPS fails the bar is if the extra Langevin and ODE cost does not translate into the blackhole PSNR gain: if it stays at $\sim 21.9$ with low chi-squared, then the stuck PSNR was never under-exploration but a genuine information limit of the sparse measurement, and no clean-variable sampler can move it.

```python
import os
import torch
from tqdm import tqdm
from algo.base import Algo
from utils.scheduler import Scheduler
from utils.diffusion import DiffusionSampler
import numpy as np


class Custom(Algo):
    """DAPS: Decoupled Annealing Posterior Sampling.
    Each annealing level: (1) unconditional PF-ODE prior update x_t -> x0hat;
    (2) clean-variable Langevin correction sampling p(x0 | x_t, y); (3) re-noise.
    The denoiser is outside the Langevin loop, so A is evaluated on the clean
    variable with no network Jacobian (robust for nonlinear operators).
    """

    # Per-problem hyperparameters. The three forward operators produce residuals
    # on very different scales, so the data-term weight and Langevin step size are
    # set per ENV (the same per-operator adaptation DPS/LGD/REDDiff all need);
    # the algorithm itself is unchanged across problems.
    #   inv-scatter: clean large EM residual  -> large data_scale.
    #   blackhole:   sparse interferometer    -> tiny data_scale.
    #   inpainting:  small pixel-domain loss (sigma_y=0.05) -> moderate data_scale.
    PROBLEM_CONFIGS = {
        'inv-scatter': {'data_scale': 1.0, 'lgvd_lr': 1e-3, 'num_annealing_steps': 200},
        'blackhole': {'data_scale': 1e-5, 'lgvd_lr': 1e-3, 'num_annealing_steps': 200},
        'inpainting': {'data_scale': 1.0 / (2 * 0.05 ** 2), 'lgvd_lr': 5e-4,
                       'num_annealing_steps': 200},
    }

    def __init__(self, net, forward_op,
                 diffusion_scheduler_config=None,
                 num_annealing_steps=200,
                 lgvd_lr=1e-3,
                 lgvd_steps=100,
                 data_scale=1.0,
                 sigma_y=0.05,
                 lr_min_ratio=0.01,
                 ode_steps=5,
                 **kwargs):
        super(Custom, self).__init__(net, forward_op)
        env = os.environ.get('ENV', '')
        if env in self.PROBLEM_CONFIGS:
            cfg = self.PROBLEM_CONFIGS[env]
            data_scale = cfg.get('data_scale', data_scale)
            lgvd_lr = cfg.get('lgvd_lr', lgvd_lr)
            num_annealing_steps = cfg.get('num_annealing_steps', num_annealing_steps)
        self.data_scale = data_scale
        self.lgvd_lr = lgvd_lr
        self.lgvd_steps = lgvd_steps
        self.sigma_y = sigma_y
        self.lr_min_ratio = lr_min_ratio
        self.ode_steps = ode_steps
        # Outer annealing schedule over noise levels (VP).
        self.annealing_scheduler = Scheduler(
            num_steps=num_annealing_steps, schedule='vp',
            timestep='vp', scaling='vp'
        )
        # Config for the inner unconditional PF-ODE sub-sampler (sigma_max set per level).
        self.diffusion_scheduler_config = diffusion_scheduler_config or {
            'num_steps': self.ode_steps, 'schedule': 'vp',
            'timestep': 'vp', 'scaling': 'vp'
        }

    def langevin_sample(self, x0hat, observation, sigma):
        """Sample p(x0 | x_t, y) on the clean variable by Langevin dynamics.
        Score = -(x0 - x0hat)/sigma^2 - data_scale * grad||A(x0) - y||^2."""
        x = x0hat.clone().detach()
        sigma2 = float(sigma) ** 2 + 1e-8
        for j in range(self.lgvd_steps):
            ratio = j / self.lgvd_steps
            lr = self.lgvd_lr * (self.lr_min_ratio + (1 - self.lr_min_ratio)
                                 * 0.5 * (1 + np.cos(np.pi * ratio)))
            x = x.detach().requires_grad_(True)
            data_grad, _ = self.forward_op.gradient(x, observation, return_loss=True)
            data_grad = torch.nan_to_num(data_grad, nan=0.0, posinf=0.0, neginf=0.0)
            grad = (x.detach() - x0hat) / sigma2 + self.data_scale * data_grad
            x = x.detach() - lr * grad + np.sqrt(2 * lr) * torch.randn_like(x)
        return x.detach()

    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        if num_samples > 1:
            observation = observation.repeat(num_samples, 1, 1, 1)
        sigmas = self.annealing_scheduler.sigma_steps
        num_steps = self.annealing_scheduler.num_steps

        x_t = torch.randn(
            num_samples, self.net.img_channels,
            self.net.img_resolution, self.net.img_resolution,
            device=device
        ) * self.annealing_scheduler.sigma_max

        pbar = tqdm(range(num_steps))
        for i in pbar:
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1] if i + 1 < len(sigmas) else 0.0

            # (1) PRIOR UPDATE: unconditional PF-ODE from x_t (inner sigma_max = sigma).
            diff_sched = Scheduler(sigma_max=float(sigma), **self.diffusion_scheduler_config)
            with torch.no_grad():
                x0hat = DiffusionSampler(diff_sched).sample(self.net, x_t, SDE=False)
            x0hat = x0hat.detach()

            # (2) LIKELIHOOD CORRECTION: Langevin sample of p(x0 | x_t, y) on the clean var.
            x0y = self.langevin_sample(x0hat, observation, sigma)

            # (3) RE-NOISE to the next, lower annealing level.
            x_t = x0y + torch.randn_like(x0y) * float(sigma_next)

            loss = self.forward_op.loss(x0y, observation).mean()
            pbar.set_description(
                f'Annealing {i + 1}/{num_steps}. '
                f'Data fitting loss: {torch.sqrt(loss.clamp(min=0))}'
            )
        return x_t
```
