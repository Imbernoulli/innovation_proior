DPS's measured numbers show exactly where the point estimate helps and where it hurts. It did well on $\texttt{blackhole}$ — PSNR $25.60$, $\texttt{cp\_chi2}$ $5.89$, $\texttt{camp\_chi2}$ $4.11$ — because the sparse-interferometry likelihood forgives a soft fidelity gradient; it landed a respectable but unspectacular PSNR $25.20$ / SSIM $0.83$ on $\texttt{inv-scatter}$ where a single point-estimate scale is a blunt instrument on the clean EM operator; and it was weakest on $\texttt{inpainting}$ at PSNR $18.48$, where the low guidance scale needed to keep the soft gradient stable also left the masked region filled in only loosely. That split traces to one approximation: DPS evaluates the likelihood gradient at the posterior *mean* $\hat{x}_0$, which collapses real mode structure exactly where the reverse process decides global structure — at high noise. The exact object is $p_t(y\mid x_t) = \mathbb{E}_{x_0\sim p(x_0\mid x_t)}[p(y\mid x_0)]$; replacing $\mathbb{E}[p(y\mid x_0)]$ by $p(y\mid \mathbb{E}[x_0])$ is a Jensen swap of a nonlinear function and an expectation, exact only when the function is affine or the distribution is a point. In a two-mode toy where the truth is computable, the point estimate gets the bias backwards at both ends: at high noise the MMSE estimate sits in the middle on the steep part of the loss gradient and DPS reports *large* guidance while the true smeared expectation is *small*; at low noise $\hat{x}_0$ has committed to one mode, deep in a flat region, so DPS reports *small* guidance while the tight true posterior is *large*. Wherever the loss gradient is curved across the support of $p(x_0\mid x_t)$, the mean of the gradient and the gradient at the mean disagree, and the disagreement scales with the posterior spread — huge at high noise. That is structural, not a tuning miss, and it is the "least reliable at high noise" failure the uneven results confirm.

I propose LGD — Loss-Guided Diffusion — which keeps DPS's single-denoiser-call structure but stops collapsing the posterior to a point. Instead of a delta I use a *spread-out* Gaussian surrogate $q(x_0\mid x_t) = \mathcal{N}(\mu(x_t), r_t^2 I)$ for the denoising posterior and estimate $\mathbb{E}_{x_0\sim q}[p(y\mid x_0)]$ honestly. The center is forced by a variational argument: maximizing the expected log-likelihood at fixed covariance reduces to $\min_\mu \mathbb{E}_{x_0\sim p(x_0\mid x_t)}\|\mu - x_0\|^2$, whose minimizer is the mean, so $\mu(x_t) = E[x_0\mid x_t] = \hat{x}_0$ — the same Tweedie estimate DPS uses is the *provably optimal* mean of any fixed-covariance Gaussian surrogate, and it comes from the one denoiser call I am already making. The entire difference from DPS is giving it a width. I set $r_t$ from the noising model: with $x_t = x_0 + \sigma_t\varepsilon$ and a roughly unit-scale prior, the posterior over $x_0$ is a Gaussian-conjugate balance of prior precision ($\approx 1$) and likelihood precision ($1/\sigma_t^2$), giving variance $\sigma_t^2/(1+\sigma_t^2)$, i.e.
$$r_t = \frac{\sigma_t}{\sqrt{1+\sigma_t^2}}.$$
The limits are exactly right and load-bearing: at high noise $r_t\to 1$, as wide as the prior, because a noisy observation says almost nothing about $x_0$; at low noise $r_t\to\sigma_t\to 0$, collapsing to a delta. That collapse is the reassuring part — as $r_t\to 0$ the surrogate *becomes* the DPS delta, so I recover the point estimate exactly where it was already accurate (the tight-posterior $\texttt{blackhole}$-like regime) and only widen it where DPS was broken. The direction is justified by a total-variation bound: $|\mathbb{E}_p[p(y\mid x_0)] - \mathbb{E}_q[p(y\mid x_0)]| \le 2M\cdot\mathrm{TV}(p,q)$ with $M = \max p(y\mid x_0)$, and the DPS delta is maximally far from a continuous posterior ($\mathrm{TV} = 1$), so a Gaussian with about the right mean *and* width is closer in TV and lowers the worst-case guidance bias.

To compute $\mathbb{E}_{x_0\sim q}[p(y\mid x_0)]$ and its gradient with one network call I Monte-Carlo it: draw $n$ samples and average. The exact estimator needs care, because the quantity is $\nabla_{x_t}\log\mathbb{E}_{x_0\sim q}[\exp(-\ell_y(x_0))]$ with the expectation *inside* the log — a log-mean-exp whose gradient is a softmin-weighted average of the per-sample loss gradients, smaller-loss samples weighted more; naively averaging the loss gradients estimates $\nabla\mathbb{E}[\ell]$ and is the same Jensen confusion one level down. The samples are reparameterized $x^{(i)} = \hat{x}_0 + r_t\varepsilon^{(i)}$, so all $n$ branch off the *same* denoiser output and I backpropagate through the diffusion network exactly *once* regardless of $n$; only the cheap operator gradients scale with $n$. That single-backward fact is what makes the spread affordable on top of DPS. Here I have to be honest about the gap between the exact estimator and what this harness runs: the operator interface gives me $\nabla\|A(x)-y\|^2$ and the squared-residual loss per call, and the loop the harness realizes takes the plain *arithmetic mean* of the per-sample residual gradients before the single VJP, $g = \frac{1}{n}\sum_i \nabla\|A(x^{(i)})-y\|^2$ (with $\texttt{batch\_grad=True}$ evaluating the operator on the whole sample batch at once). That is a least-squares mean-gradient approximation specialized to the squared-residual interface, *not* the softmin-weighted log-mean-exp — so what this LGD actually does is restore the posterior spread, the whole point versus DPS, while aggregating it with a mean-gradient surrogate rather than the exact weighting. The rest is inherited from DPS deliberately: the same root-loss normalization $0.5/\sqrt{\texttt{avg\_loss}}$ to turn $\nabla\|r\|^2$ into $\nabla\|r\|$, the same EDM SDE/ODE step, and the same subtraction of the loss-gradient update to move toward better fit. The width $r_t$ and the sample count $n$ are the only genuinely new degrees of freedom over DPS; everything else is the DPS step with a spread surrogate in front of it. The per-$\texttt{ENV}$ table again sets $\texttt{guidance\_scale}$ and $\texttt{num\_mc\_samples}$ per operator — $\texttt{inv-scatter}$ $3200.0$ with $20$ samples (the clean large residual tolerates strong guidance and benefits from variance reduction), $\texttt{blackhole}$ $10^{-3}$ with $5$, $\texttt{inpainting}$ $1.0$ with $5$ (the small pixel-domain loss, where the inv-scatter-tuned $3200$ would diverge immediately).

Reading DPS's shape, I expect the spread plus stronger tuned guidance to help most on $\texttt{inv-scatter}$ — the cleanest PSNR win — and to lift $\texttt{inpainting}$ modestly by improving the global structure of the fill at high noise. The case I genuinely worry about is $\texttt{blackhole}$: DPS already did *well* there precisely because its soft point estimate suited the sparse likelihood, so dropping a sampled surrogate with a re-tuned scale in front of an already-correct regime could perturb it. If the chi-squared values climb and the blackhole PSNR falls below DPS's $25.60$, that is the spread hurting where the point estimate was right, and the diagnosis for the next step would be that per-step guidance with a sampled surrogate is still fighting the same coupled prior-and-likelihood update — pointing toward *decoupling* the prior step from the likelihood correction rather than refining the guidance term further.

```python
import os
import torch
from tqdm import tqdm
from algo.base import Algo
from utils.scheduler import Scheduler
from utils.diffusion import DiffusionSampler
import numpy as np


class Custom(Algo):
    """LGD: Loss-Guided Diffusion.
    Uses Monte Carlo gradient estimation for measurement guidance.
    """

    # Per-problem task-local hyperparameters, initialized from InverseBench-style
    # inverse problem settings and then adjusted for this benchmark harness.
    # inpainting: ffhq256 box-mask, sigma_noise=0.05. Pixel-domain image prior
    #   has raw loss_scale ~1e3 at early steps so the 3200 default (tuned for
    #   the electromagnetic inv-scatter forward op) diverges immediately. Use
    #   guidance_scale=1.0 in the normalized image range, matching typical
    #   DPS-family values for natural-image inverse problems. num_mc_samples=5
    #   is a conservative middle ground — higher reduces variance but costs
    #   memory linearly in the samples dimension.
    PROBLEM_CONFIGS = {
        'inv-scatter': {'guidance_scale': 3200.0, 'num_mc_samples': 20},
        'navier-stokes': {'guidance_scale': 3e-3, 'num_mc_samples': 3},
        'blackhole': {'guidance_scale': 1e-3, 'num_mc_samples': 5},
        'acoustic': {'guidance_scale': 1.0, 'num_mc_samples': 3, 'num_steps': 100},
        'inpainting': {'guidance_scale': 1.0, 'num_mc_samples': 5},
    }

    def __init__(self, net, forward_op,
                 diffusion_scheduler_config=None,
                 guidance_scale=3200.0,
                 num_mc_samples=20,
                 batch_grad=True,
                 sde=True,
                 **kwargs):
        super(Custom, self).__init__(net, forward_op)
        # Apply per-problem overrides
        env = os.environ.get('ENV', '')
        if env in self.PROBLEM_CONFIGS:
            cfg = self.PROBLEM_CONFIGS[env]
            guidance_scale = cfg.get('guidance_scale', guidance_scale)
            num_mc_samples = cfg.get('num_mc_samples', num_mc_samples)
        self.scale = guidance_scale
        self.diffusion_scheduler_config = diffusion_scheduler_config or {
            'num_steps': 1000, 'schedule': 'vp', 'timestep': 'vp', 'scaling': 'vp'
        }
        # Override num_steps for expensive problems
        if env in self.PROBLEM_CONFIGS and 'num_steps' in self.PROBLEM_CONFIGS[env]:
            self.diffusion_scheduler_config['num_steps'] = self.PROBLEM_CONFIGS[env]['num_steps']
        self.scheduler = Scheduler(**self.diffusion_scheduler_config)
        self.sde = sde
        self.num_samples = num_mc_samples
        self.batch_grad = batch_grad

    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
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
            rt = sigma / np.sqrt(1 + sigma ** 2)

            denoised = self.net(
                x_cur / self.scheduler.scaling_steps[i],
                torch.as_tensor(sigma).to(x_cur.device)
            )

            samples = denoised + torch.randn(
                (self.num_samples, *denoised.shape[1:]), device=device
            ) * rt

            if self.batch_grad:
                gradient, loss_scale = self.forward_op.gradient(
                    samples, observation, return_loss=True
                )
                avg_loss = loss_scale
            else:
                gradients = torch.empty(
                    (self.num_samples, *denoised.shape[1:]), device=device
                )
                losses = np.empty(self.num_samples)
                for j in range(self.num_samples):
                    gradient, loss_scale = self.forward_op.gradient(
                        samples[j:j+1], observation, return_loss=True
                    )
                    gradients[j] = gradient
                    losses[j] = loss_scale
                avg_loss = losses.mean()
                gradient = gradients

            avg_grad = torch.mean(gradient, dim=0, keepdim=True).detach()

            ll_grad = torch.autograd.grad(denoised, x_cur, avg_grad)[0]
            ll_grad = ll_grad * 0.5 / torch.sqrt(avg_loss)

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
                x_next = (x_cur * scaling_factor + factor * score
                          + np.sqrt(factor) * epsilon)
            else:
                x_next = x_cur * scaling_factor + factor * score * 0.5
            x_next -= ll_grad * self.scale

        return x_next
```
