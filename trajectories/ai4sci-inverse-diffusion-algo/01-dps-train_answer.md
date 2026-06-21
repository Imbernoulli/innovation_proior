The floor I have to start from is the scaffold default — return random noise, no conditioning at all — and the question at this first rung is the most basic honest one: can I condition a frozen diffusion prior on a measurement using only the unconditional score the network already gives me, the known forward operator $A$, and no retraining, in a way that survives measurement noise and a nonlinear $A$? Two of my three problems are exactly those hard cases, so any rule that needs a linear, easily-projectable operator or a clean measurement is dead on arrival. The reverse diffusion is driven by the prior score $\nabla_{x_t}\log p_t(x_t)$, which the denoiser supplies through Tweedie ($E[x_0\mid x_t] = x_t + \sigma^2 \nabla_{x_t}\log p_t(x_t)$ in the EDM scaling), and to condition on $y$ Bayes splits the posterior score cleanly: $\nabla_{x_t}\log p_t(x_t\mid y) = \nabla_{x_t}\log p_t(x_t) + \nabla_{x_t}\log p_t(y\mid x_t)$. The first term is free; the second, the time-level likelihood score, is the whole game and has no closed form, because $y$ depends on the noised iterate $x_t$ only through the unknown clean $x_0$ — there is no direct $x_t \to y$ edge in the graph $y \leftarrow x_0 \to x_t$. The standard dodges both fail my requirements: dropping the likelihood and *projecting* onto $\{x : Ax = y\}$ burns measurement noise into the reconstruction (fatal for the $\sigma=0.05$ inpainting noise) and presupposes a projectable linear operator (the scattering model is a nonlinear EM solver); the spectral SNIPS/DDRM route needs an explicit SVD of $A$ that a PDE solver or a sparse interferometer simply does not give me.

I propose DPS — Diffusion Posterior Sampling — which makes that intractable term tractable honestly rather than dodging it. The key is to marginalize the clean signal: since $y$ and $x_t$ are conditionally independent given $x_0$, the exact identity is
$$p_t(y\mid x_t) = \mathbb{E}_{x_0\sim p(x_0\mid x_t)}\big[\,p(y\mid x_0)\,\big],$$
the expectation, over the denoising posterior $p(x_0\mid x_t)$, of the *clean* likelihood $p(y\mid x_0)$ which I do know in closed form. The whole difficulty is now relocated into $p(x_0\mid x_t)$, intractable as a distribution — but I do not need the distribution, only an expectation against it, and there is exactly one functional of it I can compute: its mean. Because the forward process is Gaussian, Tweedie's formula gives $E[x_0\mid x_t]$ as an explicit function of the score, which is precisely what $\texttt{net}(x_t,\sigma)$ returns. So I swap the expectation of the function for the function of the expectation,
$$p_t(y\mid x_t) \approx p\big(y \mid \hat{x}_0(x_t)\big), \qquad \hat{x}_0 = E[x_0\mid x_t],$$
a point estimate that collapses the denoising posterior to its mean and evaluates the likelihood there. This is a Jensen-gap approximation; bounding the gap with the Lipschitz constant of the Gaussian density shows it *shrinks as the measurement noise grows*, which is the opposite of what one might fear and exactly why this point estimate is well-suited to the noisy regime that broke projection. Differentiating the surrogate, for Gaussian noise $\log p(y\mid\hat{x}_0) = -\|y - A(\hat{x}_0)\|^2/(2\sigma_y^2) + \text{const}$, gives the tractable likelihood score $\nabla_{x_t}\log p_t(y\mid x_t) \approx -\tfrac{1}{2\sigma_y^2}\nabla_{x_t}\|y - A(\hat{x}_0(x_t))\|^2$. The variable matters: the gradient is with respect to $x_t$, and $\hat{x}_0$ is a function of $x_t$ *through the denoiser*, so $\nabla_{x_t}$ is a backpropagation through the network and through Tweedie. That is the entire trick for general $A$ — no SVD, no projection, no hand-written transpose — I only need $A$ differentiable so autodiff can carry the operator's gradient at $\hat{x}_0$ back to $x_t$. A nonlinear $A$ is no harder than a linear one, which is the one property that gives this a chance on the scattering operator. The cost is one denoiser backward per reverse step.

In this task's edit surface the operator does not hand back $\nabla_{x_t}$ directly; it returns $\nabla_{\hat{x}_0}\|A(\hat{x}_0)-y\|^2$ and the loss at the denoised estimate. I turn that into the $x_t$-gradient with one vector-Jacobian product, $\texttt{autograd.grad(denoised, x\_cur, gradient)}$, pushing the operator gradient back through the denoiser. The second load-bearing detail is the step size. The clean derivation gives a fixed coefficient $1/(2\sigma_y^2)$, but across a thousand reverse steps the squared-residual gradient $\nabla\|r\|^2 = 2\|r\|\nabla\|r\|$ swings over orders of magnitude — large early when $\hat{x}_0$ is a blurry mean, tiny late when it is sharp — so a fixed coefficient makes the effective guidance wildly uneven. I want a roughly constant *effect* per step, so I residual-normalize: multiplying by $0.5/\|r\|$ converts $\nabla\|r\|^2$ into $\nabla\|r\|$, the gradient of the *root* loss. Since the operator returns $\texttt{loss} = \|r\|^2$, I multiply by $0.5/\sqrt{\texttt{loss}}$ (clamped away from zero), and the update becomes a constant step on the residual *norm* governed by one stable knob $\texttt{self.scale}$ rather than a coefficient tied to the noise scale. This is what lets one rule run on operators whose residuals live on different scales, and it is why a small $\texttt{PROBLEM\_CONFIGS}$ table keyed off $\texttt{ENV}$ sets the guidance scale per operator — $50$ on $\texttt{inv-scatter}$ (clean large EM residual), $10^{-3}$ on $\texttt{blackhole}$ (sparse interferometer, residual on a wholly different scale), $1.0$ on $\texttt{inpainting}$ (small pixel-domain loss, where the default $50$ explodes); this $\texttt{ENV}$-keyed table is the harness adapting one rule to three operators, not part of the method itself. There is a $\texttt{clip\_grad}$ flag left *off* here on purpose, because clipping the raw gradient norm *before* the $0.5/\sqrt{\texttt{loss}}$ rescaling would change the effective guidance scale and corrupt the tuned values — it is opt-in only for operators that produce NaNs — while an always-on $\texttt{nan\_to\_num}$ is the cheap safety net. The reverse update is the EDM SDE step, $\texttt{score} = (\texttt{denoised} - x_\text{cur}/\texttt{scaling})/\sigma^2/\texttt{scaling}$ then $x_\text{next} = x_\text{cur}\cdot\texttt{scaling\_factor} + \texttt{factor}\cdot\texttt{score} + \sqrt{\texttt{factor}}\,\epsilon$, after which I *subtract* the normalized fidelity gradient times the scale — the sign is the easy thing to get backwards, since the score points up the prior and the loss gradient points toward worse fit, so subtracting moves toward better fit. One scaling subtlety I must respect: the denoiser is called on $x_\text{cur}/\texttt{scaling\_steps}[i]$, because the scheduler keeps the iterate in a scaled variable.

What I expect from this floor follows directly from the one approximation: the point estimate is an excellent stand-in when the denoising posterior is tight (low noise, late in sampling) and least reliable when it is broad (high noise, early), exactly where the reverse process decides global structure. So DPS should be competent but uneven — strongest where the operator is clean and the residual well-behaved or where the sparse likelihood forgives a soft fidelity gradient, weakest where a single tuned scale must keep the guidance simultaneously stable and accurate. That diagnosis already names the next move: the limitation is the *point estimate* — collapsing the denoising posterior to its mean — so the way forward is to stop collapsing it to a point and restore the spread the high-noise regime demands.

```python
import os
import torch
from tqdm import tqdm
from algo.base import Algo
from utils.scheduler import Scheduler
from utils.diffusion import DiffusionSampler
import numpy as np


class Custom(Algo):
    """DPS: Diffusion Posterior Sampling.
    Score-based guidance using the gradient of the data likelihood.
    Requires forward_op.gradient() — best for differentiable forward operators.
    """

    # Per-problem optimized hyperparameters
    # inv-scatter: linear forward op, gradient clean → high guidance works
    # navier-stokes: PDE solver forward op, gradient VERY noisy/unstable →
    #   low guidance + gradient clipping to prevent NaN divergence
    # blackhole: non-trivial forward op → moderate guidance
    # clip_grad: ONLY enable for problems where the forward-op gradient is
    #   numerically unstable (e.g. NS PDE solver producing NaN). Clipping the
    #   raw ll_grad norm before the 1/sqrt(loss_scale) rescaling changes the
    #   effective guidance scale, so leave it OFF for well-behaved problems
    #   (inv-scatter / blackhole) to preserve their tuned guidance_scale.
    PROBLEM_CONFIGS = {
        'inv-scatter': {'guidance_scale': 50.0, 'clip_grad': False},
        'blackhole': {'guidance_scale': 1e-3, 'clip_grad': False},
        # FFHQ256 box-inpaint with sigma_noise=0.05. The default 50.0 (tuned for
        # the inv-scatter forward op) explodes here because the pixel-domain
        # data fitting loss is much smaller. guidance_scale=1.0 matches typical
        # DPS values for natural-image inverse problems.
        'inpainting': {'guidance_scale': 1.0, 'clip_grad': False},
    }

    def __init__(self, net, forward_op,
                 diffusion_scheduler_config=None,
                 guidance_scale=50.0,
                 sde=True,
                 **kwargs):
        super(Custom, self).__init__(net, forward_op)
        # Apply per-problem overrides
        env = os.environ.get('ENV', '')
        self.clip_grad = False
        if env in self.PROBLEM_CONFIGS:
            cfg = self.PROBLEM_CONFIGS[env]
            guidance_scale = cfg.get('guidance_scale', guidance_scale)
            self.clip_grad = cfg.get('clip_grad', False)
        self.scale = guidance_scale
        self.diffusion_scheduler_config = diffusion_scheduler_config or {
            'num_steps': 1000, 'schedule': 'vp', 'timestep': 'vp', 'scaling': 'vp'
        }
        # Override num_steps for expensive problems
        if env in self.PROBLEM_CONFIGS and 'num_steps' in self.PROBLEM_CONFIGS[env]:
            self.diffusion_scheduler_config['num_steps'] = self.PROBLEM_CONFIGS[env]['num_steps']
        self.scheduler = Scheduler(**self.diffusion_scheduler_config)
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

            denoised = self.net(
                x_cur / self.scheduler.scaling_steps[i],
                torch.as_tensor(sigma).to(x_cur.device)
            )
            gradient, loss_scale = self.forward_op.gradient(
                denoised, observation, return_loss=True
            )

            ll_grad = torch.autograd.grad(denoised, x_cur, gradient)[0]
            # Clip gradient to prevent NaN (only needed for NS solver / acoustic);
            # for well-behaved problems (inv-scatter, blackhole) this would
            # corrupt the tuned guidance scale, so the clip is opt-in.
            if self.clip_grad:
                grad_norm = ll_grad.norm()
                max_grad_norm = 1.0
                if grad_norm > max_grad_norm:
                    ll_grad = ll_grad * (max_grad_norm / grad_norm)
            # Always replace NaN/Inf gradients with zero (cheap + safe)
            ll_grad = torch.nan_to_num(ll_grad, nan=0.0, posinf=0.0, neginf=0.0)
            ll_grad = ll_grad * 0.5 / torch.sqrt(loss_scale).clamp(min=1e-6)

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
