LGD's numbers confirm both the gamble and the worry. The wins came where I expected: $\texttt{inv-scatter}$ jumped from DPS's PSNR $25.20$ to $30.20$ and SSIM from $0.83$ to $0.90$ — restoring the posterior spread plus stronger tuned guidance is what the clean EM operator wanted — and $\texttt{inpainting}$ lifted modestly from $18.48$ to $20.50$. But $\texttt{blackhole}$ *regressed* badly: PSNR fell from DPS's $25.60$ to $21.50$, and the chi-squared values blew up, $\texttt{cp\_chi2}$ from $5.89$ to $17.58$ and $\texttt{camp\_chi2}$ from $4.11$ to $11.46$. That is the spread hurting precisely where DPS's point estimate was already correct, and the cause is structural. Both DPS and LGD do the same risky thing at every reverse step: they bend the *same coupled* update with a measurement-guidance term, where the prior score lives on the noisy latent $x_t$ and the likelihood lives on the clean signal $x_0$, jammed into one step — which is why the guidance scale must be re-tuned per operator and why a surrogate that helps one operator wrecks another. Both also pay a denoiser Jacobian (LGD's $\texttt{autograd.grad(denoised, x\_cur, \dots)}$ is the same expensive VJP as DPS's). The fragility and the cost are two faces of one fact: per-step guidance *through* the network is a touchy, coupled operation. So the question is no longer how to refine the guidance term, but whether to stop bending the reverse step at all.

I propose RED-diff — Regularization by Denoising with Diffusion priors — which changes the frame from trajectory guidance to direct inference on the clean $x_0$. I have a frozen prior $p(x_0)$, a known likelihood $p(y\mid x_0)$, and the only obstacle to the posterior is the normalizer $p(y)$ — the textbook setting for variational inference. Posit a Gaussian $q(x_0\mid y) = \mathcal{N}(\mu, \sigma^2 I)$ and fit it by minimizing $\mathrm{KL}(q \,\|\, p(x_0\mid y))$; the KL is mode-seeking, so $q$ locks onto a dominant mode, which for an ill-posed problem is the feature I want — one plausible reconstruction, not a hedge. Bayes-expanding,
$$\mathrm{KL}(q \,\|\, p(x_0\mid y)) = -\mathbb{E}_q[\log p(y\mid x_0)] + \mathrm{KL}(q \,\|\, p(x_0)) + \log p(y),$$
and $\log p(y)$ drops as constant in $q$. The first term is friendly — $\tfrac{1}{2\sigma_v^2}\mathbb{E}_q\|y - f(x_0)\|^2$, just pull $\mu$ toward data-fitting signals. The hard term is $\mathrm{KL}(q\,\|\,p(x_0))$, because $p(x_0)$ is the diffusion prior, available only as a score along the noising trajectory, not as a density at $x_0$ — the intractability is relocated, not removed. The maximum-likelihood theory of diffusion supplies the missing representation: a KL between two distributions diffused by the same forward SDE equals a time-integrated weighted score-matching loss,
$$\mathrm{KL}(q(x_0\mid y)\,\|\,p(x_0)) = \int \tfrac{\beta(t)}{2}\,\mathbb{E}_{q(x_t\mid y)}\big[\|\nabla\log q(x_t\mid y) - \nabla\log p(x_t)\|^2\big]\,dt,$$
in which every piece is computable: $\nabla\log p(x_t)$ is the pretrained score $-\epsilon_\theta/\sigma_t$ that $\texttt{net}$ supplies, and $q(x_t\mid y)$, the diffused Gaussian, has a closed-form score. Taking $\sigma\to 0$ — a Dirac $q = \delta(x_0 - \mu)$, since isotropic Gaussian dispersion would leave the natural-image manifold anyway — gives $x_t = \alpha_t\mu + \sigma_t\epsilon$, $\nabla\log q(x_t\mid y) = -\epsilon/\sigma_t$, and the score difference $(\epsilon_\theta - \epsilon)/\sigma_t$. The regularizer becomes an expected squared *noise residual*, $\mathbb{E}_{t,\epsilon}[\,\text{weight}\cdot\|\epsilon_\theta(x_t;t) - \epsilon\|^2\,]$: find $\mu$ that reconstructs $y$ while its denoiser's predicted noise matches the noise actually injected, across the whole trajectory.

The trap — and the whole reason this is worth doing — is that differentiating the regularizer naively runs $\nabla_\mu\|\epsilon_\theta(\alpha_t\mu + \sigma_t\epsilon; t) - \epsilon\|^2$, where $\epsilon_\theta$ depends on $\mu$ through $x_t$, so I would be backpropagating through the pretrained network again — exactly the DPS/LGD Jacobian I am trying to escape. The escape comes from the companion identity $d\mathrm{KL}_t/dt = -\tfrac{\beta(t)}{2}\mathbb{E}\|\nabla\log q - \nabla\log p\|^2$, which writes the regularizer as $-\int\omega(t)\,(d\mathrm{KL}_t/dt)\,dt$. Integration by parts gives $-[\omega(t)\mathrm{KL}_t] + \int\omega'(t)\mathrm{KL}_t\,dt$; the boundary term dies at $t = T$ (the forward process destroys all signal, $\mathrm{KL}_T = 0$) and dies at $t = 0$ *provided I choose any weighting with $\omega(0) = 0$*. Under that condition, differentiating under the integral and substituting the two scores collapses the gradient to a plain $\epsilon_\theta(x_t;t) - \epsilon$ in which $\epsilon_\theta$ appears as a *value*, not as something I differentiate — it came out of $\nabla\log p$, the gradient of the frozen prior log-density. The reparameterization gradient legitimately lands on the *stop-gradiented* residual $\epsilon_\theta - \epsilon$. That is the entire payoff: the regularizer's gradient costs **one forward pass of the frozen denoiser per step and no backprop through it**, where DPS and LGD both needed a VJP; the innocuous $\omega(0) = 0$ condition is the structural reason it is cheap. This residual shape — a denoiser-based regularizer whose gradient is a simple residual — is the form of Regularization by Denoising, where one minimizes $\text{loss} + \tfrac{\lambda}{2}x^\top(x - f_\text{den}(x))$ and the gradient collapses to $x - f_\text{den}(x)$. But it is *not* classical RED: RED uses a single deterministic denoiser at one noise level and injects no noise (a fixed-point/MAP use of one operator), whereas this is generative — it adds fresh noise at *every* level across the *entire* trajectory and aggregates the ensemble, so the iterate can navigate the manifold. The trajectory and noise injection are what make it move, and that is what I lean on against LGD's fragility: not a per-step guidance fighting a coupled update, but an optimization over $\mu$ regularized by the whole denoiser ensemble.

The weighting needs one more idea, because the noise residual $\epsilon_\theta - \epsilon$ blows up as $t\to 0$ for VP diffusion, so equal weighting lets the tiny-noise steps dominate. I recast the regularizer into the signal domain, where it is comparable to $\|y - f(\mu)\|^2$. Tweedie gives $\hat{\mu}_t = (x_t - \sigma_t\epsilon_\theta)/\alpha_t$, and substituting $x_t = \alpha_t\mu + \sigma_t\epsilon$ yields the exact identity $\mu - \hat{\mu}_t = (\sigma_t/\alpha_t)(\epsilon_\theta - \epsilon)$ — the signal residual is the noise residual times $\sigma_t/\alpha_t = 1/\mathrm{SNR}_t$. To convert the noise-domain gradient $\lambda_t(\epsilon_\theta - \epsilon)$ into the signal-domain residual $\lambda(\mu - \hat{\mu}_t)$ with a *constant* $\lambda$, set $\lambda_t = \lambda/\mathrm{SNR}_t$. This is not a hack but the exact rescaling that puts both terms in the same units and removes the small-$t$ blow-up; it also upweights high-noise early steps (coarse structure) and downweights low-noise late steps (detail), so one knob $\lambda$ trades fidelity against the prior, and it says to step $t$ in *descending* order like a reverse sampler. In this task's edit surface the operational no-Jacobian point is made concrete: $\texttt{Custom}$ keeps $\mu$ as the optimized clean estimate and hands it *directly* to $\texttt{forward\_op.gradient(mu, observation, return\_loss=True)}$, so the data gradient is computed on $\mu$ with no denoiser in the path. Each step it samples $\epsilon$, noises $\mu$ to $\texttt{xt} = \texttt{scaling}\cdot(\mu + \sigma\epsilon)$, runs the denoiser once for the *detached* $\texttt{pred\_epsilon}$, forms $\texttt{gradient} = \texttt{observation\_weight}\cdot\nabla_\mu\|f(\mu) - y\|^2 + \texttt{lam}\cdot(\texttt{pred\_epsilon} - \epsilon)$, assigns it to $\texttt{mu.grad}$, and lets Adam step. The scheduler's $\sigma$ *is* $1/\mathrm{SNR}$, so the $\texttt{'linear'}$ schedule realizes $\lambda/\mathrm{SNR}_t$; the benchmark default here is $\texttt{'constant'}$. The per-$\texttt{ENV}$ table is now about the optimization rather than a guidance scale: $\texttt{inv-scatter}$ uses $\texttt{observation\_weight}=1500$, $\texttt{base\_lr}=0.04$, $\texttt{base\_lambda}=5\!\times\!10^{-4}$; $\texttt{blackhole}$ uses $\texttt{observation\_weight}=10^{-4}$, $\texttt{base\_lr}=10^{-2}$, $\texttt{base\_lambda}=0.25$; $\texttt{inpainting}$ is deliberately *omitted* because the defaults already work on FFHQ box-inpaint and an override would only risk regressing it. The data-weight swing ($1500$ vs $10^{-4}$) is the variational analogue of the earlier guidance-scale swing, but it now sits on a *decoupled, stable* objective — the structural difference I am betting on. Adam runs with $\texttt{betas}=(0.9, 0.99)$ and no weight decay, since the diffusion regularizer *is* the prior.

Reading LGD's shape against this frame: on $\texttt{inv-scatter}$ the decoupled fit with a strong data weight and a tiny prior weight should fit the clean operator far harder without per-step instability — I expect a large PSNR gain, possibly into the high $30$s. On $\texttt{blackhole}$, the game LGD broke, the tiny $\texttt{observation\_weight}$ should stop the over-fitting that blew up the chi-squared and bring $\texttt{cp\_chi2}$/$\texttt{camp\_chi2}$ back toward or below DPS's level, even if the blackhole PSNR stays near LGD's because the prior weight dominates. On $\texttt{inpainting}$, the default-config fit should improve both PSNR and LPIPS, a perceptually clean fill without burning in the $\sigma = 0.05$ noise. The one way this fails is if committing to a single Gaussian mode under-explores and lands a *blurrier* reconstruction than LGD's per-step guidance — if PSNR rises but LPIPS does not improve, that mode-seeking blur is the tell, and the next move would be to restore stochastic exploration on the clean variable rather than optimize a single point estimate of $\mu$.

```python
import os
import torch
import tqdm
from algo.base import Algo
from utils.scheduler import Scheduler
from utils.diffusion import DiffusionSampler
import numpy as np


class Custom(Algo):
    """REDDiff: Regularization by Denoising with Diffusion priors.
    Optimization-based approach using diffusion score as regularizer.
    """

    # Per-problem task-local hyperparameters, initialized from InverseBench-style
    # inverse problem settings and then adjusted for this benchmark harness.
    # 'inpainting' is intentionally omitted: the default __init__ values
    # (observation_weight=1500, base_lr=0.04, base_lambda=5e-4) already work
    # well on FFHQ256 box-inpaint (REDDiff achieves PSNR~22 with these), so
    # adding an override here would only risk regressing the result.
    PROBLEM_CONFIGS = {
        'inv-scatter': {'observation_weight': 1500.0, 'base_lr': 0.04, 'base_lambda': 5e-4},
        'blackhole': {'observation_weight': 1e-4, 'base_lr': 1e-2, 'base_lambda': 0.25},
    }

    def __init__(self, net, forward_op,
                 num_steps=1000,
                 observation_weight=1500.0,
                 base_lambda=5e-4,
                 base_lr=0.04,
                 lambda_scheduling_type='constant',
                 **kwargs):
        super(Custom, self).__init__(net, forward_op)
        # Apply per-problem overrides
        env = os.environ.get('ENV', '')
        if env in self.PROBLEM_CONFIGS:
            cfg = self.PROBLEM_CONFIGS[env]
            observation_weight = cfg.get('observation_weight', observation_weight)
            base_lr = cfg.get('base_lr', base_lr)
            base_lambda = cfg.get('base_lambda', base_lambda)
            num_steps = cfg.get('num_steps', num_steps)
        self.net.eval().requires_grad_(False)

        self.scheduler = Scheduler(
            num_steps=num_steps, schedule='vp',
            timestep='vp', scaling='vp'
        )
        self.base_lr = base_lr
        self.observation_weight = observation_weight
        if lambda_scheduling_type == 'linear':
            self.lambda_fn = lambda sigma: sigma * base_lambda
        elif lambda_scheduling_type == 'sqrt':
            self.lambda_fn = lambda sigma: torch.sqrt(sigma) * base_lambda
        elif lambda_scheduling_type == 'constant':
            self.lambda_fn = lambda sigma: base_lambda
        else:
            raise NotImplementedError

    def pred_epsilon(self, model, x, sigma):
        sigma = torch.as_tensor(sigma).to(x.device)
        d = model(x, sigma)
        return (x - d) / sigma

    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        num_steps = self.scheduler.num_steps
        pbar = tqdm.trange(num_steps)
        if num_samples > 1:
            observation = observation.repeat(num_samples, 1, 1, 1)

        mu = torch.zeros(
            num_samples, self.net.img_channels,
            self.net.img_resolution, self.net.img_resolution,
            device=device
        ).requires_grad_(True)
        optimizer = torch.optim.Adam([mu], lr=self.base_lr, betas=(0.9, 0.99))

        for step in pbar:
            with torch.no_grad():
                sigma = self.scheduler.sigma_steps[step]
                scaling = self.scheduler.scaling_steps[step]
                epsilon = torch.randn_like(mu)
                xt = scaling * (mu + sigma * epsilon)
                pred_epsilon = self.pred_epsilon(self.net, xt, sigma).detach()

            lam = self.lambda_fn(sigma)
            optimizer.zero_grad()

            gradient, loss_scale = self.forward_op.gradient(
                mu, observation, return_loss=True
            )
            gradient = (gradient * self.observation_weight
                        + lam * (pred_epsilon - epsilon))
            mu.grad = gradient

            optimizer.step()
            pbar.set_description(
                f'Iteration {step + 1}/{num_steps}. '
                f'Data fitting loss: {torch.sqrt(loss_scale)}'
            )
        return mu
```
