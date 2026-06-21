The setting is the recurring shape of scientific measurement: an unknown signal $x_0$ — a face, a brain slice, a permittivity map — seen only through a degraded, noisy observation $y = f(x_0) + v$ with $v \sim \mathcal{N}(0, \sigma_v^2 I)$ and a known forward operator $f$ that may be linear (a mask, a blur, a subsampled Fourier transform) or genuinely nonlinear (clipping, phase loss, scattering). The map is many-to-one, so recovery is ill-posed and demands a strong prior on what real signals look like. In Bayesian terms I want to sample $p(x_0 \mid y) \propto p(y \mid x_0)\, p(x_0)$, where the likelihood $p(y \mid x_0) = \mathcal{N}(f(x_0), \sigma_v^2 I)$ is handed to me by the noise model. The prior I already own, and own in abundance: a pretrained denoising diffusion model for the signal class, a network $\epsilon_\theta(x_t; t)$ I never want to retrain. The prize is a plug-and-play sampler that takes any such frozen prior and any operator $f$ and produces posterior samples, universally across linear and nonlinear tasks, cheaply enough for a single GPU, with a small number of interpretable knobs.

The difficulty is entirely in how the diffusion model delivers its prior — as a score along a noising trajectory $x_t = \alpha_t x_0 + \sigma_t \epsilon$, not as a density I can evaluate at $x_0$. To run reverse diffusion conditioned on $y$ I would need the conditional score at every noise level, which Bayes splits cleanly into $\nabla_{x_t} \log p(x_t \mid y) = \nabla_{x_t} \log p(y \mid x_t) + \nabla_{x_t} \log p(x_t)$. The prior term is free, it is $-\epsilon_\theta(x_t;t)/\sigma_t$. The likelihood term is the wall, because the quantity I need is $p(y \mid x_t) = \int p(y \mid x_0)\, p(x_0 \mid x_t)\, dx_0$, and the denoising posterior $p(x_0 \mid x_t)$ is in general a complicated multimodal distribution — a noisy $x_t$ could have come from many clean images — so $p(y \mid x_t)$ has no closed form and its gradient cannot be estimated without task-specific training. Every existing plug-and-play sampler is a different way of coping with this one intractable term, and each pays. DPS replaces the whole denoising posterior by a point mass at its Tweedie mean $\hat x_0 = E[x_0 \mid x_t] = (x_t - \sigma_t \epsilon_\theta)/\alpha_t$, so $p(y \mid x_t) \approx p(y \mid \hat x_0)$ and the guidance becomes the analytic $-(1/\sigma_v^2)\nabla_{x_t}\|y - f(\hat x_0(x_t))\|^2$; this is general and handles nonlinear $f$, but the point estimate is least reliable exactly when $p(x_0 \mid x_t)$ is broad and multimodal — the regime where intermediate steps decide global structure — and, operationally worse, because $\hat x_0$ flows through the network the guidance gradient forces a backprop through the denoiser at every step, a score-Jacobian that is memory-hungry, slow, unstable, and touchy about schedule and initialization. $\Pi$GDM sharpens the guidance by treating $p(x_0 \mid x_t)$ as Gaussian and folding in $f$'s pseudoinverse, but it buys that sharpness by specializing to linear and semi-linear operators, so general nonlinear $f$ is out, and it still rests on a unimodal posterior-score approximation that it differentiates through the model. Both leading methods do the same risky thing: approximate the intractable trajectory score, then differentiate the frozen network.

I propose RED-diff. The move is to stop approximating the trajectory score altogether and instead pose inference directly on the clean signal $x_0$. I have a frozen prior $p(x_0)$ and a known likelihood $p(y \mid x_0)$; the only obstacle to the posterior is the normalizer, which is the textbook setting for variational inference. So I posit a tractable Gaussian family $q(x_0 \mid y) = \mathcal{N}(\mu, \sigma^2 I)$ and fit it by minimizing $\mathrm{KL}(q(x_0 \mid y)\,\|\,p(x_0 \mid y))$. The mode-seeking nature of this forward KL is a feature here: for an ill-posed problem I want one plausible reconstruction that is both consistent with $y$ and high-probability under the prior, not a hedge over modes. Bayes-expanding $p(x_0 \mid y) = p(y \mid x_0)\,p(x_0)/p(y)$ gives

$$\mathrm{KL}(q(x_0 \mid y)\,\|\,p(x_0 \mid y)) = -E_q[\log p(y \mid x_0)] + \mathrm{KL}(q(x_0 \mid y)\,\|\,p(x_0)) + \log p(y),$$

where $\log p(y)$ is constant in $\mu, \sigma$ and drops out. The first term, under Gaussian noise, is just a reconstruction loss $\tfrac{1}{2\sigma_v^2}E_q[\|y - f(x_0)\|^2]$ pulling $\mu$ toward signals that fit $y$. The second term, $\mathrm{KL}(q \,\|\, p(x_0))$, is where the prior lives and where intractability appears to have only relocated, since $p(x_0)$ is the diffusion prior I cannot evaluate as a density.

The maximum-likelihood theory of diffusion supplies exactly the missing representation. A KL between two distributions each diffused by the same forward SDE equals a time-integrated, weighted score-matching loss along the trajectory (Song et al. 2021; the latent cross-entropy version, Vahdat et al. 2021),

$$\mathrm{KL}(q(x_0 \mid y)\,\|\,p(x_0)) = \int_0^T \frac{\beta(t)}{2}\, E_{q(x_t \mid y)}\!\left[\,\|\nabla_{x_t}\log q(x_t \mid y) - \nabla_{x_t}\log p(x_t)\|^2\,\right] dt,$$

and now every piece is computable: $\nabla_{x_t}\log p(x_t) = -\epsilon_\theta(x_t;t)/\sigma_t$ is the frozen net, and $q(x_t \mid y)$, the diffused Gaussian, is itself Gaussian, $\mathcal{N}(\alpha_t \mu, (\alpha_t^2\sigma^2 + \sigma_t^2)I)$, with closed-form score. Taking $\sigma \to 0$ — a Dirac $q = \delta(x_0 - \mu)$, deferring the question of whether dispersion buys anything — collapses $q(x_t \mid y)$ to $\mathcal{N}(\alpha_t \mu, \sigma_t^2 I)$ with score $-\epsilon/\sigma_t$, so the score difference is $(\epsilon_\theta(x_t;t) - \epsilon)/\sigma_t$ and the objective becomes a measurement-fitting term plus an expected squared noise residual,

$$\min_\mu \;\|y - f(\mu)\|^2 + E_{t,\epsilon}\!\left[\,2\,\omega(t)\,(\sigma_v/\sigma_t)^2\,\|\epsilon_\theta(x_t;t) - \epsilon\|^2\,\right], \qquad x_t = \alpha_t \mu + \sigma_t \epsilon.$$

This finds an image $\mu$ that reconstructs $y$ through $f$ while demanding the denoiser's predicted noise match the injected noise across the trajectory, and its ensemble-over-steps structure begs for stochastic optimization. But there is a trap: differentiating $\|\epsilon_\theta(\alpha_t\mu + \sigma_t\epsilon;t) - \epsilon\|^2$ naively runs through $\epsilon_\theta$ and reinvents exactly the score-Jacobian I came to escape. The whole method lives or dies on getting this gradient without backpropagating through the network.

What rescues it is the companion identity $\,d\,\mathrm{KL}(q(x_t\mid y)\,\|\,p(x_t))/dt = -\tfrac{\beta(t)}{2}E[\,\|\nabla\log q - \nabla\log p\|^2\,]\,$, which lets me reweight the integral by an arbitrary $\omega(t)$ and integrate by parts: $-\int_0^T \omega(t)\,(d\,\mathrm{KL}_t/dt)\,dt = -[\omega(t)\,\mathrm{KL}_t]_0^T + \int_0^T \omega'(t)\,\mathrm{KL}_t\,dt$. At $t = T$ the forward process has destroyed all signal so $q(x_T \mid y) = p(x_T)$ and $\mathrm{KL}_T = 0$; the $t=0$ end of the boundary term dies only if I demand $\omega(0) = 0$. With any such weighting the boundary vanishes, and differentiating the surviving integral with $\sigma = 0$, the reparameterization $x_t = \alpha_t\mu + \sigma_t\epsilon$, $dx_t/d\mu = \alpha_t I$, and the two scores substituted in gives

$$\nabla_\mu\,\mathrm{reg} = E_{t \sim U[0,T],\,\epsilon \sim \mathcal{N}(0,I)}\!\left[\,\lambda_t\,(\epsilon_\theta(x_t;t) - \epsilon)\,\right], \qquad \lambda_t := 2T\sigma_v^2\,(\alpha_t/\sigma_t)\,\omega'(t).$$

Here is the entire payoff: because the gradient came out of the difference of two scores, $\epsilon_\theta$ enters as a stop-gradient *value*, not as something I differentiate. The variational regularizer's gradient costs one forward pass of the frozen denoiser per step and no Jacobian, where DPS and $\Pi$GDM need a vector-Jacobian product. The seemingly innocuous condition $\omega(0)=0$ is the structural linchpin — without it the integration by parts leaves a $\mathrm{KL}_0$ term that reintroduces the very dependence I am avoiding. (Since $E[\epsilon]=0$ the $-\epsilon$ term vanishes in expectation, but I keep it per-sample as a zero-mean control variate.) Equivalently I can read off a per-step linear surrogate $\|y - f(\mu)\|^2 + \lambda_t\,(\mathrm{sg}[\epsilon_\theta(x_t;t) - \epsilon])^\top \mu$, whose gradient reproduces the above and makes explicit that the residual is constant with respect to $\mu$.

That residual structure is the signature of Regularization by Denoising: a denoiser-based penalty whose gradient is a plain residual with no differentiation of the denoiser — RED's $x - f_{\text{den}}(x)$, here $\epsilon_\theta(x_t;t) - \epsilon$ — and the regularizer is small in the two RED regimes, when the diffusion reaches its fixed point $\epsilon_\theta = \epsilon$ or when the residual is pure noise orthogonal to the signal. Hence the name. But this is RED made generative: classical RED uses a single deterministic denoiser at one noise level with no noise injected, a fixed-point/MAP procedure, whereas here fresh noise is added to the input of every denoiser across the entire trajectory and their feedback is aggregated, so the iterates can actually navigate toward the prior's high-density region using high-noise denoisers for coarse structure and low-noise denoisers for detail.

The weighting needs care, because the noise residual $\epsilon_\theta(x_t;t) - \epsilon$ blows up as $t \to 0$ (the signal-to-noise ratio climbs faster than the residual shrinks), so equal weighting lets tiny-noise steps dominate and destabilize. The principled fix recasts the regularizer in the signal domain, where it is directly comparable to $\|y - f(\mu)\|^2$. Tweedie gives the MMSE estimate $\hat\mu_t = E[x_0 \mid x_t] = (x_t - \sigma_t\epsilon_\theta)/\alpha_t$, and substituting $x_t = \alpha_t\mu + \sigma_t\epsilon$ yields the exact identity

$$\mu - \hat\mu_t = \frac{\sigma_t}{\alpha_t}\,(\epsilon_\theta(x_t;t) - \epsilon),$$

so the signal residual is the noise residual times $\sigma_t/\alpha_t = 1/\mathrm{SNR}_t$ with $\mathrm{SNR}_t := \alpha_t/\sigma_t$. To express the noise-domain gradient $\lambda_t(\epsilon_\theta - \epsilon)$ as a signal-domain residual $\lambda(\mu - \hat\mu_t)$ with a single constant $\lambda$, I set $\lambda_t = \lambda/\mathrm{SNR}_t = \lambda\,(\sigma_t/\alpha_t)$. This $1/\mathrm{SNR}$ weighting is not a hack but the exact rescaling that puts the regularizer in the units of the fitting term and collapses the small-$t$ blow-up; it also upweights the high-noise early steps that build coarse semantic structure and downweights the low-noise late steps that only add detail, so a single $\lambda$ controls the prior/likelihood trade-off (larger leans on the prior, smaller on the data) and the right ordering is to step $t$ descending from $T$ to $0$, laying down structure before refining. As for the dispersion $\sigma$ I discarded: keeping it is cheap — with $\eta_t := (1 + \sigma^2(\alpha_t/\sigma_t)^2)^{1/2}$ the mean gradient is unchanged in expectation since $E[\epsilon]=0$, and the $\sigma$-gradient is closed form — but I keep $\sigma = 0$ deliberately, because isotropic Gaussian perturbation of an image does not move it to another legitimate image, it just pushes it off the natural-image manifold; diversity comes instead from the random $\epsilon$ draws and the optimizer's stochasticity.

Sampling has thus become stochastic optimization: initialize $\mu$ (zeros, or a pseudo-inverse warm start), and each step sample $t$ and $\epsilon$, form $x_t$, run the denoiser once for $\epsilon_\theta$, add the data-fitting gradient to $\lambda_t(\epsilon_\theta - \epsilon)$, and take an Adam step. Adam is the natural default — its per-coordinate adaptive steps absorb the very different scales of the $f$-dependent fitting gradient and the residual; I use $\beta = (0.9, 0.99)$, a slightly shorter second-moment window than the usual $0.999$ because the objective shifts as $t$ descends, and no weight decay, since the diffusion regularizer *is* the prior and there is no shrinkage on $\mu$.

```python
import torch
import tqdm
from .base import Algo
import wandb
from utils.scheduler import Scheduler


class REDDiff(Algo):
    def __init__(self, net, forward_op, num_steps=1000, observation_weight=1.0,
                 base_lambda=0.25, base_lr=0.5, lambda_scheduling_type='constant'):
        super(REDDiff, self).__init__(net, forward_op)
        self.net = net
        self.net.eval().requires_grad_(False)
        self.forward_op = forward_op
        self.scheduler = Scheduler(num_steps=num_steps, schedule='vp',
                                   timestep='vp', scaling='vp')
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

        mu = torch.zeros(num_samples, self.net.img_channels,
                         self.net.img_resolution, self.net.img_resolution,
                         device=device).requires_grad_(True)
        optimizer = torch.optim.Adam([mu], lr=self.base_lr, betas=(0.9, 0.99))

        for step in pbar:
            with torch.no_grad():
                sigma, scaling = self.scheduler.sigma_steps[step], self.scheduler.scaling_steps[step]
                epsilon = torch.randn_like(mu)
                xt = scaling * (mu + sigma * epsilon)
                pred_epsilon = self.pred_epsilon(self.net, xt, sigma).detach()

            lam = self.lambda_fn(sigma)  # sigma equals 1/SNR; the linear option gives lambda/SNR_t.
            optimizer.zero_grad()

            gradient, loss_scale = self.forward_op.gradient(mu, observation,
                                                            return_loss=True)
            gradient = gradient * self.observation_weight + lam * (pred_epsilon - epsilon)
            mu.grad = gradient

            optimizer.step()
            pbar.set_description(
                f'Iteration {step + 1}/{num_steps}. '
                f'Data fitting loss: {torch.sqrt(loss_scale)}')
            if wandb.run is not None:
                wandb.log({'data_fitting_loss': torch.sqrt(loss_scale)}, step=step)
        return mu
```
