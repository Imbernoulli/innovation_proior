Two of the strongest "perturb-then-reverse" generative families sit on my desk doing what looks like the same thing in two different costumes. The noise-conditional score network adds Gaussian noise at a ladder of standard deviations $\sigma_1 < \cdots < \sigma_N$, trains one network to estimate $\nabla_x \log p_\sigma$ at every level, and samples by annealed Langevin dynamics walking down the ladder. The diffusion probabilistic model runs a discrete Markov chain $x_i = \sqrt{1-\beta_i}\,x_{i-1} + \sqrt{\beta_i}\,z$, fits a variational reverse chain by a reweighted ELBO, and samples ancestrally. Different forward process, different loss, different sampler — and yet both losses, written out, are weighted sums of denoising score matching terms whose optimum is the score of the noised data, the first weighted by $\sigma_i^2$ and the second by $1-\alpha_i$, each of which is proportional to one over the expected squared norm of the conditional score. That is too much coincidence: they are learning the same object. The real question is therefore not which costume wins but what the thing underneath is, and whether the discreteness — the hand-picked ladder, the per-scale loss, the bespoke sampler — is load-bearing or merely an artifact of being forced to write everything as a finite chain. The arbitrariness all lives in that discreteness: why $N$ scales, why geometric spacing for one and a near-arithmetic $\beta$ schedule for the other, how to interpolate when I want more sampling steps. Score-based models also give no exact likelihood, unlike normalizing flows, and every conditional task today needs a freshly retrained conditional model. I want a single framework that bridges data to a tractable prior without any chosen ladder, unifies the two samplers, hands me an exact likelihood, and conditions from one unconditional model.

I propose to throw away the ladder and let the noise grow continuously. Index everything by a real time $t$ and let the perturbed distribution be a continuum $p_t$ that slides from the data at $t=0$ to a simple prior at $t=T$. A continuous-time process driven by noise is a stochastic differential equation, so I write the forward corruption in Itô form
$$\mathrm{d}x = f(x,t)\,\mathrm{d}t + g(t)\,\mathrm{d}w,$$
with $w$ standard Brownian motion, $f$ an affine drift, $g$ a scalar diffusion coefficient, and — crucially — no trainable parameters at all: it is a fixed prescription chosen so that the marginal $p_T$ is something I can sample trivially. Creating noise from data is the easy direction. The hard direction is reversing it, and my instinct says noise is irreversible. But Anderson's 1982 result says otherwise: a diffusion run backward in time is still a diffusion,
$$\mathrm{d}x = \left[\,f(x,t) - g(t)^2\,\nabla_x \log p_t(x)\,\right]\mathrm{d}t + g(t)\,\mathrm{d}\bar{w},$$
with $\bar{w}$ a reverse-time Wiener process and $\mathrm{d}t$ an infinitesimal negative step. The diffusion coefficient is unchanged and the drift picks up exactly one new term, $-g^2\,\nabla_x \log p_t(x)$, the *score* of the marginal at time $t$ — the only ingredient that depends on the data. So the whole problem collapses to estimating $\nabla_x \log p_t(x)$ for all $x$ and $t$; with that function I plug into the reverse SDE and integrate from $T$ back to $0$. Intuitively the score un-stirs because $\nabla \log p_t$ points uphill toward higher density, toward the data manifold, and since $\mathrm{d}t$ is negative the contribution $(-g^2\nabla\log p_t)\mathrm{d}t$ is really an uphill push whose strength matches the forward smearing. I call the whole construction Score SDE.

To get the score I train a network $s_\theta(x,t)$ by score matching. The naive Fisher divergence $\mathbb{E}_{p_t}\|s_\theta - \nabla\log p_t\|^2$ still hides the intractable marginal score, but Vincent's denoising identity generalizes to continuous time verbatim: minimizing against the marginal score is equivalent, up to a $\theta$-free constant, to regressing against the *conditional* score of the transition kernel,
$$\theta^* = \arg\min_\theta\ \mathbb{E}_t\Big\{\lambda(t)\,\mathbb{E}_{x(0)}\,\mathbb{E}_{x(t)\mid x(0)}\big\|s_\theta(x(t),t) - \nabla_{x(t)}\log p_{0t}(x(t)\mid x(0))\big\|^2\Big\},$$
with $t\sim \mathcal{U}(0,T)$. The swap is legitimate because $\nabla\log p_t(\tilde x) = \mathbb{E}_{x(0)\mid\tilde x}[\nabla\log p_{0t}(\tilde x\mid x(0))]$, so taking $\mathbb{E}_{p_t}$ of the cross term cancels $p_t$ and turns the marginal score into the conditional one averaged over the joint. The conditional score is trivial when the kernel is Gaussian: if $p_{0t}(x(t)\mid x(0)) = \mathcal{N}(\mu_t x(0),\sigma_t^2 I)$ then $x(t) = \mu_t x(0) + \sigma_t z$ and $\nabla_{x(t)}\log p_{0t} = -z/\sigma_t$, just minus the scaled noise I injected. The kernel is Gaussian exactly when the drift is affine, in which case the mean and covariance of $p_{0t}$ obey linear ODEs with closed-form solutions — so I restrict to affine drift and never simulate the forward process to train. The weight $\lambda(t)$ matters because the target $-z/\sigma_t$ has wildly different magnitudes across $t$; choosing $\lambda(t)\propto 1/\mathbb{E}\|\nabla\log p_{0t}\|^2 = \sigma_t^2$ makes the per-time loss $\mathbb{E}\|\sigma_t s_\theta + z\|^2$, order one at every $t$. This is exactly the $\sigma_i^2$ and $1-\alpha_i$ weights of the two discrete models, now explained: scale-invariance in $t$.

Which SDE? Deriving the two I already have in costume tells me. The additive-noise ladder, realized as a noise-only Markov chain $x_i = x_{i-1} + \sqrt{\sigma_i^2-\sigma_{i-1}^2}\,z$ and taken to the continuous limit, becomes $\mathrm{d}x = \sqrt{\mathrm{d}[\sigma^2(t)]/\mathrm{d}t}\,\mathrm{d}w$ with no drift; the conditional variance grows without bound, so I call it the Variance Exploding SDE with prior $\mathcal{N}(0,\sigma_{\max}^2 I)$. For the geometric schedule $\sigma(t)=\sigma_{\min}(\sigma_{\max}/\sigma_{\min})^t$ the diffusion is $g(t)=\sigma(t)\sqrt{2\log(\sigma_{\max}/\sigma_{\min})}$. The signal-decay chain, with $\beta_i$ rescaled by the step size and Taylor-expanding $\sqrt{1-\beta\Delta t}\approx 1-\tfrac12\beta\Delta t$, becomes $\mathrm{d}x = -\tfrac12\beta(t)x\,\mathrm{d}t + \sqrt{\beta(t)}\,\mathrm{d}w$. The affine drift gives the variance ODE $\mathrm{d}\Sigma/\mathrm{d}t = \beta(t)(I-\Sigma)$, whose solution $\Sigma(t)=I+e^{-\int_0^t\beta}(\Sigma(0)-I)$ is bounded and stays exactly $I$ if started at $I$ — the Variance Preserving SDE with prior $\mathcal{N}(0,I)$ and kernel $\mathcal{N}(x(0)e^{-\frac12\int\beta},(1-e^{-\int\beta})I)$. Both continuous kernels track their discrete originals at $N=1000$ essentially exactly, so the unification is an identity in the limit, not a hand-wave. Freed from matching an existing model, I can now invent a third SDE: keep the VP drift so the mean decays identically, but shrink the diffusion to inject less noise. Take $\mathrm{d}x = -\tfrac12\beta(t)x\,\mathrm{d}t + \sqrt{\beta(t)(1-e^{-2\int_0^t\beta})}\,\mathrm{d}w$. With $h(t)=\int_0^t\beta$ the perturbation-kernel covariance solves $\mathrm{d}\Sigma/\mathrm{d}t = -\beta\Sigma + \beta(1-e^{-2h})$; the integrating factor $e^h$ gives $\Sigma e^h = e^h+e^{-h}+C$, and $\Sigma(0)=0$ fixes $C=-2$, so $\Sigma(t)=[1-e^{-h}]^2$. Since $1-e^{-h}\in[0,1]$, this is $\le$ the VP variance $1-e^{-h}$ at every $t$ with equality only at the endpoints, and it still tends to $I$. Less injected noise, same mean dynamics, same endpoint; I call it the sub-VP SDE. This settles a code convention too: VP's `marginal_prob` returns $\sqrt{1-e^{-h}}$ because its variance is $1-e^{-h}$, while sub-VP returns $1-e^{-h}$ with no square root because its variance is already $[1-e^{-h}]^2$.

For sampling I could throw a black-box solver at the reverse SDE, but I have two pieces of information a generic solver lacks. First, I have the score $s_\theta\approx\nabla\log p_t$ itself, so at any time slice I can run Langevin dynamics $x \leftarrow x + \varepsilon s_\theta(x,t) + \sqrt{2\varepsilon}\,z$ to sample $p_t$ directly. So I alternate a numerical reverse-SDE step (the *predictor*, advancing $t\to t-\Delta t$) with a few Langevin steps (the *corrector*, re-projecting onto $p_{t-\Delta t}$): a Predictor–Corrector sampler. The old samplers are its degenerate corners — annealed Langevin is corrector-only with identity predictor, ancestral diffusion is predictor-only with identity corrector — so PC contains both and lets me spend compute on whichever side helps. The Langevin step is set from a target signal-to-noise ratio $r$ via $\varepsilon = 2\alpha(r\|z\|/\|s_\theta\|)^2$ so it stays stable across scales, and the predictor mirrors the forward discretization automatically ($x_i = x_{i+1} - f_{i+1} + G_{i+1}G_{i+1}^\top s_\theta + G_{i+1}z$); ancestral sampling is just a particular discretization of the reverse VP SDE. A single final denoising step without re-adding noise removes the residual high-frequency noise that is invisible to the eye but poison to FID. Second, the reverse need not be stochastic at all. The Fokker–Planck equation $\partial_t p_t = -\sum_i\partial_{x_i}[f_i p_t] + \tfrac12\sum_{ij}\partial^2_{x_i x_j}[(GG^\top)_{ij}p_t]$ can be folded into a single divergence by pulling one derivative out and using $\partial_{x_j}p_t = p_t\,\partial_{x_j}\log p_t$, turning the second-order term into $\tfrac12\sum_i\partial_{x_i}[p_t(\nabla\cdot[GG^\top]+GG^\top\nabla\log p_t)_i]$. The whole right-hand side becomes a continuity equation for the deterministic flow
$$\mathrm{d}x = \left[\,f(x,t) - \tfrac12\,g(t)^2\,\nabla_x \log p_t(x)\,\right]\mathrm{d}t,$$
which I call the probability flow ODE: same form as the reverse SDE drift but half the score coefficient and no noise — the factor of $\tfrac12$ is exactly what the algebra produces. It carries the same marginals deterministically, so it is a neural ODE, and that hands me exact likelihood via the instantaneous change of variables $\log p_0(x(0)) = \log p_T(x(T)) + \int_0^T \nabla\cdot\tilde f_\theta\,\mathrm{d}t$, with the divergence estimated by the Skilling–Hutchinson trace estimator $\nabla\cdot\tilde f_\theta = \mathbb{E}_v[v^\top\nabla\tilde f_\theta\,v]$, one vector-Jacobian product per evaluation, unbiased. It also gives uniquely identifiable latents (the forward process has no parameters), latent manipulation, and fast adaptive black-box sampling with RK45. Finally, conditioning drops out of Bayes on the marginals: the conditional reverse SDE uses $\nabla\log p_t(x\mid y) = \nabla\log p_t(x) + \nabla\log p_t(y\mid x)$, the trained unconditional score plus a cheap conditional term — a time-dependent classifier for class-conditional generation, or an unconditional-model approximation for inpainting and colorization — so one unconditional model serves every task with no per-task retraining.

```python
import abc
import numpy as np
import torch
from scipy import integrate

class SDE(abc.ABC):
    def __init__(self, N): self.N = N
    @property
    @abc.abstractmethod
    def T(self): ...
    @abc.abstractmethod
    def sde(self, x, t): ...
    @abc.abstractmethod
    def marginal_prob(self, x, t): ...
    @abc.abstractmethod
    def prior_sampling(self, shape): ...
    @abc.abstractmethod
    def prior_logp(self, z): ...

    def discretize(self, x, t):
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        return drift * dt, diffusion * torch.sqrt(torch.tensor(dt, device=t.device))

    def reverse(self, score_fn, probability_flow=False):
        N, T = self.N, self.T
        sde_fn, discretize_fn = self.sde, self.discretize
        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow
            @property
            def T(self):
                return T
            def sde(self, x, t):
                drift, diffusion = sde_fn(x, t)
                score = score_fn(x, t)
                drift = drift - diffusion[:, None, None, None] ** 2 * score \
                        * (0.5 if self.probability_flow else 1.)
                diffusion = 0. if self.probability_flow else diffusion
                return drift, diffusion
            def discretize(self, x, t):
                f, G = discretize_fn(x, t)
                score = score_fn(x, t)
                rev_f = f - G[:, None, None, None] ** 2 * score \
                        * (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G
        return RSDE()

class VPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        super().__init__(N)
        self.beta_0, self.beta_1 = beta_min, beta_max
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    @property
    def T(self): return 1
    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        return -0.5 * beta_t[:, None, None, None] * x, torch.sqrt(beta_t)
    def marginal_prob(self, x, t):
        log_mean = -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean[:, None, None, None]) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean))
        return mean, std
    def prior_sampling(self, shape): return torch.randn(*shape)
    def prior_logp(self, z):
        n = np.prod(z.shape[1:])
        return -n / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
    def discretize(self, x, t):
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        return torch.sqrt(alpha)[:, None, None, None] * x - x, torch.sqrt(beta)

class VESDE(SDE):
    def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
        super().__init__(N)
        self.sigma_min, self.sigma_max = sigma_min, sigma_max
        self.discrete_sigmas = torch.exp(torch.linspace(np.log(sigma_min), np.log(sigma_max), N))
    @property
    def T(self): return 1
    def sde(self, x, t):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        diffusion = sigma * torch.sqrt(torch.tensor(
            2 * (np.log(self.sigma_max) - np.log(self.sigma_min)), device=t.device))
        return torch.zeros_like(x), diffusion
    def marginal_prob(self, x, t):
        return x, self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    def prior_sampling(self, shape): return torch.randn(*shape) * self.sigma_max
    def prior_logp(self, z):
        n = np.prod(z.shape[1:])
        return -n / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(
            z ** 2, dim=(1, 2, 3)) / (2 * self.sigma_max ** 2)
    def discretize(self, x, t):
        timestep = (t * (self.N - 1) / self.T).long()
        sigma = self.discrete_sigmas.to(t.device)[timestep]
        adjacent = torch.where(
            timestep == 0, torch.zeros_like(t), self.discrete_sigmas.to(t.device)[timestep - 1])
        return torch.zeros_like(x), torch.sqrt(sigma ** 2 - adjacent ** 2)

class subVPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        super().__init__(N)
        self.beta_0, self.beta_1 = beta_min, beta_max
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas
    @property
    def T(self): return 1
    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        discount = 1. - torch.exp(-2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2)
        return -0.5 * beta_t[:, None, None, None] * x, torch.sqrt(beta_t * discount)
    def marginal_prob(self, x, t):
        log_mean = -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean)[:, None, None, None] * x
        # The kernel variance is (1 - exp(-int beta))^2, so this is the std.
        std = 1 - torch.exp(2. * log_mean)
        return mean, std
    def prior_sampling(self, shape): return torch.randn(*shape)
    def prior_logp(self, z):
        n = np.prod(z.shape[1:])
        return -n / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.

def get_loss_fn(sde, eps=1e-5, likelihood_weighting=False):
    def loss_fn(model, batch):
        t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
        z = torch.randn_like(batch)
        mean, std = sde.marginal_prob(batch, t)
        x_t = mean + std[:, None, None, None] * z
        score = model(x_t, t)
        if likelihood_weighting:
            g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
            losses = torch.square(score + z / std[:, None, None, None])
            losses = losses.reshape(losses.shape[0], -1).sum(dim=-1) * g2
        else:
            losses = torch.square(score * std[:, None, None, None] + z)
            losses = losses.reshape(losses.shape[0], -1).sum(dim=-1)
        return torch.mean(losses)
    return loss_fn

def reverse_diffusion_predictor(rsde, x, t):
    f, G = rsde.discretize(x, t)
    x_mean = x - f
    x = x_mean + G[:, None, None, None] * torch.randn_like(x)
    return x, x_mean

def langevin_corrector(score_fn, sde, x, t, snr, n_steps):
    if hasattr(sde, "alphas"):
        timestep = (t * (sde.N - 1) / sde.T).long()
        alpha = sde.alphas.to(t.device)[timestep]
    else:
        alpha = torch.ones_like(t)
    for _ in range(n_steps):
        grad, noise = score_fn(x, t), torch.randn_like(x)
        gn = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
        nn = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        step = (snr * nn / gn) ** 2 * 2 * alpha
        x_mean = x + step[:, None, None, None] * grad
        x = x_mean + torch.sqrt(step * 2)[:, None, None, None] * noise
    return x, x_mean

def pc_sample(model, sde, shape, snr=0.16, n_steps=1, eps=1e-3, device='cpu'):
    score_fn = lambda x, t: model(x, t)
    rsde = sde.reverse(score_fn)
    x = sde.prior_sampling(shape).to(device)
    timesteps = torch.linspace(1, eps, sde.N, device=device)
    for i in range(sde.N):
        vec_t = torch.ones(shape[0], device=device) * timesteps[i]
        x, x_mean = langevin_corrector(score_fn, sde, x, vec_t, snr, n_steps)
        x, x_mean = reverse_diffusion_predictor(rsde, x, vec_t)
    return x_mean                                                            # final denoising

def ode_sample(model, sde, shape, eps=1e-3, device='cpu'):
    rsde = sde.reverse(lambda x, t: model(x, t), probability_flow=True)
    def ode_func(t, xf):
        x = torch.tensor(xf.reshape(shape), dtype=torch.float32, device=device)
        vec_t = torch.ones(shape[0], device=device) * t
        return rsde.sde(x, vec_t)[0].detach().cpu().numpy().reshape(-1)
    x0 = sde.prior_sampling(shape).cpu().numpy().reshape(-1)
    sol = integrate.solve_ivp(ode_func, (1, eps), x0, rtol=1e-5, atol=1e-5, method='RK45')
    return torch.tensor(sol.y[:, -1].reshape(shape))

def divergence_fn(fn):
    def div(x, t, noise):
        with torch.enable_grad():
            x.requires_grad_(True)
            y = torch.sum(fn(x, t) * noise)
            grad = torch.autograd.grad(y, x)[0]
        x.requires_grad_(False)
        return torch.sum(grad * noise, dim=tuple(range(1, len(x.shape))))
    return div
```
