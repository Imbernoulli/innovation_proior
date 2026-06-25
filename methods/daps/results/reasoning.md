Let me start from the one structural fact every training-free posterior sampler keeps tripping over. I have a pretrained denoiser `D_theta(x_t, sigma) = E[x_0 | x_t]` that gives me the prior's score, and a known likelihood `p(y | x_0) = N(A(x_0), sigma_y^2 I)`, and I want `x_0 ~ p(x_0 | y)`. To run the reverse diffusion conditioned on `y` I need the time-level posterior score, which Bayes splits into the free prior score plus the likelihood score `∇_{x_t} log p_t(y | x_t)`. That term is intractable because `p_t(y | x_t) = E_{x_0 ~ p(x_0|x_t)}[ p(y | x_0) ]` involves the multimodal denoising posterior `p(x_0 | x_t)`. DPS approximates the expectation by the likelihood at the Tweedie mean; LGD spreads the mean into a Gaussian surrogate and Monte-Carlos it; ΠGDM inverts `A` through a pseudoinverse. All of them are the same shape: approximate a guidance term and inject it into a single reverse step on `x_t`. I want to ask whether that shape is the actual problem, rather than the particular approximation inside it.

Here is the thing that has been bothering me about the shape. The prior's score is a function of the *noisy* latent `x_t` — that is what the diffusion model was trained to know. The likelihood `p(y | x_0)` is a function of the *clean* signal `x_0` — that is where the forward operator lives. When DPS or LGD bend a single reverse step, they are forcing a clean-variable correction and a noisy-variable dynamics into the same update on `x_t`, via `x_0_hat = D_theta(x_t)` and a backprop through the network. The two demands are not the same demand. The likelihood wants `A(x_0) ≈ y` — a hard constraint on the clean signal. The prior wants `x_t` to follow the reverse dynamics — a soft pull on the noisy latent. Jamming them together is why the guidance scale has to be retuned per operator and per noise level, and why a nonlinear operator, whose residual gradient is wild, destabilizes the whole reverse trajectory: a bad guidance step at noise level `t` corrupts `x_t`, which corrupts every later step, because the steps are *coupled* — each one inherits the last one's `x_t`. The error does not stay local; it propagates down the trajectory.

So let me try the opposite move: *decouple* the steps. What if consecutive points along the sampling trajectory are allowed to differ a lot from one another, instead of each being a small perturbation of the last? Concretely, what if at each noise level I do not carry `x_t` forward by a small reverse step, but instead (a) extract a *clean* estimate from `x_t` using the prior, (b) correct that clean estimate against the likelihood as hard as I want, on the clean variable where the likelihood actually lives, and then (c) re-noise the corrected clean estimate to the *next, lower* noise level to get the next `x_t`? Then the likelihood correction never touches the noisy-latent dynamics, and a bad correction at one level does not poison the latent that feeds the next level, because the next `x_t` is freshly re-noised from a corrected clean estimate. The trajectory is decoupled: each `x_t` is generated anew from a clean correction, not inherited as a small step.

Let me make the three pieces precise. Piece (a), the clean estimate. I could use the one-step Tweedie mean `D_theta(x_t, sigma)`, which is what DPS uses. But at high noise the one-step mean is a blurry average over the whole posterior — it throws away the prior's ability to resolve structure. Since I have decoupled the steps, I can afford to spend more of the prior here: solve the *unconditional* probability-flow ODE starting at `x_t`, running the prior's own reverse dynamics from `sigma` down to clean, to get a sharp sample `x_0_hat(x_t)` of `p(x_0 | x_t)`. This is the prior update, and it is unconditional — it knows nothing about `y`. It is a sub-sampler with its own (small) number of function evaluations, and because the outer loop is decoupled I do not need it to be exact; a handful of Euler ODE steps suffices to turn `x_t` into a clean estimate that respects the prior far better than a single Tweedie call.

Piece (b), the likelihood correction. I now have `x_0_hat`, the prior's clean estimate, and I want to pull it toward agreeing with `y` — but not collapse it to a single point, because I want a *sampler*, not RED-diff's mode-seeking optimization. So sample from the conditional `p(x_0 | x_t, y)`. Write its score. By Bayes, `p(x_0 | x_t, y) ∝ p(x_0 | x_t) p(y | x_0)`. The first factor `p(x_0 | x_t)` is the denoising posterior; I approximate it as Gaussian centered at the prior's clean estimate, `p(x_0 | x_t) ≈ N(x_0_hat, r_t^2 I)`, with `r_t` the posterior width — the same conjugate-balance reasoning that gives `r_t ≈ sigma_t` (at the level's noise scale, the posterior std tracks the noise). The second factor is the Gaussian likelihood. So
`log p(x_0 | x_t, y) = − ||x_0 − x_0_hat||^2 / (2 r_t^2) − ||A(x_0) − y||^2 / (2 sigma_y^2) + const`,
and its gradient in `x_0` is `−(x_0 − x_0_hat)/r_t^2 − ∇_{x_0} ||A(x_0) − y||^2 / (2 sigma_y^2)`. This is a *clean-variable* score — `A` is evaluated at `x_0` directly, not through the denoiser — so there is no network Jacobian anywhere in the correction, and a nonlinear `A` is no harder than a linear one as long as I can differentiate it.

Now sample that conditional. The score is all I need for Langevin dynamics: iterate `x_0 ← x_0 + eta ∇_{x_0} log p(x_0 | x_t, y) + sqrt(2 eta) z`, `z ~ N(0, I)`, for `N` steps. Written out with the two terms,
`x_0^{(j+1)} = x_0^{(j)} − eta (x_0^{(j)} − x_0_hat)/r_t^2 − eta ∇_{x_0} ||A(x_0^{(j)}) − y||^2 / (2 sigma_y^2) + sqrt(2 eta) epsilon_j`,
starting from `x_0^{(0)} = x_0_hat`. The first term is a quadratic pull back toward the prior's clean estimate (a soft prior anchor with strength `1/r_t^2`), the second is the data-fidelity gradient on the clean variable, the third is the Langevin noise that keeps it a *sampler* rather than an optimizer. This is the crux of the decoupling: the likelihood correction is a full MCMC sample of the clean conditional, done entirely on `x_0`, with the prior entering only as the anchor `x_0_hat` and its width `r_t`. The expensive denoiser is not in this loop at all — only the cheap forward operator — so I can afford many Langevin steps (I use 100) to actually mix.

Piece (c), re-noise. I now have `x_0^{(N)}`, a sample of `p(x_0 | x_t, y)`. To produce the next outer iterate at the next lower noise level `sigma_next`, I simply add fresh Gaussian noise at that scale: `x_{t-1} ~ N(x_0^{(N)}, sigma_next^2 I)`, i.e. `x_{t-1} = x_0^{(N)} + sigma_next epsilon`. This is the re-noising that closes the decoupled cycle: the next `x_t` is built from a *corrected clean* estimate, not inherited as a small reverse step, which is precisely why a bad correction does not propagate — the noise wipes the slate to the next level's scale.

Let me check the annealing actually converges to the posterior, because the whole point is that the time-marginals anneal to `p(x_0 | y)` as the noise drops. At the largest `sigma`, `x_t` is essentially pure noise, the prior update's ODE samples the prior broadly, and the likelihood correction has a wide anchor (`r_t` large), so the conditional sample is dominated by the data term — it explores. As `sigma` decreases, the anchor tightens (`r_t → 0`), the prior update's clean estimate sharpens, and the conditional `p(x_0 | x_t, y)` concentrates; at `sigma → 0` the re-noising adds vanishing noise and the conditional is the true posterior `p(x_0 | y)`. So the sequence of conditionals is an annealing path from a data-dominated broad distribution to the sharp posterior — the same logic as annealed Langevin, but with the prior re-injected at every level through the ODE clean estimate. The decoupling is what lets consecutive marginals differ a lot (each is freshly re-noised) while still annealing to the target.

Now I have to be careful about the step size `eta`, because Langevin in high dimensions is touchy. The prior-anchor term has strength `1/r_t^2 ≈ 1/sigma^2`, which is enormous at small `sigma`, so a fixed `eta` would make the anchor term explode late in annealing. The fix is to scale the Langevin step with the noise level — let `eta` decay as the annealing proceeds (the reference ties the step size to `sigma` so the anchor term stays `O(1)`), so the inner sampler stays stable from the noisy start to the clean end. The number of Langevin steps `N` trades mixing for cost; since the denoiser is outside the inner loop, `N = 100` is affordable. The number of annealing levels and the ODE sub-sampler's NFE set the total cost; the EDM rho-schedule from `sigma_max ≈ 80` down to `sigma_min` is the standard choice.

Let me confirm this clears the failure modes I started with. Nonlinearity: `A` is only ever evaluated and differentiated at the *clean* `x_0` inside the Langevin loop, never backpropagated through the denoiser, so a nonlinear operator is just a different `∇||A(x_0) − y||^2` — no Jacobian-through-the-network, the thing that made DPS/LGD fragile on phase retrieval. Coupling: the prior update (unconditional ODE) and the likelihood correction (clean-variable Langevin) are separate operations on separate variables, stitched by re-noising, so a bad correction does not corrupt the latent dynamics — the error does not propagate down the trajectory the way it does in a coupled single-step guidance. Mode collapse: unlike RED-diff's `sigma → 0` mode-seeking optimization, the Langevin noise keeps the correction a genuine sample, so the method explores the posterior instead of committing to one blurred mode. And the only knobs are interpretable: the Langevin step size and count, the number of annealing levels, the ODE NFE, and the known `sigma_y`.

So let me write the sampler in the form I would actually run, with the unconditional ODE sub-sampler, the clean-variable Langevin inner loop, and the re-noising. The forward operator gives me `∇_{x_0} ||A(x_0) − y||^2` and the loss, so the data term in the Langevin update is that gradient divided by `2 sigma_y^2`:

```python
import torch
import numpy as np
from .base import Algo
from utils.scheduler import Scheduler
from utils.diffusion import DiffusionSampler


class LangevinDynamics:
    """Sample p(x0 | x_t, y) on the CLEAN variable: a quadratic anchor to the
    prior clean estimate plus the data-fidelity gradient, driven by Langevin noise.
    The denoiser is NOT in this loop — only the cheap forward operator."""

    def __init__(self, num_steps, lr, sigma_y, lr_min_ratio=0.01):
        self.num_steps = num_steps
        self.lr = lr
        self.sigma_y = sigma_y
        self.lr_min_ratio = lr_min_ratio

    def sample(self, x0hat, forward_op, observation, sigma):
        x = x0hat.clone().detach()
        for j in range(self.num_steps):
            # cosine-style decay of the Langevin step so the 1/sigma^2 anchor stays O(1)
            ratio = j / self.num_steps
            lr = self.lr * (self.lr_min_ratio + (1 - self.lr_min_ratio)
                            * 0.5 * (1 + np.cos(np.pi * ratio)))
            x = x.detach().requires_grad_(True)
            data_grad, _ = forward_op.gradient(x, observation, return_loss=True)
            # score of p(x0 | x_t, y): prior anchor + data term
            grad = (x - x0hat) / (sigma ** 2) + data_grad / (2 * self.sigma_y ** 2)
            x = x.detach() - lr * grad + np.sqrt(2 * lr) * torch.randn_like(x)
        return x.detach()


class DAPS(Algo):
    """Decoupled Annealing Posterior Sampling. Each annealing level:
    (1) unconditional ODE prior update x_t -> x0hat; (2) clean-variable Langevin
    correction sampling p(x0 | x_t, y); (3) re-noise to the next level."""

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

            # (1) PRIOR UPDATE: unconditional PF-ODE from x_t (sigma_max = sigma)
            diff_sched = Scheduler(sigma_max=sigma, **self.diffusion_scheduler_config)
            x0hat = DiffusionSampler(diff_sched).sample(self.net, x_t, SDE=False)

            # (2) LIKELIHOOD CORRECTION: Langevin sample of p(x0 | x_t, y) on x0
            x0y = self.lgvd.sample(x0hat, self.forward_op, observation, sigma)

            # (3) RE-NOISE to the next, lower annealing level
            x_t = x0y + torch.randn_like(x0y) * sigma_next

        return x_t
```

Let me trace the causal chain once more. Every training-free solver hits the intractable time-level
likelihood score; the ones that keep nonlinearity generality (DPS, LGD) cope by approximating a guidance
term inside a *single, coupled* reverse step on the noisy latent — where the clean-variable likelihood and
the noisy-variable prior dynamics fight, and a bad guidance step propagates down the trajectory. So I
decoupled the steps: at each annealing level, run the prior's own unconditional ODE to turn `x_t` into a
sharp *clean* estimate `x_0_hat` (the prior update, knowing nothing of `y`); sample the clean conditional
`p(x_0 | x_t, y) ∝ N(x_0_hat, r_t^2) p(y | x_0)` by Langevin dynamics *on the clean variable*, where `A`
is evaluated directly and no denoiser Jacobian appears (the likelihood correction); then re-noise the
corrected clean sample to the next lower level to make the next `x_t`. The re-noising is what decouples —
each iterate is freshly built from a clean correction rather than inherited as a small step, so errors do
not propagate, nonlinear operators are handled without backprop through the network, and the Langevin
noise keeps it a true sampler that anneals to `p(x_0 | y)` instead of collapsing to a mode.
