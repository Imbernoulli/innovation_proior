Let me start from the one structural fact every training-free posterior sampler keeps tripping over. I have a pretrained denoiser `D_theta(x_t, sigma) = E[x_0 | x_t]` that gives me the prior's score, and a known likelihood `p(y | x_0) = N(A(x_0), sigma_y^2 I)`, and I want `x_0 ~ p(x_0 | y)`. To run the reverse diffusion conditioned on `y` I need the time-level posterior score, which Bayes splits into the free prior score plus the likelihood score `∇_{x_t} log p_t(y | x_t)`. That term is intractable because `p_t(y | x_t) = E_{x_0 ~ p(x_0|x_t)}[ p(y | x_0) ]` involves the multimodal denoising posterior `p(x_0 | x_t)`. DPS approximates the expectation by the likelihood at the Tweedie mean; LGD spreads the mean into a Gaussian surrogate and Monte-Carlos it; ΠGDM inverts `A` through a pseudoinverse. All of them are the same shape: approximate a guidance term and inject it into a single reverse step on `x_t`. I want to ask whether that shape is the actual problem, rather than the particular approximation inside it.

Here is the thing that has been bothering me about the shape. The prior's score is a function of the *noisy* latent `x_t` — that is what the diffusion model was trained to know. The likelihood `p(y | x_0)` is a function of the *clean* signal `x_0` — that is where the forward operator lives. When DPS or LGD bend a single reverse step, they are forcing a clean-variable correction and a noisy-variable dynamics into the same update on `x_t`, via `x_0_hat = D_theta(x_t)` and a backprop through the network. The two demands are not the same demand. The likelihood wants `A(x_0) ≈ y` — a hard constraint on the clean signal. The prior wants `x_t` to follow the reverse dynamics — a soft pull on the noisy latent. Jamming them together is why the guidance scale has to be retuned per operator and per noise level, and why a nonlinear operator, whose residual gradient is wild, destabilizes the whole reverse trajectory: a bad guidance step at noise level `t` corrupts `x_t`, which corrupts every later step, because the steps are *coupled* — each one inherits the last one's `x_t`. The error does not stay local; it propagates down the trajectory.

What would it take to break that coupling? The reason a bad step propagates is that the *only* thing carried from one level to the next is the noisy latent `x_t`, and every correction has to be written into it. Suppose instead I let consecutive points along the trajectory differ a lot from one another, rather than each being a small perturbation of the last. Then at each noise level I would (a) extract a *clean* estimate from `x_t` using the prior, (b) correct that clean estimate against the likelihood as hard as I want, on the clean variable where the likelihood actually lives, and then (c) re-noise the corrected clean estimate to the *next, lower* noise level to get the next `x_t`. If that works, the likelihood correction never touches the noisy-latent dynamics, and a bad correction at one level cannot poison the latent that feeds the next level, because the next `x_t` is freshly re-noised from a corrected clean estimate rather than inherited. Whether it actually works depends on each of the three pieces being well-defined and on the sequence of clean conditionals annealing to the posterior — neither of which is obvious yet, so let me build the pieces and then check.

Piece (a), the clean estimate. I could use the one-step Tweedie mean `D_theta(x_t, sigma)`, which is what DPS uses. But at high noise the one-step mean is a blurry average over the whole posterior — it throws away the prior's ability to resolve structure. Since I have stopped requiring each step to be small, I can afford to spend more of the prior here: solve the *unconditional* probability-flow ODE starting at `x_t`, running the prior's own reverse dynamics from `sigma` down to clean, to get a sharp sample `x_0_hat(x_t)` of `p(x_0 | x_t)`. This is the prior update, and it is unconditional — it knows nothing about `y`. It is a sub-sampler with its own (small) number of function evaluations, and because the outer loop no longer chains small steps I do not need it to be exact; a handful of Euler ODE steps suffices to turn `x_t` into a clean estimate that respects the prior far better than a single Tweedie call.

Piece (b), the likelihood correction. I now have `x_0_hat`, the prior's clean estimate, and I want to pull it toward agreeing with `y` — but not collapse it to a single point, because I want a *sampler*, not RED-diff's mode-seeking optimization. So sample from the conditional `p(x_0 | x_t, y)`. Write its score. By Bayes, `p(x_0 | x_t, y) ∝ p(x_0 | x_t) p(y | x_0)`. The first factor `p(x_0 | x_t)` is the denoising posterior; the cheapest usable surrogate is a Gaussian centered at the prior's clean estimate, `p(x_0 | x_t) ≈ N(x_0_hat, r_t^2 I)`. What is `r_t`? Before I write it as `≈ sigma_t` from memory, let me actually derive the width in the one case it is closed form. Take a Gaussian prior `x_0 ~ N(0, s_0^2)` and the VE forward `x_t = x_0 + sigma epsilon`. Then `p(x_0 | x_t)` is exactly Gaussian with precision `1/s_0^2 + 1/sigma^2`, so its variance is `r_t^2 = 1/(1/s_0^2 + 1/sigma^2)`. Plugging in `sigma = 0.7`: with `s_0 = 1` I get `r_t = 0.573`, ratio `r_t/sigma = 0.82` — not yet `sigma`. But the regime that matters here is high noise, where the prior is broad relative to `sigma`, i.e. `s_0 ≫ sigma`; taking `s_0 = 5, 50, 10^6` at the same `sigma = 0.7` gives `r_t = 0.693, 0.700, 0.700`. So `r_t → sigma` exactly as the prior width dominates the noise, which is precisely the regime each annealing level sits in. Good — `r_t ≈ sigma` is not a guess, it is the broad-prior limit of the conjugate posterior, and it will be loosest (most approximate) at small `sigma`, which I should keep in mind. The second factor is the Gaussian likelihood. So

`log p(x_0 | x_t, y) = − ||x_0 − x_0_hat||^2 / (2 r_t^2) − ||A(x_0) − y||^2 / (2 sigma_y^2) + const`,

and its gradient in `x_0` is `−(x_0 − x_0_hat)/r_t^2 − ∇_{x_0} ||A(x_0) − y||^2 / (2 sigma_y^2)`. This is a *clean-variable* score — `A` is evaluated at `x_0` directly, not through the denoiser — so there is no network Jacobian anywhere in the correction, and a nonlinear `A` is no harder than a linear one as long as I can differentiate it.

Now sample that conditional. The score is all I need for Langevin dynamics: iterate `x_0 ← x_0 + eta ∇_{x_0} log p(x_0 | x_t, y) + sqrt(2 eta) z`, `z ~ N(0, I)`, for `N` steps. Written out with the two terms,

`x_0^{(j+1)} = x_0^{(j)} − eta (x_0^{(j)} − x_0_hat)/r_t^2 − eta ∇_{x_0} ||A(x_0^{(j)}) − y||^2 / (2 sigma_y^2) + sqrt(2 eta) epsilon_j`,

starting from `x_0^{(0)} = x_0_hat`. The first term is a quadratic pull back toward the prior's clean estimate (a soft prior anchor with strength `1/r_t^2`), the second is the data-fidelity gradient on the clean variable, the third is the Langevin noise that keeps it a *sampler* rather than an optimizer. Before I trust this loop I should check it actually targets the conditional I wrote down, not just something near it. Take a scalar linear-Gaussian instance where the target is closed form: `x_0_hat = 0.3`, `r_t = sigma = 0.7`, likelihood `y = a x_0 + n` with `a = 2`, `sigma_y = 0.05`, observed `y = 1.1`. The exact posterior of `x_0` has precision `1/r_t^2 + a^2/sigma_y^2` and mean `(x_0_hat/r_t^2 + a y/sigma_y^2)` over that precision, which works out to mean `0.5497`, std `0.0250`. Running the exact update above — with `∇||A x − y||^2 = 2a(a x − y)` divided by `2 sigma_y^2`, constant `eta = 2e-4`, and collecting samples after burn-in — gives empirical mean `0.5498`, std `0.0273`. The mean lands essentially on top of the closed form; the std is slightly inflated, which is the expected discretization bias of unadjusted Langevin at finite `eta` (it shrinks with `eta`). So the loop does sample `p(x_0 | x_t, y)`, with the prior entering only as the anchor `x_0_hat` and its width `r_t`. The expensive denoiser is not in this loop at all — only the cheap forward operator — so I can afford many Langevin steps (I use 100) to actually mix.

Piece (c), re-noise. I now have `x_0^{(N)}`, a sample of `p(x_0 | x_t, y)`. To produce the next outer iterate at the next lower noise level `sigma_next`, I add fresh Gaussian noise at that scale: `x_{t-1} ~ N(x_0^{(N)}, sigma_next^2 I)`, i.e. `x_{t-1} = x_0^{(N)} + sigma_next epsilon`. This re-noising is what closes the cycle without re-introducing the coupling: the next `x_t` is built from a *corrected clean* estimate, not inherited as a small reverse step, so a bad correction does not propagate — the fresh noise wipes the slate to the next level's scale rather than carrying the latent forward.

Now the part I deferred: does the sequence of conditionals actually anneal to `p(x_0 | y)`, or have I just built a stable loop that converges to the wrong thing? The qualitative story is that at large `sigma` the anchor is wide and the data term dominates (explore), and as `sigma → 0` the anchor tightens, the conditional concentrates, the re-noising adds vanishing noise, and the conditional becomes the true posterior. Let me put numbers on the "data dominates / anchor tightens" crossover, because the strengths are very lopsided and I do not want to hand-wave it. The anchor strength is `1/r_t^2 ≈ 1/sigma^2`; the data strength in the same units is `a^2/sigma_y^2`, which with `a = 2`, `sigma_y = 0.05` is `1600`. The fraction of the posterior mean's weight on the data term is `data/(anchor+data)`: at `sigma = 80` it is `1.0000`, at `sigma = 1` it is `0.9994`, at `sigma = 0.1` it is `0.941`, and only at `sigma = 0.01` does it drop to `0.138`. So the conditional is overwhelmingly data-driven across almost the entire schedule, and the prior anchor only seizes control in the last fraction of annealing, near `sigma = 0`. That is exactly what I want — the data constraint is enforced early and held throughout, and the prior only does the final-detail cleanup where `r_t ≈ sigma` is loosest anyway — but it is more lopsided than I would have guessed, and it tells me the schedule must reach genuinely small `sigma` for the prior to matter at all. At `sigma → 0` the re-noising noise vanishes and `p(x_0 | x_t, y)` is the true `p(x_0 | y)`. So the conditionals do form an annealing path to the target, with the prior re-injected at every level through the ODE clean estimate; the decoupling is what lets consecutive marginals differ a lot (each freshly re-noised) while still landing on the posterior.

Now I have to be careful about the step size `eta`, and the weighting numbers above tell me exactly why. The prior-anchor term has strength `1/r_t^2 ≈ 1/sigma^2`, which is `10^4` at `sigma = 0.01` versus `~10^{-4}` at `sigma = 80` — eight orders of magnitude across the schedule. A fixed `eta` tuned for the noisy start would make the anchor term explode at the clean end. So the Langevin step must scale with the noise level — let `eta` decay as the annealing proceeds so the anchor term stays `O(1)` and the inner sampler stays stable from the noisy start to the clean end. The number of Langevin steps `N` trades mixing for cost; since the denoiser is outside the inner loop, `N = 100` is affordable. The number of annealing levels and the ODE sub-sampler's NFE set the total cost; the EDM rho-schedule from `sigma_max ≈ 80` down to `sigma_min` is the standard choice, and the crossover numbers above are why `sigma_min` must be small.

Let me line this up against the three failure modes I started from. Nonlinearity: `A` is only ever evaluated and differentiated at the *clean* `x_0` inside the Langevin loop, never backpropagated through the denoiser, so a nonlinear operator is just a different `∇||A(x_0) − y||^2` — no Jacobian-through-the-network, the thing that made DPS/LGD fragile on phase retrieval. Coupling: the prior update (unconditional ODE) and the likelihood correction (clean-variable Langevin) are separate operations on separate variables, stitched by re-noising, so a bad correction does not corrupt the latent dynamics — the error does not propagate down the trajectory the way it does in a coupled single-step guidance. Mode collapse: unlike RED-diff's `sigma → 0` mode-seeking optimization, the Langevin noise keeps the correction a genuine sample, so the method explores the posterior instead of committing to one blurred mode. And the only knobs are interpretable: the Langevin step size and count, the number of annealing levels, the ODE NFE, and the known `sigma_y`.

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
