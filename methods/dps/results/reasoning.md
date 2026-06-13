Let me start from what I actually have and what I actually want. I have a diffusion model that has learned a prior over some signal distribution — a network `s_theta(x_t, t)` that approximates `grad_{x_t} log p_t(x_t)`, the score of the data smoothed by Gaussian noise at level `t`. Running its reverse SDE, `dx = [-(beta/2) x - beta * grad log p_t(x_t)] dt + sqrt(beta) dw_bar`, integrates noise back to a clean sample of `p(x)`. Separately I have a measurement `y = A(x_0) + n` from a known forward operator `A` with detector noise `n`, and I want to recover `x_0`. Since `A` is many-to-one this is ill-posed, so I shouldn't chase a single inverse; the honest object is the posterior `p(x_0 | y) ∝ p(y | x_0) p(x_0)`, with my diffusion model serving as the prior `p(x_0)`. So the real question is: can I sample the posterior using only the unconditional score I already have plus the known `A`, without retraining anything per task, and have it actually work when the measurement is noisy and when `A` is nonlinear?

The Bayesian decomposition that makes this look easy is just the gradient of Bayes' rule applied at every noise level. Take logs of `p_t(x_t | y) ∝ p_t(x_t) p_t(y | x_t)` and differentiate in `x_t`:

  grad_{x_t} log p_t(x_t | y) = grad_{x_t} log p_t(x_t) + grad_{x_t} log p_t(y | x_t).

The first term is exactly the score my network gives me. So a posterior sampler is just the unconditional reverse SDE with one extra drift term added, the **likelihood score** `grad_{x_t} log p_t(y | x_t)`:

  dx = [-(beta/2) x - beta ( s_theta(x_t,t) + grad_{x_t} log p_t(y | x_t) )] dt + sqrt(beta) dw_bar.

If I could compute that second term I'd be done. So let me try to write it down. I know the measurement model `p(y | x_0)` — for white Gaussian noise it's `N(y | A(x_0), sigma^2 I)`. But the SDE wants `p_t(y | x_t)`: the likelihood of `y` given the *noised* iterate `x_t`, not the clean `x_0`. And there is no direct relationship between `y` and `x_t`. Draw the dependency graph: `y <- x_0 -> x_t`. The measurement edge `x_0 -> y` is known, the noising edge `x_0 -> x_t` is the Gaussian forward process and known, but there's no `x_t -> y` edge at all. `y` only knows about `x_t` through the unknown clean `x_0` sitting between them. So `p_t(y | x_t)` is intractable — it has no closed form, because to get it I'd have to integrate over all the `x_0` that could have produced `x_t`. That's the wall. Everything in this problem is downstream of this one intractable term.

How have people gotten around it? The most common move is to refuse to compute it. Drop the likelihood term, take a plain unconditional reverse-diffusion step, and then **project** the result onto the measurement subspace `C = {x : A x = y}` — enforce data consistency by fiat. That's score-SDE / ILVR / the POCS family. It's clean and it works beautifully when there's no noise. But I have to be careful about what projection *means* when `n != 0`. Projecting onto `{A x = y}` says "make `A x` equal the measured `y` exactly." If `y` is corrupted — `y = A x_0 + n` — then I'm forcing my reconstruction to reproduce the noise. Worse: for operators like super-resolution or deblurring, the projection step applies the transpose `A^T`, and `A^T n` doesn't stay small — for these ill-conditioned operators it gets *amplified*. So the sample overfits the noise, drifts off the set of plausible images, and accumulates error step after step. It's well documented that these projection methods fall apart on noisy measurements for exactly this reason. And there's a second problem: a projection onto `{A x = y}` only makes sense when `A` is something I can project against — linear, with a definable subspace. For a nonlinear `A`, "the measurement subspace" isn't even a subspace. So projection is out on both of my requirements at once.

The spectral approach — SNIPS, DDRM — is smarter about noise: diagonalize `A` by its SVD, push the diffusion into the SVD basis, and there the measurement-domain Gaussian noise becomes spectral-domain noise you can handle in closed form. That genuinely deals with `n`. But it buys noise-robustness with the SVD, and the SVD of a forward operator is exactly the thing I can't always get. For a separable Gaussian blur, fine; for motion blur, a PDE solver, a Fourier-magnitude operator, the SVD is prohibitive or undefined. So spectral methods trade my nonlinearity requirement away to satisfy my noise requirement. I want both.

Then there's the line that tries to actually *approximate* the likelihood score instead of dodging it. For linear `A` and Gaussian noise, the clean-model likelihood score is `grad_x log p(y | x) = A^H (y - A x) / sigma^2` — the gradient of `-||y - A x||^2 / 2 sigma^2`. Robust-CSGM just uses that inside annealed Langevin. But that's the likelihood at `t = 0`, the clean model. At every intermediate noise level it's simply the wrong term, and they know it — they patch it by inflating the denominator, `A^H (y - A x) / (sigma^2 + gamma_t^2)`, with `gamma_t` a hand-tuned sequence decaying to zero, a heuristic that pretends the effective noise is higher early in sampling. It's linear-only and the `gamma_t` schedule is a fudge. I'd rather have the *right* time-level term, or at least a principled approximation to it, than a corrected wrong one.

So none of the dodges satisfy me. Let me go back and stare at the intractable term itself and try to make it tractable honestly. The reason `p_t(y | x_t)` has no closed form is the missing `x_t -> y` edge — `y` depends on `x_t` only through `x_0`. But that's also a hint: I should integrate over the thing in between. Marginalize `x_0`:

  p(y | x_t) = ∫ p(y | x_0, x_t) p(x_0 | x_t) dx_0.

Now use the conditional independence the graph hands me: given `x_0`, the measurement `y` and the noised iterate `x_t` are independent — `y` is `A(x_0) + n`, `x_t` is `x_0 + (forward-process noise)`, and the two noises are unrelated. So `p(y | x_0, x_t) = p(y | x_0)`, and

  p(y | x_t) = ∫ p(y | x_0) p(x_0 | x_t) dx_0 = E_{x_0 ~ p(x_0 | x_t)} [ p(y | x_0) ].

This is genuinely exact, no approximation yet. The intractable likelihood at noise level `t` is the *expectation*, over the reverse/denoising posterior `p(x_0 | x_t)`, of the clean likelihood `p(y | x_0)` — which I do know in closed form. The whole difficulty is now relocated into one object: `p(x_0 | x_t)`, the distribution of clean signals consistent with the noised iterate. And in general that posterior is still intractable; it's the blue dotted edge `x_t -> x_0` in the graph, the inverse of the forward noising, which is exactly what diffusion is hard about.

But — and this is the thing — I don't actually need the whole distribution `p(x_0 | x_t)`. I need an *expectation against* it. And there is one functional of `p(x_0 | x_t)` I can compute in closed form: its **mean**. Because the forward process is Gaussian, `x_t = sqrt(alpha_bar) x_0 + sqrt(1 - alpha_bar) z`, the posterior mean `E[x_0 | x_t]` is given by Tweedie's formula. Let me actually derive that rather than quote it, because the derivation tells me *why* it's the score that shows up, and I'll lean on the same machinery later.

Tweedie in its general empirical-Bayes form: suppose `p(x_t | x_0)` is an exponential family in the canonical parameter `x_0`,

  p(x_t | x_0) = p_0(x_t) exp( x_0^T T(x_t) - phi(x_0) ),

with `p_0` the base measure, `T` the sufficient statistic, `phi` the log-partition. I want `E[x_0 | x_t]`. Write the marginal `p(x_t) = ∫ p(x_t | x_0) p(x_0) dx_0` and differentiate in `x_t`:

  grad_{x_t} p(x_t) = ∫ grad_{x_t} p(x_t | x_0) p(x_0) dx_0
                    = ∫ [ (grad_{x_t} log p_0(x_t)) + (grad_{x_t} T(x_t))^T x_0 ] p(x_t | x_0) p(x_0) dx_0,

because `grad_{x_t} log p(x_t | x_0) = grad log p_0(x_t) + (grad T(x_t))^T x_0` (the `phi(x_0)` term has no `x_t`). Divide through by `p(x_t)`; the first piece gives `grad log p_0(x_t)` times `p(x_t)/p(x_t)`, the second gives `(grad T(x_t))^T` times `∫ x_0 p(x_0 | x_t) dx_0 = (grad T)^T E[x_0 | x_t]`. So

  grad_{x_t} log p(x_t) = grad_{x_t} log p_0(x_t) + (grad_{x_t} T(x_t))^T E[x_0 | x_t].

That's the exponential-family Tweedie identity: the posterior mean of the canonical parameter is recovered from the score of the marginal minus the score of the base measure. Now specialize to my Gaussian forward process. With `x_t | x_0 ~ N(sqrt(alpha_bar) x_0, (1 - alpha_bar) I)`,

  p(x_t | x_0) = (2 pi (1 - alpha_bar))^{-d/2} exp( -||x_t - sqrt(alpha_bar) x_0||^2 / (2 (1 - alpha_bar)) ).

Expand the square: `||x_t - sqrt(alpha_bar) x_0||^2 = ||x_t||^2 - 2 sqrt(alpha_bar) x_0^T x_t + alpha_bar ||x_0||^2`. Matching to the exponential-family form `p_0(x_t) exp(x_0^T T(x_t) - phi(x_0))`, I read off

  p_0(x_t) = (2 pi (1 - alpha_bar))^{-d/2} exp( -||x_t||^2 / (2(1 - alpha_bar)) ),
  T(x_t)   = sqrt(alpha_bar) x_t / (1 - alpha_bar),
  phi(x_0) = alpha_bar ||x_0||^2 / (2 (1 - alpha_bar)).

So `grad_{x_t} log p_0(x_t) = -x_t / (1 - alpha_bar)` and `grad_{x_t} T(x_t) = sqrt(alpha_bar)/(1 - alpha_bar) I`. Plug into the identity:

  grad_{x_t} log p_t(x_t) = -x_t/(1 - alpha_bar) + ( sqrt(alpha_bar)/(1 - alpha_bar) ) E[x_0 | x_t].

Solve for the posterior mean:

  E[x_0 | x_t] = (1 / sqrt(alpha_bar)) ( x_t + (1 - alpha_bar) grad_{x_t} log p_t(x_t) ).

There it is — Tweedie. And it's beautiful for me: the posterior mean of the clean signal is an explicit, closed-form function of the very score my network estimates. Replace `grad log p_t` by `s_theta` and I get, at any noise level, for free, inside the loop,

  x_0_hat := E[x_0 | x_t] ≈ (1/sqrt(alpha_bar)) ( x_t + (1 - alpha_bar) s_theta(x_t, t) ).

Now back to the intractable likelihood. I have `p(y | x_t) = E_{x_0 ~ p(x_0|x_t)}[ p(y | x_0) ]` exactly, and I can compute the *mean* of that posterior, `x_0_hat`, but not the full expectation. The cheapest possible move is to swap them: approximate the expectation of the function by the function of the expectation,

  p(y | x_t) = E[ p(y | x_0) ] ≈ p( y | E[x_0 | x_t] ) = p(y | x_0_hat).

That is, replace the outer expectation of `p(y | x_0)` over the posterior with the inner expectation of `x_0` evaluated once. It's a point-estimate approximation: collapse `p(x_0 | x_t)` to its mean and evaluate the likelihood there. Whether this is legitimate is precisely a question about how far `E[f(x_0)]` sits from `f(E[x_0])` — a Jensen gap. So let me not hand-wave it; let me bound it, because the size of that gap tells me when this whole scheme is trustworthy.

Define the Jensen gap `J(f, p) = E[f(x_0)] - f(E[x_0])` for `f(x_0) := p(y | x_0)`. In the Gaussian case `f(x_0) = h(A(x_0))` where `h` is the `N(y, sigma^2 I)` density as a function of its mean argument, so `h(A(x_0)) = (2 pi sigma^2)^{-n/2} exp(-||y - A(x_0)||^2 / 2 sigma^2)`. I want to bound

  J = | E[ h(A(x_0)) ] - h(A(x_0_hat)) |,    x_0_hat = E[x_0 | x_t].

Bring the absolute value inside the expectation (it can only grow):

  J ≤ ∫ | h(A(x_0)) - h(A(x_0_hat)) | p(x_0 | x_t) dx_0.

Now I need `h` to be Lipschitz so I can pull `||A(x_0) - A(x_0_hat)||` out of the density. So let me bound the gradient of an isotropic Gaussian density `h_sigma(u) = (2 pi sigma^2)^(-q/2) exp(-||u - y||^2 / 2 sigma^2)` over the measurement space. Its gradient is

  grad_u h_sigma(u) = -(u - y)/sigma^2 * h_sigma(u).

The norm is `(r/sigma^2) (2 pi sigma^2)^(-q/2) exp(-r^2 / 2 sigma^2)` with `r = ||u - y||`, and the scalar part `r exp(-r^2/2 sigma^2)` is maximized at `r = sigma`. Therefore

  L_sigma := sup_u ||grad h_sigma(u)|| = e^{-1/2} (2 pi)^(-q/2) sigma^(-(q+1)).

That is the Lipschitz constant I need under the Euclidean norm for the normalized `q`-dimensional density. If I use another norm I may pick up a `sqrt(q)` or `q` factor, but the noise-scale behavior of the normalized density is the thing I must not lose: the constant behaves like `sigma^{-(q+1)}`, so it gets large as `sigma` becomes tiny and decays to zero as `sigma` grows. By the mean value theorem,

  |h_sigma(u) - h_sigma(v)| ≤ L_sigma ||u - v||.

Carrying that bound forward,

  J ≤ L_sigma ∫ ||A(x_0) - A(x_0_hat)|| p(x_0 | x_t) dx_0.

Next, `A` itself: by the mean value / intermediate value theorem, `||A(x_0) - A(x_0_hat)|| ≤ ||grad_x A(x)|| · ||x_0 - x_0_hat||` where `||grad_x A(x)|| := max_x ||grad_x A(x)||` is the operator's max Jacobian norm. Pull it out:

  J ≤ L_sigma ||grad A|| ∫ ||x_0 - x_0_hat|| p(x_0 | x_t) dx_0 = L_sigma ||grad A|| m_1,

with `m_1 := ∫ ||x_0 - x_0_hat|| p(x_0 | x_t) dx_0` the first absolute central moment of the denoising posterior. So, writing `L` out,

  J ≤ e^{-1/2} (2 pi)^(-q/2) sigma^(-(q+1)) ||grad_x A(x)|| m_1,

where `q` is the measurement dimension. Now read this bound, because it's telling me when the approximation is safe. `||grad A||` is finite for essentially every forward operator I care about — it's the Lipschitz constant of `A`, which has nothing to do with the *ill-posedness* of the inverse problem (ill-posedness is unboundedness of `A^{-1}`, a totally different object). `m_1` is finite for any reasonable posterior — it's just how spread out the plausible clean signals are around their mean. So the delicate factor is the Gaussian-density Lipschitz constant, and this normalized-density bound tends to zero as `sigma -> infinity`. The Jensen gap therefore shrinks as measurement noise grows. That's the opposite of what I'd have feared — I'd have guessed more noise makes everything worse — and it's exactly why this point-estimate approximation is well-suited to the noisy regime that broke the projection methods. At very small `sigma`, the density is sharp and the Lipschitz constant is large, so the point-estimate approximation has less room for error only if the denoising posterior is already tight enough to make `m_1` small. The same construction goes through for a Poisson measurement distribution with its own Lipschitz constant, so the argument isn't married to Gaussian noise.

Good — the approximation is principled. So I get a tractable likelihood score by differentiating the surrogate:

  grad_{x_t} log p_t(y | x_t) ≈ grad_{x_t} log p(y | x_0_hat(x_t)).

For Gaussian noise, `log p(y | x_0_hat) = -||y - A(x_0_hat)||^2 / (2 sigma^2) + const`, so with the unhalved squared residual loss,

  grad_{x_t} log p_t(y | x_t) ≈ -(1/(2 sigma^2)) grad_{x_t} ||y - A(x_0_hat(x_t))||^2.

If instead I define the data term as `0.5 ||r||^2`, the coefficient is `1/sigma^2`; those are the same likelihood gradient written under two loss conventions. I want to be careful about the variable, because I'm differentiating with respect to `x_t`, not `x_0_hat`. The estimate `x_0_hat` is a *function* of `x_t` — it runs `x_t` through the score network. So `grad_{x_t}` is a backpropagation through the network and through the Tweedie formula. That's the whole trick to handling general `A`: I never need an SVD or a projection or a transpose by hand; I just need `A` to be differentiable so autodiff can carry `grad_{x_0_hat} ||y - A(x_0_hat)||^2` back through `x_0_hat(x_t)` to `x_t`. A nonlinear `A` is no harder than a linear one as long as I can backprop through it. The conditional score I'll inject is therefore

  grad_{x_t} log p_t(x_t | y) ≈ s_theta(x_t, t) - rho * grad_{x_t} ||y - A(x_0_hat)||^2,   rho = 1/(2 sigma^2)

for the unhalved squared-loss gradient. In a sampler this constant is going to be folded into the discrete guidance scale anyway, but I want the calculus right before I start absorbing constants.

Before I commit, let me handle the Poisson case too, because shot noise is the realistic model for photon-counting measurements and signal-dependent noise is supposed to be a strength here. The Poisson likelihood is `p(y | x_0) = prod_j [A(x_0)]_j^{y_j} exp(-[A(x_0)]_j) / y_j!`. If I write `mu_j = [A(x_0_hat)]_j`, the direct surrogate log-gradient contains the factor `(y_j / mu_j - 1) grad mu_j`; that division by `mu_j` is numerically fragile wherever the predicted intensity is near zero. So the direct Poisson likelihood is a dead end for an iterative sampler. Instead use the standard Gaussian approximation to Poisson, valid once the counts aren't tiny: a Poisson with mean `mu` is approximately `N(mu, mu)`, so

  p(y | x_0) ≈ prod_j (2 pi [A(x_0)]_j)^{-1/2} exp( -(y_j - [A(x_0)]_j)^2 / (2 [A(x_0)]_j) ).

I could keep the `[A(x_0)]_j` in the denominator, but that signal-dependent variance term itself depends on `x_0` and makes the weighting jump around, which destabilizes the iteration. The clean fix is the shot-noise approximation `[A(x_0)]_j ≈ y_j` in the variance only (the high-SNR substitution that's standard for detection): replace the per-bin variance `[A(x_0)]_j` by the measured `y_j`, giving a *fixed* weight per bin. The surrogate log likelihood is now `-sum_j (y_j - mu_j)^2 / (2 y_j) + const`, so

  grad_{x_t} log p_t(y | x_t) ≈ -rho grad_{x_t} ||y - A(x_0_hat)||^2_Lambda,   [Lambda]_{jj} = 1/(2 y_j),

a weighted least squares with frozen weights; `rho` is only a remaining global guidance scale because the per-bin likelihood weights already live in `Lambda`. So the unified rule is: Gaussian -> plain `||y - A(x_0_hat)||^2`, Poisson -> weighted `||y - A(x_0_hat)||^2_Lambda`, otherwise identical.

Now I drop this into the sampler. The unconditional reverse step (ancestral DDPM, or the equivalent EDM-scaled update) produces an intermediate `x'_{i-1}` from `x_i`, `x_0_hat`, and fresh noise; then I take the fidelity-gradient step:

  x_{i-1} = x'_{i-1} - zeta_i grad_{x_i} ||y - A(x_0_hat)||^2.

And here's the practical question: what is `zeta_i`? The clean derivation gives a fixed noise-variance coefficient. Let me think about whether that can be right across a thousand reverse steps. Early in sampling `x_t` is almost pure noise, `x_0_hat` is a blurry mean, and `||y - A(x_0_hat)||` is large; late in sampling `x_0_hat` is sharp and the residual is small. A fixed coefficient multiplies a squared-residual gradient whose magnitude swings over orders of magnitude across the trajectory, so the effective guidance strength is wildly uneven — too weak when the residual is huge, or saturating and noise-amplifying when it's cranked up. I want the guidance to have a roughly constant *effect* per step, not a constant coefficient. The natural normalization: divide by the residual norm. Set

  zeta_i = zeta' / ||y - A(x_0_hat(x_i))||,

with `zeta'` a single constant. Watch what that does to the step. The squared-norm gradient is `grad ||r||^2 = 2 ||r|| grad||r||` where `r = y - A(x_0_hat)`; multiplying by `zeta_i = zeta'/||r||` gives `2 zeta' grad||r||`. So the residual-normalized step size on the squared loss is the same as a constant step on the un-squared residual norm, with the factor of `2` absorbed into the name of the constant. That's the right invariant: the guidance now depends on the direction of the data mismatch and a fixed budget, with the magnitude self-normalized, so it behaves consistently from the noisy start to the clean end. A literal noise-variance coefficient is tied to the measurement-noise scale rather than the changing residual scale along the trajectory; the root-residual step gives me one stable knob, with too small a value ignoring the measurement and too large a value forcing saturation and noise amplification.

Now I want to sanity-check this against the closest ancestor, because if I've found the right thing then a known method should fall out inside it. Take the noiseless limit. My Gaussian fidelity step is `grad_{x_i} ||y - A(x_0_hat)||^2` evaluated through the Tweedie estimate. That is exactly the manifold-constrained-gradient term, with the weighting `W = I`. And there's a geometric story for why this particular gradient is the right thing, which I can reconstruct: a single denoising step via the score acts like an orthogonal projection of `x_t` onto the data manifold `M` — the score only resolves the component normal to `M`, since two clean signals differing along a tangent direction look the same to the smoothed density. The measurement is what discriminates points the score can't: and the gradient `grad_{x_i} ||y - A(x_0_hat)||^2`, taken through `x_0_hat`, is precisely the projection of the data-fidelity force onto the *tangent space* `T_{x_0_hat} M`. So denoise = project onto `M`, fidelity gradient = step along `M`; the two together walk along the manifold toward measurement consistency. So far MCG and I agree. Where do we differ? MCG, after that tangent gradient step, additionally projects the iterate onto the measurement subspace `{A x = y}` to nail exact data consistency. In the noiseless case that hard projection is a legitimate exact-consistency correction, so adding it to this gradient recovers the full MCG sampler. With noisy `y`, the same projection becomes the wrong constraint: forcing `A x = y` exactly re-introduces the overfitting-and-amplification pathology, yanks the sample away from the manifold to satisfy corrupted data, and accumulates error. So the noisy generalization is to keep the tangent fidelity gradient that uses the measurement to move along the manifold, but never hard-project onto corrupted data. Geometrically, I stay near `M` throughout; projection-based MCG repeatedly risks stepping off it. That's why dropping the projection is the whole point in the noisy regime, and it also frees me from MCG's per-application choice of `W`.

Let me also confirm I've dealt with the two failure modes I started with. Noise: I never enforce exact consistency — I add a soft likelihood gradient whose Jensen-gap bound decreases with `sigma`, and the residual-normalized step avoids scaling directly with the squared residual. Nonlinearity: the only thing I ask of `A` is differentiability for the backprop `grad_{x_t} ||y - A(x_0_hat(x_t))||^2`; no SVD, no projection-onto-subspace, no linearity. That covers Fourier phase retrieval (`A(x) = |F x|`), neural-network forward models, and any differentiable measurement map autodiff can traverse.

So let me write the sampler in the concrete EDM-scheduled form I'd actually run, because the scaffold already gives me the scheduler, the denoiser returning `x_0_hat`, the unconditional reverse step, and a `forward_op` that returns `grad_{x_0_hat} ||A(x_0_hat) - y||^2` together with the loss `||A(x_0_hat) - y||^2`. The two implementation subtleties I need to get exactly right: (1) the gradient w.r.t. `x_t` is obtained by a vector-Jacobian product — I ask the operator for `grad_{x_0_hat} ||A(x_0_hat) - y||^2`, then backprop it through `x_0_hat(x_t)` to `x_t` via `autograd.grad(denoised, x_cur, that_gradient)`; (2) the residual normalization. The operator hands me `loss_scale = ||A(x_0_hat) - y||^2` and the gradient of that squared loss. To turn the squared-loss gradient into the norm-gradient form, I multiply by `0.5 / sqrt(loss_scale)` — because `grad sqrt(L) = grad L / (2 sqrt(L))`. That factor corresponds to the squared-loss step size `1/(2||r||)`; `self.scale` is the remaining constant root-loss guidance budget:

```python
import torch
from tqdm import tqdm
from .base import Algo
from utils.scheduler import Scheduler
import numpy as np


class DPS(Algo):
    """Diffusion Posterior Sampling: unconditional reverse step + a soft
    likelihood-gradient guidance term computed through the Tweedie estimate.
    Works for any differentiable forward operator (autodiff handles A)."""

    def __init__(self, net, forward_op, diffusion_scheduler_config, guidance_scale, sde=True):
        super().__init__(net, forward_op)
        self.scale = guidance_scale                     # zeta': the one constant step-size budget
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.scheduler = Scheduler(**diffusion_scheduler_config)
        self.sde = sde

    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        if num_samples > 1:
            observation = observation.repeat(num_samples, 1, 1, 1)
        # start from pure Gaussian noise at the largest noise level
        x_initial = torch.randn(num_samples, self.net.img_channels,
                                self.net.img_resolution, self.net.img_resolution,
                                device=device) * self.scheduler.sigma_max
        x_next = x_initial
        x_next.requires_grad = True

        pbar = tqdm(range(self.scheduler.num_steps))
        for i in pbar:
            x_cur = x_next.detach().requires_grad_(True)   # keep a graph for the VJP through x_0_hat

            sigma = self.scheduler.sigma_steps[i]
            factor = self.scheduler.factor_steps[i]
            scaling_factor = self.scheduler.scaling_factor[i]

            # Tweedie posterior mean x_0_hat = E[x_0 | x_t], a function of x_cur
            denoised = self.net(x_cur / self.scheduler.scaling_steps[i],
                                torch.as_tensor(sigma).to(x_cur.device))

            # grad_{x_0_hat} ||A(x_0_hat) - y||^2  and  loss = ||A(x_0_hat) - y||^2
            gradient, loss_scale = self.forward_op.gradient(denoised, observation, return_loss=True)

            # backprop that through x_0_hat(x_cur): VJP gives grad_{x_cur} ||y - A(x_0_hat)||^2
            ll_grad = torch.autograd.grad(denoised, x_cur, gradient)[0]
            # residual-normalize: 0.5/sqrt(loss) turns grad||.||^2 into grad||.||
            ll_grad = ll_grad * 0.5 / torch.sqrt(loss_scale)

            # unconditional reverse step: score in EDM scaling, then SDE / PF-ODE update
            score = ((denoised - x_cur / self.scheduler.scaling_steps[i])
                     / sigma ** 2 / self.scheduler.scaling_steps[i])
            pbar.set_description(f'Iteration {i + 1}/{self.scheduler.num_steps}. '
                                 f'Data fitting loss: {torch.sqrt(loss_scale)}')

            if self.sde:
                epsilon = torch.randn_like(x_cur)
                x_next = x_cur * scaling_factor + factor * score + np.sqrt(factor) * epsilon
            else:
                x_next = x_cur * scaling_factor + factor * score * 0.5

            # likelihood guidance: subtract zeta' * (residual-normalized fidelity gradient)
            x_next = x_next - ll_grad * self.scale
        return x_next
```

Let me trace the causal chain one more time so I'm sure the parts hang together. I wanted to sample the posterior `p(x_0 | y)` with a pre-trained score; Bayes says that's the unconditional reverse SDE plus a likelihood-score drift `grad_{x_t} log p_t(y | x_t)`; that drift is intractable because `y` depends on `x_t` only through the unknown `x_0`. Projection methods dodge it but break under noise and demand linear `A`; spectral methods need an SVD; the linear-likelihood approximation is the wrong time-level term patched by a heuristic. So I marginalized `x_0` and got the exact `p(y | x_t) = E_{x_0 | x_t}[ p(y | x_0) ]`, then noticed I can't compute the posterior `p(x_0 | x_t)` but I *can* compute its mean in closed form — Tweedie's formula, derived from the exponential-family score identity, gives `x_0_hat = E[x_0 | x_t]` straight from the score network. Approximating the expectation of the likelihood by the likelihood at the mean is a Jensen-gap approximation, and bounding that gap with the Lipschitz constant of the Gaussian density and the operator's Jacobian gives `J <= L_sigma ||grad A|| m_1`, with `L_sigma` tending to zero as measurement noise grows. Differentiating the surrogate gives a tractable likelihood gradient through `x_0_hat(x_t)` — an ordinary squared loss for Gaussian noise, a weighted squared loss for the shot-noise approximation to Poisson — so any differentiable nonlinear `A` is fine, with no SVD and no projection. A fixed noise-variance coefficient is not the scale I want across the whole trajectory, so I normalize by the residual norm; in code the `0.5/sqrt(loss)` factor turns the squared-loss VJP into the root-loss VJP, and `self.scale` is the single guidance knob. The Gaussian guidance term recovers the MCG tangent gradient with `W = I`; adding the hard measurement projection in the noiseless setting recovers the full MCG sampler, while dropping that projection is what prevents corrupted measurements from being enforced exactly. The result is a single training-free sampler that conditions a diffusion prior on general noisy nonlinear measurements.
