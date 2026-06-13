Let me start from what actually hurts. I have an unknown signal `x_0` and a degraded, noisy observation `y = f(x_0) + v` with `v ~ N(0, sigma_v^2 I)`, the operator `f` known but the map many-to-one so the problem is ill-posed. I have a magnificent prior sitting on my disk: a pretrained diffusion model for this signal class, a network `epsilon_theta(x_t; t)` that I never want to retrain. The Bayesian thing to do is sample the posterior `p(x_0 | y) ∝ p(y | x_0) p(x_0)`. The likelihood `p(y | x_0) = N(f(x_0), sigma_v^2 I)` is trivial. The prior `p(x_0)` is exactly what the diffusion model encodes. So in principle I have both factors. The entire difficulty is that the diffusion model only gives me the prior in a peculiar form — as a score along a noising trajectory — and stitching the likelihood into that trajectory is where everybody bleeds.

Let me be precise about why. To run the reverse diffusion conditioned on `y` I need the conditional score at every noise level, and Bayes splits it cleanly:

  ∇_{x_t} log p(x_t | y) = ∇_{x_t} log p(y | x_t) + ∇_{x_t} log p(x_t).

The second term is free — that's `-epsilon_theta(x_t; t) / sigma_t`, the pretrained net. The first term is the wall. It looks like it should be as simple as the likelihood, but it isn't, because `x_t` is the *noised* signal, not `x_0`. What I actually need is

  p(y | x_t) = ∫ p(y | x_0) p(x_0 | x_t) dx_0,

and the denoising posterior `p(x_0 | x_t)` is, in general, a complicated multimodal distribution — given a very noisy `x_t` there are many clean images it could have come from. So `p(y | x_t)` has no closed form and I cannot estimate its gradient without training something task-specific, which defeats the plug-and-play goal.

How does the field currently get past this? DPS approximates the integral by its mean. Replace the whole posterior `p(x_0 | x_t)` by a point mass at its mean, the Tweedie/MMSE estimate `x̂_0 = E[x_0 | x_t] = (x_t - sigma_t epsilon_theta) / alpha_t`, so that `p(y | x_t) ≈ p(y | x̂_0)`. Then under Gaussian noise the guidance term is analytic, `∇_{x_t} log p(y|x_t) ≈ -(1/sigma_v^2) ∇_{x_t} || y - f(x̂_0(x_t)) ||^2`, and you add it to the prior score each reverse step. It's general, it handles nonlinear `f`, and it works. But two things gnaw at me. The approximation `p(y|x_t) ≈ p(y|x̂_0)` collapses the mode structure of `p(x_0|x_t)` to a single point, so it is least reliable when the denoising posterior is broad or strongly multimodal, exactly the regime where many intermediate reverse-diffusion steps are trying to decide global structure. And the operational killer: `x̂_0` is a function of `x_t` *through the network*, so `∇_{x_t} || y - f(x̂_0(x_t)) ||^2` forces me to backpropagate through the diffusion denoiser at every single step. That's a score-Jacobian / vector-Jacobian product — memory-hungry, slow, and unstable, and it makes the method touchy about the step-size schedule and the initialization. ΠGDM sharpens the guidance by treating `p(x_0|x_t)` as a Gaussian and inverting `f` through its pseudoinverse, but it buys its sharpness by specializing to linear (and semi-linear, like JPEG) operators, so general nonlinear `f` is out — and it's still a unimodal posterior-score approximation that I differentiate through the model to compute.

So both leading methods are doing the same risky thing: they *approximate the intractable posterior score* and then differentiate the network. Stare at that. The reason the posterior score is intractable is the multimodal `p(x_0|x_t)`. What if I don't try to approximate that score at all? What if I never form `p(x_t | y)` along the trajectory, and instead pose the whole thing as inference over the clean `x_0` directly?

I have a frozen prior `p(x_0)` and a known likelihood `p(y|x_0)`; the only obstacle to writing the posterior is the normalizer `p(y)`. That's the textbook setting for variational inference. So let me posit a tractable family `q(x_0 | y)` and fit it to the true posterior by minimizing

  KL( q(x_0 | y) || p(x_0 | y) ).

Take `q := N(mu, sigma^2 I)`, a Gaussian. `KL(q || p)` is mode-seeking, so `q` will lock onto a dominant mode of the posterior — the image that is both consistent with the observation and high-probability under the prior. That's a feature, not a bug: for an ill-posed problem I want one plausible reconstruction, not a hedge over modes. Now expand the objective with Bayes, `p(x_0 | y) = p(y | x_0) p(x_0) / p(y)`:

  KL( q(x_0|y) || p(x_0|y) ) = E_q[ log q(x_0|y) - log p(y|x_0) - log p(x_0) + log p(y) ]
                             = -E_q[ log p(y|x_0) ] + KL( q(x_0|y) || p(x_0) ) + log p(y).

The last term `log p(y)` is constant in `q` — it doesn't depend on `mu` or `sigma` at all — so it drops out of the minimization. What's left is the evidence lower bound, two terms:

  min_{mu, sigma}  -E_q[ log p(y|x_0) ] + KL( q(x_0|y) || p(x_0) ).

The first term is friendly. With Gaussian measurement noise, `log p(y|x_0) = -(1/(2 sigma_v^2)) ||y - f(x_0)||^2 + const`, so `-E_q[log p(y|x_0)] = (1/(2 sigma_v^2)) E_q[ ||y - f(x_0)||^2 ] + const`. That's just a reconstruction loss — pull `mu` toward signals that fit the observation. The second term, `KL(q || p(x_0))`, is the hard one again, and for a good reason: `p(x_0)` is the diffusion prior, which I only have as a score along the noising trajectory, not as a density I can evaluate at `x_0`. So I've relocated the intractability but not removed it. Wall — for a moment.

The maximum-likelihood theory of diffusion models gives me exactly the missing representation. There's a result (Song et al. 2021; the latent-space cross-entropy version is Vahdat et al. 2021) that a KL between two distributions, *each diffused by the same forward SDE*, equals a time-integrated weighted score-matching loss along the trajectory:

  KL( q(x_0|y) || p(x_0) ) = ∫_0^T (beta(t)/2) E_{q(x_t|y)} [ || ∇_{x_t} log q(x_t|y) - ∇_{x_t} log p(x_t) ||^2 ] dt.

This is gold, because every piece is now something I can compute. `∇_{x_t} log p(x_t)` is the pretrained score `-epsilon_theta(x_t;t)/sigma_t`. And `q(x_t|y)` — the diffused version of my Gaussian `q(x_0|y)` — is itself Gaussian: push `N(mu, sigma^2 I)` through `x_t = alpha_t x_0 + sigma_t epsilon` and you get `q(x_t|y) = N(alpha_t mu, (alpha_t^2 sigma^2 + sigma_t^2) I)`, whose score I can write down in closed form. So the intractable KL has turned into something I evaluate by sampling a time `t`, sampling noise, running the frozen denoiser once, and taking a squared residual. Combining the two terms, the variational objective is

  min_{mu, sigma}  E_q[ ||y - f(x_0)||^2 / (2 sigma_v^2) ]
                 + ∫_0^T (beta(t)/2) E_{q(x_t|y)} [ || ∇_{x_t} log q(x_t|y) - ∇_{x_t} log p(x_t) ||^2 ] dt.

Now simplify. The dispersion `sigma` of `q` is awkward and I suspect it isn't doing much for images — I'll come back to it — so let me take `sigma -> 0`, a Dirac `q(x_0|y) = delta(x_0 - mu)`. Then `x_0 = mu` deterministically, `x_t = alpha_t mu + sigma_t epsilon`, and `q(x_t|y) = N(alpha_t mu, sigma_t^2 I)`. The score of that Gaussian is `∇_{x_t} log q(x_t|y) = -(x_t - alpha_t mu)/sigma_t^2 = -epsilon/sigma_t`. The prior score is `-epsilon_theta(x_t;t)/sigma_t`. Their difference is `(epsilon_theta(x_t;t) - epsilon)/sigma_t`. So the regularizer is an expected squared *noise residual*: how far the network's predicted noise is from the noise I actually injected. Written out, with a general weighting `omega(t)` folded in,

  min_mu  ||y - f(mu)||^2  +  E_{t, epsilon} [ 2 omega(t) (sigma_v/sigma_t)^2 || epsilon_theta(x_t;t) - epsilon ||^2 ],   x_t = alpha_t mu + sigma_t epsilon.

This already feels right. Minimizing it finds an image `mu` that reconstructs `y` through `f` while looking high-probability under the prior, the latter enforced by demanding the denoiser's noise prediction match the injected noise across the whole trajectory. And notice the structure of the objective is an *ensemble* over diffusion steps and noise draws — that's a sum of stochastic terms, which is screaming for stochastic optimization: sample `t` and `epsilon` each iteration, take a gradient step on `mu`, repeat.

But before I celebrate, there's a trap hiding in the regularizer. Its gradient with respect to `mu` — if I take it naively — runs `∇_mu || epsilon_theta(alpha_t mu + sigma_t epsilon; t) - epsilon ||^2`, and `epsilon_theta` depends on `mu` through `x_t`, so I'd be differentiating through the pretrained network again. I'd have reinvented exactly the DPS/ΠGDM cost I was trying to escape: a score Jacobian per step. So the variational reformulation, by itself, has bought me nothing operationally. The whole thing lives or dies on whether I can get the gradient of that regularizer *without* backpropagating through `epsilon_theta`.

Let me look hard at the time integral, because that's where the structure is. The ML-score theory comes with a companion identity — the time-derivative of the running KL is itself the instantaneous score-matching term:

  d KL( q(x_t|y) || p(x_t) ) / dt = -(beta(t)/2) E_{q(x_t|y)} [ || ∇_{x_t} log q(x_t|y) - ∇_{x_t} log p(x_t) ||^2 ].

So my regularizer is `∫_0^T (beta(t)/2) E[...] dt = -∫_0^T (d KL_t / dt) dt`. Now suppose I want a *generic* reweighting `omega(t)`, writing the weight as `omega_tilde(t) = beta(t) omega(t)/2`. Then the weighted regularizer is `-∫_0^T omega(t) (d KL_t/dt) dt`, and integration by parts gives

  -∫_0^T omega(t) (d KL_t/dt) dt = -[ omega(t) KL_t ]_0^T + ∫_0^T omega'(t) KL_t dt.

Look at the boundary term `[omega(t) KL_t]_0^T`. At `t = T` the forward process has destroyed all signal, `x_T` is pure `N(0,I)`, so `q(x_T|y) = p(x_T)` and `KL_T = 0`. At `t = 0` the only way to kill the term is to demand `omega(0) = 0`. So *if I choose any weighting with `omega(0) = 0`*, the boundary term vanishes entirely and

  reg = ∫_0^T omega'(t) E_{q(x_t|y)} [ log( q(x_t|y) / p(x_t) ) ] dt.

That `omega(0) = 0` condition isn't cosmetic — it's the linchpin, and I want to see why by actually taking the gradient. With `sigma = 0` and the reparameterization `x_t = alpha_t mu + sigma_t epsilon`, differentiate under the integral:

  ∇_mu reg = 2 sigma_v^2 ∫_0^T omega'(t) E_epsilon [ ( ∇_{x_t} log q(x_t|y) - ∇_{x_t} log p(x_t) )^T (d x_t / d mu) ] dt.

Now substitute the two scores — `∇ log q = -epsilon/sigma_t`, `∇ log p = -epsilon_theta(x_t;t)/sigma_t` — and `d x_t / d mu = alpha_t I`:

  ∇_mu reg = 2 sigma_v^2 ∫_0^T omega'(t) E_epsilon [ ( (-epsilon/sigma_t) - (-epsilon_theta(x_t;t)/sigma_t) )^T (alpha_t I) ] dt
           = 2 sigma_v^2 ∫_0^T omega'(t) (alpha_t/sigma_t) E_epsilon [ epsilon_theta(x_t;t) - epsilon ] dt.

And there it is — the thing I needed. The difference of *scores* turned the gradient into a plain difference `epsilon_theta(x_t;t) - epsilon`, in which `epsilon_theta` appears as a *value*, not as something I differentiate. The `d x_t/d mu = alpha_t I` factor is the only place `mu` enters, and it's a constant Jacobian. Rewriting the integral as an expectation over `t ~ U[0,T]` by pulling out a `1/T` and absorbing the rest,

  ∇_mu reg = E_{t ~ U[0,T], epsilon ~ N(0,I)} [ lambda_t ( epsilon_theta(x_t;t) - epsilon ) ],   lambda_t := 2 T sigma_v^2 (alpha_t/sigma_t) omega'(t).

I should double-check I haven't fooled myself: doesn't `epsilon_theta(x_t;t)` still secretly depend on `mu`? It does — but the *expression for the gradient* I derived treats it as a fixed evaluation, because the score-difference came out of `∇log q - ∇log p` where `∇log p = -epsilon_theta/sigma_t` is the gradient of the (frozen) prior log-density, evaluated at the current point, not re-differentiated. The reparameterization gradient legitimately lands on `epsilon_theta - epsilon` with `epsilon_theta` stop-gradiented. This is the entire payoff: I get the variational regularizer's gradient with **one forward pass of the frozen denoiser per step and no backprop through it**, where DPS and ΠGDM needed a vector-Jacobian product. And the `omega(0)=0` requirement is what made the boundary term die so this clean form exists; without it the integration by parts leaves a `KL_0` term that reintroduces exactly the messy dependence I'm avoiding. So a seemingly innocuous condition on the weighting is the structural reason the whole method is cheap. (As an aside, `E[epsilon] = 0`, so the `-epsilon` term contributes nothing in expectation and I could drop it — but it's a useful zero-mean control variate, so I'll keep it in the per-sample gradient.)

Now treat sampling as optimization. Since `t` is just a uniform random variable in this gradient, I sample `t` and `epsilon` each iteration and get an unbiased estimate of `∇_mu reg`; add the reconstruction gradient; hand the sum to any first-order optimizer. I can even read off a per-timestep "loss" whose gradient is this: a linear surrogate

  ||y - f(mu)||^2  +  lambda_t ( sg[ epsilon_theta(x_t;t) - epsilon ] )^T mu,

where `sg[·]` is stop-gradient, emphasizing that the score residual is a constant w.r.t. the optimization. Differentiating this gives back `∇||y-f(mu)||^2 + lambda_t (epsilon_theta - epsilon)`, as wanted. So the algorithm is: initialize `mu`; for each step sample `t` and `epsilon`, form `x_t`, run the denoiser once to get `epsilon_theta`, compute the data-fitting gradient and the residual, combine, step. That's it — no Jacobian anywhere, and I can plug in Adam, SGD, momentum, whatever optimizer I like, which makes the sampler fast and easy to tune.

Something about that per-step regularizer `(sg[epsilon_theta - epsilon])^T mu` is tickling a memory. A denoiser-based regularizer whose gradient is a simple *residual*, no differentiation of the denoiser... that's the shape of Regularization by Denoising. In RED you minimize `ℓ(y,x) + (lambda/2) x^T (x - f_den(x))`, and the celebrated trick is that under local homogeneity of the denoiser the gradient of the penalty collapses to `x - f_den(x)` — the residual between the image and its denoised self — so you never differentiate `f_den`. My regularizer gradient is `lambda_t (epsilon_theta(x_t;t) - epsilon)`, a residual between the predicted and injected noise; in the signal domain (I'll do this conversion in a second) it's a residual between `mu` and its denoised estimate. It is the same animal: a residual you read straight off the denoiser. The regularizer is small in two regimes that mirror RED's — either the diffusion reaches its fixed point `epsilon_theta(x_t;t) = epsilon` (the analogue of `x = f_den(x)`), or the residual is pure noise with no image content left, the orthogonality regime. That's why I'll name the method in this RED spirit. But it is not RED, and the differences matter. Classical RED uses a *single, deterministic* denoiser at one noise level and injects no noise into the denoiser's input. That gives me no mechanism for moving across many noise scales or using high-noise denoisers for coarse structure and low-noise denoisers for detail; it is a fixed-point/MAP-style use of one denoising operator. My version is generative: it adds fresh noise to the input of *every* denoiser across the *entire* trajectory and aggregates their feedback, so it can stochastically navigate toward the prior's high-density region. The trajectory, the noise injection, the ensemble of noise levels — that's what makes it move.

Now the weighting, which I've been carrying as an abstract `lambda_t`, because I suspect the noise-residual form is numerically treacherous. The trouble: `lambda_t = 2T sigma_v^2 (alpha_t/sigma_t) omega'(t)`, and even setting aside `omega'`, the residual `epsilon_theta(x_t;t) - epsilon` itself behaves wildly across `t`. For VP diffusion it's known that as `t -> 0` the signal-to-noise ratio climbs fast and this noise residual blows up — equally weighting denoisers across the trajectory lets the tiny-noise steps dominate and destabilize the optimization. So I want a principled way to tame the scale. The cleanest fix is to recast the regularizer in the *signal domain*, where it's directly comparable to the data-fitting term `||y - f(mu)||^2`. In the signal domain the natural penalty is

  ||y - f(mu)||^2  +  lambda ( sg[ mu - mu_hat_t ] )^T mu,

with a single constant `lambda` and `mu_hat_t` the denoiser's MMSE estimate of the clean signal. By Tweedie, `mu_hat_t = E[x_0 | x_t] = (1/alpha_t)(x_t - sigma_t epsilon_theta(x_t;t))`. So how does this signal-domain residual relate to my noise-domain residual? Substitute `x_t = alpha_t mu + sigma_t epsilon`:

  mu_hat_t = (1/alpha_t)( alpha_t mu + sigma_t epsilon - sigma_t epsilon_theta )
           = mu + (sigma_t/alpha_t)( epsilon - epsilon_theta ),

hence

  mu - mu_hat_t = (sigma_t/alpha_t)( epsilon_theta(x_t;t) - epsilon ).

That's exact and lovely. The signal residual is the noise residual times `sigma_t/alpha_t`, which is precisely `1/SNR_t` for `SNR_t := alpha_t/sigma_t`. So to convert my noise-domain gradient `lambda_t (epsilon_theta - epsilon)` into the signal-domain residual `lambda (mu - mu_hat_t)` with a *constant* `lambda`, I just set

  lambda_t = lambda / SNR_t = lambda (sigma_t / alpha_t).

The `1/SNR` weighting isn't a hack — it's the exact rescaling that maps the noise-prediction residual onto the clean-data residual, putting the regularizer in the same units as the fitting term and collapsing the blow-up at small `t`. And it has the right qualitative behavior on its own: it upweights the high-noise early steps (large `t`, small SNR), which build coarse semantic structure, and downweights the low-noise late steps (small `t`, large SNR), which only add fine detail — so a single `lambda` controls the bias-variance / fidelity-perceptual trade-off, with larger `lambda` leaning on the prior and smaller `lambda` leaning on the data. (If I ever want to push the weighting harder I can use `sqrt(1/SNR_t)` or `(1/SNR_t)^2`, but linear `1/SNR` is the one that follows from the signal-domain conversion.) This also tells me which timestep ordering to use: since different `t` govern different scales, and coarse structure should be laid down before detail, I should step `t` in *descending* order from `T` to `0`, like a standard reverse sampler — establish semantics first, refine later.

Let me reconsider the dispersion `sigma` I threw away, to be sure I'm not leaving something on the table. Keep `q = N(mu, sigma^2 I)`. The reparameterized diffused sample is `x_t = alpha_t mu + sqrt(alpha_t^2 sigma^2 + sigma_t^2) epsilon`; define `eta_t := (1 + sigma^2 (alpha_t/sigma_t)^2)^{1/2}` so `x_t = alpha_t mu + eta_t sigma_t epsilon` (and `eta_t = 1` recovers `sigma = 0`). Redo the mean gradient with the same constants as before: now `∇_{x_t} log q(x_t|y) = -epsilon / (eta_t sigma_t)`, while `∇_{x_t} log p(x_t) = -epsilon_theta/sigma_t`, so

  ∇_mu reg(mu, sigma) = 2 sigma_v^2 ∫ omega'(t) E_epsilon [ ( (-epsilon/(eta_t sigma_t)) - (-epsilon_theta/sigma_t) )^T (alpha_t I) ] dt
                      = E_{t,epsilon}[ lambda_t (epsilon_theta(x_t;t) - eta_t^{-1} epsilon) ]
                      = E_{t,epsilon}[ lambda_t epsilon_theta(x_t;t) ],

since `E[epsilon] = 0` again kills the `eta_t^{-1} epsilon` term in expectation — so the mean gradient is essentially unchanged. The dispersion gradient comes out in closed form too,

  ∇_sigma reg(mu, sigma) = sigma E_{t,epsilon}[ lambda_t 2 eta_t^{-1} (alpha_t/sigma_t) epsilon^T ( epsilon_theta(x_t;t) - eta_t^{-1} epsilon ) ],

so optimizing `sigma` would be cheap. But I don't think Gaussian dispersion is the right way to inject diversity into images: perturbing an image with isotropic Gaussian noise doesn't move you to another *legitimate* image — it just makes it noisy, off the natural-image manifold. Real diversity needs a dispersion model that respects the manifold, which is a much harder object. So I'll keep `sigma = 0` and get diversity instead from the random `epsilon` draws and the optimizer's stochasticity. The Dirac choice is deliberate, not lazy.

Let me also settle the optimizer details, since "use any first-order optimizer" is too glib. Adam is the natural default — adaptive per-coordinate step sizes handle the very different gradient scales of the fitting term (which depends on `f`) and the residual term. The learning rate is a real tuning knob, so I should expose it rather than bury it in the schedule. I'll use momentum betas like `(0.9, 0.99)` — a slightly shorter second-moment window than the usual `0.999` is reasonable here because the objective changes as `t` descends, so I don't want the denominator to be too sluggish. No weight decay — there's no parameter-shrinkage prior on `mu`, the diffusion regularizer *is* the prior. For initialization, either a pseudo-inverse warm start or zeros can fit the framework; a compact implementation can start from zeros and let the stochastic residuals plus the data gradient move `mu`.

So let me put the whole reasoning into the code I'd actually run, filling the one empty slot — the `inference` body — using the frozen denoiser, the forward operator's data-fitting gradient, the VP schedule, and an off-the-shelf Adam. The clean signal estimate I optimize is `mu`; each step I noise it to `x_t`, read the denoiser's noise prediction `epsilon_theta`, detach it, choose the configured residual weight (with the linear option giving `lambda/SNR_t` because the scheduler's `sigma` is inverse SNR), add the data-fitting gradient, and let Adam move `mu`:

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

        # Random initialization for the optimized clean estimate.
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

Let me trace the causal chain one last time. I want posterior samples from a pretrained diffusion prior and a known forward operator, but the reverse-diffusion route needs the likelihood score `∇log p(y|x_t)`, which is intractable because the denoising posterior `p(x_0|x_t)` is multimodal — and DPS and ΠGDM cope by approximating that score with a point estimate and then *differentiating through the network*, which is loose when the denoising posterior is broad and expensive because it needs a denoiser Jacobian. So instead of approximating the trajectory score, I posed inference directly on `x_0`: fit a Gaussian `q(x_0|y)` to the true posterior by minimizing `KL(q || p(x_0|y))`, which Bayes-expands into a reconstruction term plus `KL(q || p(x_0))`; the maximum-likelihood diffusion identity turns that prior-KL into a weighted score-matching integral the frozen network can evaluate. With `sigma -> 0` the objective became a measurement-fitting term plus a noise-residual regularizer, and the integration-by-parts identity — provided the weighting satisfies `omega(0) = 0` so the boundary term dies — collapsed the regularizer's gradient to `lambda_t (epsilon_theta - epsilon)`, a residual with **no denoiser Jacobian**, which is the whole advantage over DPS/ΠGDM. That residual structure is RED's, but generative: noise injected to every denoiser across the full trajectory, not one deterministic denoiser, so the iterates actually navigate the manifold. The noise residual blows up at small `t`, so I rescaled it to the signal domain via the exact identity `mu - mu_hat_t = (sigma_t/alpha_t)(epsilon_theta - epsilon)`, which forced `lambda_t = lambda/SNR_t` — a single interpretable knob that also upweights coarse-structure steps and downweights detail steps, telling me to step `t` descending. Dispersion stays at zero because Gaussian perturbations leave the image manifold. And because the gradient is a single forward pass, the whole thing drops into an off-the-shelf Adam loop: sampling has become stochastic optimization.
