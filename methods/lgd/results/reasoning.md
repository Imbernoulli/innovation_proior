I have a pretrained diffusion model sitting in front of me — a beautiful, expensive prior over some signal space, `p_0(x_0)`, baked into a denoiser `D_θ`. And I have a condition I want to impose at test time, written as a differentiable loss `l_y(x_0)` on clean signals: a measurement-consistency term `‖A(x_0) - y‖^2` for an inverse problem, a classifier's negative log-probability, a CLIP score, a "stay on this path and avoid that obstacle" penalty for a motion model. I do not want to retrain anything. I just want to draw samples that respect both the prior and the loss, which means sampling from the posterior

```
p^(l)_0(x_0 | y) ∝ p_0(x_0) · exp(-l_y(x_0)),
```

where I read `exp(-l_y(x_0))` as an (unnormalized) likelihood `p_0(y | x_0)`. The normalizer `Z = ∫ p_0 exp(-l) dx_0` I'll carry along and watch it drop out wherever it can; if `l` were so flat that `Z` failed to exist I could always add a tiny `x_0^2` to the loss to make the whole thing sub-Gaussian, but for any real measurement loss that won't be an issue.

The problem is that a diffusion sampler doesn't know about `y`. What a sampler integrates is the reverse process, and the only thing driving that reverse process is the *score* of the noised marginal at each level. So I need the conditional score `∇_{x_t} log p_t(x_t | y)` at every noise level `t`, not just at `t = 0`. Bayes' rule on the noised variable splits it cleanly:

```
∇_{x_t} log p_t(x_t | y) = ∇_{x_t} log p_t(x_t) + ∇_{x_t} log p_t(y | x_t).
```

The first term I already have — that's exactly what the pretrained model gives me, via Tweedie: `D_θ(x_t, t) = E[x_0 | x_t] = x_t + σ_t^2 ∇_{x_t} log p_t(x_t)`, so `∇_{x_t} log p_t(x_t) = (D_θ(x_t,t) - x_t)/σ_t^2`. One network call, score in hand. The second term — the *guidance* term `∇_{x_t} log p_t(y | x_t)` — is the whole game, and I don't have it. Everything from here is: how do I manufacture that term cheaply and accurately, at every noise level, from a single query of `D_θ`?

So let me write down what `p_t(y | x_t)` actually is. The condition `y` is tied to the *clean* signal, not the noisy one. In the graphical model the chain is `x_0 → x_t` (forward noising) and `x_0 → y` (the loss). Given `x_0`, the noisy `x_t` and the condition `y` are independent — `y` only cares about `x_0`. So

```
p_t(y | x_t) = ∫ p(x_0 | x_t) p_0(y | x_0) dx_0 = E_{x_0 ~ p(x_0 | x_t)}[ p_0(y | x_0) ].
```

There it is, and there's the trap. The guidance likelihood at noise level `t` is an *expectation of the clean-data likelihood over the denoising posterior* `p(x_0 | x_t)`. And `p(x_0 | x_t)` is the genuinely hard object: to sample it or evaluate it accurately I'd have to run many diffusion steps, which is precisely the cost I'm trying to avoid. If I insist on one network evaluation, the integral is intractable. I can't compute it; I have to approximate it. Fine — but the *way* I approximate it is going to decide whether this works at all.

What does the existing approach do? It takes the cheapest possible approximation of `p(x_0 | x_t)`: a single point. Tweedie hands me the MMSE estimate `x_hat_t = D_θ(x_t, t) = E[x_0 | x_t]` for free, so collapse the posterior to a delta mass there. Then

```
p_t(y | x_t) ≈ p_0(y | x_hat_t),    ∇_{x_t} log p_t(y | x_t) ≈ ∇_{x_t} log p_0(y | x_hat_t) = -∇_{x_t} l_y(x_hat_t).
```

The normalizer `Z` doesn't depend on `x_t`, so taking `∇_{x_t} log` kills it — good, one fewer thing to worry about. And the gradient `∇_{x_t} l_y(x_hat_t)` is just the loss gradient on the clean estimate, backpropagated through the denoiser to `x_t`. For a Gaussian measurement this becomes a negative squared-residual gradient through the denoiser, commonly written as `-ρ ∇_{x_t}‖y - A(x_hat_t)‖^2` with a likelihood step scale `ρ`. This is DPS, and it's genuinely useful — it handles nonlinear `A`, it handles measurement noise, it's plug-and-play. I should keep its *structure*: one network call, backprop the clean-data loss through the denoiser.

But I want to interrogate the delta. Replacing `E[exp(-l(x_0))]` by `exp(-l(E[x_0]))` is swapping the order of a nonlinear function and an expectation — that's a Jensen-type move, and Jensen swaps are only exact when the function is affine or the distribution is a point. Here the distribution `p(x_0 | x_t)` is emphatically not a point, especially at high noise where the posterior over clean signals is enormously spread out. So I expect a bias. The question is whether it's a benign bias or a structural one.

Let me build the smallest example where I can see the truth, because I can't reason about the bias in the abstract — I need a case where `p_t(y | x_t)` is computable in closed form and I can hold it up against the delta approximation. Take `x_0` one-dimensional, `p_0` a mixture of two well-separated Gaussians, say `p(x_0 | y=0) = N(-1, 0.2^2)` and `p(x_0 | y=1) = N(+1, 0.2^2)`, equal weights, and let `y ∈ {0,1}` be the mixture label. Then every noised marginal `p_t(x_t)` is itself a Gaussian mixture, and `p_t(y | x_t)` — the probability that the label is 0 given the noisy observation — is a tractable logistic function of `x_t`. So I can plot the true guidance `∇_{x_t} log p_t(y=0 | x_t)` exactly, at any `σ_t`, and compare it to the delta's `-∇_{x_t} l_y(x_hat_t)`.

Now think about what the *clean-data* guidance gradient looks like. `log p^(l)(y=0 | x_0)` is large and steep right at the decision boundary `x_0 ≈ 0` — that's where moving `x_0` flips the label — and goes flat (gradient ≈ 0) once `x_0` is deep inside either mode, because there the label is already decided. A gradient that's big near zero and dies away from zero.

So trace the delta through the schedule. At *high* noise, `σ_t` large: the MMSE estimate `x_hat_t = E[x_0 | x_t]` of a symmetric two-mode mixture is pulled toward the middle, `x_hat_t ≈ 0`. That lands the delta right on the steep part of the clean-data gradient — so the delta reports a *large* guidance. But the *true* expectation at high noise averages `exp(-l)` over a posterior `p(x_0 | x_t)` that's smeared across *both* modes; the contributions from the two sides partly cancel, and the true guidance is *small*. At *low* noise, `σ_t` small: `x_hat_t` has committed to one mode, sitting deep in a flat region, so the delta reports a *small* guidance. But the true posterior is now tight around a single clean signal, so the true expectation's gradient is *large*. The delta has it exactly backwards at both ends: **too large at high noise, too small at low noise.** And I can see this isn't an artifact of my toy — it's a direct consequence of evaluating the gradient at one point (the mean) instead of averaging the function over the spread. Wherever the loss gradient is curved across the support of `p(x_0 | x_t)`, the mean of the gradient and the gradient at the mean disagree, and the disagreement scales with how spread the posterior is — which is enormous at high noise and shrinks to nothing at low noise. That's the structural bias.

The tell that the field already knows something is off: the standard fix divides the delta's guidance by the denoiser's negative log-likelihood times a constant. The NLL is large when `σ_t` is large (the MMSE prediction is unreliable) and small when `σ_t` is small — so dividing by it shrinks the guidance at high noise and inflates it at low noise, which is exactly the direction of my correction. That heuristic *works* because it's compensating for precisely the scale error I just derived. But it's a patch bolted onto the wrong object. The honest fix is to stop collapsing the posterior to a point.

So: don't use a delta. Use a *spread-out* surrogate `q(x_0 | x_t)` for the denoising posterior, and estimate the expectation `E_{x_0 ~ q}[exp(-l(x_0))]` honestly. The structure I must preserve is the single network call — whatever `q` is, it has to be buildable from the one `D_θ(x_t, t)` evaluation, because backpropagating through the diffusion network is the expensive part and I refuse to pay for more than one.

What `q` should I pick? I want it close to the true `p(x_0 | x_t)` but cheap. The cheapest non-degenerate choice is a Gaussian, `q(x_0 | x_t) = N(µ(x_t), r_t^2 I)` — a mean and an isotropic spread. Two questions: where to center it, and how wide.

The center first. I want `q` to be the best Gaussian approximation, in the sense of putting mass where the true posterior puts mass. Take the variational view: maximize the expected log-likelihood of the true clean signals under `q`,

```
max_q  E_{x_0 ~ p(x_0 | x_t)}[ log q(x_0 | x_t) ].
```

With `q = N(µ, r_t^2 I)` at fixed covariance, `log q = -‖x_0 - µ‖^2/(2 r_t^2) + const`, so this is equivalent to

```
min_µ  E_{x_0 ~ p(x_0 | x_t)}[ ‖µ - x_0‖^2 ],
```

and the minimizer of a mean-squared distance to a random variable is its mean: `µ(x_t) = E[x_0 | x_t] = x_hat_t`. The MMSE estimate isn't just a convenient point — it's provably the *optimal mean* of any fixed-covariance Gaussian surrogate. And I get it from the one network call I'm already making. So center `q` at `x_hat_t`, same as DPS. The difference is going to be entirely in giving it a width.

The width `r_t`. I want it to be roughly the spread of the *true* denoising posterior `p(x_0 | x_t)`, because that's the object `q` is standing in for. Reason it out from the noising model. With `x_t = x_0 + σ_t ε` and a roughly unit-scale prior on `x_0` (signals normalized to order one), the posterior over `x_0` given `x_t` is a Gaussian-conjugate balance of two precisions: the prior precision (≈ 1) and the likelihood precision from the observation (`1/σ_t^2`). Posterior precision `= 1 + 1/σ_t^2 = (1 + σ_t^2)/σ_t^2`, so posterior variance `= σ_t^2 / (1 + σ_t^2)`, i.e.

```
r_t = σ_t / sqrt(1 + σ_t^2).
```

Look at the limits and they're exactly right. At high noise `σ_t → ∞`, `r_t → 1` — the surrogate is as wide as the prior, because a very noisy observation tells you almost nothing about `x_0` and the posterior is essentially the prior. At low noise `σ_t → 0`, `r_t → σ_t → 0` — the surrogate collapses to a delta, because a clean observation pins `x_0` down. And that low-noise collapse is the reassuring part: as `r_t → 0` my Gaussian surrogate *becomes* the DPS delta, so I recover the point-estimate behavior exactly where the point estimate is accurate. I'm not throwing DPS away; I'm fixing it where it's broken (high noise, wide posterior) and leaving it alone where it was fine (low noise, tight posterior).

Can I justify that a wider, better-shaped `q` is the right direction, or am I just hoping? Let me bound the error by the distance from the true posterior. The bias I care about is the gap between the true and surrogate expectations of the likelihood,

```
| E_{p(x_0|x_t)}[p_0(y|x_0)] - E_{q(x_0|x_t)}[p_0(y|x_0)] |.
```

Write the difference as a single integral and pull the bounded likelihood out. Let `M = max_{x_0} p_0(y | x_0)` (the loss is such that `exp(-l)` is bounded). Then

```
|E_p[p_0(y|x_0)] - E_q[p_0(y|x_0)]| = |∫ ( p(x_0|x_t) - q(x_0|x_t) ) p_0(y|x_0) dx_0|
   ≤ max_{x_0} p_0(y|x_0) · ∫ | p(x_0|x_t)/q(x_0|x_t) - 1 | q(x_0|x_t) dx_0
   = M · ∫ | p(x_0|x_t) - q(x_0|x_t) | dx_0
   = 2M · TV( p(x_0|x_t), q(x_0|x_t) ),
```

using `∫|p - q| = 2·TV(p, q)`. So the bias is controlled by the *total variation distance* between my surrogate and the true posterior, times a constant `2M`. That's the lever. The DPS delta is maximally far from a continuous posterior with real width: a point mass and an absolutely continuous spread-out distribution have `TV = 1`. A Gaussian with approximately the right mean *and the right width* is designed to be closer in TV, and whenever it is closer, the worst-case upper bound is smaller. This is an asymptotic statement — it's about the bias when I can evaluate `E_q` exactly — so it tells me about the *choice of `q`*, not yet about how many samples I'll need. But it settles the design question: spread is good, and matching the posterior width is the principled way to spread.

Now, how do I actually compute `E_{x_0 ~ q}[exp(-l(x_0))]` and its gradient with one network call? I can't do the integral in closed form for a general `l`. So Monte Carlo it: draw `n` i.i.d. samples from `q` and average. But I have to be careful *what* I average, because the quantity I need is

```
∇_{x_t} log p_t(y | x_t) = ∇_{x_t} log E_{x_0 ~ q}[ exp(-l_y(x_0)) ].
```

The expectation is *inside* the log. So the right estimator replaces the expectation by the sample mean and then takes the log:

```
MC_n(x_t, y) = ∇_{x_t} log( (1/n) Σ_{i=1}^n exp(-l_y(x^(i))) ),    x^(i) ~ q(x_0 | x_t).
```

This is a log-mean-exp-of-the-negative-loss. I want to be careful here, because the naive thing — average the loss gradients, `(1/n) Σ_i ∇l(x^(i))` — would be estimating `∇ E[l]`, which is a different and wrong quantity; it's the same Jensen confusion one level down. The expectation has to wrap `exp(-l)`, not `l`. Concretely: differentiating the log-mean-exp gives a *softmin-weighted* average of the per-sample loss gradients, where samples with smaller loss (larger `exp(-l)`) get more weight:

```
∇_{x_t} log( (1/n) Σ_i exp(-l(x^(i))) ) = - Σ_i w_i ∇_{x_t} l_y(x^(i)),    w_i = exp(-l(x^(i))) / Σ_j exp(-l(x^(j))).
```

That's the correct gradient of the surrogate guidance. The exact DPS formula appears when the surrogate itself collapses to a delta at `x_hat_t`; with positive `r_t`, `n = 1` is only a one-sample Gaussian estimate, so I should not confuse "one sample" with "the DPS delta." Increasing `n` only reduces the Monte Carlo error around the chosen surrogate expectation; the remaining gap to the true guidance is the bias controlled by how close `q(x_0 | x_t)` is to the real denoising posterior.

For this gradient to flow I need differentiability through the sampling, so reparameterize: `x^(i) = x_hat_t + r_t ε^(i)`, `ε^(i) ~ N(0, I)`. Every sample is a deterministic function of `x_hat_t = D_θ(x_t, t)` plus fixed noise, so the gradient path runs from each `x^(i)`, through the loss, back through the *shared* `x_hat_t`, and through `D_θ` to `x_t`. The crucial efficiency fact: there is only *one* `D_θ` in that path. All `n` samples branch off the same denoiser output. So I backpropagate through the diffusion network exactly once regardless of `n`; the only thing that scales with `n` is evaluating and differentiating the loss `l_y`, which is cheap whenever the forward operator and loss are cheap relative to a UNet backward pass.

Let me make the one-backprop concrete, because it changes how I write the code. For the exact log-mean-exp estimator, I compute the per-sample loss gradients with respect to the *samples* `x^(i)`, weight them by the softmin weights `w_i`, aggregate them into the cotangent `g = Σ_i w_i ∇_{x^(i)} l_y(x^(i))`, and then call `autograd.grad(x_hat_t, x_t, g)` — one VJP through the denoiser. The estimated score contribution is the negative of that VJP, `-J_D^T g`; if I store `J_D^T g` as a loss-gradient update, I have to subtract it in the sampler. The expensive backward happens once; `n` only multiplies the cheap loss and forward-operator gradients.

Now I have to decide what the inverse-problem implementation actually uses. Its forward operator exposes squared-residual gradients and a squared-residual loss, and the practical loop takes the arithmetic mean of the per-sample residual gradients before the one VJP:

```
g = (1/n) Σ_i ∇_{x^(i)} ‖A(x^(i)) - y‖^2.
```

That is not the exact softmin-weighted log-mean-exp gradient for an arbitrary loss; the exact version would use `w_i`. It is the least-squares implementation path I can run cheaply with this operator interface: preserve the posterior spread through the samples, average the cheap residual gradients, then root-normalize the resulting loss-gradient update. I should keep that distinction sharp — the derivation gives the log-mean-exp and its softmin weights; this code path realizes a mean-gradient approximation specialized to the squared-residual inverse-problem interface.

Now the scale of the step. The raw squared-residual guidance for a Gaussian measurement, `ρ∇‖y - A(x_hat)‖^2`, has a notoriously unstable magnitude — the likelihood scale can blow up at low noise, and `∇‖·‖^2 = 2(·)∇(·)` carries the residual magnitude, which swings wildly over a run. DPS's practical cure, which I'll inherit because it's the thing that actually makes these methods stable, is to normalize the squared-residual gradient by the residual norm: use `ζ_i = ζ' / ‖y - A(x_hat)‖` for a constant `ζ'`. Watch what that does. With `loss = ‖A(x) - y‖^2`,

```
0.5 / sqrt(loss) · ∇loss = ∇‖A(x) - y‖^2 / (2‖A(x) - y‖) = ∇‖A(x) - y‖ = ∇ sqrt(loss),
```

so multiplying the squared-loss gradient by `0.5/sqrt(loss)` turns it into the gradient of the *root* loss — the residual norm itself, not its square. That tames the magnitude: the root-loss gradient has a far gentler scale than the squared-loss gradient, it doesn't carry the extra factor of the residual, and it's the empirically stable form. On top of that I keep a single overall `guidance_scale` multiplier — the classifier-guidance lesson, that `s · ∇ log p(y|x) = ∇ log(p(y|x)^s/Z)` sharpens the conditional toward the loss's modes, so the scale is a fidelity-versus-prior-fidelity knob I tune per problem. (Where the loss already has a known scale, like super-resolution's `l = ‖y - Hx_0‖^2/(2 s_t^2)`, one folds the appropriate `s_t^2` factor in to keep the late-sampling gradients from exploding; the principle is the same — control the magnitude of the guidance, don't let `1/σ^2` run away.)

So now I assemble the whole reverse step. I'm in an EDM-style sampler: the prior is integrated by the probability-flow ODE or its stochastic SDE sibling. In the scaled-variable EDM form the update at step `i`, with schedule `σ`, scaling `s`, step size `Δt`, is built from precomputed coefficients `scaling_factor = 1 - (ṡ/s)Δt` and `factor = 2 s^2 σ̇ σ Δt`, and the score evaluated on the unscaled variable, `score = (denoised - x_cur/s)/σ^2/s`. The unconditional SDE step is `x_next = x_cur·scaling_factor + factor·score + sqrt(factor)·ε`, and the ODE (deterministic) step drops the noise and halves the score, `x_next = x_cur·scaling_factor + factor·score·0.5`. Then I subtract the loss-gradient update, which is the same as adding the estimated `∇_{x_t} log p_t(y | x_t)`, because `l` is a loss to be minimized and `log p_t(y | x_t)` moves in the `-∇l` direction. The sign matters and it's easy to get backwards: the score already points up the prior; the loss-gradient update points toward *worse* fit, so I subtract it to go toward better fit.

Let me write the per-iteration logic, filling the one empty slot — the guidance estimator — in the inference loop. Initialize from noise at `σ_max`, and at each step: detach and re-enable grad on the current iterate (so each step's backprop is fresh); read the schedule coefficients and compute `r_t = σ/sqrt(1+σ^2)`; one denoiser call to get `x_hat_t`; draw `n` reparameterized samples `x^(i) = x_hat_t + r_t ε^(i)`; get the per-sample measurement gradients and losses from the forward operator; aggregate the gradients (mean, detached) into the cotangent; push it back through the denoiser with a single VJP; normalize by `0.5/sqrt(avg_loss)` to get the stable root-loss update; take the unconditional EDM reverse step; subtract `loss_update·scale`.

```python
import torch
from tqdm import tqdm
from .base import Algo
from utils.scheduler import Scheduler
import numpy as np

import wandb


class LGD(Algo):
    def __init__(self,
                 net,
                 forward_op,
                 diffusion_scheduler_config,
                 guidance_scale,
                 num_samples=10,
                 batch_grad=True,
                 sde=True):
        super(LGD, self).__init__(net, forward_op)
        self.scale = guidance_scale
        self.diffusion_scheduler_config = diffusion_scheduler_config
        self.scheduler = Scheduler(**diffusion_scheduler_config)
        self.sde = sde
        self.num_samples = num_samples
        self.batch_grad = batch_grad

    def inference(self, observation, num_samples=1, **kwargs):
        device = self.forward_op.device
        x_initial = torch.randn(num_samples, self.net.img_channels, self.net.img_resolution,
                                self.net.img_resolution, device=device) * self.scheduler.sigma_max
        x_next = x_initial
        x_next.requires_grad = True
        pbar = tqdm(range(self.scheduler.num_steps))

        for i in pbar:
            x_cur = x_next.detach().requires_grad_(True)

            sigma, factor, scaling_factor = self.scheduler.sigma_steps[i], self.scheduler.factor_steps[i], \
                self.scheduler.scaling_factor[i]
            rt = sigma / np.sqrt(1 + sigma ** 2)

            denoised = self.net(x_cur / self.scheduler.scaling_steps[i], torch.as_tensor(sigma).to(x_cur.device))

            samples = denoised + torch.randn((self.num_samples, *denoised.shape[1:]), device=device) * rt

            if self.batch_grad:
                gradient, loss_scale = self.forward_op.gradient(samples, observation, return_loss=True)
                gradients = gradient
                avg_loss = loss_scale
            else:
                gradients = torch.empty((self.num_samples, *denoised.shape[1:]), device=device)
                losses = np.empty(self.num_samples)
                for j in range(self.num_samples):
                    gradient, loss_scale = self.forward_op.gradient(samples[j:j+1], observation, return_loss=True)
                    gradients[j] = gradient
                    losses[j] = loss_scale
                avg_loss = losses.mean()

            avg_grad = torch.mean(gradients, dim=0, keepdim=True).detach()

            ll_grad = torch.autograd.grad(denoised, x_cur, avg_grad)[0]
            ll_grad = ll_grad * 0.5 / torch.sqrt(avg_loss)

            score = (denoised - x_cur / self.scheduler.scaling_steps[i]) / sigma ** 2 / self.scheduler.scaling_steps[i]
            pbar.set_description(f'Iteration {i + 1}/{self.scheduler.num_steps}. Data fitting loss: {torch.sqrt(loss_scale)}')
            if wandb.run is not None:
                wandb.log({'data_fitting_loss': torch.sqrt(loss_scale)})

            if self.sde:
                epsilon = torch.randn_like(x_cur)
                x_next = x_cur * scaling_factor + factor * score + np.sqrt(factor) * epsilon
            else:
                x_next = x_cur * scaling_factor + factor * score * 0.5
            x_next -= ll_grad * self.scale

        return x_next
```

Let me trace the causal chain once more to be sure it holds together. I needed the conditional score; Bayes split it into the free unconditional score plus a guidance term that is an expectation of the clean-data likelihood over the intractable denoising posterior `p(x_0 | x_t)`. The cheap existing fix collapses that posterior to the Tweedie point `x_hat_t`, and the mixture-of-Gaussians diagnostic shows the resulting guidance is mis-scaled at both ends of the schedule — too large at high noise, too small at low noise — because evaluating a curved loss gradient at the posterior mean is not the same as averaging it over a spread posterior, and the spread is exactly what's large at high noise. So I keep DPS's single-network-call structure but replace the delta with a Gaussian surrogate `q = N(x_hat_t, r_t^2 I)`: the mean `x_hat_t` is the provably optimal fixed-covariance mean (the variational argument reduces to minimizing mean-squared distance, whose minimizer is the posterior mean), and the width `r_t = σ_t / sqrt(1+σ_t^2)` is the conjugate posterior std, which widens to the prior at high noise and collapses to the DPS delta at low noise. The total-variation bound gives `2M · TV(p, q)` as the worst-case bias control. For the exact estimator I differentiate a log-mean-exp of the negative loss over `n` reparameterized samples, which gives softmin weights on the per-sample loss gradients. For the inverse-problem implementation here, I use the canonical mean-gradient approximation over the sampled squared-residual gradients, then one VJP through the denoiser. The step is stabilized by the DPS root-loss normalization `0.5/sqrt(loss)` and an overall classifier-guidance-style scale, and subtracted from the standard EDM reverse SDE/ODE step. The result is a plug-and-play loss update computed from one network call per step, with posterior spread restored where the point estimate was mis-scaled.
