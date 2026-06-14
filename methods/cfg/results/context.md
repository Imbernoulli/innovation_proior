# Context: conditional diffusion sampling and the fidelity/diversity knob (circa 2021)

## Research question

Conditional diffusion models can draw samples `x ~ p(x|c)` for a class label or other
conditioning `c`, and on likelihood and coverage they are excellent. But practitioners want one
more thing that other generative families already give them: a *post-training knob* that trades
sample diversity for per-sample fidelity. In a GAN you get this with truncated sampling
(shrinking the input noise range); in a normalizing flow you get it with low-temperature
sampling (scaling down the latent noise). Both produce a smooth tradeoff curve — sweep the knob
and watch a perceptual-quality metric improve while coverage falls. Diffusion models had no such
knob, and the obvious analogues do not work: scaling the model's predicted score, or shrinking
the Gaussian noise added in the reverse process, both just yield blurry, low-quality samples.

The precise goal is therefore a controllable diffusion sampler: a single scalar that, swept from
"off" to "strong," moves the model along a fidelity/diversity tradeoff comparable to GAN
truncation — and that does so (1) *after* training, without retraining the generator per setting;
(2) ideally without bolting an extra trained component onto the pipeline; and (3) without the
sampling procedure degenerating into something that merely games the metrics used to evaluate it
(the Inception-based IS/FID scores are themselves classifier outputs). A method that needed an
auxiliary network, or whose update direction looked like an adversarial perturbation of an image
classifier, would satisfy (1) but leave (2) and (3) open.

## Background

A diffusion model is trained in continuous time over a variance-preserving forward process. Let
`x ~ p(x)` and index the latents by the log signal-to-noise ratio `lambda` running over
`[lambda_min, lambda_max]`. The forward marginals are

```
q(z_lambda | x) = N(alpha_lambda * x, sigma_lambda^2 * I),
    alpha_lambda^2 = 1/(1 + e^{-lambda}),  sigma_lambda^2 = 1 - alpha_lambda^2,
```

so that `lambda = log(alpha_lambda^2 / sigma_lambda^2)` is exactly the log-SNR of `z_lambda`, and
the forward (noising) process runs in the direction of decreasing `lambda`. The reverse process
is a Markov chain `p_theta(z_{lambda'} | z_lambda)` whose mean is obtained by plugging a denoised
estimate of `x` into the (tractable) forward posterior `q(z_{lambda'} | z_lambda, x)`; sampling
applies these transitions along an increasing `lambda` schedule from pure noise at `lambda_min`.

The network is parameterized in **epsilon-prediction** form (Ho, Jain & Abbeel 2020). Writing the
noised latent as `z_lambda = alpha_lambda * x + sigma_lambda * eps` with `eps ~ N(0, I)`, the
model `eps_theta(z_lambda)` is trained to recover the injected noise,

```
E_{eps, lambda} [ || eps_theta(z_lambda) - eps ||^2 ],
```

and the denoised estimate ("Tweedie" form) is `x_theta(z_lambda) = (z_lambda - sigma_lambda *
eps_theta(z_lambda)) / alpha_lambda`. This objective is denoising score matching (Vincent 2011;
Hyvarinen 2005) at every noise scale (Song & Ermon 2019), which ties the trained network to the
gradient of the log-density of the noised data:

```
eps_theta(z_lambda) ≈ - sigma_lambda * grad_{z_lambda} log p(z_lambda).
```

This **score / noise identity** is load-bearing: it means a denoiser *is* (up to the `-sigma`
factor) a score-function estimator, so sampling resembles Langevin dynamics on a sequence of
noise-mollified densities `p(z_lambda)` that converges to the data distribution (Song et al.
2020, score-based SDE; Song & Ermon 2019). One caveat sits in the background and will matter: the
network `eps_theta` is an *unconstrained* neural map from `z_lambda` to a vector field; there need
not exist any scalar potential whose gradient it is. A denoiser is only *approximately* a
conservative (gradient) field.

Conditioning is trivial to add at the architecture level: feed `c` to the network as an extra
input, `eps_theta(z_lambda, c)`, and train on data `(x, c)` drawn jointly. That gives a
conditional score estimator `eps_theta(z_lambda, c) ≈ -sigma_lambda * grad log p(z_lambda | c)`.

The motivating empirical facts about existing systems, knowable before any new method:

- **Other generative families already have a fidelity/diversity knob.** Truncated BigGAN (Brock
  et al. 2019) traces out an FID-vs-IS tradeoff curve as truncation is varied; low-temperature
  Glow (Kingma & Dhariwal 2018) behaves similarly. The desire is to reproduce this in diffusion.
- **The naive diffusion analogues fail.** Scaling the model's score vectors, or decreasing the
  variance of the Gaussian noise injected in the reverse process, both produce blurry, low-quality
  samples rather than a clean fidelity boost. The straightforward "turn down the temperature"
  moves do not transfer.
- **Raising a distribution to a power sharpens it.** A general fact used throughout: `p(.)^s` for
  `s > 1` shifts probability mass from low-density regions onto the modes — an inverse-temperature
  operation. In score (gradient-of-log) space, raising to the power `s` is just multiplying the
  score by `s`, because `grad log p^s = s * grad log p`.

## Baselines

**Classifier guidance (Dhariwal & Nichol, arXiv:2105.05233).** The first method to give diffusion
a working fidelity/diversity knob. Train, alongside the diffusion model, an auxiliary classifier
`p_phi(c | z_lambda)` on the *noised* latents (it must see the same noise distribution the sampler
visits, so a standard pretrained clean-image classifier cannot be dropped in). At sampling time,
add the classifier's input-gradient to the diffusion score. Starting from the conditional-reverse
identity `p(z_t | z_{t+1}, c) ∝ p(z_t | z_{t+1}) * p_phi(c | z_t)` and Taylor-expanding the
low-curvature log-classifier around the predicted mean, the guided transition is a Gaussian whose
mean is shifted along `grad_{z} log p_phi(c | z)`. In epsilon/score form, with classifier scale `g`,

```
eps_hat(z_lambda, c) = eps_theta(z_lambda, c) - g * sigma_lambda * grad_{z} log p_phi(c | z_lambda),
```

which targets the renormalized distribution `p(z_lambda | c) * p_phi(c | z_lambda)^g`. Because
`g * grad log p(c|z) = grad log [ (1/Z) p(c|z)^g ]`, the scale `g > 1` sharpens the classifier
onto its high-confidence modes — exactly the inverse-temperature sharpening that yields higher
fidelity and lower diversity, and empirically a scale above 1 was needed (at `g = 1` the samples
scored ~50% on the target class and looked wrong; at `g = 10` they hit ~100% and looked
class-consistent). Dhariwal & Nichol further observed that guidance applied to an already
class-*conditional* model worked better than guidance on an unconditional one.

This method works, but leaves three open issues. (i) It requires a *second* trained network, and
that network must be trained on noisy latents, complicating the training pipeline and ruling out
off-the-shelf classifiers. (ii) The guidance direction is the input-gradient of an image
classifier, the same object an adversarial attack ascends; since IS and FID are themselves
computed from a (different) image classifier, it is unclear whether the metric gains reflect real
perceptual improvement or partly an adversarial interaction with classifier-based metrics. (iii)
Stepping along classifier gradients also resembles GAN-style training against a discriminator,
raising the same doubt about whether the gains are "for the right reason."

**Truncation / low-temperature sampling in other model families (Brock et al. 2019; Kingma &
Dhariwal 2018).** The external reference point for what a good knob should produce: a smooth
FID-vs-IS frontier swept by a single scalar. These are GAN/flow techniques and do not transfer to
diffusion (the naive analogues blur, as above), but they define the target behavior any diffusion
knob is measured against.

**Naive score-scaling and reverse-noise reduction.** The simplest diffusion-native attempts at a
temperature knob — multiply the predicted score by a constant, or shrink the sampling noise. Both
are cheap and need no auxiliary network, and both fail: they yield blurry, low-fidelity samples
rather than the truncation-like tradeoff. They establish that a *uniform* sharpening of the
model's own score is not the right move; whatever works has to sharpen something more specific.

## Evaluation settings

The natural yardsticks already in use for this question:

- **Class-conditional ImageNet** at `64x64` and `128x128` (area-downsampled; Russakovsky et al.
  2015) — the standard testbed for fidelity/diversity tradeoffs since the BigGAN paper.
- **FID** (Heusel et al. 2017), Frechet distance between Inception features of generated vs.
  reference images; lower is better, and it is sensitive to both fidelity and coverage. **Inception
  Score** (Salimans et al. 2016); higher rewards confident, varied classification of the samples.
  An FID-vs-IS curve, swept over the guidance knob, is the comparison object.
- Protocol mirrors the classifier-guidance setup: same UNet architecture and hyperparameters as
  Dhariwal & Nichol's guided models (continuous-time training as above), 50,000 samples per
  setting for the metrics, log-SNR endpoints `lambda_min = -20`, `lambda_max = 20`, a fixed number
  of ancestral sampling steps `T`, and a constant reverse-process variance interpolation
  coefficient `v` between the two natural posterior-variance choices.

## Code framework

A conditional diffusion sampler already exists as a harness. It owns the VP schedule
(`alpha_lambda`, `sigma_lambda`), an epsilon-prediction network that takes a conditioning input,
the Tweedie map from an epsilon estimate to a denoised `x`, and a DDIM/ancestral update that
renoises from `x` back to the next latent. Two slots are left open: at sampling time, *what epsilon
estimate to step with* given the latent and the conditioning; and at training time, *how the
conditioning signal is handled* for each example. Both are filled by the method.

```python
import torch


class DiffusionSampler:
    """Pre-method conditional diffusion harness. Owns the VP schedule and an
    epsilon-prediction network eps_theta(z, lambda, c). The per-step epsilon
    used for the update is not fixed yet."""

    def __init__(self, eps_theta, schedule):
        self.eps_theta = eps_theta          # network: (z, lambda, c) -> predicted noise
        self.schedule = schedule            # provides alpha(lambda), sigma(lambda)

    def tweedie_x0(self, z, eps, lam):
        # denoised estimate x_hat = (z - sigma * eps) / alpha
        a, s = self.schedule.alpha(lam), self.schedule.sigma(lam)
        return (z - s * eps) / a

    def ddim_step(self, z, x0, lam, lam_next):
        # renoise the denoised estimate to the next (higher-SNR) latent
        a_next, s_next = self.schedule.alpha(lam_next), self.schedule.sigma(lam_next)
        eps_back = (z - self.schedule.alpha(lam) * x0) / self.schedule.sigma(lam)
        return a_next * x0 + s_next * eps_back

    @torch.no_grad()
    def sample(self, c, lambdas):
        z = torch.randn(self.shape, device=self.device)   # start from N(0, I)
        for lam, lam_next in zip(lambdas[:-1], lambdas[1:]):
            # TODO: produce the epsilon estimate to step with, given z, lam, and the
            #       conditioning c (and whatever else the network can be queried for).
            eps = None  # <-- the per-step estimate we will design
            x0 = self.tweedie_x0(z, eps, lam)
            z = self.ddim_step(z, x0, lam, lam_next)
        return self.tweedie_x0(z, eps, lambdas[-1])


# existing training step the network is fit with
def train_step(eps_theta, x, c, schedule, opt):
    lam = sample_log_snr()                                 # lambda ~ p(lambda)
    eps = torch.randn_like(x)
    z = schedule.alpha(lam) * x + schedule.sigma(lam) * eps
    # TODO: decide how the conditioning c enters this step before predicting.
    pred = eps_theta(z, lam, c)
    loss = ((pred - eps) ** 2).mean()                      # denoising MSE
    opt.zero_grad(); loss.backward(); opt.step()
    return loss
```

The sampler supplies everything except the per-step epsilon estimate; `train_step` supplies the
denoising objective except for how the conditioning is treated. Those are the slots the method
fills.
