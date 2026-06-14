# Classifier-Free Guidance (CFG), distilled

Classifier-free guidance is a post-training fidelity/diversity knob for conditional diffusion
models. It replaces the trained auxiliary classifier of classifier guidance with an *implicit*
classifier built by Bayes' rule out of the generator itself, so the guidance signal becomes a
linear combination of the model's own conditional and unconditional noise predictions — no extra
network, no classifier gradient. A single network is trained to be both conditional and
unconditional (by randomly dropping the conditioning during training), and at sampling time the
per-step epsilon is a mix of the two predictions controlled by one scalar dial.

## Problem it solves

Conditional diffusion models lacked the post-training fidelity/diversity tradeoff that GAN
truncation and flow low-temperature sampling provide. The naive analogues (scaling the predicted
score, shrinking the reverse-process noise) just blur, because they sharpen the unconditional
image prior `p(z)` uniformly rather than sharpening how strongly the sample commits to the
condition. Classifier guidance gave a working knob but needed a separate classifier trained on
noised latents, and its sampling update is a classifier input-gradient — formally an adversarial
perturbation of an image classifier, which clouds whether its IS/FID gains are genuine.

## Key idea

A classifier is implicit in any conditional+unconditional generator: `p^i(c|z) ∝ p(z|c)/p(z)`, so

```
grad_z log p^i(c|z) = grad_z log p(z|c) - grad_z log p(z)
                    = -(1/sigma_lambda) [ eps*(z, c) - eps*(z) ].
```

Substituting this implicit-classifier gradient into the classifier-guidance rule
`eps_tilde = eps(z,c) - w·sigma·grad_z log p(c|z)` makes the `sigma` factors cancel and the
classifier disappear:

```
eps_tilde(z_lambda, c) = (1 + w) · eps_theta(z_lambda, c) - w · eps_theta(z_lambda).
```

Equivalently, with **guidance scale** `s = 1 + w`:

```
eps_tilde = eps_theta(z_lambda) + s · [ eps_theta(z_lambda, c) - eps_theta(z_lambda) ]
          = eps_uc + s · (eps_c - eps_uc),
```

start from the unconditional prediction and step `s` times toward the conditional. `s = 1`
(`w = 0`) is no guidance (pure conditional); `s > 1` (`w > 0`) sharpens.

Why it works and why it is not adversarial:
- In score units, `-(eps_c - eps_uc)/sigma_lambda` is the score of `p(z|c)/p(z)` — the
  *class-specific* component beyond the unconditional image prior. Amplifying it (equivalently,
  using a **negative** coefficient `-w` on the unconditional epsilon term) sharpens class-ness, not
  image-ness. It targets `p(z|c) · p^i(c|z)^w`, raising the implicit classifier to a power, an
  inverse-temperature sharpening that trades diversity for fidelity.
- The `p(z)` denominator is the unconditional score term, not a disposable normalizer. Only the
  final normalization constants of the guided density vanish in score space, because additive
  constants differentiate to zero.
- `eps_theta` are unconstrained networks, so `eps_theta(z,c) - eps_theta(z)` need not be the
  gradient of any scalar potential; in general there is *no* classifier whose gradient it is. The
  implicit classifier is the inspiration, not a thing the sampler computes — so the update is not a
  classifier gradient and the adversarial-attack interpretation does not apply.

## Training: one network, conditioning dropout

Train a single `eps_theta(z_lambda, c)` and, with probability `p_uncond`, replace `c` by a null
token `∅`; define `eps_theta(z_lambda) := eps_theta(z_lambda, ∅)`. The MSE-optimal output given
`(z_lambda, ∅)` is `E[eps | z_lambda]`, the unconditional score; given `(z_lambda, c)` it is the
conditional score. Joint training is trivial to implement, forks no pipeline, and adds zero
parameters. `p_uncond` is a hyperparameter that allocates enough dropped-conditioning batches to
learn the marginal score without turning the training mostly unconditional.

```
Training (one-line change: drop the condition):
  repeat:
    (x, c) ~ data
    c <- ∅ with probability p_uncond
    lambda ~ p(lambda);  eps ~ N(0, I)
    z_lambda = alpha_lambda * x + sigma_lambda * eps
    grad step on || eps_theta(z_lambda, c) - eps ||^2

Sampling (mix conditional and unconditional epsilon):
  given w (or s = 1+w), c, increasing log-SNR schedule lambda_1..lambda_T:
  z ~ N(0, I)
  for t = 1..T:
    eps_c  = eps_theta(z, lambda_t, c)
    eps_uc = eps_theta(z, lambda_t, ∅)
    eps_tilde = (1 + w) * eps_c - w * eps_uc          # = eps_uc + s*(eps_c - eps_uc)
    x_tilde   = (z - sigma_{lambda_t} * eps_tilde) / alpha_{lambda_t}   # Tweedie
    z ~ q(z_{lambda_{t+1}} | z, x_tilde)  if t < T  else  z = x_tilde   # renoise with eps_tilde
  return z
```

Cost: two network passes per step (conditional and unconditional), batchable into one call.

## Working code

The per-step guidance combination as it appears in a standard text-to-image sampler (one batched
network call, chunk into unconditional/conditional, mix; then the ordinary DDIM/ancestral step):

```python
import torch


@torch.no_grad()
def cfg_sample(unet, scheduler, uc, c, shape, device,
               guidance_scale=7.5, skip=1):
    """Classifier-free guided DDIM sampling. uc, c are the unconditional (null) and
    conditional embeddings. guidance_scale s = 1 + w; s = 1 is no guidance."""
    z = torch.randn(shape, device=device)                  # z_1 ~ N(0, I)
    x0 = None
    for t in scheduler.timesteps:
        at = scheduler.alphas_cumprod[t]
        at_prev = scheduler.alphas_cumprod[t - skip] if t - skip >= 0 \
            else scheduler.final_alpha_cumprod

        # one batched pass: stack [uncond, cond]; split into the two predictions
        z_in = torch.cat([z, z], dim=0)
        emb = torch.cat([uc, c], dim=0)
        noise_uc, noise_c = unet(z_in, t, encoder_hidden_states=emb).sample.chunk(2)

        # classifier-free guidance: eps_uc + s * (eps_c - eps_uc)
        noise_pred = noise_uc + guidance_scale * (noise_c - noise_uc)

        # Tweedie: denoised estimate from the guided epsilon
        x0 = (z - (1 - at).sqrt() * noise_pred) / at.sqrt()

        # DDIM renoise toward the next latent using the guided epsilon
        z = at_prev.sqrt() * x0 + (1 - at_prev).sqrt() * noise_pred
    return x0
```

```python
def cfg_train_step(eps_theta, x, c, schedule, opt, null_token, p_uncond=0.1):
    """Joint conditional/unconditional training: drop the condition to the null token
    with probability p_uncond, otherwise the standard denoising MSE step."""
    mask = (torch.rand(x.shape[0], device=x.device) < p_uncond)
    c = torch.where(mask[:, None], null_token.expand_as(c), c)   # conditioning dropout
    lam = schedule.sample_log_snr(x.shape[0], device=x.device)   # lambda ~ p(lambda)
    eps = torch.randn_like(x)
    z = schedule.alpha(lam) * x + schedule.sigma(lam) * eps      # corrupt to log-SNR lambda
    pred = eps_theta(z, lam, c)
    loss = ((pred - eps) ** 2).mean()                           # denoising score matching
    opt.zero_grad(); loss.backward(); opt.step()
    return loss
```

## Convention note

Three equivalent ways the dial is written:
- guidance weight `w` convention: `(1 + w) eps_c - w eps_uc`; `w = 0` is no guidance.
- guidance scale `s = 1 + w` (text-to-image / diffusers form): `eps_uc + s (eps_c - eps_uc)`;
  `s = 1` is no guidance.
- barycentric weight `gamma = 1 + w`: `(1 - gamma) eps_uc + gamma eps_c`.
All three are the same combination of conditional and unconditional predictions.
