Let me start from the thing I actually want and can't have. I have a conditional diffusion model that draws `x ~ p(x|c)` for a class `c`, and on coverage and likelihood it's great. What it doesn't have is the knob every other generative family gives me for free: a dial I can turn after training that trades diversity for fidelity. A GAN gives me truncation — shrink the noise range and each sample gets cleaner while the set as a whole gets less varied. A flow gives me low-temperature sampling — scale the latent noise down, same effect. Sweep the dial and you get a smooth FID-versus-IS curve. I want that curve for diffusion, and I want it without retraining the model for each setting on the dial.

The obvious moves don't work, and I should be precise about why, because the failure is a clue. The diffusion analogue of "lower the temperature" would be: either multiply the predicted score by a constant, or shrink the Gaussian noise I add during the reverse process. Both are known to just blur — they produce low-quality, washed-out samples instead of sharpened ones. Why? Because both of those uniformly rescale the *whole* generative process. The reverse process is, near enough, Langevin sampling on the noised data density `p(z_lambda)`; if I scale the score everywhere by a constant `s`, I'm sampling from something like `p(z)^s` — I've turned down the temperature of the *entire data distribution*, foreground and background alike. Sharpening `p(z)` itself doesn't sharpen *toward the class*; it pushes on generic high-density image structure and runs straight into the known blur/low-quality failure. The temperature I want to lower isn't the temperature of `p(x)`. It's the temperature of how strongly the sample commits to the *class*. That distinction — sharpen the class-ness, not the image-ness — is the whole game, and naive score scaling fails exactly because it can't tell the two apart.

So how do I sharpen class-ness specifically? There's already a method that does it, and I should understand it cold before I try to do better, because my whole plan is going to be to mimic its effect while throwing away its machinery. Classifier guidance. The idea: alongside the diffusion model, train a classifier `p_phi(c|z_lambda)` on the noised latents, and at sampling time nudge each reverse step in the direction the classifier says makes the current latent more class-`c`. Concretely, the conditional reverse transition factors as `p(z_t|z_{t+1}, c) ∝ p(z_t|z_{t+1}) · p_phi(c|z_t)` — the unconditional transition times the classifier likelihood. Taylor-expand the log-classifier around the predicted mean (it has low curvature relative to the transition covariance when steps are small), and the product is again a Gaussian, just with its mean shifted by the covariance times `grad_z log p_phi(c|z)`. Translate that into the score/epsilon language I live in. My denoiser and the score are the same object up to a factor: `eps_theta(z_lambda) ≈ -sigma_lambda · grad_z log p(z_lambda)`, because training the denoiser by MSE *is* denoising score matching at each noise level. So a shift of the score by `grad log p_phi(c|z)` is a shift of the epsilon by `-sigma_lambda · grad log p_phi(c|z)`. With a classifier scale `g`, classifier guidance modifies the model's epsilon to

  `eps_hat(z_lambda, c) = eps_theta(z_lambda, c) - g · sigma_lambda · grad_z log p_phi(c|z_lambda)`,

and I should pin down what distribution that's sampling from, because that tells me whether the dial is the right kind. Collect the gradients: `eps_hat ≈ -sigma_lambda · grad_z [ log p(z|c) + g · log p_phi(c|z) ]`, so I'm following the score of `p(z|c) · p_phi(c|z)^g`. And `g` raising the classifier to a power is exactly the inverse-temperature sharpening I wanted — `g · grad log p(c|z) = grad log[(1/Z) p(c|z)^g]`, and `p(c|z)^g` for `g>1` dumps probability mass onto the classifier's high-confidence modes. *This* is sharpening class-ness, not image-ness, which is why it traces out a real fidelity/diversity curve where naive score scaling failed: at `g = 1` the classifier is only ~50% sure of the class and the images look off; crank it and they snap to the class. That story hangs together, but it's a story — I'll want to put numbers on the "sharpens the class, leaves the image prior alone" claim once I have a concrete form to test, because that is the exact distinction I'm betting the whole approach on.

Now why I don't want it. Three things bother me. First, it's a *second trained model*, and not a free one — it has to be trained on the noised latents `z_lambda`, because at sampling time the classifier sees noisy inputs, so I can't drop in a pretrained clean-image classifier; I have to build and train a noise-aware classifier as part of the pipeline. That's real friction. Second, and this is the one that actually worries me about the *validity* of the whole approach: the guidance direction is the input-gradient of an image classifier — `grad_z log p_phi(c|z)`. That is, structurally, exactly the object an adversarial attack ascends. I'm perturbing the image in the direction that most increases a classifier's confidence. And the metrics I'll be judged on, IS and FID, are themselves computed from an image classifier (Inception). So there's a nagging possibility that classifier guidance "improves" IS and FID partly because it's adversarially nudging samples toward whatever a classifier likes, not because the images are genuinely better. Stepping along classifier gradients also smells faintly of GAN training against a discriminator. I can't cleanly separate "real perceptual gain" from "gaming a classifier-based metric" as long as my sampling update literally is a classifier gradient. Third, smaller: there's an asymmetry that's nagging me. People found guidance works best applied to an *already conditional* model, not an unconditional one. Let me check that's consistent rather than just assert it. Applying classifier guidance with weight `w+1` to an unconditional model targets `p(z) p(c|z)^{w+1}`, and `p(z) p(c|z)^{w+1} ∝ p(z|c) p(c|z)^w` by Bayes (`p(z|c) ∝ p(z) p(c|z)`), which is classifier guidance with weight `w` on the *conditional* model. To be sure I haven't mis-handled the index shuffle, I push it down to score space and put numbers on it: the conditional-guided score is `(1+w) score_c - w score_uc`, the unconditional-guided score at scale `W=w+1` is `score_uc + W(score_c - score_uc) = (w+1) score_c - w score_uc`. With stand-in scores `score_c=-1.234`, `score_uc=0.567`, `w=3`: the first gives `4·(-1.234) - 3·(0.567) = -6.637`, the second gives `0.567 + 4·(-1.801) = -6.637`. Identical, as Bayes promised — they're the same target distribution, just relabeled — and yet the conditional version trains better in practice. Fine; I'll keep guiding the conditional model. But it tells me the conditional and unconditional models are the two natural ingredients here, and the classifier weight just shuffles between them.

That last observation is itching at me. The whole effect lives in `p(c|z)`, the classifier. But I have a *generative* model, not just a discriminative one. By Bayes' rule, a classifier is hiding inside any pair of conditional-and-unconditional generative models: `p(c|z) = p(z|c) p(c) / p(z)`, so `p(c|z) ∝ p(z|c)/p(z)` for fixed `c`. I don't need to *train* a classifier — I could *construct* one out of densities I already model. Call it the implicit classifier, `p^i(c|z) ∝ p(z|c)/p(z)`. If that works, the classifier as a separate network just evaporates.

But wait — can I even use `p^i(c|z)`? The classifier-guidance recipe needs `grad_z log p(c|z)`, the gradient. Do I have access to that through my generative models? Take the log and the gradient:

  `grad_z log p^i(c|z) = grad_z log p(z|c) - grad_z log p(z)`,

because `log p(c)` is constant in `z`, while the Bayes denominator `p(z)` is exactly the unconditional density and must not be thrown away. Good: the term that normalizes the classifier over classes is not a harmless constant; it is the score I need to subtract. Both gradients are things I have: `grad_z log p(z|c)` is (up to `-1/sigma`) my conditional epsilon, and `grad_z log p(z)` is (up to `-1/sigma`) my *unconditional* epsilon. So if I had exact scores,

  `grad_z log p^i(c|z) = -(1/sigma_lambda) [ eps*(z, c) - eps*(z) ]`.

The classifier gradient is just the *difference of my conditional and unconditional epsilon predictions*, scaled by `-1/sigma`. That is striking. The entire classifier signal — the thing I was training a whole separate network to compute the gradient of — is `eps(z,c) - eps(z)`, two forward passes of the model I already have.

Now plug this implicit classifier straight into the classifier-guidance formula and watch the machinery fall away. Classifier guidance on the conditional model with strength `w`:

  `eps_tilde(z, c) = eps*(z, c) - w · sigma_lambda · grad_z log p^i(c|z)`
                  `= eps*(z, c) - w · sigma_lambda · ( -(1/sigma_lambda) [ eps*(z,c) - eps*(z) ] )`
                  `= eps*(z, c) + w · [ eps*(z, c) - eps*(z) ]`
                  `= (1 + w) · eps*(z, c) - w · eps*(z).`

The `sigma_lambda` and the `1/sigma_lambda` cancelled exactly — they had to, one came from converting score to epsilon and the other from converting epsilon back to score. And what's left has *no classifier in it at all*. No gradient through a classifier, no `p_phi`, no second network. Just a linear combination of two epsilon predictions: take `(1+w)` of the conditional and subtract `w` of the unconditional. The classifier didn't get approximated away — it got *constructed* away, by noticing the classifier I needed was already implicit in the generator. So my sampling rule is

  `eps_tilde_theta(z_lambda, c) = (1 + w) · eps_theta(z_lambda, c) - w · eps_theta(z_lambda),`

using my model's own learned epsilon for both terms, and `w` is the dial. At `w=0` it's the plain conditional model; crank `w` up and I sharpen.

Let me double-check I haven't fooled myself about what distribution this targets, because the derivation went through "exact scores" and my model is not exact. Rewrite the combination as `eps_tilde = eps(z,c) + w[eps(z,c) - eps(z)]`. The bracket is the conditional epsilon minus the unconditional epsilon; after multiplying by `-1/sigma`, it becomes `grad log p(z|c) - grad log p(z) = grad log[p(z|c)/p(z)]`, the score of the implicit classifier, the part of the model that says "what does the label `c` add beyond the unconditional image prior." `w` amplifies precisely *that* class-specific score component and leaves the unconditional image-modeling part at strength 1. So I am sharpening class-ness and *not* image-ness — exactly the distinction the naive score-scaling failure pointed me to. Equivalently, I'm targeting `p(z|c) · p^i(c|z)^w`, the conditional density times an implicit-classifier-to-the-`w`-power, which is the same form classifier guidance had, with the trained classifier swapped for the Bayes one. The `p(z)` denominator is not ignored; it is the negative unconditional score term. What disappears are only the final normalizing constants of the guided density, because in score space multiplicative constants become additive constants and differentiate to zero.

Now the honest worry, the one I have to confront before I trust this. Did I just re-derive classifier guidance with a different classifier, and does it inherit the adversarial-attack problem? Stare at `eps_theta(z,c) - eps_theta(z)`. For the *exact* scores, yes, this equals `-sigma · grad log p^i(c|z)` and so it *is* a classifier gradient. But my `eps_theta` are unconstrained neural networks. There is no guarantee — in fact no reason — that the learned vector field `eps_theta(z,·)` is the gradient of any scalar potential. A trained denoiser is only approximately a conservative field. So `eps_theta(z,c) - eps_theta(z)` is the difference of two arbitrary learned vector fields, and there need not exist *any* classifier `p(c|z)` whose log-gradient it is. My sampler steps in the `eps_tilde` direction, and that direction in general *cannot* be written as the input-gradient of an image classifier. So the second objection to classifier guidance — that the update is an adversarial perturbation of a classifier — simply doesn't apply: there's no classifier to attack. The implicit classifier was the *inspiration* for the formula, not a thing my sampler actually computes gradients of. That gap, between `eps_tilde*` (built from exact scores, genuinely a classifier gradient) and `eps_tilde_theta` (built from network outputs, not a gradient of anything in general), is the whole point. It means I can get the fidelity/diversity tradeoff with a pure generative model whose sampling steps look nothing like classifier gradients.

I should be skeptical for one more beat: is inverting a generator by Bayes' rule even a *good* classifier? There's a known caution that discriminative models usually beat generative-model-derived classifiers, and that under model misspecification — which I certainly have — the Bayes-inverted classifier can be inconsistent, so I have no theoretical guarantee it gives a useful guidance signal. I'll grant that. But notice the flip side, which is reassuring rather than worrying: a discriminative classifier *can* take shortcuts — ignore most of the image and still classify well — and that's exactly what makes its input-gradient brittle and adversarial. A generative model gets no such shortcut; to model `p(z|c)` it has to account for the whole image. So the difference `eps(z,c) - eps(z)` is forced to be a more holistic, robust direction than a discriminative gradient. I can't *prove* the implicit classifier is good, but the reason it might be *better* than a trained one is the same reason classifier guidance felt suspect. I'll let the experiment settle whether it works; the construction is sound.

Now I need the unconditional epsilon, `eps_theta(z_lambda)`, and I'd rather not train a second network — that would reintroduce exactly the pipeline complexity I just removed. So train *one* network to be both. The unconditional model is just the conditional model with the conditioning marginalized out, `p(z) = sum_c p(z|c) p(c)`, i.e. "predict the noise given the image but *not* told the class." I can teach a single net to do both by, during training, sometimes replacing the real class `c` with a special null token `∅` and asking it to denoise anyway. Define `eps_theta(z_lambda) := eps_theta(z_lambda, c=∅)`. Why does dropping the label to `∅` with some probability actually fit the unconditional score? Because the MSE-optimal prediction given an input is the conditional mean of the target given that input: when the net sees `(z_lambda, ∅)`, its loss-minimizing output is `E[eps | z_lambda]` *marginalized over whatever generated `z_lambda`* — which is precisely the unconditional denoiser, the unconditional score. When it sees `(z_lambda, c)`, the minimizer is `E[eps | z_lambda, c]`, the conditional score. One network, one objective, two behaviors selected by whether the class slot holds `c` or `∅`. So training is the existing denoising MSE step with a single extra line: with probability `p_uncond`, set `c ← ∅` before computing the loss.

  for each minibatch:
    draw (x, c)
    with prob p_uncond: c ← ∅
    draw lambda ~ p(lambda), eps ~ N(0, I)
    z_lambda = alpha_lambda · x + sigma_lambda · eps
    step on grad_theta || eps_theta(z_lambda, c) - eps ||^2

That's it for training — genuinely a one-line change (the conditioning dropout). I briefly consider training two *separate* networks instead, which is also valid, but joint training wins on every axis I care about: it's trivial to implement, it doesn't fork the pipeline, and it adds zero parameters — the same weights serve both roles. The only cost is at sampling time, where I now need two forward passes per step (one with `c`, one with `∅`), which I'll come back to.

How much capacity should the unconditional task get — what's `p_uncond`? I don't need the unconditional model to be *great*; I need its score as a baseline to subtract, to define the *difference* direction. The classifier-guidance experience already hints that a small/low-capacity classifier can supply a useful guidance reference, and the unconditional branch here plays an analogous role. So `p_uncond` should be a hyperparameter: enough dropped-conditioning batches to learn the marginal score, but not so many that the conditional model I actually sample from is starved.

Let me get a feel for what the dial does geometrically, so I trust the sign and the magnitude. I have to think in score units, not raw epsilon units, because `eps = -sigma · score`. The guided epsilon `eps_tilde = (1+w)eps_c - w eps_uc` corresponds to the guided score `score_tilde = (1+w)score_c - w score_uc`. Picture three classes, each a little Gaussian blob. `score_c - score_uc` is the gradient of the log-ratio `log p(z|c) - log p(z)`: it points toward places that are more characteristic of this class than of images in general. Amplifying that log-ratio score pushes mass away from regions explained well by the marginal image model and toward the high-confidence interior of class `c`, concentrating each conditional into a tighter region. That confirms the sign: in epsilon units the unconditional coefficient is negative (`-w`), and in score units the unconditional likelihood is deliberately decreased while the conditional likelihood is increased. The negative coefficient is the un-obvious part; it's what makes guidance sharpen the class rather than the image.

There's a tidy sanity check hiding here. Set `w = 0`: `eps_tilde = eps(z,c)`, the plain conditional model, no guidance — the dial's "off" position is the ordinary sampler, as it must be. And the relationship to the way people often write the dial in code: factor `eps_tilde = (1+w) eps(z,c) - w eps(z)` as `eps_tilde = eps(z) + (1+w)[eps(z,c) - eps(z)]`. If I rename `s = 1 + w` — the "guidance scale" — this reads

  `eps_tilde = eps(z) + s · [ eps(z, c) - eps(z) ],`

i.e. start from the unconditional prediction and step `s` times toward the conditional. Now `s = 1` is "off" (pure conditional, since `eps(z) + 1·[eps(z,c)-eps(z)] = eps(z,c)`), `s = 0` is pure unconditional, and `s > 1` is guidance. The two parameterizations are identical — `s = 1 + w` is just an offset — and the `s` form is the convenient one to write in a sampler because it's literally "`uncond` plus scale times the `(cond − uncond)` difference."

Now the sampling loop. Nothing about the diffusion machinery changes except *which epsilon I feed into the step*. I keep the ordinary reverse process — Tweedie to get the denoised estimate, then renoise toward the next, higher-SNR latent — and I just substitute `eps_tilde` for `eps`. Per step: query the network twice, once conditioned on `c` and once on `∅`, form the guided epsilon, run it through Tweedie to get `x_tilde = (z - sigma · eps_tilde)/alpha`, and take the DDIM/ancestral step. The two passes are the price of being classifier-free: I run the diffusion model twice per step instead of running a (possibly cheaper) classifier once. I can batch the two passes into one network call by stacking the `∅` and `c` conditionings, which is the practical thing to do. (One could imagine injecting the conditioning only late in the network so the shared trunk runs once, but that's an architecture change I'll leave aside.)

Let me write the sampling step as an algorithm, mirroring exactly the conditional sampler I already have:

  given guidance w (or scale s = 1+w), condition c, increasing log-SNR schedule lambda_1..lambda_T:
    z ~ N(0, I)
    for t = 1..T:
      eps_c  = eps_theta(z, lambda_t, c)
      eps_uc = eps_theta(z, lambda_t, ∅)
      eps_tilde = (1 + w) · eps_c - w · eps_uc        # == eps_uc + s·(eps_c - eps_uc)
      x_tilde   = (z - sigma_{lambda_t} · eps_tilde) / alpha_{lambda_t}     # Tweedie
      if t < T:
        z ~ q(z_{lambda_{t+1}} | z, x_tilde)          # renoise with the guided estimate
      else:
        z = x_tilde
    return z

That's the whole method. Two parts, both tiny: at training, drop the conditioning to `∅` with probability `p_uncond`; at sampling, mix the conditional and unconditional epsilon by the dial and otherwise run the unchanged reverse process.

Let me put it into the code I'd actually run, filling the two open slots in the sampler harness — the per-step epsilon and the conditioning handling in the training step. I'll write the sampling combination in the `s = 1 + w` "scale" convention because that's the form that drops cleanly into a denoising loop, and note the `w` equivalence in a comment.

```python
import torch


class ClassifierFreeGuidedSampler:
    """Classifier-free guidance. One network eps_theta(z, lambda, c) is trained to be
    both conditional and unconditional (null token for c). At sampling time the per-step
    epsilon is a linear mix of the conditional and unconditional predictions."""

    def __init__(self, eps_theta, schedule, null_token):
        self.eps_theta = eps_theta          # shared conditional+unconditional noise predictor
        self.schedule = schedule            # alpha(lambda), sigma(lambda)
        self.null = null_token              # the unconditional class identifier (the empty token)

    def guided_eps(self, z, lam, c, guidance_scale):
        # one batched call: stack the unconditional (null) and conditional conditionings
        z_in = torch.cat([z, z], dim=0)
        c_in = torch.cat([self.null_for(z), c], dim=0)
        eps_uc, eps_c = self.eps_theta(z_in, lam, c_in).chunk(2)
        # eps_tilde = eps_uc + s * (eps_c - eps_uc),  with s = 1 + w (s=1 => no guidance)
        # equivalently (1 + w) * eps_c - w * eps_uc with w = s - 1
        return eps_uc + guidance_scale * (eps_c - eps_uc)

    def null_for(self, z):
        return self.null.expand(z.shape[0], *self.null.shape[1:])

    @torch.no_grad()
    def sample(self, c, lambdas, guidance_scale=7.5):
        z = torch.randn(self.shape, device=self.device)        # z_1 ~ N(0, I)
        x0 = None
        for i in range(len(lambdas)):
            lam = lambdas[i]
            eps_tilde = self.guided_eps(z, lam, c, guidance_scale)   # the guidance step
            a, s = self.schedule.alpha(lam), self.schedule.sigma(lam)
            x0 = (z - s * eps_tilde) / a                             # Tweedie: denoised estimate
            if i < len(lambdas) - 1:
                # renoise toward the next, higher-SNR latent using the GUIDED estimate
                lam_next = lambdas[i + 1]
                a_n, s_n = self.schedule.alpha(lam_next), self.schedule.sigma(lam_next)
                eps_back = (z - a * x0) / s
                z = a_n * x0 + s_n * eps_back
        return x0


def train_step(eps_theta, x, c, schedule, opt, null_token, p_uncond=0.1):
    # joint training: with prob p_uncond, replace the conditioning by the null token,
    # which fits the unconditional score E[eps | z] in those steps.
    mask = (torch.rand(x.shape[0], device=x.device) < p_uncond)
    c = torch.where(mask[:, None], null_token.expand_as(c), c)
    lam = schedule.sample_log_snr(x.shape[0], device=x.device)        # lambda ~ p(lambda)
    eps = torch.randn_like(x)
    z = schedule.alpha(lam) * x + schedule.sigma(lam) * eps           # corrupt to log-SNR lambda
    pred = eps_theta(z, lam, c)
    loss = ((pred - eps) ** 2).mean()                                 # denoising MSE
    opt.zero_grad(); loss.backward(); opt.step()
    return loss
```

And the standard text-to-image form, the way it actually appears inside a sampling loop, is even more compact — one batched UNet call, chunk into unconditional and conditional, mix:

```python
noise_pred = unet(torch.cat([z, z]), t, encoder_hidden_states=torch.cat([uc, c])).sample
noise_uc, noise_c = noise_pred.chunk(2)
noise_pred = noise_uc + guidance_scale * (noise_c - noise_uc)   # s = 1 + w; s = 1 is no guidance
```

So the causal chain. I wanted a post-training fidelity/diversity dial like GAN truncation, and the naive diffusion moves — scaling the score, shrinking the reverse noise — failed because they sharpen the image prior `p(z)` uniformly instead of sharpening class-ness, collapsing everything to a blur. Classifier guidance fixed that by adding a classifier's input-gradient to the score, targeting `p(z|c) p(c|z)^g`, which sharpens the classifier onto its modes — but at the cost of a second noise-trained network, and with a sampling update that *is* a classifier gradient, indistinguishable in form from an adversarial attack on the very classifier-based metrics it's scored by. Noticing that a classifier is implicit in any conditional-plus-unconditional generator by Bayes, `p(c|z) ∝ p(z|c)/p(z)`, I took its gradient and found it equals `-(1/sigma)[eps(z,c) - eps(z)]` — the difference of my own conditional and unconditional epsilon. Substituting that into the classifier-guidance formula made the `sigma` factors cancel and the classifier vanish, leaving `eps_tilde = (1+w) eps(z,c) - w eps(z)`, a pure linear mix of two predictions from one model. Because those predictions come from unconstrained networks, the mix is in general not the gradient of any classifier, so the adversarial-attack objection dissolves. I get the unconditional model for free by training one network with the conditioning randomly dropped to a null token, so the whole method is two one-line changes — dropout the condition during training, mix conditional and unconditional epsilon during sampling — applied to an otherwise unchanged denoising diffusion pipeline.
