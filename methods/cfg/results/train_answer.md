Conditional diffusion models are excellent at covering the distribution of data given a class or other condition, but they originally lacked the simple post-training knob that GANs and normalizing flows provide for trading sample diversity against per-sample fidelity. In a GAN you can shrink the input noise range; in a flow you can lower the temperature of the latent noise. Both sweep out a smooth fidelity-diversity frontier. The obvious diffusion analogues, multiplying the predicted score by a constant or reducing the variance of the Gaussian noise added during the reverse process, do not produce the same effect. They uniformly sharpen the unconditional image prior rather than sharpening how strongly the sample commits to the condition, so the result is blurry and low-quality rather than class-sharp. The goal is therefore a sampler that can steer samples toward high-confidence modes of the conditional distribution using only a single scalar that is set after training, without requiring a separate auxiliary network whose gradient might interact adversarially with classifier-based metrics.

Classifier guidance already solved part of the problem by training an auxiliary classifier on the noised latents and adding its input-gradient to the diffusion score. That does sharpen class-ness, but it needs a second network trained on the noise levels the sampler visits, and its update direction is literally the gradient through an image classifier, which is structurally the same object an adversarial attack exploits. Since metrics such as Inception Score and FID are themselves computed from image classifiers, it is hard to tell how much of the gain is genuine perceptual improvement and how much comes from gaming those classifiers. The cleaner target is to obtain the same directional push from the generative model itself, using no separate classifier and no classifier gradient.

The method is Classifier-Free Guidance (CFG). The starting observation is that any pair of conditional and unconditional generative models already contains an implicit classifier by Bayes' rule: p^i(c | z) is proportional to p(z | c) divided by p(z). Taking the score gradient gives grad_z log p^i(c | z) = grad_z log p(z | c) - grad_z log p(z). In epsilon-prediction form the score is related to the predicted noise by eps_theta(z) approximately equals minus sigma_lambda times grad_z log p(z), so the implicit-classifier gradient becomes minus one over sigma_lambda times the difference between the conditional and unconditional epsilon predictions. Substituting this into the classifier-guidance formula eps_tilde = eps(z, c) - w sigma grad_z log p(c | z) makes the sigma factors cancel exactly. The classifier disappears and the guided epsilon reduces to a linear combination of two predictions from the same model: eps_tilde = (1 + w) eps_theta(z, c) - w eps_theta(z), where w is the guidance strength. Equivalently, defining the guidance scale s = 1 + w, the update is eps_tilde = eps_theta(z) + s [ eps_theta(z, c) - eps_theta(z) ]. At s = 1 there is no guidance and the sampler is the ordinary conditional model; increasing s amplifies the class-specific component beyond the unconditional image prior and pushes samples toward the high-confidence modes of the implicit classifier, trading diversity for fidelity.

The unconditional prediction is obtained without training a second network. A single noise-prediction network is trained to handle both conditional and unconditional inputs by randomly replacing the true conditioning with a null token during training. When the network sees the null token, its MSE-optimal output is the expectation of the noise given only the noisy latent, which is exactly the unconditional score. When it sees the real condition, it learns the conditional score. The only training change is therefore a conditioning dropout step with some probability p_uncond. At sampling time the network is queried twice per diffusion step, once with the real condition and once with the null token, and the two epsilon predictions are mixed by the guidance scale. The two queries can be batched into a single forward pass by stacking the inputs.

Because the learned epsilon predictions come from unconstrained neural networks, their difference is not guaranteed to be the gradient of any scalar classifier potential. The implicit classifier provides the derivation, but the actual sampler does not compute a classifier gradient, so the adversarial-attack interpretation of classifier guidance does not apply. What remains is a pure generative mechanism: one scalar dial, one jointly trained network, and the standard reverse diffusion update with a modified epsilon estimate.

```python
import torch


class ClassifierFreeGuidedSampler:
    """One network is trained to predict noise both conditionally and
    unconditionally (via null-token dropout). At sampling time the per-step
    epsilon is a linear mix of the two predictions."""

    def __init__(self, eps_theta, schedule, null_token):
        self.eps_theta = eps_theta      # network: (z, lambda, c) -> predicted noise
        self.schedule = schedule        # provides alpha(lambda), sigma(lambda)
        self.null = null_token          # unconditional conditioning identifier

    def null_for(self, z):
        return self.null.expand(z.shape[0], *self.null.shape[1:])

    def guided_eps(self, z, lam, c, guidance_scale):
        # Batched forward pass: [unconditional, conditional].
        z_in = torch.cat([z, z], dim=0)
        c_in = torch.cat([self.null_for(z), c], dim=0)
        eps_uc, eps_c = self.eps_theta(z_in, lam, c_in).chunk(2)
        # guidance_scale s = 1 + w; s = 1 is no guidance.
        return eps_uc + guidance_scale * (eps_c - eps_uc)

    @torch.no_grad()
    def sample(self, c, lambdas, guidance_scale=7.5):
        z = torch.randn(self.shape, device=self.device)
        for i in range(len(lambdas)):
            lam = lambdas[i]
            eps_tilde = self.guided_eps(z, lam, c, guidance_scale)
            a, s = self.schedule.alpha(lam), self.schedule.sigma(lam)
            x0 = (z - s * eps_tilde) / a                 # Tweedie denoised estimate
            if i < len(lambdas) - 1:
                lam_next = lambdas[i + 1]
                a_n, s_n = self.schedule.alpha(lam_next), self.schedule.sigma(lam_next)
                eps_back = (z - a * x0) / s
                z = a_n * x0 + s_n * eps_back            # renoise toward next latent
        return x0


def cfg_train_step(eps_theta, x, c, schedule, opt, null_token, p_uncond=0.1):
    # Joint conditional/unconditional training: randomly drop the condition.
    mask = torch.rand(x.shape[0], device=x.device) < p_uncond
    c = torch.where(mask[:, None], null_token.expand_as(c), c)
    lam = schedule.sample_log_snr(x.shape[0], device=x.device)
    eps = torch.randn_like(x)
    z = schedule.alpha(lam) * x + schedule.sigma(lam) * eps
    pred = eps_theta(z, lam, c)
    loss = ((pred - eps) ** 2).mean()                    # denoising score matching
    opt.zero_grad(); loss.backward(); opt.step()
    return loss
```
