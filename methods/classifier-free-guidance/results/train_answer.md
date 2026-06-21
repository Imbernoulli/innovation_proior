Conditional diffusion models can sample a broad, diverse set of images for a given class, but the individual samples are uneven: some are sharp and unmistakably on-class, while others are blurry, malformed, or ambiguous. What is missing is a sampling-time knob that trades diversity for fidelity, like the truncation trick in GANs or low-temperature sampling in flow-based models. The obvious diffusion analogues, such as scaling the predicted score or shrinking the variance of the Gaussian noise injected at each reverse step, do not reproduce that smooth tradeoff; instead they tend to blur or wash out the samples.

Classifier guidance showed that a guidance-based tradeoff is possible, but it introduces a separate classifier trained on noisy inputs and the guidance step is literally a gradient ascent step on a classifier log-probability. That raises both a practical concern, an extra noisy-image classifier to train, and a conceptual one, the update is shaped like an adversarial perturbation of the classifier, which is uncomfortably close to the networks that compute the metrics being optimized. A cleaner solution would get the same sharpening effect without ever constructing or differentiating a classifier.

The method is classifier-free guidance. The target is the sharpened distribution p̃(x | c) ∝ p(x | c) · p(c | x)^w, where w ≥ 0 is a scalar knob. When w is zero we recover ordinary conditional sampling; as w grows we concentrate mass on regions where the classifier posterior would be high, giving sharper, more class-typical samples at the cost of diversity. In score space the tilted density has gradient ∇ log p̃(z_λ | c) = ∇ log p(z_λ | c) + w ∇ log p(c | z_λ). The conditional score is already available from the diffusion model. The classifier gradient can be rewritten by Bayes' rule: p(c | z_λ) = p(z_λ | c) p(c) / p(z_λ). The label prior p(c) does not depend on z_λ, so its gradient vanishes and we get ∇ log p(c | z_λ) = ∇ log p(z_λ | c) − ∇ log p(z_λ). Substituting this implicit-classifier expression into the tilted score collapses the whole thing to a linear combination of two generative scores: ∇ log p̃(z_λ | c) = (1 + w) ∇ log p(z_λ | c) − w ∇ log p(z_λ). There is no classifier left, only the conditional score and the unconditional score.

Because a diffusion model trained with denoising score matching predicts noise ε_θ(z_λ) ≈ −σ_λ ∇ log p(z_λ), we can translate the combined score into a noise estimate by multiplying by −σ_λ. The result is ε̃(z_λ, c) = (1 + w) ε(z_λ, c) − w ε(z_λ). Equivalently, writing the unconditional prediction as a baseline, ε̃ = ε(z_λ) + (1 + w)(ε(z_λ, c) − ε(z_λ)). At each sampling step we evaluate the conditional noise prediction and the unconditional noise prediction, then extrapolate past the conditional one, away from the unconditional one. Since both predictions come from learned neural networks whose vector fields need not be conservative, the combined direction is not generally the gradient of any classifier likelihood, so the adversarial-attack interpretation disappears.

The only remaining question is how to obtain both ε(z_λ, c) and ε(z_λ) from a single network without training a separate unconditional model. The trick is a learned null token ∅. Define ε(z_λ) := ε_θ(z_λ, c = ∅). During training, with probability p_uncond, replace the true class label c with ∅ before computing the denoising loss. On those dropped-label examples the network learns to denoise without class information; on the rest it learns the conditional task. A small p_uncond, around 0.1, is usually enough to learn a usable unconditional baseline while keeping most updates focused on the conditional model. At sampling time the same network is run twice, once with the real label and once with ∅, and the two outputs are combined with weight w.

Sweeping w gives the desired fidelity-versus-diversity curve. At w = 0 the sampler is ordinary conditional diffusion, maximizing diversity. As w increases, samples become sharper and more class-typical, raising metrics like Inception Score, while eventually collapsing variety enough to hurt Fréchet Inception Distance. The useful output is therefore a curve, not a single point, matching the behavior of truncation in GANs.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def alpha_sigma(lam):
    alpha_sq = torch.sigmoid(lam)
    return alpha_sq.sqrt(), (1.0 - alpha_sq).sqrt()


def expand_to_data(v, x):
    if v.ndim == 0:
        v = v.expand(x.shape[0])
    return v.view(-1, *([1] * (x.ndim - 1)))


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    if prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    return torch.rand(shape, device=device) < prob


class NoisePredictor(nn.Module):
    """ε_θ(z_λ, λ, c). A learned null embedding lets one network act as both
    the conditional denoiser and the unconditional baseline."""
    def __init__(self, num_classes, cond_dim, backbone):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, cond_dim)
        self.null_emb = nn.Parameter(torch.randn(cond_dim))
        self.backbone = backbone

    def forward(self, z, lam, c, cond_drop_prob=0.0):
        cond = self.class_emb(c)
        if cond_drop_prob > 0:
            keep = prob_mask_like((z.shape[0],), 1.0 - cond_drop_prob, z.device)
            null_cond = self.null_emb[None, :].expand_as(cond)
            cond = torch.where(keep[:, None], cond, null_cond)
        return self.backbone(z, lam, cond)

    @torch.no_grad()
    def forward_with_cond_scale(self, z, lam, c, cond_scale=1.0):
        eps_c = self.forward(z, lam, c, cond_drop_prob=0.0)
        if cond_scale == 1.0:
            return eps_c
        eps_uncond = self.forward(z, lam, c, cond_drop_prob=1.0)
        return eps_c + (eps_c - eps_uncond) * (cond_scale - 1.0)


def diffusion_loss(model, x, c, p_uncond=0.1):
    lam = sample_log_snr(x.shape[0], x.device)
    eps = torch.randn_like(x)
    a, s = alpha_sigma(expand_to_data(lam, x))
    z = a * x + s * eps
    eps_pred = model(z, lam, c, cond_drop_prob=p_uncond)
    return F.mse_loss(eps_pred, eps)


@torch.no_grad()
def guided_eps(model, z, lam, c, w):
    return model.forward_with_cond_scale(z, lam, c, cond_scale=1.0 + w)


def predict_x_from_eps(z, lam, eps_hat):
    a, s = alpha_sigma(expand_to_data(lam, z))
    return (z - s * eps_hat) / a


@torch.no_grad()
def sample(model, c, schedule, w, shape, v):
    z = torch.randn(shape, device=c.device)
    x_pred = None
    for lam, lam_next in schedule:
        eps_hat = guided_eps(model, z, lam, c, w)
        x_pred = predict_x_from_eps(z, lam, eps_hat)
        z = reverse_step(z, lam, lam_next, x_pred, v)
    return x_pred
```
