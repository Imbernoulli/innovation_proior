Let me start from the thing that actually bothers me. I have a class-conditional diffusion model. I can sample `x ~ p(x | c)`, and on average the samples are fine — they cover the class — but individually a lot of them are mushy or off. Meanwhile a GAN has this lovely knob: the truncation trick. Sample the GAN's latent from a normal, but reject any draw outside a radius, so you only keep the central, high-density region of the latent. Shrink that radius and the samples get cleaner, sharper, more obviously "a dog" — Inception Score climbs — at the cost of variety. Sweep the radius and you trace a whole curve trading FID against IS. Glow does the same thing by sampling its base Gaussian at temperature `T < 1`. I want that knob for diffusion.

The naive transfers don't work, and I should be honest about why before I waste time on them. A diffusion sample is built by many reverse steps, each of which uses the model's predicted noise `ε_θ(z_λ)` — equivalently a score — and then adds a fresh bit of Gaussian noise. The two obvious "make it less random" moves are: scale the predicted score by some constant > 1 to push harder toward high density, or shrink the variance of the Gaussian noise injected at each reverse step. People have tried both and they produce blurry, low-quality images, not sharp ones. So there's no single latent whose variance I can just squeeze; the randomness is spread across hundreds of steps and the marginal I'm sampling isn't a clean reparameterized Gaussian I can truncate. The truncation idea doesn't transplant.

Before I touch the guidance knob, I need the diffusion object itself pinned down, because the signs and factors will all come from this parameterization. I use a variance-preserving noising process indexed by log-SNR `λ`, with `λ_max` almost clean and `λ_min` almost pure noise:

```
q(z_λ | x) = N(α_λ x, σ_λ² I),
α_λ² = sigmoid(λ),   σ_λ² = 1 - α_λ²,   λ = log(α_λ² / σ_λ²).
```

So a noisy point is `z_λ = α_λ x + σ_λ ε`, `ε ~ N(0,I)`. The forward Markov transition from a cleaner level `λ'` down to a noisier level `λ < λ'` is

```
q(z_λ | z_{λ'}) = N((α_λ/α_{λ'}) z_{λ'}, σ²_{λ|λ'} I),
σ²_{λ|λ'} = (1 - e^{λ-λ'}) σ_λ².
```

For sampling I move the other way, along `λ_min = λ_1 < ... < λ_T = λ_max`, so I need the reverse conditional. If I condition on the clean `x`, the Gaussian algebra gives

```
q(z_{λ'} | z_λ, x) = N(μ̃_{λ'|λ}(z_λ,x), σ̃²_{λ'|λ} I),
μ̃_{λ'|λ}(z_λ,x) = e^{λ-λ'}(α_{λ'}/α_λ) z_λ
                  + (1 - e^{λ-λ'}) α_{λ'} x,
σ̃²_{λ'|λ} = (1 - e^{λ-λ'}) σ_{λ'}².
```

The exponent is easy to flip, so I check the case: during sampling `λ' > λ`, hence `e^{λ-λ'} < 1`, both coefficients are nonnegative, and the variance is positive. Good. The model does not know `x`, so it predicts `x_θ(z_λ)` and plugs that into the same posterior mean. I could predict `x` directly, but predicting the injected noise gives the score link I need:

```
x_θ(z_λ) = (z_λ - σ_λ ε_θ(z_λ)) / α_λ,
E_{ε,λ} ||ε_θ(z_λ) - ε||².
```

This MSE is denoising score matching at every noise level, so the learned noise field estimates the noisy-data score by

```
ε_θ(z_λ) ≈ -σ_λ ∇_{z_λ} log p(z_λ).
```

That is why every guidance formula can be derived in score space and then converted back to `ε` by multiplying by `-σ_λ`. The sampler then takes the ordinary reverse transition, often with a log-space variance interpolation `(σ̃²_{λ'|λ})^{1-v}(σ²_{λ|λ'})^v`; `v` only matters for finite step sizes because the variances meet in the infinitesimal limit. For the noise schedule I want bounded log-SNR endpoints and a cosine-like density, so I can sample `u ~ Uniform[0,1]` and set `λ = -2 log tan(au+b)`, with `b = arctan(exp(-λ_max/2))` and `a = arctan(exp(-λ_min/2)) - b`. None of this is the new knob; it is the machinery the knob has to respect.

So let me think about what "truncation" actually *is* distributionally, because maybe I can target the distribution directly instead of mechanically copying the GAN's procedure. Truncation concentrates mass on the high-density, high-confidence part of the distribution. For a conditional model that means: don't just sample `p(x | c)`, sample something that's been *sharpened* toward the regions that are most confidently class-`c`. A natural way to write "sharpen toward confident-class regions" is to tilt the conditional by a power of the classifier posterior:

```
p̃(x | c) ∝ p(x | c) · p(c | x)^w ,   w ≥ 0.
```

When `w = 0` this is just `p(x | c)`. As `w` grows, I multiply in more and more of `p(c | x)^w`, which is large exactly where a classifier is very sure the image is class `c` — the typical, unambiguous, high-fidelity examples — and small in the ambiguous tails. That's precisely the diversity-down, fidelity-up behavior I want, and `w` is the knob. Good. Now: how do I sample from `p̃` with a diffusion model?

Diffusion gives me scores, so let me move into score space. I take the log and the gradient in the noisy variable `z_λ` (I'll work at a generic noise level `λ`; everything is per-level):

```
∇ log p̃(z_λ | c) = ∇ log p(z_λ | c) + w ∇ log p(c | z_λ).
```

The `∝` constant vanishes under the gradient, so the tilted score is just the ordinary conditional score plus `w` times the gradient of the classifier's log-posterior. And my model already gives me `∇ log p(z_λ | c)` — that's `−ε_θ(z_λ, c) / σ_λ`, from the score-matching identity. So the only new object I need is `∇ log p(c | z_λ)`, the gradient of a classifier that reads a *noisy* image at level `λ` and outputs class probabilities.

This is exactly classifier guidance. Train a classifier `p_φ(c | z_λ)` and form the modified noise estimate
```
ε̃_θ(z_λ, c) = ε_θ(z_λ, c) − w σ_λ ∇_{z_λ} log p_φ(c | z_λ).
```
Let me check the sign and the `σ_λ` factor, because this is the kind of thing I'll get wrong if I'm not careful. The combined score I want is `∇ log p(z_λ|c) + w ∇ log p(c|z_λ)`. To turn a score back into an `ε`-style quantity I multiply by `−σ_λ` (since `ε = −σ_λ ∇ log p`). So
```
ε̃ = −σ_λ[ ∇ log p(z_λ|c) + w ∇ log p(c|z_λ) ]
   = ε_θ(z_λ, c) − w σ_λ ∇ log p(c|z_λ).
```
Yes — the conditional `ε` minus `w σ_λ` times the classifier gradient. Signs and factor check out, and this samples from `p̃(z_λ|c) ∝ p(z_λ|c) p_φ(c|z_λ)^w`. Sweep `w`, get the IS/FID curve. It works; this is the existing answer.

But I don't like it, and the reasons are worth pinning down because they're going to tell me what a better method has to avoid. First, that classifier `p_φ(c | z_λ)` has to be trained on *noisy* images — at every noise level `λ`, from nearly-clean down to near-pure-noise. I can't grab a pretrained ImageNet classifier off the shelf; standard classifiers have only ever seen clean images and fall apart on `z_λ` at low SNR. So I have to build and train a second model, on noised data, spanning the whole schedule. That's a whole extra pipeline bolted onto a diffusion model that was, by itself, simple to train.

Second, and this one really nags: the guidance step moves `z_λ` in the direction `∇_{z_λ} log p_φ(c | z_λ)` — the input gradient that *most increases the classifier's confidence in class `c`*. That is, definitionally, a step of a gradient-based adversarial attack on `p_φ`. And the metrics I'm trying to improve — IS and FID — are themselves computed from a fixed Inception classifier's features and logits. So there's a gnawing doubt: maybe classifier guidance scores well on IS/FID partly because it's *adversarially nudging images to please exactly the kind of classifier those metrics are built on*, not because the images are genuinely better. I can't rule that out as long as my sampling direction literally is a classifier gradient.

So the goal sharpens: I want the `p̃(x|c) ∝ p(x|c) p(c|x)^w` sharpening effect, but I want to compute it *without ever training or differentiating a classifier*. The classifier is the source of both problems — the noisy-data training and the adversarial-attack interpretation. Can I get `∇ log p(c | z_λ)` from things I already have?

Stare at `p(c | z_λ)`. I keep treating it as an opaque discriminative model I have to train. But it isn't opaque — it's tied to the generative quantities by Bayes' rule:

```
p(c | z_λ) = p(z_λ | c) p(c) / p(z_λ).
```

Take logs: `log p(c | z_λ) = log p(z_λ | c) − log p(z_λ) + log p(c)`. Now take the gradient in `z_λ`. The label prior `p(c)` doesn't depend on `z_λ` at all, so `∇_{z_λ} log p(c) = 0` and it just disappears:

```
∇_{z_λ} log p(c | z_λ) = ∇_{z_λ} log p(z_λ | c) − ∇_{z_λ} log p(z_λ).
```

There it is. The classifier gradient I thought I needed a whole separate model for is *the conditional score minus the unconditional score*. Both of those are generative scores. I already have the conditional one. The only thing I don't yet have is the unconditional score `∇ log p(z_λ)` — and that's not a classifier, it's just a diffusion model with no label. This is an *implicit classifier*: I never train `p(c|z)` directly; I read it off the two generative models through Bayes.

Now substitute this back into the tilted score and see what falls out:

```
∇ log p̃(z_λ | c) = ∇ log p(z_λ | c) + w[ ∇ log p(z_λ | c) − ∇ log p(z_λ) ]
                  = (1 + w) ∇ log p(z_λ | c) − w ∇ log p(z_λ).
```

The whole tilted score is a *linear combination of two generative scores* — the conditional one with weight `(1+w)`, the unconditional one with weight `−w`. No classifier anywhere. Let me push it into `ε`-space, which is what the sampler actually consumes. Both scores convert by `ε = −σ_λ ∇ log p`: write `ε(z_λ, c) = −σ_λ ∇ log p(z_λ|c)` and `ε(z_λ) = −σ_λ ∇ log p(z_λ)`. Multiply the combined score by `−σ_λ`:

```
ε̃(z_λ, c) = −σ_λ[ (1+w) ∇ log p(z_λ|c) − w ∇ log p(z_λ) ]
          = (1 + w) ε(z_λ, c) − w ε(z_λ).
```

So the recipe is: at each sampling step, evaluate the conditional noise prediction and the unconditional one, and combine them as `(1+w) ε_c − w ε_∅`. That's it. When `w = 0`, `ε̃ = ε_c` — plain conditional sampling, no guidance. As `w` grows, I extrapolate *past* the conditional prediction, away from the unconditional one. Let me sanity-check the algebra by rewriting it around the unconditional baseline: `(1+w)ε_c − w ε_∅ = ε_∅ + (1+w)ε_c − (1+w)ε_∅ = ε_∅ + (1+w)(ε_c − ε_∅)`. So it's the unconditional prediction plus `(1+w)` times the conditional-minus-unconditional difference vector — push `(w+1)` times harder along "what the label adds." Same expression, and it makes the geometry obvious: `ε_c − ε_∅` is the direction the label pulls in, and I'm amplifying that pull. (The two equivalent forms also explain a small puzzle from classifier guidance: guiding the conditional model with weight `w` equals guiding the *unconditional* model with weight `w+1`, since `p(z|c)p(c|z)^w ∝ p(z)p(c|z)^{w+1}`. I'll stay with the conditional formulation, which is the one that works best in practice.)

Hold on — is `ε̃` actually a classifier-guided score, just with the classifier hidden? I should be careful here, because it would be too good if it were. The implicit classifier `p(c|z) ∝ p(z|c)/p(z)` is exact *only if* `ε(z,c)` and `ε(z)` are the true scores. But my `ε_θ(z,c)` and `ε_θ(z)` are outputs of unconstrained neural networks. They're not the gradient of any scalar potential — a generic network output isn't a conservative vector field — so the difference `ε_θ(z,c) − ε_θ(z)` is *not* in general the gradient of any function `log p(c|z)`. So `ε̃_θ` is not literally a classifier-guided score; there's no real classifier whose gradient it is. It's *inspired by* the implicit classifier but is its own object built from two generative predictions.

And that's not a defect — it's exactly the property I wanted. Because the step direction is not, in general, a classifier gradient, the adversarial-attack interpretation loses its object: there is no classifier in the sampler to attack. If this still produces the IS/FID tradeoff, then I'll have shown the sharpening effect can come from a pure generative model with a sampling direction that need not be a classifier gradient. (I should keep one honest reservation in mind: inverting a generative model by Bayes doesn't *guarantee* a good classifier — when the generative model is misspecified, Bayes-derived classifiers can behave badly, as Grandvalet & Bengio 2004 and Grünwald & Langford 2007 point out. So this is an empirical bet; I'll have to verify the tradeoff actually shows up, not assume it.)

Now the remaining practical gap: the formula needs both `ε(z_λ, c)` and the unconditional `ε(z_λ)`. Do I have to train a whole second, label-free diffusion model? That would trade the extra-classifier complaint for an extra-unconditional-model complaint — no good. But the unconditional score is just the conditional model evaluated at "no class." So let me give the network a designated null token `∅` for the class slot and *define* the unconditional model as the same network fed `∅`:

```
ε_θ(z_λ) := ε_θ(z_λ, c = ∅).
```

One network produces both — feed it the real label for `ε_c`, feed it `∅` for `ε_∅`. The only thing left is to make sure training actually teaches it the unconditional task too. That's a one-line change: during training, with some probability `p_uncond`, throw away the real label and replace `c` with `∅` before the usual denoising step. On those examples the network learns to denoise without knowing the class — that's exactly `ε_θ(z_λ, ∅)`. The rest of the time it sees the true label and learns `ε_θ(z_λ, c)`. So a single denoising network plus a tiny learned null embedding gives me both predictions, without a second model or classifier pipeline.

How much should `p_uncond` be? The unconditional model is only ever used as the *baseline* I extrapolate away from — I don't need it to be a great unconditional generator, I just need a usable `ε_∅` to define the guidance direction. Too little dropout starves the unconditional estimate; too much dropout spends too many updates on the label-free task and weakens the conditional model I actually sample from. So I should treat `p_uncond` as a small hyperparameter, with something like `0.1` as a sensible starting point rather than a half-and-half split.

Let me also make sure I understand *why* this sharpens, in plain terms, because the negative sign on `ε_∅` is unusual. Going back to the distribution: `p̃(z|c) ∝ p(z|c)^{1+w} / p(z)^w`. So I'm raising the conditional density to a power above 1 (concentrating on its modes) *and dividing by* a power of the unconditional density. That division is the interesting part — it actively pushes *down* the probability of samples that are generic, that look plausible under *any* class, and rewards samples that are distinctively class-`c`. The `−w ε(z_λ)` term is the score-space face of that "decrease the unconditional likelihood" move. Increasing the conditional likelihood while *decreasing* the unconditional one — a negative score term — is what concentrates mass onto class-typical, high-confidence images. That's the fidelity boost, and it's why `w` behaves like a temperature knob.

What do I expect when I sweep `w`? At `w = 0` I'm back to ordinary conditional sampling: best diversity, modest fidelity. As `w` rises, samples should get sharper and more obviously on-class — IS should climb. FID is subtler: a little guidance may fix the worst, least-typical samples, but too much guidance should collapse diversity enough to hurt distributional match. So I should evaluate a curve, not a single setting: the knob is only useful if it traces the same fidelity-versus-diversity frontier that truncation traces.

Now let me write it as code, grounded in how a clean conditional diffusion model is actually built. Two touch points only: conditioning dropout in training, and the linear combination at sampling.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def expand_to_data(v, x):
    if v.ndim == 0:
        v = v.expand(x.shape[0])
    return v.view(-1, *([1] * (x.ndim - 1)))


def alpha_sigma(lam):
    # variance-preserving: alpha_lambda^2 = sigmoid(lambda), sigma_lambda^2 = 1 - alpha^2
    alpha_sq = torch.sigmoid(lam)
    return alpha_sq.sqrt(), (1.0 - alpha_sq).sqrt()


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    if prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    return torch.rand(shape, device=device) < prob


class NoisePredictor(nn.Module):
    """ε_θ(z_λ, λ, c). A learned null embedding lets one network serve as
    both the conditional and unconditional denoiser."""
    def __init__(self, num_classes, cond_dim, backbone):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, cond_dim)
        self.null_emb = nn.Parameter(torch.randn(cond_dim))
        self.backbone = backbone  # consumes (z, lambda, conditioning embedding)

    def forward(self, z, lam, c, cond_drop_prob=0.0):
        cond = self.class_emb(c)
        if cond_drop_prob > 0:
            # conditioning dropout: replace c with the null embedding on a random subset
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
    # corrupt to a random log-SNR level and regress the noise (denoising score matching)
    lam = sample_log_snr(x.shape[0], x.device)          # λ ~ p(λ)
    eps = torch.randn_like(x)
    alpha, sigma = alpha_sigma(expand_to_data(lam, x))
    z = alpha * x + sigma * eps                         # z_λ = α_λ x + σ_λ ε
    # randomly drop the label to the null embedding with probability p_uncond
    eps_pred = model(z, lam, c, cond_drop_prob=p_uncond)
    return F.mse_loss(eps_pred, eps)


@torch.no_grad()
def guided_eps(model, z, lam, c, w):
    # canonical implementations usually expose cond_scale = w + 1
    return model.forward_with_cond_scale(z, lam, c, cond_scale=1.0 + w)


def predict_x_from_eps(z, lam, eps_hat):
    alpha, sigma = alpha_sigma(expand_to_data(lam, z))
    return (z - sigma * eps_hat) / alpha                # x_θ = (z_λ − σ_λ ε̃)/α_λ


@torch.no_grad()
def sample(model, c, schedule, w, shape, v):
    z = torch.randn(shape, device=c.device)             # z ~ N(0, I)
    x_pred = None
    for lam, lam_next in schedule:                      # increasing log-SNR
        eps_hat = guided_eps(model, z, lam, c, w)       # form ε̃ first...
        x_pred = predict_x_from_eps(z, lam, eps_hat)    # ...then an ordinary reverse step
        z = reverse_step(z, lam, lam_next, x_pred, v)
    return x_pred
```

Tracing the whole chain back: I wanted a GAN-style truncation knob for diffusion, and the naive moves (scaling scores, shrinking noise) fail. Truncation is distributionally a sharpening `p̃(x|c) ∝ p(x|c) p(c|x)^w`, whose score is `∇log p(z|c) + w ∇log p(c|z)`. Classifier guidance gets `∇log p(c|z)` from a separately trained noisy-data classifier — but that's an extra pipeline and the step is a classifier-gradient adversarial direction. Bayes rewrites `∇log p(c|z) = ∇log p(z|c) − ∇log p(z)`, the implicit classifier, eliminating the real one; substituting collapses the tilted score to `(1+w)∇log p(z|c) − w∇log p(z)`, i.e. in noise space `ε̃ = (1+w)ε_c − w ε_∅`. A single network with a null token, trained by dropping the label with probability `p_uncond`, supplies both `ε_c` and `ε_∅`, so the whole thing is a one-line change at training and a one-line combination at sampling; since the learned vector fields need not be conservative, the update is not generally the gradient of any classifier likelihood.
