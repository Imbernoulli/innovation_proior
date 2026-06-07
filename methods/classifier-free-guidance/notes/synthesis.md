# Synthesis — Classifier-Free Diffusion Guidance (CFG)

## The pain point (research question)
Other generative model families have a knob that trades **diversity for fidelity** at sampling time:
- BigGAN: the **truncation trick** — sample the latent z from a truncated normal (resample any |z| > threshold). Lower threshold → higher IS, lower FID-on-recall, less diversity.
- Glow / normalizing flows: **low-temperature sampling** — scale the base-distribution std by T<1.

Diffusion models had no such knob. Naive analogues fail (Dhariwal & Nichol 2021 report this empirically):
- Scaling the model score / ε by a constant → blurry, washed-out samples.
- Reducing the variance of the injected Gaussian noise in the reverse step → also degrades quality.

So: **can we get a temperature/truncation-like fidelity↔diversity knob for diffusion?**

## Background (load-bearing ancestors)

### DDPM ε-prediction (Ho et al. 2020)
Forward (VP): `z_λ = α_λ x + σ_λ ε`, ε~N(0,I). Train a network ε_θ(z_λ) to predict the noise via simple MSE `E‖ε_θ(z_λ) − ε‖²`. Reverse step plugs `x_θ = (z_λ − σ_λ ε_θ)/α_λ` into the tractable posterior q(z_{λ'}|z_λ, x). This is the parameterization CFG uses (continuous-time, log-SNR λ, VP, from Kingma et al. 2021 VDM / Song et al. 2020).

### Denoising score matching → score interpretation (Vincent 2011; Song & Ermon 2019; Song et al. 2020)
The MSE noise-prediction loss IS denoising score matching at each noise level. Consequence:
**ε_θ(z_λ) ≈ −σ_λ ∇_{z_λ} log p(z_λ)**, i.e. the network's ε is (minus) the score scaled by σ_λ. This is the bridge between ε-space (what we train) and score-space (∇log p, what guidance manipulates). The sign/factor: score = −ε/σ_λ.
(Caveat the paper notes: ε_θ from an unconstrained net need not be the gradient of any scalar potential — it's a non-conservative vector field. Matters later.)

For the conditional model: ε_θ(z_λ, c) ≈ −σ_λ ∇_{z_λ} log p(z_λ | c).

### Classifier guidance (Dhariwal & Nichol 2021) — the baseline being reacted to
Take a conditional diffusion score and add the gradient of a **separately trained** classifier p_φ(c | z_λ) that must be trained on **noisy** images z_λ:
```
ε̃_θ(z_λ, c) = ε_θ(z_λ, c) − w σ_λ ∇_{z_λ} log p_φ(c | z_λ)
            ≈ −σ_λ ∇_{z_λ}[ log p(z_λ | c) + w log p_φ(c | z_λ) ]
```
Samples come from `p̃(z_λ|c) ∝ p(z_λ|c) · p_φ(c|z_λ)^w`. Larger w → up-weight inputs the classifier is confident about → IS up, diversity down. Verified against openai/guided-diffusion `condition_score`: `eps = eps − sqrt(1−ᾱ)·cond_fn` with `sqrt(1−ᾱ)=σ_λ`. Sign and σ factor confirmed.

**Why undesirable (the motivation):**
1. Needs an **extra classifier**, and it must be trained on noisy data across all noise levels — can't drop in a pretrained ImageNet classifier. Complicates the pipeline.
2. Stepping along ∇log p_φ(c|x) resembles a **gradient-based adversarial attack** on the classifier → raises doubt whether the IS/FID gains are "real" or just fooling the very classifiers IS/FID are built on.
3. Resembles GAN-like behavior (stepping in a discriminator gradient).

## The derivation (the heart)

### Step A — reframe classifier guidance as sampling a tilted distribution
We want to sample `p̃(z|c) ∝ p(z|c) p(c|z)^w`. In score space:
`∇log p̃(z|c) = ∇log p(z|c) + w ∇log p(c|z)`.

### Step B — Bayes turns the classifier into a ratio of generative scores
Bayes: `p(c|z) = p(z|c) p(c) / p(z)`, so `log p(c|z) = log p(z|c) − log p(z) + log p(c)`. The last term is constant in z, so:
```
∇_z log p(c|z) = ∇_z log p(z|c) − ∇_z log p(z).
```
This is an **implicit classifier** built purely from the conditional and unconditional generative scores — no separate classifier.

### Step C — substitute into the tilted score
```
∇log p̃(z|c) = ∇log p(z|c) + w[∇log p(z|c) − ∇log p(z)]
            = (1+w) ∇log p(z|c) − w ∇log p(z).
```

### Step D — convert scores → ε via ε = −σ_λ ∇log p
Multiply by −σ_λ (so the combined object is again an ε-prediction):
```
ε̃(z_λ, c) = (1+w) ε(z_λ, c) − w ε(z_λ).        [paper Eq. (6), alg 2]
```
Equivalent rearrangement: `ε̃ = ε(z_λ) + (w+1)(ε(z_λ,c) − ε(z_λ))` — wait, check: (1+w)ε_c − w ε_∅ = ε_∅ + (1+w)ε_c − (1+w)ε_∅ = ε_∅ + (1+w)(ε_c − ε_∅). Yes. So the "guidance scale" `s = w+1` multiplies the conditional-minus-unconditional delta. **Convention check:** paper's w: w=0 → ε̃ = ε_c (plain conditional). lucidrains `cond_scale = w+1`: cond_scale=1 → plain conditional; `scaled = logits + (logits−null)(cond_scale−1)` = ε_∅ + cond_scale·(ε_c−ε_∅)... let me recheck: logits=ε_c, null=ε_∅, update=ε_c−ε_∅, scaled = ε_c + (ε_c−ε_∅)(cond_scale−1) = ε_c·cond_scale − ε_∅(cond_scale−1) = (cond_scale)ε_c − (cond_scale−1)ε_∅. With cond_scale = 1+w: (1+w)ε_c − w ε_∅. **Matches exactly.** Diffusers `guidance_scale g`: `ε̃ = ε_∅ + g(ε_c − ε_∅)`, so g = w+1 too.

### Step E — getting both ε(z,c) and ε(z) from ONE network: conditioning dropout
We need an unconditional score ε(z_λ) = ε(z_λ, c=∅) alongside ε(z_λ, c). Train a single net; during training replace c with a learned null token ∅ with probability p_uncond. Then the same weights give both. (Could train two separate nets — but joint training is one line of code, no extra params, no extra pipeline.) Empirically p_uncond ∈ {0.1, 0.2} works; 0.5 is worse — only a small slice of capacity needs to go to the unconditional task.

### Step F — the implicit-classifier caveat (honesty / why it isn't a contradiction)
ε̃ here is NOT literally a classifier-guided score: ε_θ(z,c) − ε_θ(z) is the difference of two unconstrained-network outputs, generally not the gradient of any scalar potential (non-conservative), so no real classifier p(c|z) has it as its gradient. CFG is *inspired by* the implicit classifier but is its own object. Bonus: this means the step direction is provably NOT a classifier gradient → cannot be dismissed as an adversarial attack, answering the motivation's doubt. Also Bayes-inverted classifiers from a (likely misspecified) generative model carry no guarantees (Grandvalet & Bengio 2004; Grünwald & Langford 2007) — so it's an empirical bet, justified by the IS/FID results.

### The fidelity↔diversity knob
`p̃(z|c) ∝ p(z|c)^{1+w} / p(z)^w`. Raising p(z|c) and *lowering* p(z) (negative score term — novel) sharpens onto class-typical, high-confidence modes → IS↑ monotone, FID non-monotone (best FID at small w≈0.1–0.3, then worsens as diversity collapses).

## Design-decision → why table

| Decision | Why this, why not the alternative |
|---|---|
| ε-prediction parameterization | DDPM showed MSE-on-noise = denoising score matching, gives ε ≈ −σ∇log p; lets guidance be a linear combo in ε-space. Predicting x directly loses the clean score link. |
| Guide an already-conditional model (mix ε_c and ε_∅) rather than guide an unconditional model with (w+1) | Algebraically `p(z|c)p(c|z)^w ∝ p(z)p(c|z)^{w+1}` — same target. Dhariwal & Nichol got best results guiding the conditional model, so CFG keeps that setup. |
| Implicit classifier via Bayes (∇log p(c|z)=∇log p(z|c)−∇log p(z)) | Eliminates the separate classifier entirely — the whole point. Removes noisy-classifier training, the adversarial-attack interpretation, and extra params. |
| Single net + conditioning dropout (replace c→∅ w.p. p_uncond) | One-line change, zero extra params, no pipeline complication. Alternative (two separate nets) doubles storage and infra for no quality gain. |
| Learned null token ∅ for "unconditional" | Gives the net an explicit, learnable representation of "no class"; reuses the exact same architecture/forward pass. |
| p_uncond ≈ 0.1–0.2 | Only a small share of capacity needs to serve the unconditional task; p_uncond=0.5 wastes capacity and hurts the whole IS/FID frontier. |
| Combine as ε̃=(1+w)ε_c − w ε_∅ at sample time | Direct consequence of Step C/D. Convention: w=0 ⇒ plain conditional; w>0 ⇒ guidance. Implementations use scale s=w+1. |
| Two forward passes per step (cond + uncond) | Cost of having no classifier. Could be cut by batching c and ∅ together, or injecting conditioning late; the simple version just runs both. |
| Negative weight on unconditional score | Decreasing the *unconditional* likelihood (push away from generic samples) while raising the conditional — the mechanism of the fidelity boost. |

## Canonical code grounding
lucidrains `denoising-diffusion-pytorch/classifier_free_guidance.py`:
- Training: `forward()` does `keep_mask = prob_mask_like(b, 1−cond_drop_prob)`, replaces class emb with `null_classes_emb` where dropped → conditioning dropout (Alg 1).
- Sampling: `forward_with_cond_scale`: `logits=forward(cond_drop_prob=0)`, `null=forward(cond_drop_prob=1)`, `scaled = logits + (logits−null)(cond_scale−1)` = (1+w)ε_c − w ε_∅ with cond_scale=w+1 (Alg 2 / Eq 6).
- Loss: `p_losses` MSE on noise.
openai/guided-diffusion `condition_score` confirms the classifier-guidance baseline `eps − σ_λ ∇log p_φ(c|x)`.
