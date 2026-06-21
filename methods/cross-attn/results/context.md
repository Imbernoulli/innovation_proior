# Context: conditioning a diffusion denoiser on a label (circa 2020-2021)

## Research question

A diffusion model generates an image by repeatedly denoising. The trained object is a network
`eps_theta(x_t, t)` that, given a noisy image `x_t` and the noise level `t`, predicts the noise that
was added; sampling runs this network backwards from pure noise down to a clean image. That much is
unconditional — it draws *some* image from the data distribution. Here the target is **conditional**
generation: we want `p(x | c)` for a side input `c` — at minimum a class label, possibly also richer
conditions like a text caption or a spatial layout — and the only handle on the generative process is
the denoiser. The design question is mechanical and specific: *by what architectural operation does the
condition `c` enter `eps_theta`, turning it into `eps_theta(x_t, t, c)`*?

## Background

**The denoiser and its training.** Diffusion models (Sohl-Dickstein et al. 2015; Ho, Jain & Abbeel
2020) fix a forward process that gradually adds Gaussian noise to data on a variance schedule
`beta_1..beta_T`, with `alpha_t = 1 - beta_t` and `abar_t = prod_{s<=t} alpha_s`. A key closed form is
that the noised image at any step is available in one shot,

```
q(x_t | x_0) = N(x_t; sqrt(abar_t) x_0, (1 - abar_t) I),
x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps,   eps ~ N(0, I).
```

Ho et al. showed the reverse process is most easily learned by *predicting the noise* `eps` rather than
the posterior mean, which collapses the variational objective to a plain regression,

```
L_simple = E_{t, x_0, eps} [ || eps - eps_theta( sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps,  t ) ||^2 ].
```

The backbone `eps_theta` is a UNet (Ronneberger et al. 2015): an encoder that downsamples through stages
of residual blocks, a bottleneck, and a decoder that upsamples with skip connections from the encoder.
The network uses **group normalization** throughout (it does not depend on batch statistics, so it
behaves at small batch and per-image), and **self-attention layers at a few low resolutions** (e.g.
16x16) so that distant spatial positions can communicate.

**How a condition is normally fed in: the timestep, by FiLM.** The denoiser already conditions on one
scalar — the noise level `t`. The standard mechanism is feature-wise linear modulation (Perez et al.,
AAAI 2018): the integer `t` is turned into a sinusoidal embedding, pushed through a small MLP that
outputs a per-channel **scale** and **shift**, and these modulate the group-normalized features inside
each residual block, `h <- GroupNorm(h) * (1 + scale) + shift`. This is the established way "extra
information" reaches the convolutional stream, and it generalizes immediately to a class label: embed
the class, add its embedding to the timestep embedding, and let the same FiLM machinery carry both. It
is cheap and stable, which is exactly why it is the default.

**Scaled dot-product attention.** A separate, very general way for one set of vectors to read from
another comes from the attention mechanism (Vaswani et al. 2017). Given a set of *query* vectors packed
in `Q`, *key* vectors in `K`, and *value* vectors in `V`,

```
Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V.
```

Each query forms a dot product with every key; a softmax over those scores gives weights; the output for
that query is the weighted sum of the values. The output has one row per query, of the value dimension.
The `1/sqrt(d_k)` scaling has an effect on training: if query and key components are roughly independent
with unit variance, the dot product `q . k = sum_{i=1}^{d_k} q_i k_i` has variance `d_k`, so for large
`d_k` the logits grow, the softmax saturates near one-hot, and its gradient nearly vanishes; dividing by
`sqrt(d_k)` restores unit-variance logits. **Multi-head** attention runs `h` such functions in parallel on
separately projected, lower-dimensional copies of `Q, K, V` (each of width `d_model/h`), concatenates, and
projects once more, so the model can attend in several representation subspaces at once instead of being
forced to average them into one. Vaswani et al. distinguish three uses: self-attention within the encoder,
masked self-attention within the decoder, and *encoder-decoder attention*, where the queries come from one
sequence and the keys/values come from a **different** sequence, letting every position in the first read
over all positions in the second. Attention is agnostic to the length of the key/value set: the same layer
accepts one key or many.

## Baselines

These are the conditioning mechanisms a new injection method is measured against.

**FiLM / adaptive group-norm conditioning (Perez et al., AAAI 2018; the diffusion default).** Embed the
condition, map it through an MLP to a per-channel scale and shift, and apply that affine to the
group-normalized features inside each residual block. For a class label, add the class embedding to the
timestep embedding and reuse the timestep's FiLM path. Core idea: condition by *modulating feature
statistics* with a single per-channel scale/shift applied identically at every spatial location, taking a
fixed-size condition vector.

**Adaptive group normalization + classifier guidance (Dhariwal & Nichol 2021, "Diffusion Models Beat
GANs").** A strong class-conditional pixel-space diffusion model. Conditioning is AdaGN: the combined
class-and-timestep embedding produces per-channel scale/shift on the group norm (a more careful FiLM), in
a UNet with attention at several resolutions (32, 16, 8). To sharpen class adherence at sampling time it
adds *classifier guidance*: a separate classifier `p_phi(y | x_t)`, trained on noised images, whose
log-probability gradient `grad_{x_t} log p_phi(y | x_t)` is added to the predicted score to shift the
reverse-process mean toward the class. Core idea: AdaGN affine inside the network, plus a separately
trained noise-robust classifier back-propagated through at each sampling step.

**Input concatenation (used for spatially-aligned conditions, e.g. SR3, Saharia et al. 2021).** When the
condition is itself an image-shaped map aligned to the output (a low-resolution image for
super-resolution, a mask for inpainting), simply concatenate it to the noisy input along the channel axis
and let the first convolution mix them. Core idea: condition by *stacking it onto the input*; it applies
when the condition is spatially aligned and image-shaped.

## Evaluation settings

The natural yardsticks already in use for class-conditional image generation:

- **Datasets.** Class-conditional CIFAR-10 (32x32, 10 classes) is the small-scale standard; ImageNet at
  resolutions like 64/128/256, and LSUN / CelebA-HQ for unconditional or attribute-conditional, are the
  larger ones; captioned sets such as MS-COCO are the text-to-image yardstick. The condition `c` is the
  class index (or caption, or layout) that ships with each image.
- **Training protocol.** eps-prediction MSE on noised images with timesteps drawn uniformly, a linear
  `beta` schedule (e.g. `beta_1 = 1e-4` to `beta_T = 2e-2`, `T = 1000`), the Adam/AdamW optimizer, and an
  exponential moving average of the weights used for evaluation.
- **Sampling.** A deterministic, few-step sampler — DDIM (Song, Meng & Ermon 2020), a non-Markovian
  reverse process that shares the same trained denoiser but reaches good samples in tens of steps rather
  than a thousand — run class-conditionally (the same class index fed at every step).
- **Metric.** Frechet Inception Distance between a large pool of generated samples and the training-set
  images, computed with a fixed Inception feature extractor (e.g. clean-fid); lower is better. The
  improvement must come from the conditioning design, holding the dataset, labels, loss, optimizer,
  sampler, and metric fixed.

## Code framework

The conditioning method plugs into an existing class-conditional DDPM harness: a diffusers-style UNet
denoiser backbone, the standard eps-prediction MSE training loop, a DDIM sampler, and FID scoring are all
already in place and fixed. Two things already exist for the condition to ride on: a learnable
`nn.Embedding` that maps a class index to a vector the size of the timestep embedding, and the UNet's own
timestep/FiLM path. What is *not* settled is the operator that decides how the class embedding actually
reaches the spatial features — that operator is exactly what is to be designed. So the substrate exposes
two empty slots: one for how (if at all) the class embedding is combined with the timestep embedding
before the blocks, and one module applied to a feature map after each block.

```python
import torch
import torch.nn as nn


# --- already exists: the denoiser backbone, training loop, sampler, FID ---
# UNet2DModel backbone (group norm throughout, self-attention at low resolution),
# eps-prediction MSE loss, AdamW + weight EMA, DDIM 50-step class-conditional sampler,
# clean-fid scoring. All fixed; only the two slots below are open.


def prepare_conditioning(time_emb, class_emb):
    """How the class embedding is combined with the timestep embedding before blocks.

    time_emb:  [B, D] timestep (noise-level) embedding
    class_emb: [B, D] class embedding
    returns:   [B, D] embedding fed to the residual blocks' FiLM path
    """
    # TODO: decide what (if anything) the class embedding contributes here.
    raise NotImplementedError


class ClassConditioner(nn.Module):
    """The conditioning operator applied to a feature map after each UNet block.

    forward(h, class_emb):
        h:         [B, C, H, W] feature map from the block
        class_emb: [B, D] class embedding
        returns:   [B, C, H, W] conditioned feature map
    """
    def __init__(self, channels, cond_dim):
        super().__init__()
        # TODO: the conditioning operator we will design.
        raise NotImplementedError

    def forward(self, h, class_emb):
        # TODO: apply the class embedding to the feature map h.
        raise NotImplementedError


# existing UNet wrapper: embeds the class, calls prepare_conditioning to form the
# block embedding, runs the backbone, and applies a ClassConditioner after each block.
class ConditionalUNet(nn.Module):
    def __init__(self, backbone, num_classes, cond_dim):
        super().__init__()
        self.backbone = backbone
        self.class_embed = nn.Embedding(num_classes, cond_dim)
        # one ClassConditioner per (down / mid / up) block, sized to that block's channels
        # ... (constructed against the backbone's per-block channel widths)

    def forward(self, x, t, class_id):
        t_emb = self.backbone.time_embedding(t)            # noise-level embedding (existing)
        class_emb = self.class_embed(class_id)             # [B, D]
        emb = prepare_conditioning(t_emb, class_emb)       # slot 1
        # backbone forward, inserting self.<block>_cond[i](h, class_emb) after each block  # slot 2
        return predicted_eps
```

The two `raise NotImplementedError` slots are exactly what the conditioning design fills in.
