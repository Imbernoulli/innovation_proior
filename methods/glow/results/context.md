## Research question

How can we build a generative model of high-dimensional natural images that (a) lets us evaluate the **exact** log-likelihood of a datapoint, (b) lets us infer the **exact** latent code of a datapoint with no approximation, (c) **trains and samples in parallel** on modern hardware, and (d) actually produces sharp, realistic, high-resolution samples — all at once?

Each existing family gives up at least one of these. Autoregressive models give exact likelihood but sample one subpixel at a time, so synthesis is serial and painfully slow on a 256×256 image. Variational autoencoders sample in parallel but only optimize a *lower bound* on the likelihood and only infer the latent *approximately*. Adversarial models make beautiful images but have no encoder, no tractable likelihood, and no way to even measure how well they fit the data. A model built from exactly-invertible transformations could in principle deliver all four properties simultaneously, but the existing instances of that idea lagged behind on likelihood and had never been shown to scale to realistic high-resolution synthesis. The question is whether the invertible-transformation approach can be pushed to close those gaps.

## Background

**The maximum-likelihood / change-of-variables setting.** Let `x` be a datapoint with unknown density `p*(x)`. We pick a model `p_θ(x)` and minimize the average negative log-likelihood `L = (1/N) Σ_i −log p_θ(x^(i))`, which is exactly the expected compression cost in nats (or bits, after dividing by `log 2`); reporting it per dimension gives **bits per dimension**.

A clean way to define `p_θ` is to start from a simple latent density and push it through an invertible map. Let `z ~ p(z)` with `p(z)` a tractable density such as a spherical Gaussian `N(0, I)`, and let `x = g(z)` be a bijection with inverse `z = f(x) = g^{-1}(x)`. The **change-of-variables formula** then gives the model density exactly:

```
p_θ(x) = p(z) · |det(dz/dx)|,   z = f(x)
log p_θ(x) = log p(z) + log|det(dz/dx)|.
```

The term `log|det(dz/dx)|` — the **log-determinant** of the Jacobian of `f` — is the change in log-density induced by the volume distortion of the map. If `f` is built as a composition `f = f_1 ∘ f_2 ∘ ... ∘ f_K` with intermediate states `h_0 = x, h_1, ..., h_K = z`, the log-determinant decomposes into a sum,

```
log p_θ(x) = log p(z) + Σ_{i=1}^K log|det(dh_i / dh_{i-1})|,
```

so each invertible layer contributes one log-determinant. Such a composition of invertible maps is a **normalizing flow** (Rezende & Mohamed, 2015). The entire difficulty is that a general `c×c` (or `D×D`) Jacobian determinant costs `O(D^3)`; the field's central trick (Deco & Brauer, 1995; Dinh et al., 2014; Rezende & Mohamed, 2015) is to choose layers whose Jacobian is **triangular**, so the determinant is the product of the diagonal and

```
log|det(dh_i/dh_{i-1})| = sum(log |diag(dh_i/dh_{i-1})|),
```

an `O(D)` quantity.

**Dequantization.** Image pixels are 8-bit integers, but a continuous density placed on a discrete grid can put unbounded density on the grid points and cheat the likelihood. Adding independent uniform noise `u ~ U([0, a)^M)` to form `x̃ = x + u` (with `a` set by the discretization level) turns the continuous-density objective into a lower bound on the discrete-data log-likelihood:

```
log P(x) = log ∫_bin p(x') dx'
         = M log a + log E_u[p(x + u)]
         ≥ M log a + E_u[log p(x + u)].
```

Thus the negative log-likelihood objective carries the fixed constant `−M log a`; with `a = 1/n_bins`, the log-likelihood accumulator adds `M log a = −M log(n_bins)`.

**Why this family is attractive.** Because `f` is an exact bijection, latent inference `z = f(x)` is exact (no approximate posterior as in VAEs, no missing encoder as in adversarial models), and the continuous-density term `log p_θ(x)` is exact once the data have been dequantized. Both directions are feed-forward neural nets, so training and sampling parallelize on a GPU — unlike autoregressive models, whose sampling is inherently sequential in the number of pixels. And because the layers are invertible, activations can be recomputed on the backward pass rather than stored, giving memory cost roughly constant in depth (Gomez et al., 2017, RevNets).

**Diagnostic observations that shape the design.** Two facts about the existing systems matter. First, a coupling layer leaves half the variables unchanged, so the permutation placed between coupling layers decides which variables can condition which later variables; fixed reversal and fixed random shuffling are arbitrary, non-learned routing choices. Second, batch normalization, the standard tool for training deep nets, injects activation noise whose variance is inversely proportional to the per-processing-unit minibatch size; for high-resolution images memory forces a minibatch of size 1 per GPU, the regime where batch-normalization statistics are at their noisiest and least reliable.

## Baselines

**NICE — Non-linear Independent Components Estimation (Dinh, Krueger & Bengio, 2014).** The founding instance of this approach. Its workhorse is the **additive coupling layer**: split the input into two parts `(x_a, x_b)`, leave one part untouched and shift the other by an arbitrary neural net of the untouched part,

```
y_a = x_a,    y_b = x_b + m(x_a),
```

with inverse `x_a = y_a, x_b = y_b − m(y_a)`. Because `m` is only ever evaluated, never inverted, it can be any network. The Jacobian is triangular with **ones on the diagonal**, so its determinant is `1` and its log-determinant is `0` — the map is volume-preserving. Stacking such layers requires swapping which half is updated between layers (otherwise the same variables would only condition while the other half alone changed), so NICE alternates the partition. To let the model rescale volume at all, NICE appends a single learned **diagonal scaling** `z = s ⊙ h` at the end, contributing `Σ log|s|` to the log-determinant. **Gap it leaves:** the per-layer transformation is purely additive (no per-coupling scaling), the mixing between layers is a hand-fixed partition, and it was demonstrated on small data rather than convolutional image models at scale.

**RealNVP — Density estimation using Real NVP (Dinh, Sohl-Dickstein & Bengio, 2016).** Extends NICE in the directions that make it a real image model.

- **Affine coupling layer.** Generalize the additive shift to a scale *and* shift, both functions of the untouched half:

  ```
  y_a = x_a,    y_b = (x_b + t(x_a)) ⊙ exp(s(x_a)),
  ```

  inverted by `x_a = y_a, x_b = y_b ⊙ exp(−s(y_a)) − t(y_a)`. The Jacobian is block-triangular with identity and diagonal block `diag(exp(s(x_a)))`, so the **log-determinant is `Σ s(x_a)`** — the layer can now change volume (additive coupling is the special case `s ≡ 0`).
- **Masking.** Rather than a literal split, RealNVP uses binary masks in two patterns — a spatial **checkerboard** mask and a **channel** mask — and alternates them so every variable is eventually transformed.
- **Squeeze.** A reshape that trades space for channels, `h × w × c → (h/2) × (w/2) × 4c`, so deeper layers have channels to apply channel-wise coupling to.
- **Multi-scale architecture.** After several flow steps at a given resolution, **factor out** half of the dimensions as part of the latent `z` (modeled by a Gaussian) and continue transforming only the other half at the next, coarser scale. This produces a coarse-to-fine latent and reduces compute and memory at deep layers.
- **Batch normalization** is used both inside the coupling networks and as a flow layer whose scaling contributes to the Jacobian, to make deep flows trainable.

**Gaps RealNVP leaves open.** (1) The only mixing of variables *between* coupling layers is a **fixed permutation** — reversing the channel order, or the fixed alternation of checkerboard/channel masks. A fixed permutation is a hand-chosen, non-learned choice of which variables flow into the next coupling's conditioning half; it is unclear it is the best mixing. (2) Its normalization layer is **batch normalization**, whose noise grows as the per-PU batch shrinks — exactly the high-resolution, batch-size-1 regime. (3) Maintaining **two mask types** (checkerboard and channel) is architectural overhead.

**Autoregressive and adversarial models, as foils.** PixelRNN/PixelCNN and WaveNet (van den Oord et al., 2016; 2016) give exact likelihoods and are also invertible in a sense, but sample one element at a time, so synthesis of a high-resolution image takes orders of magnitude longer and their hidden layers have no usable marginal latent space. Adversarial models (Goodfellow et al., 2014; Karras et al., 2017) synthesize striking images but have no encoder, no exact likelihood, often lack full support over the data (Grover et al., 2018), and are hard to assess for overfitting.

## Evaluation settings

The natural yardstick is **average negative log-likelihood in bits per dimension** on standard natural-image benchmarks, lower being better, computed on a held-out test set with the dequantization-and-constant convention above.

- **CIFAR-10** (Krizhevsky, 2009): 32×32 color images.
- **ImageNet**, downsampled to **32×32** and **64×64** (van den Oord et al., 2016).
- **LSUN** (bedroom, tower, church-outdoor; Yu et al., 2015): downsampled to 96×96 with random 64×64 crops.
- **CelebA-HQ** (Karras et al., 2017): 30000 high-resolution face images, used at 256×256 for high-resolution qualitative study (samples, latent interpolation, attribute manipulation), with a train/validation split.

Preprocessing follows the RealNVP protocol; some studies use reduced-bit (5-bit) images. The optimizer is Adam (Kingma & Ba, 2015) with `α = 0.001` and default `β`. Models are described by a depth-per-level `K` and a number of levels `L`. A qualitative knob is **sampling temperature** `T`: for a spherical Gaussian prior, sampling `z` from `N(0, T^2 I)` is equivalent in latent space to using a density proportional to the unit-Gaussian density raised to `1/T^2`.

## Code framework

The existing pieces are a data pipeline that loads images and dequantizes them, a generic invertible-layer abstraction with forward/inverse and a running log-determinant accumulator, the change-of-variables objective in bits/dim, the squeeze/multi-scale plumbing inherited from prior flows, convolutional heads for conditional Gaussian priors, and the Adam training loop. The unresolved design slot is the content of one generic invertible step.

```python
import math
import torch
import torch.nn as nn

# ---- data: dequantize discrete pixels into a continuous density problem ----
def preprocess(x, n_bits=8):
    n_bins = 2 ** n_bits
    x = x.float()
    if n_bits < 8:
        x = torch.floor(x / 2 ** (8 - n_bits))
    x = x / n_bins - 0.5
    x = x + torch.rand_like(x) / n_bins          # uniform dequantization
    return x, n_bins

# ---- generic invertible layer: forward returns (output, logdet contribution) ----
class FlowModule(nn.Module):
    def forward(self, x, logdet):
        raise NotImplementedError
    def reverse(self, y):
        raise NotImplementedError

# ---- squeeze / multi-scale plumbing inherited from prior flows ----
def squeeze2d(x, factor=2):
    b, c, h, w = x.shape
    x = x.view(b, c, h // factor, factor, w // factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.view(b, c * factor * factor, h // factor, w // factor)

def unsqueeze2d(x, factor=2):
    b, c, h, w = x.shape
    x = x.view(b, c // factor ** 2, factor, factor, h, w)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    return x.view(b, c // factor ** 2, h * factor, w * factor)

# ---- convolutional output head for Gaussian prior parameters ----
class OutputConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, logscale_factor=3.0):
        super().__init__()
        pass  # TODO

    def forward(self, x):
        pass  # TODO

# ---- open invertible step ----
class FlowStep(FlowModule):
    """One generic step of the flow."""
    def __init__(self, channels, width=512):
        super().__init__()
        pass  # TODO

    def forward(self, x, logdet):
        pass  # TODO: transform x, add this step's log|det Jacobian| to logdet

    def reverse(self, y):
        pass  # TODO: exact inverse

# ---- latent prior + objective (change of variables, in bits/dim) ----
def gaussian_logp(z, mean, log_sd):
    return (-0.5 * (math.log(2 * math.pi) + 2 * log_sd
                    + (z - mean) ** 2 / torch.exp(2 * log_sd)))

class ImageFlow(nn.Module):
    def __init__(self, in_ch=3, depth=32, levels=3, width=512):
        super().__init__()
        self.levels = levels
        self.blocks = nn.ModuleList()
        self.split_priors = nn.ModuleList()
        channels = in_ch * 4
        for level in range(levels):
            self.blocks.append(nn.ModuleList([FlowStep(channels, width) for _ in range(depth)]))
            if level < levels - 1:
                self.split_priors.append(OutputConv2d(channels // 2, channels))
                channels *= 2
        self.top_prior = OutputConv2d(channels, 2 * channels)

    def forward(self, x):
        pass  # TODO: squeeze, run FlowStep blocks, split latents, score Gaussian priors

    def reverse(self, z, eps=None, eps_std=1.0):
        pass  # TODO: sample/merge latents, reverse FlowStep blocks, unsqueeze

def loss_bits_per_dim(logdet, log_p, n_bins, n_pixels):
    lower_bound = log_p + logdet - math.log(n_bins) * n_pixels
    return (-lower_bound / (math.log(2) * n_pixels)).mean()

# ---- training loop (already standard) ----
def train_step(model, batch, opt):
    x, n_bins = preprocess(batch)
    _, logdet, log_p = model(x)                   # stacks of FlowStep + squeeze/split
    loss = loss_bits_per_dim(logdet, log_p, n_bins, x[0].numel())
    opt.zero_grad(); loss.backward(); opt.step()
    return loss
```
