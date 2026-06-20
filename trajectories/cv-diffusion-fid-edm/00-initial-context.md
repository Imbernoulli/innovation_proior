## Research question

A diffusion model generates an image by learning a denoiser and then running a sampler that walks
pure noise down to a clean picture. On CIFar-10 at $32\times32$ the quality of those pictures is
measured by **FID** — Fréchet Inception Distance against the real CIFAR-10 statistics, computed over
**50,000 generated samples**, lower is better. The single thing being designed here is the
**training design of the denoiser**: how the network is parameterized as a function of noise level,
what regression target and per-noise-level loss weighting it is trained against, how the training
noise levels are sampled, and how the data is augmented. The sampler is held fixed at a strong
deterministic setting (a 2nd-order solver, **35 network evaluations per image**, NFE = 35), and so are
the dataset and the FID protocol. The question is how low the FID can be pushed on CIFAR-10 — in both
the class-conditional and the unconditional setting — by redesigning *how the denoiser is trained*,
not by changing the sampler.

This matters because the field's recipes for training the denoiser had accreted, one paper at a time,
into a tangle: a particular forward-process schedule, a particular thing the network is asked to
predict, a particular implicit loss weighting across noise levels, a particular noise-level sampling
distribution — each justified locally, none of them obviously the best joint choice. A solution would
have to disentangle these training-time decisions from one another and from the sampler, and find the
setting of each that the FID actually prefers.

## Background

**The forward process and the score/denoising view.** A datapoint $\mathbf{y}$ is corrupted by adding
i.i.d. Gaussian noise of standard deviation $\sigma$: $\mathbf{x} = \mathbf{y} + \sigma\boldsymbol{\epsilon}$,
$\boldsymbol{\epsilon}\sim\mathcal{N}(0,\mathbf{I})$. Sweeping $\sigma$ from near zero up to a large
maximum gives a continuum of progressively noisier marginals $p(\mathbf{x};\sigma)$. Sohl-Dickstein et
al. (2015) and Ho et al. (2020) cast generation as learning to reverse this corruption; Song et al.
(2021, *Score-Based Generative Modeling through SDEs*) showed the discrete-time chains and the
continuous score-matching view are two formulations of the same object, with two standard
parameterizations of the noise schedule — **variance-preserving (VP)**, where the signal is rescaled
so the noisy marginal keeps unit variance, and **variance-exploding (VE)**, where clean signal is kept
and ever-larger noise is added. Either way, training reduces to fitting a denoiser
$D(\mathbf{x};\sigma)\approx\mathbb{E}[\mathbf{y}\mid\mathbf{x}]$, and the score of the noisy marginal is
$\nabla_{\mathbf{x}}\log p(\mathbf{x};\sigma)=(D(\mathbf{x};\sigma)-\mathbf{x})/\sigma^2$.

**Why the across-noise-level conditioning is the hard part.** A single network must serve every noise
level. At tiny $\sigma$ the input is almost the clean image and the denoiser's job is nearly the
identity; at huge $\sigma$ the input is almost pure noise and the conditional mean
$\mathbb{E}[\mathbf{y}\mid\mathbf{x}]$ is barely identifiable and has large variance. The magnitude of the
ideal target, the magnitude of the network's input, the irreducible error, and therefore the natural
loss scale, all swing by orders of magnitude across $\sigma$. Any training recipe makes — explicitly
or implicitly — four coupled decisions about this: (i) how the raw network output is mapped to the
denoiser (input/output scaling as a function of $\sigma$); (ii) what regression target the loss
compares against; (iii) how the per-$\sigma$ loss terms are weighted; (iv) which $\sigma$ values are
drawn during training. These decisions had each been made inside the VP/VE chains rather than as free,
separable knobs.

**The metric and the diagnostic pattern it exposes.** FID over 50k samples is dominated by global
image statistics, which are set early in sampling — at the high-noise steps — and refined late. So a
denoiser that is well-conditioned at high noise but mediocre at low noise, or vice versa, leaves a
visible FID signature; the recipe's implicit weighting across $\sigma$ is not a cosmetic choice but the
thing the metric is most sensitive to. It was already understood that the diffusion losses, written as
an $\mathbf{x}_0$- or $\boldsymbol{\epsilon}$-regression under a flat or schedule-derived weight,
distribute their effective emphasis across noise levels very unevenly, and that the network spends a
disproportionate share of its capacity on noise levels that contribute little to sample quality.

## Baselines

- **DDPM (Ho, Jain & Abbeel 2020).** Trains a noise predictor
  $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)$ on the variance-preserving chain
  $\mathbf{x}_t=\sqrt{\bar\alpha_t}\,\mathbf{x}_0+\sqrt{1-\bar\alpha_t}\,\boldsymbol{\epsilon}$ with a
  linear $\beta$ schedule ($\beta_1=10^{-4}\to\beta_T=0.02$, $T=1000$). Its objective is the
  *simplified* loss $L_{\text{simple}}=\mathbb{E}_{t,\mathbf{x}_0,\boldsymbol{\epsilon}}\big[\|\boldsymbol{\epsilon}-\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)\|^2\big]$,
  i.e. an *unweighted* $\boldsymbol{\epsilon}$-MSE with the variational per-$t$ weight discarded. The
  reverse-process variances are fixed to untrained constants ($\sigma_t^2=\beta_t$ or $\tilde\beta_t$),
  and sampling runs the full $T$-step ancestral chain. Gorgeous samples, GAN-free stability — but the
  $\boldsymbol{\epsilon}$ target and the constant input pose a regression whose scale and difficulty
  vary sharply across $t$, the flat weight emphasizes the low-noise near-identity terms where the
  return is small, and the fixed reverse variance is a guess rather than a fit, so the model leaves
  quality on the table at the noise levels FID cares about.

- **Improved DDPM (Nichol & Dhariwal 2021).** Reacts to two limitations of the DDPM recipe. First,
  the linear $\beta$ schedule destroys signal too quickly near the end of the forward process, wasting
  the highest-noise steps; it replaces it with a **cosine schedule**
  $\bar\alpha_t=f(t)/f(0)$, $f(t)=\cos^2\!\big(\tfrac{t/T+s}{1+s}\cdot\tfrac{\pi}{2}\big)$, $s=0.008$.
  Second, the fixed reverse variance is suboptimal for likelihood; it **learns** the reverse-process
  variance per coordinate by interpolating in log space,
  $\Sigma_\theta=\exp\!\big(v\log\beta_t+(1-v)\log\tilde\beta_t\big)$, trained by a hybrid objective
  $L_{\text{hybrid}}=L_{\text{simple}}+\lambda L_{\text{vlb}}$ ($\lambda=0.001$, with a stop-gradient
  so the variational term shapes only the variance), and importance-samples the timesteps for the
  variational term. Gap: the gains are real but modest on CIFAR-10 FID, the recipe is now a stack of
  separate fixes layered onto the same $\boldsymbol{\epsilon}$ parameterization and flat-ish
  $\boldsymbol{\epsilon}$-MSE, and the across-$\sigma$ weighting and the network's input/output scaling
  are still inherited from the chain rather than chosen for the regression.

- **ADM / guided-diffusion (Dhariwal & Nichol 2021).** Holds the diffusion training recipe roughly
  fixed and pours effort into the **denoiser architecture** — a heavily tuned U-Net: attention at
  several resolutions ($32\times32$, $16\times16$, $8\times8$) with a fixed 64 channels per head,
  BigGAN-style residual blocks for up/downsampling, residual connections rescaled by $1/\sqrt2$, and
  **adaptive group normalization (AdaGN)** that injects the (timestep, class) embedding as a learned
  per-channel scale and shift, $\mathrm{AdaGN}(\mathbf{h},\mathbf{e})=\mathbf{e}_s\cdot\mathrm{GroupNorm}(\mathbf{h})+\mathbf{e}_b$.
  This architecture sets a strong bar on ImageNet at $64\times64$. Gap: it improves the *backbone* but
  leaves the *training-design* decisions — what the network predicts, how the loss is weighted across
  $\sigma$, how $\sigma$ is sampled, how the data is augmented without leaking into samples — exactly
  where DDPM and iDDPM left them. The architecture is a fixed, strong substrate; the open question is
  whether the way it is *trained* as a denoiser is itself near-optimal.

## Evaluation settings

CIFAR-10 at $32\times32$, both **class-conditional** and **unconditional**. Quality is FID against the
CIFAR-10 reference statistics, computed from **50,000 generated images** (the standard protocol; FID is
sensitive to sample count and seed, so the figure is taken as the minimum over a few independent 50k
draws). Sampling uses a fixed strong deterministic solver at **NFE = 35** network evaluations per
image, identical across every training recipe compared, so that differences in FID are attributable to
the *training* of the denoiser and not to the sampler. For broader generalization, the same
training-design changes are also evaluated on a class-conditional model at $64\times64$ resolution
where a strong pre-trained architecture already exists, again by FID.

## Code framework

The substrate is a from-scratch diffusion trainer. A denoiser network is wrapped by a
**preconditioning** module that, given a noisy image $\mathbf{x}$ and its noise level $\sigma$, returns
the denoised estimate $D(\mathbf{x};\sigma)$; a **loss** object draws a noise level, corrupts a clean
image, queries the denoiser and returns a per-pixel loss; an optional **augmentation** pipe perturbs
the clean image before noising. The U-Net backbones (a DDPM++/score-SDE-style net and an ADM-style net)
already exist and are not the object of design here. What is open is the *training design* — the slots
below, written in pre-method terms with empty bodies.

```python
# Preconditioning: wrap a raw U-Net F_theta as a denoiser D(x; sigma).
class Precond(torch.nn.Module):
    def __init__(self, img_resolution, img_channels, label_dim=0, model_type='SongUNet', **model_kwargs):
        super().__init__()
        self.model = globals()[model_type](img_resolution=img_resolution, in_channels=img_channels,
                                           out_channels=img_channels, label_dim=label_dim, **model_kwargs)

    def forward(self, x, sigma, class_labels=None, **model_kwargs):
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        # TODO: choose how the noise level sigma maps to
        #   - the scaling applied to the network input,
        #   - the scaling/skip applied to the network output to form D(x; sigma),
        #   - the scalar conditioning fed to the network in place of sigma.
        raise NotImplementedError


# Loss: noise an image, query the denoiser, return the per-pixel regression loss.
class Loss:
    def __call__(self, net, images, labels=None, augment_pipe=None):
        # TODO: choose
        #   - the distribution from which the training noise level sigma is drawn,
        #   - the per-noise-level weighting applied to the squared error,
        #   - the regression target the denoiser output is compared against.
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        raise NotImplementedError


# Augmentation: optionally perturb the clean image before noising.
class AugmentPipe:
    def __call__(self, images):
        # TODO: decide whether/how to augment the training images, and
        #   what (if anything) must be done so the augmentation does not
        #   change the distribution the model is asked to generate.
        raise NotImplementedError
```

The denoiser's raw backbone $F_\theta$ and the U-Net definitions are fixed. Every rung below is a choice
of how to fill the three slots — the preconditioning $D(\mathbf{x};\sigma)$, the loss (noise
distribution, weighting, target), and the augmentation — and nothing else about the sampler, dataset,
or FID protocol changes.
