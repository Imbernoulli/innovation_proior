## Research question

A diffusion model generates an image by training a denoiser and then running a sampler that walks pure noise down to a clean picture. On CIFAR-10 at $32\times32$, quality is measured by **FID** — Fréchet Inception Distance against the real CIFAR-10 statistics, computed over **50,000 generated samples**; lower is better. The sampler is held fixed at a strong deterministic setting (a 2nd-order solver, **35 network evaluations per image**, NFE = 35), as are the dataset and the FID protocol. The only open design is the **training of the denoiser**: how the network is parameterized as a function of noise level, what regression target and per-noise-level loss weighting it is trained against, how the training noise levels are sampled, and how the data is augmented. The question is how low the FID can be pushed on CIFAR-10 — in both class-conditional and unconditional settings — by redesigning *how the denoiser is trained*, not by changing the sampler.

## Prior art / Background / Baselines

**Forward process and denoising view.** A datapoint $\mathbf{y}$ is corrupted by adding i.i.d. Gaussian noise of standard deviation $\sigma$: $\mathbf{x} = \mathbf{y} + \sigma\boldsymbol{\epsilon}$, $\boldsymbol{\epsilon}\sim\mathcal{N}(0,\mathbf{I})$. Training reduces to fitting a denoiser $D(\mathbf{x};\sigma)\approx\mathbb{E}[\mathbf{y}\mid\mathbf{x}]$; the score of the noisy marginal is $\nabla_{\mathbf{x}}\log p(\mathbf{x};\sigma)=(D(\mathbf{x};\sigma)-\mathbf{x})/\sigma^2$. The standard formulations use either a **variance-preserving (VP)** chain, where the signal is rescaled so the noisy marginal keeps unit variance, or a **variance-exploding (VE)** chain, where clean signal is kept and ever-larger noise is added.

**Across-noise-level conditioning.** A single network must serve every noise level. At tiny $\sigma$ the input is almost the clean image and the denoiser's job is nearly the identity; at huge $\sigma$ the input is almost pure noise and the conditional mean $\mathbb{E}[\mathbf{y}\mid\mathbf{x}]$ has large variance. The magnitude of the ideal target, the magnitude of the network's input, the irreducible error, and the natural loss scale all vary across $\sigma$. Any training recipe makes — explicitly or implicitly — four coupled decisions about this: (i) how the raw network output is mapped to the denoiser; (ii) what regression target the loss compares against; (iii) how the per-$\sigma$ loss terms are weighted; (iv) which $\sigma$ values are drawn during training.

**DDPM.** Trains a noise predictor $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)$ on a VP chain with a linear $\beta$ schedule, using an unweighted $\boldsymbol{\epsilon}$-MSE and fixed reverse-process variances.

**Improved DDPM.** Replaces the linear schedule with a cosine schedule and learns the reverse-process variance per coordinate through a hybrid objective with importance-sampled timesteps.

**ADM / guided-diffusion.** Holds the diffusion training recipe roughly fixed and invests effort in the denoiser architecture — a heavily tuned U-Net with attention at several resolutions, BigGAN-style residual blocks, residual connections rescaled by $1/\sqrt2$, and adaptive group normalization (AdaGN) that injects the (timestep, class) embedding as a learned per-channel scale and shift.

## Fixed substrate / Code framework

The substrate is a from-scratch diffusion trainer. The U-Net backbones (a DDPM++/score-SDE-style net and an ADM-style net) already exist and are not the object of design here. The sampler, the dataset, and the FID protocol are also fixed. The trainer wraps a raw network $F_\theta$ in three pieces: a **preconditioning** module that, given a noisy image $\mathbf{x}$ and its noise level $\sigma$, returns the denoised estimate $D(\mathbf{x};\sigma)$; a **loss** object that draws a noise level, corrupts a clean image, queries the denoiser, and returns a per-pixel loss; and an optional **augmentation** pipe that perturbs the clean image before noising.

## Editable interface

What is open is the *training design* — the slots below, written in pre-method terms with empty bodies.

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

Every rung is a choice of how to fill the three slots — the preconditioning $D(\mathbf{x};\sigma)$, the loss (noise distribution, weighting, target), and the augmentation — and nothing else about the sampler, dataset, or FID protocol changes.

## Evaluation settings

CIFAR-10 at $32\times32$, both **class-conditional** and **unconditional**. Quality is FID against the CIFAR-10 reference statistics, computed from **50,000 generated images** (the standard protocol; FID is sensitive to sample count and seed, so the figure is taken as the minimum over a few independent 50k draws). Sampling uses the same fixed strong deterministic solver at **NFE = 35** network evaluations per image for every training recipe, so differences in FID are attributable to the *training* of the denoiser and not to the sampler. For broader generalization, the same training-design changes are also evaluated on a class-conditional model at $64\times64$ resolution where a strong pre-trained architecture already exists, again by FID.
