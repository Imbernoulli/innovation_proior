# Context: scalable Gaussian-bottleneck image modeling

## Research question

We have a large i.i.d. collection of images `x^(i)` and want to fit a directed latent-variable model with a simple continuous prior `p(z)` and a neural conditional distribution `p_theta(x|z)`. The model should support three operations at once: learn the generative parameters from data, infer a compact code for a new image quickly, and generate or reconstruct images from codes.

The hard part is the integral over the latent variable. Once `p_theta(x|z)` is a nonlinear neural network, the marginal likelihood

```text
p_theta(x) = integral p(z) p_theta(x|z) dz
```

is generally unavailable in closed form. The posterior also depends on that same marginal,

```text
p_theta(z|x) = p_theta(x|z) p(z) / p_theta(x),
```

so exact posterior inference is unavailable too. The desired training procedure therefore has to avoid exact marginal and posterior evaluations, work with minibatches, and update the encoder and decoder by ordinary backpropagation through the fixed image model.

## Background

The usual probabilistic goal is maximum likelihood or MAP estimation of the generative parameters. A flexible decoder can represent complex image distributions by transforming simple latent noise into structured pixels, so expressiveness is not the central bottleneck. The bottleneck is optimization: the learning signal must pass through expectations over latent variables whose distribution changes with the encoder parameters.

A variational approximation is the standard way to make posterior inference tractable. It introduces a tractable distribution `q(z|x)` and compares it with either the true posterior or a simple prior by KL divergence. In classical settings this comparison often gives analytic coordinate updates. In a neural likelihood setting those expectations no longer have closed forms, and a separate variational parameter vector per datapoint is too expensive for a large image dataset.

The observation model fixes what a pixel reconstruction penalty means. If real-valued pixels are modeled with a fixed-variance Gaussian decoder,

```text
p_theta(x|z) = N(x; f_theta(z), sigma_x^2 I),
```

then the negative log likelihood is, up to an additive constant,

```text
(1 / (2 sigma_x^2)) * ||x - f_theta(z)||_2^2.
```

Thus squared pixel error is not just a heuristic distance; it is the likelihood term for a Gaussian decoder. If the decoder likelihood were Laplace the corresponding penalty would be absolute error, and if the pixels were binary the corresponding penalty would be Bernoulli cross entropy.

## Baselines

**Expectation-Maximization.** EM alternates an E-step that evaluates posterior expectations with an M-step that updates the model. It is not usable here because the nonlinear decoder makes `p_theta(z|x)` intractable.

**Conjugate or coordinate-ascent variational Bayes.** Mean-field coordinate updates are effective when expectations of the complete-data log density are analytic. A neural decoder breaks those analytic expectations, and per-datapoint variational parameters do not scale well to large image collections.

**Score-function stochastic gradients.** The likelihood-ratio identity can differentiate an expectation whose sampling distribution depends on parameters:

```text
grad_phi E_q[f(z)] = E_q[f(z) grad_phi log q_phi(z|x)].
```

It is unbiased and applies broadly, including to discrete latent variables, but it is high variance in continuous neural models because each sample contributes only a scalar weight `f(z)` times a score. It does not use the derivative of the decoder output with respect to the latent sample, even though backpropagation can compute that derivative.

**Wake-sleep and related recognition-network models.** These pair a generative model with an inference network, which is the right computational shape for fast coding. Their limitation is that the two phases optimize different criteria rather than one shared lower-bound objective, so the encoder and decoder updates can point at different targets.

**Monte Carlo EM and per-datapoint MCMC.** Sampling an approximate posterior chain for each image can be accurate but is too slow for online or minibatch training at image-dataset scale.

**Plain and regularized autoencoders.** A deterministic encoder-decoder trained only by reconstruction loss gives fast reconstructions, but it is not by itself a normalized latent-variable model with a simple sampling procedure. Sparse, denoising, and contractive variants add useful regularizers, but their weights are hand-set rather than dictated by the probabilistic model.

## Evaluation settings

The relevant experimental setting is an image reconstruction and generation problem with continuous latent codes. Historical benchmarks include MNIST and Frey Face; a modern reconstruction harness may use CIFAR-10-sized RGB images. The useful objective-level diagnostics are lower-bound estimates and, when possible, marginal-likelihood estimates in very low latent dimension. The useful reconstruction diagnostics are reconstruction FID, PSNR, and SSIM, all computed under an identical architecture, optimizer, schedule, and data pipeline across candidate objectives.

The training protocol is minibatch stochastic optimization. Each image is encoded into a diagonal Gaussian bottleneck distribution, a latent code is sampled, the decoder reconstructs the image, and a scalar objective supplies the gradient for both encoder and decoder. The method to be filled in is only that scalar objective; the encode-sample-decode path is already part of the harness.

## Code framework

The local code substrate is a convolutional encoder-decoder with a Gaussian bottleneck. The encoder returns a `posterior` object with `posterior.mean`, `posterior.logvar`, `posterior.sample()`, and `posterior.kl()`. The local `DiagonalGaussianDistribution` implementation samples with

```python
std = torch.exp(0.5 * logvar)
z = mean + std * eps
```

where `eps` is standard normal noise on the same device and dtype as the parameters. With no comparison distribution supplied, its `kl()` method returns the positive per-sample divergence to the standard normal prior:

```python
0.5 * torch.sum(mean.pow(2) + var - 1.0 - logvar, dim=[1, 2, 3])
```

The objective slot receives the already-decoded reconstruction, the target image, the posterior object, and the current step, and must return a scalar tensor plus metrics.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TrainingObjective(nn.Module):
    """Scalar loss for the fixed Gaussian-bottleneck image model.

    forward() receives:
        recon:     [B, 3, 32, 32] reconstructed images
        target:    [B, 3, 32, 32] original images
        posterior: diagonal Gaussian bottleneck distribution exposing
                   mean, logvar, sample(), and kl()
        step:      current training step
    and returns (loss_tensor, metrics_dict).
    """

    def __init__(self, device):
        super().__init__()
        pass

    def forward(self, recon, target, posterior, step):
        pass


def train(model, objective, data_loader, optimizer):
    for step, x in enumerate(data_loader):
        recon, posterior = model(x)
        loss, metrics = objective(recon, x, posterior, step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
