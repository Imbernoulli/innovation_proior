# Context

## Research question

How can an autoencoder-style generative model learn a *useful* latent representation — one that keeps the high-level, semantically important structure of the data (objects in images, phonemes and words in speech, the gist of a scene) rather than spending its capacity on local noise and imperceptible detail — while still being trainable by gradient descent and competitive on maximum likelihood?

## Background

**Variational autoencoders.** The variational autoencoder (Kingma & Welling, 2013; Rezende, Mohamed & Wierstra, 2014) frames an autoencoder as approximate inference in a latent-variable model. There is a prior `p(z)`, a decoder likelihood `p(x|z)`, and an encoder/inference network `q(z|x)` that approximates the true posterior. Training maximizes the evidence lower bound

`log p(x) ≥ E_{q(z|x)}[log p(x|z)] − KL(q(z|x) || p(z))`.

The first term is a reconstruction term; the second pulls the approximate posterior toward the prior and acts as a regularizer. Typically `q(z|x)` and `p(z)` are diagonal Gaussians, which is what makes the reparameterization trick applicable: sampling `z = μ(x) + σ(x)·ε` with `ε∼N(0,I)` moves the randomness off the parameters so that gradients of the reconstruction term flow into the encoder with low variance. Extensions enrich the prior and posterior — autoregressive priors/posteriors (Gregor et al., 2013), normalizing flows (Rezende & Mohamed, 2015; Dinh et al., 2016), inverse-autoregressive posteriors (Kingma et al., 2016) — but they keep the latents continuous.

**Posterior collapse.** A well-documented failure mode appears when the decoder `p(x|z)` is made very expressive (an autoregressive decoder such as an LSTM, a dilated-convolutional model, or a PixelCNN). The KL term `KL(q(z|x)||p(z))` is minimized (driven to zero) by setting `q(z|x)=p(z)` — i.e. by making the posterior independent of `x`, so the latent carries no information. This was observed for sentence VAEs with LSTM decoders (Bowman et al., 2015), for dilated-convolutional text decoders (Yang et al., 2017), and analyzed directly in the variational-lossy-autoencoder study (Chen et al., 2016), which argues that with a sufficiently strong decoder the model prefers to push information *out* of the latent.

**The discreteness/differentiability obstacle.** A discretization step — pick the nearest symbol, take an argmax/argmin, sample a category — is piecewise constant, so its gradient with respect to its input is zero almost everywhere. Backpropagation through it delivers no signal to whatever produced the input. Two families of workarounds existed. (i) *Score-function (REINFORCE-style) estimators*: differentiate the expectation by sampling, which is unbiased but high variance, requiring variance-reduction machinery. (ii) *Straight-through estimation* (Bengio, Léonard & Courville, 2013): on the forward pass use the hard discrete value, but on the backward pass pretend the discretization was the identity and copy the gradient straight through. It is biased but low variance and simple.

**Autoregressive models over discrete variables.** Even though discrete *latents* were hard, autoregressive models over discrete *observations* were strong and mature: PixelRNN/PixelCNN for images (van den Oord et al., 2016) factorize `p(x) = Π_i p(x_i | x_{<i})` with masked convolutions over a discretized pixel grid, and WaveNet (van den Oord et al., 2016) does the same for raw audio with dilated causal convolutions. These give expressive, tractable distributions over discrete sequences/grids.

**Vector quantization.** Vector quantization is a classical dictionary-learning/clustering idea: maintain a finite codebook of prototype vectors and represent any input vector by the index of its nearest prototype. The codebook that minimizes squared reconstruction error is exactly the k-means solution — each prototype is the mean of the vectors assigned to it. This gives a principled, non-parametric way to turn a continuous vector into a discrete symbol.

## Baselines

**Standard (continuous) VAE.** Encoder outputs a Gaussian `q(z|x)=N(μ(x),σ²(x))`; sample with the reparameterization trick; decode; maximize the ELBO with a Gaussian/normalizing-flow prior. Continuous latents with low-variance gradients.

**NVIL (Mnih & Gregor, 2014).** Trains discrete-latent belief networks by optimizing the single-sample variational bound with a score-function (REINFORCE) gradient estimator, plus learned baselines and variance-reduction tricks.

**VIMCO (Mnih & Rezende, 2016).** Optimizes a *multi-sample* (importance-weighted, à la Burda et al., 2015) objective for discrete latents, using the other samples in the batch as a per-sample baseline to cut variance.

**Gumbel-softmax / Concrete relaxation (Jang et al., 2016; Maddison et al., 2016).** Replaces the discrete categorical sample with a continuous relaxation controlled by a temperature, so the reparameterization trick applies; the temperature is annealed toward a hard categorical.

**Soft-to-hard vector quantization for compression (Agustsson et al., 2017).** A continuous relaxation of vector quantization annealed over training to a hard clustering, used for learned image/network compression. The reported recipe trains an autoencoder first, then applies VQ to the encoder activations, then fine-tunes the whole network with the soft-to-hard relaxation at a small learning rate.

**Image-compression scalar quantization (Theis et al., 2017).** Uses scalar quantization of activations before arithmetic coding for lossy image compression. Related in spirit (quantize encoder activations) but scalar, and aimed at compression rather than representation learning / generation.

## Evaluation settings

- **Datasets.** CIFAR-10 (32×32 natural images) for the likelihood comparison against continuous VAEs; ImageNet downsampled to 128×128 and 84×84 frames from the DeepMind Lab environment for high-resolution images and for action-conditional video; raw speech from VCTK (109 speakers) and a larger 460-speaker corpus / LibriSpeech for audio, with ground-truth phoneme sequences available (used only as an external probe, never for training).
- **Metrics.** Negative log-likelihood in bits/dim for images (lower bounds reported for latent-variable models); reconstruction quality; for the unsupervised-speech probe, accuracy of a fixed mapping from discrete latent values to the 41 phoneme classes against a random-latent chance baseline; qualitative coherence of samples and of speaker conversion.
- **Protocol.** Train encoder/decoder by gradient descent (Adam, learning rate 2e-4, batch size 128, ~250k steps for the CIFAR comparison). A common encoder/decoder backbone of strided convolutions and residual blocks is shared across the latent-variable models being compared, varying only the latent capacity (number of latents; for discrete models also the alphabet/codebook size K). If the missing bottleneck supplies a grid or sequence of discrete codes, the existing autoregressive machinery can be used to model those codes for generation.

## Code framework

The pieces below already exist before the method: a convolutional encoder/decoder, a reconstruction loss, an optimizer, a training loop, and a prior model that can score or sample discrete symbol grids/sequences. What does not yet exist is the bottleneck that turns the encoder's continuous output into a discrete latent and back -- that is the one empty slot.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualStack(nn.Module):
    """Stack of residual blocks: ReLU, 3x3 conv, ReLU, 1x1 conv (existing primitive)."""
    def __init__(self, in_ch, hidden, num_res):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(in_ch, hidden, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden, in_ch, 1),
            ) for _ in range(num_res)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return F.relu(x)


class Encoder(nn.Module):
    """Strided convs down to a feature map (existing primitive)."""
    def __init__(self, in_ch, hidden, latent_dim, num_res):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 4, stride=2, padding=1),
            ResidualStack(hidden, hidden, num_res),
            nn.Conv2d(hidden, latent_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """Residual blocks then transposed convs back to image (existing primitive)."""
    def __init__(self, latent_dim, hidden, out_ch, num_res):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_dim, hidden, 3, padding=1),
            ResidualStack(hidden, hidden, num_res),
            nn.ConvTranspose2d(hidden, hidden, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(hidden, out_ch, 4, stride=2, padding=1),
        )

    def forward(self, z):
        return self.net(z)


class LatentBottleneck(nn.Module):
    """The slot the method fills in: map the encoder's continuous feature map to a
    latent the decoder can consume, and return whatever extra training signal the
    bottleneck needs. Pre-method this is just a placeholder."""
    def __init__(self, latent_dim):
        super().__init__()
        # TODO: the contribution goes here.
        pass

    def forward(self, z_e):
        # TODO: turn the continuous z_e into the latent passed to the decoder,
        # and return any auxiliary loss the bottleneck contributes.
        raise NotImplementedError


class Model(nn.Module):
    def __init__(self, in_ch, hidden, latent_dim, num_res):
        super().__init__()
        self.encoder = Encoder(in_ch, hidden, latent_dim, num_res)
        self.bottleneck = LatentBottleneck(latent_dim)
        self.decoder = Decoder(latent_dim, hidden, in_ch, num_res)

    def forward(self, x):
        z_e = self.encoder(x)
        z, aux_loss = self.bottleneck(z_e)   # TODO: defined once the bottleneck exists
        x_rec = self.decoder(z)
        return x_rec, aux_loss


def reconstruction_loss(x_rec, x):
    return F.mse_loss(x_rec, x)


def train_step(model, x, optimizer):
    optimizer.zero_grad()
    x_rec, aux_loss = model(x)
    loss = reconstruction_loss(x_rec, x) + aux_loss
    loss.backward()
    optimizer.step()
    return loss


def fit_prior_over_latents(latent_codes):
    """Fit an autoregressive model over a supplied grid/sequence of discrete
    latents. Masked-convolutional and dilated-causal models already exist;
    pre-method, the missing piece is how those latents are produced."""
    # TODO
    pass
```
