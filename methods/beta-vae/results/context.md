# Context

## Research question

How can a model learn, with no supervision, an *interpretable, factorised* latent representation of image data — one where each latent unit responds to a single underlying generative factor of the world (position, scale, rotation, lighting, identity, …) and stays invariant to the others — and how can the degree of such *disentanglement* be measured so that models can be compared and tuned?

The world's images are generated from a small set of independent factors of variation; a representation that recovers them in separate, axis-aligned coordinates would generalise across tasks, support transfer and zero-shot inference (recombining known factors into unseen configurations), and enable novelty detection. The setting is *unsupervised*: a freshly initialised learner faces complex data with no a-priori knowledge of how many generative factors exist or what they are, and little-to-no labelling to discover them. Two sub-problems are in play together: a learning pressure that encourages each latent to capture one independent factor, and a way to *quantify* disentanglement so that methods can be compared.

## Background

**The generative-factor view.** Assume the data is produced by a true simulator from a set of ground-truth factors: a subset of conditionally *independent*, interpretable factors `v ∈ ℝ^K` (object shape, position, scale, rotation, lighting, …) and a subset of conditionally *dependent* factors `w ∈ ℝ^H`, with `p(x | v, w) = Sim(v, w)`. A *disentangled* representation (Bengio et al., 2013) is one whose single latent units are sensitive to changes in single generative factors while remaining relatively invariant to changes in the others. The aim is a latent `z ∈ ℝ^M` (with `M ≥ K`) whose inference distribution `q(z|x)` captures the independent factors `v` in separate coordinates, leaving the dependent factors `w` entangled in some other subset.

**Why disentanglement helps.** Knowledge about one factor then generalises to novel configurations of the others; downstream tasks become learnable with simple (even linear) decision rules; and the representation can be reused across tasks (transfer), recombined for unseen inputs (zero-shot), or used to flag inputs that don't fit the learned factors (novelty).

**The state of unsupervised factor learning.** Earlier approaches typically operate with a-priori knowledge of the number and/or nature of the generative factors, or on simple datasets like FreyFaces/MNIST. Independent-component / principal-component methods (ICA, PCA) project data onto independent or uncorrelated bases; the recovered components are statistically independent or uncorrelated rather than aligned to the interpretable generative factors.

**The variational autoencoder substrate.** The VAE (Kingma & Welling, 2013; Rezende, Mohamed & Wierstra, 2014) provides a scalable, stable, unsupervised deep generative model with an inference network. It posits a prior `p(z)`, a decoder `p_θ(x|z)`, and an encoder `q_φ(z|x)`, and maximises the evidence lower bound
`E_{q_φ(z|x)}[log p_θ(x|z)] − D_KL(q_φ(z|x) ‖ p(z))`,
trained by the reparameterisation trick `z = μ(x) + σ(x)·ε`, `ε ∼ N(0, I)`. The first term reconstructs; the second pulls the posterior toward the prior.

## Baselines

**Standard VAE (Kingma & Welling, 2013; Rezende et al., 2014).** Maximise the ELBO above with `q_φ(z|x) = N(μ(x), σ²(x))` and `p(z) = N(0, I)`; the KL has a closed form for diagonal Gaussians. Scalable, stable, unsupervised, with an inference network.

**InfoGAN (Chen et al., 2016).** Augments a GAN with a recognition network and an objective that maximises the mutual information between a subset of the noise variables and the recognition output, so those noise variables come to control interpretable factors — fully unsupervised.

**DC-IGN (Kulkarni et al., 2015).** An inverse-graphics autoencoder trained *semi-supervised*: minibatches are structured so that exactly one generative factor varies, teaching specific latents to encode specific factors.

**PCA / ICA.** Linear decompositions onto uncorrelated / independent components.

## Evaluation settings

- **Datasets.** A synthetic dataset of 2D shapes — 737,280 binary 64×64 images, the Cartesian product of shape ∈ {heart, oval, square} (3 values), position-X (32), position-Y (32), scale (6), and rotation (40 values over 2π) — chosen because it has exactly five known independent generative factors and no confounds, giving ground truth for an objective disentanglement comparison. Also CelebA, 3D chairs, and 3D faces for qualitative inspection of learned factor traversals.
- **Metrics.** A quantitative disentanglement score; qualitative latent-traversal inspection (fix all latents but one, sweep that one over a range, view the decoded images).
- **Protocol.** Train encoder/decoder by gradient descent. On the 2D-shapes encoder/decoder of fully-connected layers (FC 1200, 1200; 10 latents; Adagrad lr 1e-2; Bernoulli decoder). On CelebA/chairs/faces, a convolutional encoder (four 32-channel 4×4 stride-2 convs, then FC 256; 32 latents) with a mirror-image deconvolutional decoder and Adam at 1e-4. Prior `p(z) = N(0, I)` throughout.

## Code framework

Pre-existing primitives: PyTorch `nn.Module`, `Conv2d` / `ConvTranspose2d`, `Linear`, the Adam/Adagrad optimisers, and the closed-form Gaussian-vs-`N(0,I)` KL. The VAE scaffold — encoder to `(μ, log σ²)`, reparameterised sample, decoder, ELBO loss — already exists.

```python
import torch
from torch import nn
import torch.nn.functional as F

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = torch.randn_like(std)
    return mu + std * eps                      # z = mu + sigma * eps

class VAE(nn.Module):
    def __init__(self, z_dim=10, nc=3):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),  nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(256, z_dim * 2),             # outputs mu and logvar
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256), nn.Unflatten(1, (256, 1, 1)), nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4), nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),
        )

    def forward(self, x):
        dist = self.encoder(x)
        mu, logvar = dist[:, :self.z_dim], dist[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        return self.decoder(z), mu, logvar

def reconstruction_loss(x, x_recon, distribution="bernoulli"):
    B = x.size(0)
    if distribution == "bernoulli":
        return F.binary_cross_entropy_with_logits(x_recon, x, reduction="sum") / B
    return F.mse_loss(x_recon, x, reduction="sum") / B

def kl_divergence(mu, logvar):
    # closed-form KL( N(mu, sigma^2) || N(0, I) )
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return klds.sum(1).mean(0)

def vae_objective(recon_loss, total_kld):
    # TODO: combine the reconstruction and KL terms into the training objective.
    pass
```
